"""Bounded, observed initialization contact retreat; no pose resets/attachments.

The original raised-hand IK/initializer is unchanged. This microstage runs
after it, before settle/BC. Raw setup contact evidence is never erased. A Cup
already tipping cannot be restored by pretending an arm retreat succeeded.
"""
import json
from pathlib import Path

import numpy as np


class ContactObserver:
    def __init__(self):self.events=[]

    def record(self,headers,data,decode_path,time,phase):
        for header in headers:
            actors=[str(decode_path(header.actor0)),str(decode_path(header.actor1))]
            if not any('/cup_V1' in a for a in actors) or not any('/World/G1/' in a and '_hand_' in a for a in actors):
                continue
            points=[data[i] for i in range(header.contact_data_offset,header.contact_data_offset+header.num_contact_data)]
            self.events.append(dict(time=float(time),phase=phase,actors=actors,
                reports=len(points),impulse_Ns=sum(float(np.linalg.norm(tuple(c.impulse))) for c in points),
                minimum_separation_m=min([float(c.separation) for c in points],default=None)))

    def latest(self,time,dt):
        return [x for x in self.events if x['reports'] and x['time']>=time-dt-1e-7]


def retreat_step(q,palm_position,cup_center,torso_world,model,params,used_m):
    """One outward Cartesian IK correction; clamp joint slew and total FK travel."""
    q=np.asarray(q,dtype=float)
    away=np.asarray(palm_position)-np.asarray(cup_center)
    away/=max(float(np.linalg.norm(away)),1e-12)
    remaining=max(0.,params.init_max_retreat_m-used_m)
    distance=min(params.init_retreat_step_m,remaining)
    world=model.forward(q) @ torso_world
    target=world.copy();target[3,:3]+=away*distance
    solved,ik=model.solve(target @ np.linalg.inv(torso_world),q,starts=1,orientation_weight=.12,posture_weight=.002)
    dq=np.clip(solved-q,-params.slew_max_rad,params.slew_max_rad)
    for _ in range(20):
        movement=(model.forward(q+dq) @ torso_world)[3,:3]-world[3,:3]
        length=float(np.linalg.norm(movement))
        if length<=distance+1e-10 and float(np.dot(movement,away))>=-1e-12:
            break
        dq*=.5
    else:
        dq[:]=0.;movement=np.zeros(3);length=0.
    return q+dq,dict(away_direction=away.tolist(),requested_distance_m=distance,
        achieved_displacement_m=movement.tolist(),achieved_distance_m=length,
        cumulative_retreat_m=used_m+length,maximum_joint_delta_rad=float(np.max(abs(dq))),ik=ik)


def run_init_contact_resolve(robot,cup,sim,stage,asset,model,arm_ids,finger_ids,palm_id,torso_id,
        observer,params,out,local_cup_center,initial_physics_steps=2):
    import torch
    from pipeline.ue_isaac_pipeline.arm_kinematics import pose_matrix
    from training.privileged import geometric_center
    from training.safe_execute import NativeScene
    out=Path(out)
    method=out/'init-resolve-diagnostics';method.mkdir(exist_ok=False)
    nominal=robot.data.joint_pos[0].cpu().numpy()
    from training.constants import ARM_NAMES
    ids=[robot.joint_names.index(n) for n in ARM_NAMES]
    scene=NativeScene(robot,stage,asset,params,nominal[ids],robot.data.soft_joint_pos_limits[0].cpu().numpy()[ids],method)
    rows=[];used=0.;clear=0;triggered=False
    before=robot.data.body_pos_w[0,palm_id].cpu().numpy().copy()
    dt=float(sim.get_physics_dt())
    def state():
        pose=cup.data.root_state_w[0,:7].cpu().numpy()
        tilt=float(np.degrees(np.arccos(np.clip(pose_matrix(pose[:3],pose[3:])[2,2],-1,1))))
        center=geometric_center(pose,local_cup_center)
        contacts=observer.latest(float(sim.current_time),dt)
        overlaps=[]
        for spec in scene.specs:
            if not spec['name'].startswith(robot.body_names[palm_id].split('_hand_')[0]+'_hand_'):
                continue
            body=robot.body_names.index(spec['name'])
            actual=robot.data.body_state_w[0,body,:7].cpu().numpy()
            matrix=pose_matrix(actual[:3],actual[3:])
            point=(np.r_[spec['center'],1.] @ matrix)[:3]
            hits=[]
            def cb(hit):
                if '/cup_V1' in str(hit.collision) or '/cup_V1' in str(hit.rigid_body):
                    hits.append(dict(collider=str(hit.collision),rigid_body=str(hit.rigid_body)))
                return True
            scene.query.overlap_sphere(spec['radius'],scene.carb.Float3(*map(float,point)),cb,False)
            if hits:overlaps.append(dict(link=spec['name'],hits=hits))
        return dict(tilt_deg=tilt,cup_pose=pose.tolist(),cup_center=center.tolist(),contacts=contacts,
            cup_velocity_world=cup.data.root_state_w[0,7:13].cpu().tolist(),
            overlap_spheres=overlaps,sim_time=float(sim.current_time),palm=robot.data.body_state_w[0,palm_id,:7].cpu().tolist())
    def dangerous(events):
        return any((e['minimum_separation_m'] is not None and e['minimum_separation_m'] < -params.init_penetration_m)
            or e['impulse_Ns']>params.init_impulse_Ns for e in events)
    initial=state()
    # Historical raw setup contacts trigger resolution even if the new raised
    # hand is already separated; their impulse may still be tipping the Cup.
    triggered=dangerous(observer.events) or bool(initial['overlap_spheres']) or initial['tilt_deg']>params.init_tip_gate_deg
    result=dict(status='STARTING',microstage='init_contact_resolve',initial=initial,
        original_initializer_executed_first=True,initial_physics_steps=initial_physics_steps,
        fully_open_fingers='Calibrated Dex3 straight/open joint targets0, limited to native joint bounds',
        physics_dt=dt,rows=rows,maximum_retreat_m=params.init_max_retreat_m,
        limitation='Arm-only retreat cannot remove momentum already introduced by original setup contacts; Cup pose is never reset')
    failure=None
    try:
        for step in range(params.init_max_physics_steps):
            current=state()
            target=robot.data.joint_pos[0].cpu().numpy().copy()
            correction=None
            if triggered:
                torso=robot.data.body_state_w[0,torso_id,:7].cpu().numpy()
                q,correction=retreat_step(target[arm_ids],np.array(current['palm'][:3]),np.array(current['cup_center']),
                    pose_matrix(torso[:3],torso[3:]),model,params,used)
                target[arm_ids]=q;used=correction['cumulative_retreat_m']
            limits=robot.data.soft_joint_pos_limits[0].cpu().numpy()
            target[finger_ids]=np.clip(np.zeros(len(finger_ids)),limits[finger_ids,0],limits[finger_ids,1])
            robot.set_joint_position_target(torch.tensor(target,dtype=torch.float32,device=robot.device).unsqueeze(0))
            robot.write_data_to_sim();sim.step(render=False);robot.update(dt);cup.update(dt)
            after=state()
            triggered=triggered or dangerous(after['contacts']) or bool(after['overlap_spheres']) or after['tilt_deg']>params.init_tip_gate_deg
            clear=clear+1 if not after['contacts'] and not after['overlap_spheres'] and after['tilt_deg']<=params.init_clear_tilt_deg else 0
            row=dict(step=step,before=current,after=after,retreat=correction,used_retreat_m=used,
                full_joint_target=target.tolist(),measured_joints=robot.data.joint_pos[0].cpu().tolist(),
                sdk_joint_target_buffer=robot.data.joint_pos_target[0].cpu().tolist(),
                actual_palm_displacement_m=float(np.linalg.norm(np.array(after['palm'][:3])-before)),
                clear_steps=clear,triggered=triggered)
            rows.append(row)
            with (out/'init-contact-execution.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
            # B2 deliberately rejects a Cup still tilted after an outward attempt,
            # BEFORE reaching the old25-degree policy failure threshold.
            if triggered and correction is not None and after['tilt_deg']>params.init_tip_gate_deg:
                failure='INIT_CONTACT_TIP';break
            if row['actual_palm_displacement_m']>params.init_max_retreat_m+1e-6:
                failure='INIT_RESOLVE_TIMEOUT';break
            if clear>=params.init_clear_steps:
                result['status']='RESOLVED';break
            if not triggered and initial_physics_steps+step+1>=params.init_probe_steps:
                result['status']='NO_CONTACT_DETECTED';break
        else:failure='INIT_CONTACT_TIP' if rows[-1]['after']['tilt_deg']>params.init_tip_gate_deg else 'INIT_RESOLVE_TIMEOUT'
        if failure:result.update(status=failure)
        result.update(used_retreat_m=used,physics_steps=len(rows),final=rows[-1]['after'] if rows else initial,
            initial_contacts_all=observer.events,BC_inference_calls=0)
        return result
    finally:
        (out/'init-contact-resolve.json').write_text(json.dumps(result,indent=2),encoding='utf8')
