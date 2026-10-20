"""Additive privileged grasp servo and conservative execution filter; no training/rendering.

This is an intervention on a frozen BC policy, NOT a pure BC result. Collision
queries use cooked native environment geometry and USD-derived link spheres.
No USD, robot state, Cup pose, original initializer or audit is rewritten.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from training.privileged import ArmPolicy as FrozenPolicy
from training.constants import ARM_NAMES


GRASP_PHASES = frozenset(('settle_open_grasp', 'raise_clear_of_counter',
    'walk_with_raised_hand', 'extend_above', 'lower_open', 'reach', 'grasp', 'close'))


@dataclass(frozen=True)
class Parameters:
    kp: float = 0.65
    max_corr_m: float = 0.006
    max_ik_joint_correction_rad: float = 0.10
    alignment_tolerance_m: float = 0.010
    alignment_rotation_rad: float = 0.20
    slew_max_rad: float = 0.008
    collision_shrink: float = 0.5
    collision_trials: int = 9
    collision_path_samples: int = 3
    tilt_rate_deg_s: float = 45.0
    closing_speed_m_s: float = 0.06
    balance_envelope_rad: float = 0.75
    max_arm_extension_m: float = 0.78
    stalled_controls: int = 100
    control_dt: float = 0.02
    ik_starts: int = 1
    ik_orientation_weight: float = 0.12
    baseline_tol_m: float = 0.001
    escape_max_steps: int = 50
    sphere_mode: str = 'circumsphere'
    halfaxis_margin_m: float = 0.01
    init_probe_steps: int = 3
    init_max_physics_steps: int = 40
    init_clear_steps: int = 10
    init_max_retreat_m: float = 0.012
    init_retreat_step_m: float = 0.001
    init_penetration_m: float = 0.001
    init_impulse_Ns: float = 0.02
    init_clear_tilt_deg: float = 2.0
    init_tip_gate_deg: float = 3.0
    close_tilt_deg: float = 3.0
    close_tilt_rate_deg_s: float = 5.0
    close_wait_controls: int = 20
    inactive_arm_envelope_rad: float | None = None


def baseline_exemption(position, baseline_position, collider_center, tolerance):
    """Exact plan24 unnormalized dot formula (m^2, legacy tolerance name says m)."""
    dot = float(np.dot(np.asarray(position)-baseline_position,np.asarray(baseline_position)-collider_center))
    return dot >= -tolerance, dot


def advance_escape(previous_count, status, maximum):
    count = previous_count+1 if status=='ESCAPE_BASELINE_CONTACT' else 0
    if count > maximum:
        raise RuntimeError('SAFETY_ESCAPE_TIMEOUT')
    return count


def close_gate(aligned, tilt_deg, rate_deg_s, params):
    return bool(aligned and tilt_deg<=params.close_tilt_deg and abs(rate_deg_s)<=params.close_tilt_rate_deg_s)


def gate_finger_targets(targets,phase,allowed):
    targets=np.asarray(targets,dtype=float).copy()
    if phase in ('close','grasp') and not allowed:
        targets[:]=0.
    return targets


def servo_correct(action, cup_rel_palm, phase):
    """IK correction added to BC, bounded in both Cartesian and joint spaces.

    cup_rel_palm is a measured-state context. goal_world uses the calibrated
    grasp-center offset, not an incorrect assumption that Cup center = palm.
    predicted target FK is corrected, so residual BC displacement cannot undo
    alignment unnoticed. The underlying BC target is retained in the receipt.
    """
    c = cup_rel_palm
    params = c['params']
    action = np.asarray(action, dtype=float).copy()
    report = dict(enabled=phase in GRASP_PHASES, correction_norm_rad=0., correction_m=[0.,0.,0.])
    if not report['enabled']:
        return action, report
    from pipeline.ue_isaac_pipeline.arm_kinematics import pose_matrix
    model, ids = c['model'], c['active_ids']
    # The first pilot exposed unreachable raw BC targets (22cm error). Apply
    # the SAME disclosed execution slew bound before linearizing IK, rather
    # than solving around an action the filter would never execute.
    if 'previous_target' in c:
        previous = np.asarray(c['previous_target'])
        action[ids] = previous[ids] + np.clip(action[ids]-previous[ids],
            -params.slew_max_rad, params.slew_max_rad)
        report['ik_basis'] = 'BC proposal after identical per-step slew bound'
    world = model.forward(action[ids]) @ c['torso_world']
    error = np.asarray(c['goal_world']) - world[3,:3]
    correction = params.kp * error
    correction *= min(1., params.max_corr_m / max(float(np.linalg.norm(correction)), 1e-12))
    target_world = pose_matrix(world[3,:3]+correction, c['goal_rotation'])
    solved, ik = model.solve(target_world @ np.linalg.inv(c['torso_world']), action[ids],
        starts=params.ik_starts, orientation_weight=params.ik_orientation_weight, posture_weight=0.002)
    dq = np.clip(solved-action[ids], -params.max_ik_joint_correction_rad, params.max_ik_joint_correction_rad)
    # Joint clipping can distort an IK solution. Bound the achieved Cartesian
    # displacement too, not just the requested correction vector.
    for _ in range(16):
        achieved = (model.forward(action[ids]+dq) @ c['torso_world'])[3,:3]-world[3,:3]
        if np.linalg.norm(achieved) <= params.max_corr_m+1e-10:
            break
        dq *= .5
    else:
        dq[:] = 0.
    report['achieved_correction_m'] = ((model.forward(action[ids]+dq) @ c['torso_world'])[3,:3]-world[3,:3]).tolist()
    action[ids] += dq
    report.update(correction_norm_rad=float(np.linalg.norm(dq)), correction_m=correction.tolist(),
        predicted_goal_error_m=float(np.linalg.norm(error)), measured_goal_error_m=c['measured_error_m'], ik=ik)
    return action, report


def safety_filter(action, prev_action, scene):
    """Joint envelope/slew, closing-speed/tilt guard, then real-geometry overlaps.

    Failure to find any clear candidate returns HOLD_UNCERTIFIED, never a false
    collision-free claim. Three interpolation samples are conservative discrete
    checks, not a continuous full-body certificate.
    """
    p = scene.params
    previous = np.asarray(prev_action, dtype=float)
    requested = np.asarray(action, dtype=float)
    if requested.shape != previous.shape or not np.isfinite(requested).all():
        raise RuntimeError('Nonfinite or malformed safe-execution action')
    lo = np.maximum(scene.limits[:,0], scene.nominal-p.balance_envelope_rad)
    hi = np.minimum(scene.limits[:,1], scene.nominal+p.balance_envelope_rad)
    if p.inactive_arm_envelope_rad is not None and hasattr(scene,'inactive_ids'):
        ids=scene.inactive_ids
        lo[ids]=np.maximum(lo[ids],scene.nominal[ids]-p.inactive_arm_envelope_rad)
        hi[ids]=np.minimum(hi[ids],scene.nominal[ids]+p.inactive_arm_envelope_rad)
    bounded = np.clip(requested, lo, hi)
    candidate = previous + np.clip(bounded-previous, -p.slew_max_rad, p.slew_max_rad)
    speed = scene.closing_speed(candidate, previous)
    factor = min(1., p.closing_speed_m_s / max(speed, 1e-12)) if scene.grasp_phase else 1.
    tilt_guard = scene.grasp_phase and scene.tilt_rate > p.tilt_rate_deg_s
    if tilt_guard:
        factor = 0.
    candidate = previous + factor*(candidate-previous)
    attempts = []
    delta = candidate-previous
    for attempt in range(p.collision_trials+1):
        alpha = p.collision_shrink**attempt if attempt < p.collision_trials else 0.
        proposed = previous+alpha*delta
        collisions = scene.check_path(previous, proposed, p.collision_path_samples)
        exemptions=list(getattr(scene,'path_exemptions',[]))
        attempts.append(dict(alpha=alpha, collisions=collisions, exemptions=exemptions))
        if not collisions:
            return proposed, dict(status='CLEAR_DISCRETE_LINK_SPHERES',
                exempted_baseline=bool(exemptions),baseline_exemptions=exemptions,
                slew_limited=bool(np.max(abs(bounded-previous)) > p.slew_max_rad),
                envelope_limited=bool(np.max(abs(bounded-requested)) > 1e-10),
                tilt_guard=bool(tilt_guard), measured_tilt_rate_deg_s=float(scene.tilt_rate),
                requested_closing_speed_m_s=float(speed), closing_factor=float(factor),
                shrink_alpha=float(alpha), attempts=attempts,
                maximum_target_delta_rad=float(np.max(abs(proposed-previous))))
    # New geometry blockers always prohibit escape. Unknown blockers/envelope
    # failures are also hard blockers, never silently treated as old contact.
    blocked=[x for a in attempts for x in a['collisions']]
    all_baseline=bool(blocked) and all(x.get('baseline_approaching') is True for x in blocked)
    if all_baseline:
        candidates=[(scene.escape_score(previous+a['alpha']*delta),a) for a in attempts if a['alpha']>0]
        score,chosen=max(candidates,key=lambda x:x[0])
        proposed=previous+chosen['alpha']*delta
        return proposed,dict(status='ESCAPE_BASELINE_CONTACT',attempts=attempts,
            minimum_baseline_center_distance_m=float(score),shrink_alpha=chosen['alpha'],
            exempted_baseline=False,tilt_guard=bool(tilt_guard),
            maximum_target_delta_rad=float(np.max(abs(proposed-previous))))
    status='HOLD_UNCERTIFIED' if any(x.get('new_collider') for x in blocked) else 'BLOCKED_NONCONTACT_CONSTRAINT'
    # Legacy mock adapters with no typed geometry blockers remain hard holds.
    if not hasattr(scene,'baseline'):
        status='HOLD_UNCERTIFIED'
    return previous.copy(), dict(status=status, attempts=attempts,
        tilt_guard=bool(tilt_guard), measured_tilt_rate_deg_s=float(scene.tilt_rate),
        requested_closing_speed_m_s=float(speed), closing_factor=float(factor),
        maximum_target_delta_rad=0.)


class NativeScene:
    """Live PhysX query adapter. Never force-reloads the running physics scene."""
    def __init__(self, robot, stage, asset, params, nominal, limits, output):
        import carb
        import omni.physx
        from pxr import Usd, UsdGeom, UsdPhysics, Gf
        from pipeline.ue_isaac_pipeline.arm_kinematics import UsdArmKinematics
        self.carb = carb
        self.query = omni.physx.get_physx_scene_query_interface()
        self.params, self.nominal, self.limits = params, nominal.copy(), limits
        self.robot = robot
        self.arm_ids = [robot.joint_names.index(n) for n in ARM_NAMES]
        self.specs = []
        self.baseline = {}
        self.collider_centers = {}
        asset_stage = Usd.Stage.Open(str(asset))
        cache = UsdGeom.BBoxCache(0, ['default','render','proxy','guide'], False, True)
        names = [side+'_'+suffix for side in ('left','right')
            for suffix in ('shoulder_yaw_link','elbow_link','wrist_yaw_link','hand_palm_link',
                'hand_index_0_link','hand_index_1_link','hand_middle_0_link','hand_middle_1_link','hand_thumb_2_link')]
        bodies = {x.GetName(): x for x in asset_stage.Traverse() if x.HasAPI(UsdPhysics.RigidBodyAPI)}
        for name in names:
            link = bodies[name]
            bounds = Gf.Range3d()
            colliders = []
            for prim in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):
                if prim != link and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    box = cache.ComputeRelativeBound(prim,link).ComputeAlignedRange()
                    if not box.IsEmpty():
                        bounds.UnionWith(box)
                        colliders.append(str(prim.GetPath()))
            assert not bounds.IsEmpty(), 'Missing collision geometry: '+name
            lo,hi = np.array(bounds.GetMin()),np.array(bounds.GetMax())
            radius = float(np.linalg.norm(hi-lo)/2)
            if params.sphere_mode=='halfaxis' and ('hand_' in name):
                radius=float(np.max(hi-lo)/2+params.halfaxis_margin_m)
            elif params.sphere_mode not in ('circumsphere','halfaxis'):
                raise ValueError('Unknown sphere_mode')
            assert 0 < radius < .5
            model = UsdArmKinematics(asset,end=name)
            self.specs.append(dict(name=name, model=model, center=(lo+hi)/2, radius=radius,
                ids=[robot.joint_names.index(n) for n in model.joint_names], collision_prims=colliders))
        # Static furniture centers are cached once; per-control baseline contact
        # membership and link centers are refreshed separately from live state.
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI) and not str(prim.GetPath()).startswith('/World/G1') and not str(prim.GetPath()).startswith('/World/Environment/Source/cup_V1'):
                box=cache.ComputeWorldBound(prim).ComputeAlignedRange()
                if not box.IsEmpty():
                    self.collider_centers[str(prim.GetPath())]=(np.array(box.GetMin())+np.array(box.GetMax()))/2
        floor = stage.GetPrimAtPath('/World/Environment/Source/diban')
        box = cache.ComputeWorldBound(floor).ComputeAlignedRange()
        point = (np.array(box.GetMin())+np.array(box.GetMax()))/2
        hits = self.hits(.03,point)
        assert hits, 'Native PhysX floor positive control empty; refuse uncooked queries'
        (Path(output)/'collision-method.json').write_text(json.dumps(dict(
            method='Native PhysX overlap_sphere, 18 USD collision-geometry circumspheres, three joint-path samples; self/Cup excluded',
            limitations='Discrete upper-limb spheres; static furniture AABB centers cached at startup; centroid direction is not signed surface distance; no continuous whole-body certificate. Baseline membership refresh cannot certify contacts induced by locomotion.',
            sphere_mode=params.sphere_mode,baseline_dot_formula='dot(p-p_baseline,p_baseline-c_center)>=-baseline_tol_m',
            baseline_dot_units='m^2; legacy parameter baseline_tol_m is not an exact1mm distance',
            cached_static_collider_centers={k:v.tolist() for k,v in self.collider_centers.items()},
            positive_control=dict(point_m=point.tolist(),hits=hits),
            link_spheres=[dict(name=s['name'],center_local_m=s['center'].tolist(),radius_m=s['radius'],
                collision_prims=s['collision_prims']) for s in self.specs]),indent=2),encoding='utf8')

    def hits(self,radius,position):
        hits = []
        def callback(hit):
            paths = [str(hit.collision),str(hit.rigid_body)]
            if not any(x.startswith('/World/G1') or x.startswith('/World/Environment/Source/cup_V1') for x in paths):
                hits.append(dict(collider=paths[0],rigid_body=paths[1]))
            return True
        self.query.overlap_sphere(float(radius),self.carb.Float3(*map(float,position)),callback,False)
        return hits

    def baseline_contacts(self):
        self.baseline={}
        for spec in self.specs:
            world=spec['model'].forward(self.full_q[spec['ids']]) @ self.torso_world
            position=(np.r_[spec['center'],1.] @ world)[:3]
            contacts={}
            for hit in self.hits(spec['radius'],position):
                path=hit['collider']
                if path not in self.collider_centers:
                    raise RuntimeError('Missing cached static collider center: '+path)
                contacts[path]=position.copy()
            self.baseline[spec['name']]=contacts
        return self.baseline

    def escape_score(self,proposed):
        q=self.full_q.copy();q[self.arm_ids]=proposed
        distances=[]
        for spec in self.specs:
            world=spec['model'].forward(q[spec['ids']]) @ self.torso_world
            position=(np.r_[spec['center'],1.] @ world)[:3]
            distances.extend(float(np.linalg.norm(position-self.collider_centers[path])) for path in self.baseline[spec['name']])
        return min(distances) if distances else float('-inf')

    def check_path(self,previous,proposed,samples):
        self.path_exemptions=[]
        blockers=[]
        for alpha in np.linspace(1./samples,1.,samples):
            q = self.full_q.copy()
            q[self.arm_ids] = previous+alpha*(proposed-previous)
            for spec in self.specs:
                world = spec['model'].forward(q[spec['ids']]) @ self.torso_world
                position = (np.r_[spec['center'],1.] @ world)[:3]
                if spec['name'].endswith('hand_palm_link') and np.linalg.norm(position-self.torso_world[3,:3]) > self.params.max_arm_extension_m:
                    blockers.append(dict(link=spec['name'],reason='arm_extension_envelope',alpha=float(alpha)))
                hits = self.hits(spec['radius'],position)
                for hit in hits:
                    path=hit['collider'];baseline=self.baseline.get(spec['name'],{})
                    if path not in baseline:
                        blockers.append(dict(link=spec['name'],alpha=float(alpha),hits=[hit],new_collider=True))
                    else:
                        exempt,dot=baseline_exemption(position,baseline[path],self.collider_centers[path],self.params.baseline_tol_m)
                        entry=dict(link=spec['name'],alpha=float(alpha),hits=[hit],baseline_dot_m2=dot,
                            exempted_baseline=exempt,baseline_approaching=not exempt)
                        (self.path_exemptions if exempt else blockers).append(entry)
        return blockers

    def closing_speed(self,candidate,previous):
        before = (self.active_model.forward(previous[self.active_ids]) @ self.torso_world)[3,:3]
        after = (self.active_model.forward(candidate[self.active_ids]) @ self.torso_world)[3,:3]
        direction = self.goal_world-self.actual_palm
        direction /= max(float(np.linalg.norm(direction)),1e-12)
        predicted = float(np.dot(after-before,direction)/self.params.control_dt)
        measured = float(np.dot(self.relative_velocity,direction))
        return max(0.,predicted,measured)


class ArmPolicy(FrozenPolicy):
    def __init__(self,checkpoint,robot,sim,stage,diagnostics,asset,parameters):
        super().__init__(checkpoint,robot,sim,stage,diagnostics)
        self.params = Parameters(**json.loads(Path(parameters).read_text(encoding='utf8')))
        nominal = robot.data.joint_pos[0].cpu().numpy()[self.joint_ids].copy()
        limits = robot.data.soft_joint_pos_limits[0].cpu().numpy()[self.joint_ids]
        self.scene = NativeScene(robot,stage,asset,self.params,nominal,limits,self.out)
        self.previous = nominal.copy()
        self.torso_id = robot.body_names.index('torso_link')
        self.previous_tilt = None
        self.previous_relative = None
        self.stalled = 0
        self.finger_close_allowed = False
        self.escape_controls = 0
        self.close_blocked_controls = 0
        (self.out/'safe-execution-parameters.json').write_text(json.dumps(asdict(self.params),indent=2),encoding='utf8')

    def safe_targets(self,robot,step,seconds,phase,cup,goal_world,goal_rotation,model,hand):
        from pipeline.ue_isaac_pipeline.arm_kinematics import pose_matrix
        raw = super().targets(robot,step,seconds,phase,cup).copy()
        s = self.scene
        torso = robot.data.body_state_w[0,self.torso_id,:7].cpu().numpy()
        palm_id = robot.body_names.index(hand+'_hand_palm_link')
        palm = robot.data.body_state_w[0,palm_id,:7].cpu().numpy()
        cup_pose = cup.data.root_state_w[0,:7].cpu().numpy()
        relative = palm[:3]-cup_pose[:3]
        relative_velocity = np.zeros(3) if self.previous_relative is None else (relative-self.previous_relative)/self.params.control_dt
        self.previous_relative = relative.copy()
        cup_matrix = pose_matrix(cup_pose[:3],cup_pose[3:])
        tilt = float(np.degrees(np.arccos(np.clip(cup_matrix[2,2],-1,1))))
        signed_rate = 0. if self.previous_tilt is None else (tilt-self.previous_tilt)/self.params.control_dt
        rate=max(0.,signed_rate)
        self.previous_tilt = tilt
        active_ids = np.arange(7) if hand=='left' else np.arange(7,14)
        s.full_q = robot.data.joint_pos[0].cpu().numpy().copy()
        s.torso_world = pose_matrix(torso[:3],torso[3:])
        s.active_model,s.active_ids = model,active_ids
        s.inactive_ids=np.arange(7,14) if hand=='left' else np.arange(7)
        s.goal_world,s.actual_palm = np.array(goal_world),palm[:3]
        s.relative_velocity,s.tilt_rate,s.grasp_phase = relative_velocity,rate,phase in GRASP_PHASES
        s.baseline_contacts()
        error = float(np.linalg.norm(s.goal_world-palm[:3]))
        rotation = pose_matrix([0,0,0],goal_rotation)
        from scipy.spatial.transform import Rotation
        rotation_error = float(np.linalg.norm(Rotation.from_matrix(rotation[:3,:3].T @ pose_matrix([0,0,0],palm[3:])[:3,:3]).as_rotvec()))
        aligned=error <= self.params.alignment_tolerance_m and rotation_error <= self.params.alignment_rotation_rad
        self.finger_close_allowed = close_gate(aligned,tilt,signed_rate,self.params)
        corrected,servo = servo_correct(raw,dict(params=self.params,model=model,active_ids=active_ids,
            torso_world=s.torso_world,goal_world=s.goal_world,goal_rotation=goal_rotation,measured_error_m=error,
            previous_target=self.previous),phase)
        safe,filtered = safety_filter(corrected,self.previous,s)
        close_blocked=phase in ('close','grasp') and not self.finger_close_allowed
        self.close_blocked_controls=self.close_blocked_controls+1 if close_blocked else 0
        if close_blocked:
            safe=self.previous.copy()
            filtered['close_gate_holds_target']=True
            filtered['maximum_target_delta_rad']=0.
        self.stalled = self.stalled+1 if np.max(abs(safe-self.previous)) < 1e-7 else 0
        row = dict(step=step,physical_seconds=step*self.params.control_dt,phase=phase,nature='BC + privileged servo + safety filter',
            bc_target=raw.tolist(),servo_target=corrected.tolist(),previous_safe_target=self.previous.tolist(),
            executed_target=safe.tolist(),measured_arm_q=s.full_q[self.joint_ids].tolist(),
            measured_palm=palm.tolist(),cup_pose=cup_pose.tolist(),goal_world=s.goal_world.tolist(),
            finger_close_allowed=bool(self.finger_close_allowed),alignment_error_m=error,
            alignment_rotation_rad=rotation_error,cup_tilt_deg=tilt,cup_tilt_rate_deg_s=signed_rate,servo=servo,filter=filtered,
            stalled_controls=self.stalled,bc_inference_calls=self.calls,
            escape_controls=self.escape_controls,close_blocked_controls=self.close_blocked_controls,
            baseline_contacts={name:{path:center.tolist() for path,center in contacts.items()} for name,contacts in s.baseline.items()})
        with (self.out/'execution.jsonl').open('a',encoding='utf8') as f:
            f.write(json.dumps(row)+'\n')
        self.previous = safe.copy()
        self.escape_controls=advance_escape(self.escape_controls,filtered['status'],self.params.escape_max_steps)
        if self.close_blocked_controls>=self.params.close_wait_controls:
            raise RuntimeError('CLOSE_GATE_BLOCKED')
        if self.stalled >= self.params.stalled_controls:
            raise RuntimeError('BASELINE_BLOCK_STALL: '+filtered['status'])
        return safe
