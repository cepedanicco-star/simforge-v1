"""50-D privileged-state BC: no image, mask, boxes, render product or annotator."""
import json
from pathlib import Path

import numpy as np

from .constants import ARM_NAMES, PALMS, PHASES


def network(torch):
    return torch.nn.Sequential(torch.nn.Linear(50,256),torch.nn.ReLU(),torch.nn.Linear(256,256),
        torch.nn.ReLU(),torch.nn.Linear(256,112))


def geometric_center(pose, local_center):
    from pxr import Gf
    q=Gf.Quatd(float(pose[3]),Gf.Vec3d(*map(float,pose[4:])))
    q=q.GetNormalized()
    return np.asarray(Gf.Rotation(q).TransformDir(Gf.Vec3d(*map(float,local_center))))+pose[:3]


def state_features(cup,palms,joints,fingers,phase,seconds,local_center):
    from pxr import Gf
    cq=Gf.Quatd(float(cup[3]),Gf.Vec3d(*map(float,cup[4:])))
    relatives=[]
    for palm in palms:
        pq=Gf.Quatd(float(palm[3]),Gf.Vec3d(*map(float,palm[4:])))
        inv=pq.GetInverse();relative=inv*cq
        position=Gf.Rotation(inv).TransformDir(Gf.Vec3d(*map(float,cup[:3]-palm[:3])))
        relatives.extend([*position,relative.GetReal(),*relative.GetImaginary()])
    phase_index=PHASES.index(phase) if phase in PHASES else 15
    x=np.r_[geometric_center(cup,local_center),relatives,fingers,np.eye(16)[phase_index],seconds/60.,joints].astype(np.float32)
    assert x.shape==(50,) and np.isfinite(x).all()
    return x


class ArmPolicy:
    def __init__(self,checkpoint,robot,sim,stage,diagnostics):
        import torch
        self.torch=torch;torch.set_num_threads(1)
        self.state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert self.state['nature']=='privileged-state BC' and self.state['input_dim']==50
        assert tuple(self.state['arm_names'])==ARM_NAMES and self.state['converged']
        self.model=network(torch).cpu();self.model.load_state_dict(self.state['model']);self.model.eval()
        self.joint_ids=np.array([robot.joint_names.index(n) for n in ARM_NAMES])
        self.palm_ids=[robot.body_names.index(n) for n in PALMS]
        self.fingers=[[i for i,n in enumerate(robot.joint_names) if n.startswith(side+'_hand_')] for side in ('left','right')]
        self.out=Path(diagnostics);self.out.mkdir(parents=True,exist_ok=False)
        self.calls=0;self.last_step=-2;self.clock_step=None
        (self.out/'observation-contract.json').write_text(json.dumps(dict(nature='privileged-state BC',input_dim=50,
            render_products_created=0,orchestrator_calls=0,cup_center_local_m=self.state['cup_center_local_m'].tolist(),
            source='Measured PhysX rigid-body and joint state, USD-derived fixed local cup center; no trained visual detector')),encoding='utf8')

    def targets(self,robot,step,unused_controller_seconds,phase,cup):
        seconds=step/50.
        if self.clock_step is not None and step!=self.clock_step+1:
            raise RuntimeError('Privileged physical clock skipped/stalled')
        self.clock_step=step
        if step-self.last_step>=2:
            all_q=robot.data.joint_pos[0].cpu().numpy();q=all_q[self.joint_ids]
            pose=np.r_[cup.data.root_pos_w[0].cpu().numpy(),cup.data.root_quat_w[0].cpu().numpy()]
            palms=np.c_[robot.data.body_pos_w[0,self.palm_ids].cpu().numpy(),robot.data.body_quat_w[0,self.palm_ids].cpu().numpy()]
            x=state_features(pose,palms,q,[float(all_q[i].mean()) for i in self.fingers],phase,seconds,self.state['cup_center_local_m'])
            with self.torch.inference_mode():
                scaled=self.model(self.torch.as_tensor((x-self.state['obs_mean'])/self.state['obs_std'])[None]).numpy().reshape(8,14)
            delta=np.clip(scaled,-1,1)*self.state['joint_ranges']
            if not np.isfinite(delta).all():raise RuntimeError('Nonfinite privileged action chunk')
            limits=robot.data.soft_joint_pos_limits[0].cpu().numpy()[self.joint_ids]
            self.chunk=np.clip(q[None]+delta,limits[:,0],limits[:,1]);self.last_step=step;self.calls+=1
            with (self.out/'decisions.jsonl').open('a',encoding='utf8') as f:
                f.write(json.dumps(dict(step=step,seconds=seconds,phase=phase,cup_center=x[:3].tolist(),
                    targets_first_two=self.chunk[:2].tolist(),nature='privileged-state BC',render_calls=0))+'\n')
        return self.chunk[min(step-self.last_step,1)]
