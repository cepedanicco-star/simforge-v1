"""Extract measured state/action chunks with whole-trajectory held-out splits."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .constants import ARM_NAMES,PALMS
from .privileged import state_features


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',nargs='+',required=True,help='Native recording directories')
    p.add_argument('--validation',nargs='+',required=True)
    p.add_argument('--joint-limits',required=True,help='JSON lower/upper in frozen 14-arm order')
    p.add_argument('--cup-center-local',type=float,nargs=3,required=True)
    p.add_argument('--out',required=True);args=p.parse_args()
    train=[Path(x).resolve(strict=True) for x in args.train];val=[Path(x).resolve(strict=True) for x in args.validation]
    if set(train)&set(val):raise ValueError('A recording cannot belong to both splits')
    fingerprints=[hashlib.sha256((d/'physics-body-states.npz').read_bytes()).hexdigest() for d in train+val]
    if len(set(fingerprints))!=len(fingerprints):raise ValueError('Duplicate deterministic trajectories are not independent data')
    target=Path(args.out)
    if target.exists():raise FileExistsError(target)
    limits=json.loads(Path(args.joint_limits).read_text(encoding='utf8'))
    ranges=np.asarray(limits['upper'],np.float32)-np.asarray(limits['lower'],np.float32)
    rows={k:[] for k in ('obs','act_chunks','act_chunk_valid','train_mask','validation_mask','trajectory_id','segment')}
    for number,folder in enumerate(train+val):
        report=json.loads((folder/'probe-report.json').read_text(encoding='utf8'))
        audit=json.loads((folder/'independent-recording-audit.json').read_text(encoding='utf8'))
        if not report.get('full_transfer_completed') or not audit.get('full_transfer_audit_passed'):
            raise ValueError('Expert recording needs successful physical audit')
        if audit.get('source_npz_sha256')!=fingerprints[number]:
            raise ValueError('Physical audit does not match this recording')
        with np.load(folder/'physics-body-states.npz',allow_pickle=False) as d:
            if int(d['control_hz'])!=50:raise ValueError('Frozen observation clock requires 50Hz')
            jnames=d['joint_names'].tolist();bnames=d['body_names'].tolist()
            arms=[jnames.index(n) for n in ARM_NAMES];palms=[bnames.index(n) for n in PALMS]
            fingers=[[i for i,n in enumerate(jnames) if n.startswith(s+'_hand_')] for s in ('left','right')]
            for i in range(len(d['cup'])-1):
                phase=str(d['phases'][i]);q=d['joints'][i,arms]
                future=i+np.arange(1,9);safe=np.minimum(future,len(d['cup'])-1)
                valid=(future<len(d['cup']))&(d['phases'][safe]==phase)
                if not valid.any():continue
                obs=state_features(d['cup'][i],d['states'][i,palms],q,
                    [float(d['joints'][i,ids].mean()) for ids in fingers],phase,(i+1)/50,args.cup_center_local)
                values=dict(obs=obs,act_chunks=d['motor_targets_rad'][safe][:,arms]-q,
                    act_chunk_valid=valid,train_mask=number<len(train),validation_mask=number>=len(train),
                    trajectory_id=f'trajectory-{number:04d}',segment='place' if phase in
                    ('move_above_desk','lower_to_desk','release_on_desk','withdraw_hand','verify_placement') else
                    'reach' if phase in ('settle_open_grasp','close','lift') else 'carry')
                for k,v in values.items():rows[k].append(v)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as stream:
        np.savez_compressed(stream,**{k:np.asarray(v) for k,v in rows.items()},joint_ranges=ranges,
            cup_center_local_m=np.asarray(args.cup_center_local,np.float32))
    print(json.dumps(dict(status='EXTRACTED_MEASURED_ONLY',rows=len(rows['obs']),trajectories=len(train)+len(val),
        train_trajectories=len(train),validation_trajectories=len(val),new_robot_success_rate=None),indent=2))


if __name__=='__main__':main()
