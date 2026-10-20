"""Portable 50-D privileged BC trainer. Offline convergence is not task success."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from .constants import ARM_NAMES


def validate_data(data):
    required=('obs','act_chunks','act_chunk_valid','train_mask','validation_mask','joint_ranges','cup_center_local_m')
    if not all(k in data for k in required):raise ValueError('Missing BC dataset keys')
    n=len(data['obs'])
    if data['obs'].shape!=(n,50) or data['act_chunks'].shape!=(n,8,14) or data['act_chunk_valid'].shape!=(n,8):
        raise ValueError('Expected obs[N,50], act_chunks[N,8,14], valid[N,8]')
    train,val=[np.asarray(data[k],dtype=bool) for k in ('train_mask','validation_mask')]
    if train.shape!=(n,) or val.shape!=(n,) or not train.any() or not val.any() or (train&val).any():
        raise ValueError('Train/validation masks invalid or overlapping')
    if 'trajectory_id' not in data:raise ValueError('trajectory_id required for leakage checks')
    groups=np.asarray(data['trajectory_id'])
    if groups.shape!=(n,) or set(groups[train])&set(groups[val]):raise ValueError('Trajectory leakage across splits')
    if any(not np.isfinite(data[k]).all() for k in ('obs','act_chunks','joint_ranges','cup_center_local_m')):
        raise ValueError('Non-finite BC data')
    if data['joint_ranges'].shape!=(14,) or np.any(data['joint_ranges']<=0):raise ValueError('Invalid joint ranges')
    if data['cup_center_local_m'].shape!=(3,):raise ValueError('Invalid cup local center')
    valid=np.asarray(data['act_chunk_valid'])
    if not np.isin(valid,[0,1]).all() or not (valid[train].sum(1)>0).all() or not (valid[val].sum(1)>0).all():
        raise ValueError('Each selected row needs at least one real action-chunk target')
    if 'synthetic_mask' in data:
        synthetic=np.asarray(data['synthetic_mask'],dtype=bool)
        if synthetic.shape!=(n,) or (synthetic&val).any():raise ValueError('Synthetic samples cannot enter validation')
    return train,val


def train(data,output,*,epochs=2500,patience=120,seed=1904,seconds=300,synthetic_weight=.5):
    from .torch_runtime import import_torch
    torch=import_torch()
    from .privileged import network
    if epochs<1 or patience<1 or seconds<=0:raise ValueError('Invalid training budget')
    ti,vi=validate_data(data);out=Path(output)
    if out.exists():raise FileExistsError('Fresh training output required')
    out.mkdir(parents=True)
    x=np.asarray(data['obs'],dtype=np.float32);y=np.asarray(data['act_chunks'],dtype=np.float32)
    real_train=ti&~np.asarray(data.get('synthetic_mask',np.zeros(len(x),bool)),dtype=bool)
    if not real_train.any():raise ValueError('At least one real training trajectory required')
    mean=x[real_train].mean(0);std=np.maximum(x[real_train].std(0),1e-6)
    ranges=np.asarray(data['joint_ranges'],dtype=np.float32)
    xn=torch.as_tensor((x-mean)/std);yn=torch.as_tensor(y/ranges)
    mask=torch.as_tensor(np.asarray(data['act_chunk_valid'],dtype=np.float32))
    weights=np.ones(len(x),np.float32)
    if 'synthetic_mask' in data:weights[np.asarray(data['synthetic_mask'],bool)]=synthetic_weight
    if 'segment' in data:
        for segment in ('reach','carry','place'):
            selected=ti&(np.asarray(data['segment'])==segment)
            if not selected.any():raise ValueError('Missing training segment '+segment)
            weights[selected]*=ti.sum()/(3*weights[selected].sum())
    wt=torch.as_tensor(weights);train_ids=np.flatnonzero(ti);val_ids=np.flatnonzero(vi)
    torch.set_num_threads(4);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=network(torch);optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    def loss(ids,weighted=False):
        pred=model(xn[ids]).reshape(-1,8,14)
        raw=torch.nn.functional.smooth_l1_loss(pred,yn[ids],reduction='none')*mask[ids,:,None]
        per=raw.sum((1,2))/(mask[ids].sum(1).clamp_min(1)*14)
        return (per*wt[ids]).mean() if weighted else per.mean()
    model.eval()
    with torch.inference_mode():initial=float(loss(val_ids))
    best=significant=initial;stale=0;history=[];best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    started=time.monotonic()
    for epoch in range(epochs):
        if time.monotonic()-started>=seconds:break
        model.train();order=rng.permutation(train_ids)
        for at in range(0,len(order),128):
            ids=order[at:at+128];optimizer.zero_grad(set_to_none=True);value=loss(ids,True)
            if not torch.isfinite(value):raise ValueError('Nonfinite training loss')
            value.backward();optimizer.step()
        model.eval()
        with torch.inference_mode():val=float(loss(val_ids))
        if not np.isfinite(val):raise ValueError('Nonfinite validation loss')
        if val<best:best=val;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        if val<significant-1e-5:significant=val;stale=0
        else:stale+=1
        history.append((epoch,val,best,stale))
        if stale>=patience:break
    converged=best<initial*.9 and stale>=patience
    model.load_state_dict(best_state)
    with torch.inference_mode():pred=model(xn[val_ids]).numpy().reshape(-1,8,14)*ranges
    target_mask=np.asarray(data['act_chunk_valid'])[val_ids,:,None]
    rmse=float(np.sqrt((((pred-y[val_ids])**2)*target_mask).sum()/(target_mask.sum()*14)))
    state=dict(model=best_state,obs_mean=mean,obs_std=std,joint_ranges=ranges,arm_names=ARM_NAMES,K=8,h=2,
        cup_center_local_m=np.asarray(data['cup_center_local_m']),input_dim=50,nature='privileged-state BC',
        converged=converged,normalization_real_train_only=True)
    torch.save(state,out/'checkpoint.pt')
    report=dict(status='CONVERGED' if converged else 'NOT_CONVERGED_STOPPED',epochs=len(history),
        train_rows=int(ti.sum()),validation_rows=int(vi.sum()),initial_loss=initial,best_loss=best,
        validation_rmse_rad=rmse,converged=converged,physical_success_rate=None,
        note='New offline training output, not the historical frozen checkpoint or a physical success claim')
    (out/'training-summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    (out/'loss.csv').write_text('epoch,val_loss,best_loss,stale\n'+''.join(','.join(map(str,r))+'\n' for r in history),encoding='utf8')
    return report


def main():
    import yaml
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--dataset');p.add_argument('--out');args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf8'))
    data=np.load(args.dataset or cfg['dataset'],allow_pickle=False)
    report=train(data,args.out or cfg['output'],epochs=cfg.get('epochs',2500),patience=cfg.get('patience',120),
        seed=cfg.get('seed',1904),seconds=cfg.get('seconds',300),synthetic_weight=cfg.get('synthetic_weight',.5))
    print(json.dumps(report,indent=2))
    if not report['converged']:raise SystemExit(2)


if __name__=='__main__':main()
