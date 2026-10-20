"""Dataset split/clock contracts, without an Isaac or trained-weight dependency."""
import numpy as np
import pytest
from training.train import validate_data
from training.evaluate import command_for
from pipeline.m3_accept import compare_geometry


def data():
    return dict(obs=np.zeros((4,50),np.float32),act_chunks=np.zeros((4,8,14),np.float32),
        act_chunk_valid=np.ones((4,8),bool),train_mask=np.array([1,1,0,0],bool),
        validation_mask=np.array([0,0,1,1],bool),trajectory_id=np.array(['train','train','val','val']),
        joint_ranges=np.ones(14),cup_center_local_m=np.zeros(3))


def test_training_accepts_disjoint_measured_trajectories():
    train,val=validate_data(data());assert train.sum()==2 and val.sum()==2


def test_training_rejects_trajectory_leakage():
    d=data();d['trajectory_id'][:]='same'
    with pytest.raises(ValueError,match='leakage'):validate_data(d)


def test_training_rejects_synthetic_validation_and_empty_chunks():
    d=data();d['synthetic_mask']=np.array([0,0,1,0],bool)
    with pytest.raises(ValueError,match='Synthetic'):validate_data(d)
    d=data();d['act_chunk_valid'][0]=False
    with pytest.raises(ValueError,match='real action'):validate_data(d)


def test_modes_select_separate_native_workers_and_parameters(tmp_path):
    args=tmp_path/'args.yaml';args.write_text('scene: sample.usda\n',encoding='utf8')
    cfg=dict(arguments_file=str(args),native_python='python',checkpoint='checkpoint.pt')
    bc,bc_args=command_for(cfg,'bc',tmp_path/'episode')
    safe,safe_args=command_for(cfg,'safe',tmp_path/'episode')
    expert,expert_args=command_for(cfg,'expert',tmp_path/'episode')
    pre,pre_args=command_for(cfg,'preflight',tmp_path/'episode')
    assert bc[1].endswith('probe_apartment_g1_privileged.py') and 'safe_parameters' not in bc_args
    assert safe[1].endswith('probe_apartment_g1_safe.py') and 'safe_parameters' in safe_args
    assert expert[1].endswith('probe_apartment_g1.py') and 'bc_checkpoint' not in expert_args
    assert pre_args['scenario_preflight_only'] is True and pre_args['cup_offset_y']==.12


def test_native_geometry_repeat_rejects_count_and_transform_difference():
    box=dict(native_prim_path='/World/Cup',center=[0,0,1],extent=[.1,.1,.1],rotation=np.eye(3).tolist())
    assert compare_geometry([box],[box])
    assert not compare_geometry([box],[])
    changed={**box,'center':[.001,0,1]}
    assert not compare_geometry([box],[changed])
