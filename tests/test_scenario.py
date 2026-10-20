import json
import numpy as np
import pytest
from pxr import Usd,UsdGeom,Gf
from pipeline.simforge.bc.scenario import CUP,apply_offsets,start_y,classify_initial_contacts,write_scenario,surface_distance

def stage():
    s=Usd.Stage.CreateInMemory();UsdGeom.Xform.Define(s,CUP)
    UsdGeom.Xformable(s.GetPrimAtPath(CUP)).AddTranslateOp().Set((1.,2.,3.))
    return s

def test_zero_default_is_exact_v1_no_authored_change():
    s=stage();old=s.GetRootLayer().ExportToString();row=apply_offsets(s)
    assert old==s.GetRootLayer().ExportToString()
    assert row['after_position_m']==row['before_position_m'] and not row['authored_offset']
    assert start_y(-1.56,0.)==-1.56

def test_offsets_in_world_coordinates_with_rotated_parent():
    s=stage();parent=s.GetPrimAtPath('/World/Environment/Source')
    UsdGeom.Xformable(parent).AddRotateZOp().Set(90.)
    row=apply_offsets(s,.1,.12)
    np.testing.assert_allclose(np.array(row['after_position_m'])-row['before_position_m'],[.1,.12,0.])
    actual=UsdGeom.XformCache(0).GetLocalToWorldTransform(s.GetPrimAtPath(CUP)).ExtractTranslation()
    np.testing.assert_allclose(actual,row['after_position_m'])

def test_scenario_json_saved_once_with_actual_units(tmp_path):
    p=tmp_path/'scenario-v2.json';data=dict(t0_cup_cabinet_surface_distance_m=.02,G0_pass=False)
    write_scenario(p,data);assert json.loads(p.read_text())==data
    with pytest.raises(AssertionError):write_scenario(p,data)

def test_s2_counts_contact_even_with_zero_impulse_not_policy_failure():
    assert classify_initial_contacts([dict(reports=1,impulse_Ns=0.)])=='init_infeasible_v2'
    assert classify_initial_contacts([dict(reports=0)])=='no_hand_cup_contact_in_original_setup'

def test_start_offset_is_only_scenario_knob():
    assert start_y(-1.56,.04)==pytest.approx(-1.52)
    assert start_y(-1.56,-.04)==pytest.approx(-1.60)

def test_surface_distance_parallel_contact_and_intersection():
    a=np.array([[[0,0,0],[1,0,0],[0,1,0]]],float)
    assert surface_distance(a,a+[0,0,.12])==pytest.approx(.12)
    assert surface_distance(a,a)==pytest.approx(0.)
    b=np.array([[[.2,.2,-1],[.2,.2,1],[.8,.2,0]]])
    assert surface_distance(a,b)==pytest.approx(0.,abs=1e-9)

def test_support_contact_cannot_be_eight_cm_whole_object_clearance():
    a=np.array([[[0,0,0],[1,0,0],[0,1,0]]],float)
    assert surface_distance(a,a)<.08
