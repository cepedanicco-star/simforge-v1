import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1])]
from pipeline.simforge.bc.safe_execute import Parameters, safety_filter, servo_correct
from pipeline.simforge.bc.safe_execute import NativeScene,baseline_exemption,advance_escape,close_gate,gate_finger_targets
from pipeline.simforge.bc.contact_resolve import retreat_step

def scene(**changes):
    p=Parameters(**changes)
    return SimpleNamespace(params=p,nominal=np.zeros(14),limits=np.tile([-2.,2.],(14,1)),
        grasp_phase=True,tilt_rate=0.,closing_speed=lambda a,b:0.,check_path=lambda a,b,n:[])

def test_combined_envelope_and_slew():
    s=scene()
    result,log=safety_filter(np.ones(14)*3,np.zeros(14),s)
    assert np.max(abs(result))==pytest.approx(.008)
    assert log['envelope_limited'] and log['slew_limited']

def test_actual_query_shrinks_until_clear():
    s=scene();seen=[]
    def check(a,b,n):
        seen.append(b.copy())
        return [{'real_geometry_hit':True}] if b[0]>.0021 else []
    s.check_path=check
    result,log=safety_filter(np.ones(14),np.zeros(14),s)
    assert result[0]==pytest.approx(.002)
    assert log['shrink_alpha']==.25 and len(seen)==3

def test_no_free_pose_never_claims_safe():
    s=scene();s.check_path=lambda a,b,n:[{'hit':'cabinet'}]
    result,log=safety_filter(np.ones(14),np.zeros(14),s)
    assert np.array_equal(result,np.zeros(14))
    assert log['status']=='HOLD_UNCERTIFIED'

def test_tilt_freeze_and_closing_slowdown():
    s=scene();s.tilt_rate=46
    result,log=safety_filter(np.ones(14),np.zeros(14),s)
    assert not result.any() and log['tilt_guard']
    s.tilt_rate=0;s.closing_speed=lambda a,b:.12
    result,log=safety_filter(np.ones(14),np.zeros(14),s)
    assert result[0]==pytest.approx(.004) and log['closing_factor']==.5

def test_carry_returns_bc_unchanged_without_ik():
    result,log=servo_correct(np.arange(14.),{'params':Parameters()},'navigate_apartment_with_cup')
    assert np.array_equal(result,np.arange(14.)) and not log['enabled']

def test_servo_cartesian_clamp_and_additive_joint_correction():
    class IK:
        def forward(self,q):return np.eye(4)
        def solve(self,target,seed,**kw):
            assert np.linalg.norm(target[3,:3])<=.006000001
            return seed+.5,{'feasible':True}
    a=np.zeros(14)
    result,log=servo_correct(a,dict(params=Parameters(),model=IK(),active_ids=np.arange(7),
        torso_world=np.eye(4),goal_world=np.ones(3),goal_rotation=[1,0,0,0],measured_error_m=1.),'close')
    assert np.allclose(result[:7],.1) and not result[7:].any()
    assert log['correction_norm_rad']>0

def test_nonfinite_action_fail_stops():
    with pytest.raises(RuntimeError):safety_filter(np.full(14,np.nan),np.zeros(14),scene())

def test_servo_actual_displacement_is_bounded_after_joint_clipping():
    class IK:
        def forward(self,q):
            m=np.eye(4);m[3,0]=q[0];return m
        def solve(self,target,seed,**kw):return seed+.5,{'feasible':False}
    result,log=servo_correct(np.zeros(14),dict(params=Parameters(),model=IK(),active_ids=np.arange(7),
        torso_world=np.eye(4),goal_world=np.ones(3),goal_rotation=[1,0,0,0],measured_error_m=1.,
        previous_target=np.zeros(14)),'close')
    assert abs(result[0])<=.006
    assert np.linalg.norm(log['achieved_correction_m'])<=.006


def geometry_scene():
    class Model:
        def forward(self,q):
            m=np.eye(4);m[3,0]=q[0];return m
    s=object.__new__(NativeScene)
    s.params=Parameters();s.full_q=np.r_[1.,np.zeros(13)];s.arm_ids=np.arange(14)
    s.specs=[dict(name='test_link',model=Model(),ids=[0],center=np.zeros(3),radius=.1)]
    s.torso_world=np.eye(4)
    s.baseline={'test_link':{'cabinet':np.array([1.,0,0])}}
    s.collider_centers={'cabinet':np.zeros(3),'chair':np.array([1.,1,0])}
    s.hits=lambda r,p:[dict(collider='cabinet')]
    s.nominal=s.full_q.copy();s.limits=np.tile([-2.,2.],(14,1))
    s.grasp_phase=False;s.tilt_rate=0.;s.closing_speed=lambda a,b:0.
    return s


def test_baseline_away_and_tangent_are_exempt_with_receipt():
    s=geometry_scene();previous=s.full_q.copy();a=previous.copy();a[0]+=.1
    result,log=safety_filter(a,previous,s)
    assert result[0]>previous[0] and log['exempted_baseline']
    assert all(x['exempted_baseline'] for x in log['baseline_exemptions'])
    assert baseline_exemption(np.array([1.,.2,0]),np.array([1.,0,0]),np.zeros(3),.001)[0]


def test_new_collider_never_receives_baseline_exemption():
    s=geometry_scene();s.hits=lambda r,p:[dict(collider='chair')]
    result,log=safety_filter(s.full_q+.1,s.full_q,s)
    assert log['status']=='HOLD_UNCERTIFIED' and np.array_equal(result,s.full_q)


def test_approaching_baseline_blocked_away_clear():
    s=geometry_scene();a=s.full_q.copy();a[0]-=.1
    blocked=s.check_path(s.full_q,a,1)
    assert blocked and blocked[0]['baseline_approaching']
    a[0]=1.1
    assert s.check_path(s.full_q,a,1)==[] and s.path_exemptions


def test_baseline_membership_refreshes_actual_static_pose_not_target():
    s=geometry_scene();s.full_q[0]=1.2
    snapshot=s.baseline_contacts()
    assert np.array_equal(snapshot['test_link']['cabinet'],[1.2,0,0])
    assert np.array_equal(s.collider_centers['cabinet'],[0,0,0])


def test_escape_old_contacts_only_and_new_collider_priority():
    s=geometry_scene();previous=s.full_q.copy();previous[0]=.8
    goal=previous.copy();goal[0]=.9
    result,log=safety_filter(goal,previous,s)
    assert log['status']=='ESCAPE_BASELINE_CONTACT' and result[0]>previous[0]
    s.hits=lambda r,p:[dict(collider='cabinet' if p[0]>.805 else 'chair')]
    result,log=safety_filter(goal,previous,s)
    assert log['status']=='HOLD_UNCERTIFIED'


def test_escape_timeout_is_strictly_after_fifty_steps():
    assert advance_escape(49,'ESCAPE_BASELINE_CONTACT',50)==50
    with pytest.raises(RuntimeError,match='SAFETY_ESCAPE_TIMEOUT'):
        advance_escape(50,'ESCAPE_BASELINE_CONTACT',50)
    assert advance_escape(49,'CLEAR_DISCRETE_LINK_SPHERES',50)==0


def test_close_gate_opens_fingers_at_ten_degrees_and_for_fast_untilting():
    p=Parameters()
    assert not gate_finger_targets(np.ones(7),'close',close_gate(True,10.,0.,p)).any()
    assert not gate_finger_targets(np.ones(7),'grasp',close_gate(True,1.,-6.,p)).any()
    assert gate_finger_targets(np.ones(7),'close',close_gate(True,2.,4.,p)).all()


def test_init_retreat_outward_direction_joint_slew_and_total_twelve_mm():
    class Model:
        def forward(self,q):
            m=np.eye(4);m[3,0]=q[0];return m
        def solve(self,target,seed,**kwargs):
            q=seed.copy();q[0]=target[3,0];return q,dict(feasible=True)
    p=Parameters();q=np.r_[1.,np.zeros(6)];used=0.
    for _ in range(30):
        new,log=retreat_step(q,np.array([q[0],0,0]),np.zeros(3),np.eye(4),Model(),p,used)
        assert np.max(abs(new-q))<=p.slew_max_rad
        assert np.dot(log['achieved_displacement_m'],[q[0],0,0])>=0.
        used=log['cumulative_retreat_m'];assert used<=.012+1e-9
        q=new
    assert q[0]>1. and used==pytest.approx(.012)


def test_inactive_balance_guard_limits_without_disabling_active_arm():
    s=scene(inactive_arm_envelope_rad=.04);s.inactive_ids=np.arange(7,14)
    previous=np.zeros(14);previous[7:]=.04
    result,log=safety_filter(np.ones(14),previous,s)
    assert np.allclose(result[7:],.04) and np.all(result[:7]>0)
    assert log['envelope_limited']
