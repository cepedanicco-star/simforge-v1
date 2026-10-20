import copy

import numpy as np
import pytest

from pipeline.simforge.eval.grasp_metrics import evaluate
from pipeline.simforge.physics.sweep import cup_jitter, scaled_inertia


CONFIG = dict(
    physics_hz=200,
    metrics=dict(lift_m=0.05, max_slip_mm=5, slip_window_s=0.5, hold_s=2,
                 minimum_opposed_force_N=0.05, maximum_support_force_N=0.05,
                 takeoff_m=0.001, settle_linear_speed_m_s=0.02,
                 settle_angular_speed_rad_s=0.15, settle_window_s=0.5),
    contact=dict(force_definition="sum of filtered normal forces"),
)


def measured_samples(n=400):
    cup = np.zeros((n, 13))
    cup[:, 2] = 0.05
    cup[:, 3] = 1
    palm = np.zeros_like(cup)
    palm[:, 3] = 1
    forces = np.zeros((n, 2, 3))
    forces[:, 0, 0] = 1
    forces[:, 1, 0] = -1
    return dict(time_s=np.arange(n)/200, cup_state=cup, palm_state=palm,
                finger_forces_N=forces, support_force_N=np.zeros((n, 3)),
                sensor_names=np.asarray(["left_hand_thumb_1_link", "left_hand_index_1_link"]),
                initial_cup_pose=np.asarray([0, 0, 0, 1, 0, 0, 0]))


def test_exact_lift_and_hold_boundaries_pass():
    result = evaluate(measured_samples(), CONFIG)
    assert result["success"]
    assert result["longest_retention_s"] == pytest.approx(2)
    assert result["maximum_lift_mm"] == pytest.approx(50)


def test_one_missing_physics_sample_below_hold_duration_fails():
    result = evaluate(measured_samples(399), CONFIG)
    assert not result["success"]
    assert "retention_below_2s" in result["failure_reasons"]


def test_lift_just_below_five_centimetres_fails():
    data = measured_samples()
    data["cup_state"][:, 2] -= 0.000001
    assert not evaluate(data, CONFIG)["success"]


@pytest.mark.parametrize("slip_mm,passed", [(5, True), (5.001, False)])
def test_slip_boundary_is_not_relaxed(slip_mm, passed):
    data = measured_samples()
    data["cup_state"][:, 0] = np.minimum(data["time_s"]/.5, 1)*slip_mm/1000
    assert evaluate(data, CONFIG)["success"] is passed


def test_rigid_cup_motion_with_rotating_palm_is_not_slip():
    data = measured_samples()
    angle = np.minimum(data["time_s"]/.5, 1)*np.pi/2
    data["palm_state"][:, 3] = np.cos(angle/2)
    data["palm_state"][:, 6] = np.sin(angle/2)
    data["cup_state"][:, 0] = .04*np.cos(angle)
    data["cup_state"][:, 1] = .04*np.sin(angle)
    result = evaluate(data, CONFIG)
    assert result["success"]
    assert result["slip_distance_mm"] < 1e-10


def test_missing_contact_is_undefined_slip_not_false_success():
    data = measured_samples()
    data["finger_forces_N"][:] = 0
    result = evaluate(data, CONFIG)
    assert not result["success"] and result["slip_distance_mm"] is None


def test_time_gaps_cannot_count_as_continuous_retention():
    data = measured_samples()
    data["time_s"][200:] += .1
    with pytest.raises(ValueError, match="uniform timestep"):
        evaluate(data, CONFIG)


def test_success_threshold_changes_are_rejected():
    altered = copy.deepcopy(CONFIG)
    altered["metrics"]["max_slip_mm"] = 6
    with pytest.raises(ValueError, match="approved M2"):
        evaluate(measured_samples(), altered)


def test_mass_scaling_preserves_normalized_full_inertia():
    matrix = np.asarray([[.003, .0002, -.0001], [.0002, .004, 0], [-.0001, 0, .002]])
    scaled = scaled_inertia(matrix, 1, .25)
    np.testing.assert_allclose(scaled/1, matrix/.25, rtol=1e-6)
    assert np.linalg.eigvalsh(scaled).min() > 0
    with pytest.raises(ValueError, match="Mass"):
        scaled_inertia(matrix, 0, .25)


def test_declared_jitter_is_tabletop_xy_only_and_bounded():
    values = np.asarray([cup_jitter(seed,.002) for seed in range(5)])
    assert values.shape == (5, 2) and np.abs(values).max() <= .002
    assert len(np.unique(values, axis=0)) == 5
