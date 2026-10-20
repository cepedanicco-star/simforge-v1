import numpy as np
import pytest

from pipeline.simforge.randomize.camera_constraints import inspect_position, select_position


def test_reject_inside_and_less_than_clearance_without_moving_boundaries():
    from types import SimpleNamespace

    def wall_surface(radius, point, callback, any_hit):
        assert radius == .15 and any_hit is False
        if abs(point[0]) <= radius:
            callback(SimpleNamespace(collision="/wall", rigid_body="/wall"))
            return 1
        return 0

    assert not inspect_position([.149, 0, 0], wall_surface)["accepted"]
    assert inspect_position([.151, 0, 0], wall_surface)["accepted"]
    assert inspect_position([.5, .5, .5], wall_surface)["accepted"]
    selected = select_position([0., .5, .5], [[0., 0, 0], [.2, 0, 0]], wall_surface)
    assert selected["selected_index"] == 1
    with pytest.raises(ValueError, match="No camera position"):
        select_position([0., .5, .5], np.zeros((128, 3)), wall_surface)
    with pytest.raises(ValueError, match="exactly 0.15"):
        inspect_position([.5, .5, .5], wall_surface, .1)


def test_query_count_without_hit_details_is_not_accepted_as_free_space():
    with pytest.raises(ValueError, match="count/callback mismatch"):
        inspect_position([1, 1, 1], lambda *args: 1)


def test_lens_values_use_same_time_code_for_animated_camera():
    from pxr import Gf, Usd, UsdGeom
    stage = Usd.Stage.CreateInMemory()
    camera = UsdGeom.Camera.Define(stage, "/Camera")
    camera.GetFocalLengthAttr().Set(50.)
    camera.GetHorizontalApertureAttr().Set(20.955)
    camera.GetFocalLengthAttr().Set(.5, 0)
    camera.GetHorizontalApertureAttr().Set(.4752, 0)
    # Exercise the D5 implementation with only camera randomization enabled.
    target = UsdGeom.Xform.Define(stage, "/cup")
    target.AddTranslateOp().Set(Gf.Vec3d(0, 0, 1))
    config = dict(dimensions=dict(D4=dict(target_prim="/cup"), D5={}))
    sample = dict(dimensions={f"D{i}": dict(enabled=i == 5) for i in range(1, 7)})
    sample["dimensions"]["D5"]["values"] = dict(position_m=[0, 0, 0], euler_rad=[0, 0, 0], fov_scale=1.)
    from pipeline.simforge.randomize.apply_visual import apply
    result = apply(stage, config, sample, "/Camera")
    assert abs(result["D5"]["original_horizontal_fov_rad"] - .887223) < 1e-5
    assert abs(camera.GetFocalLengthAttr().Get(0) - .5) < 1e-6
    assert camera.GetFocalLengthAttr().Get() == 50.
