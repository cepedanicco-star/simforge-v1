import copy

from pipeline.simforge.randomize.sampler import compare_values, sample


def config():
    dimensions = {
        f"D{i}": dict(enabled=True, timing="episode_start", distribution="uniform", center={})
        for i in range(1, 7)
    }
    dimensions["D1"].update(illuminance_lux=[500, 3000], temperature_K=[3000, 8000])
    dimensions["D2"].update(roughness=[0.2, 0.9], hue_degrees=[-15, 15])
    dimensions["D3"].update(count=[0, 3], proposal_count=128)
    dimensions["D4"].update(
        position_m=[-0.05, 0.05], axes=["x", "y"], z_rule="preserve_support_height", yaw_degrees=[-30, 30]
    )
    dimensions["D5"].update(position_m=[-0.05, 0.05], euler_rad=[-0.1, 0.1], fov_scale=[0.95, 1.05])
    dimensions["D6"].update(
        mass_scale=[0.7, 1.3], friction=[0.4, 1], nominal_mass_kg=0.25, nominal_friction=0.5
    )
    return dict(global_seed=42, dimensions=dimensions)


def test_repeated_seed_is_exact_for_all_six_dimensions():
    cfg = config()
    for i in range(6):
        assert compare_values(sample(cfg, i), sample(cfg, i)) == []
        assert sample(cfg, i)["seed"] == 42 + i


def test_disabling_physics_preserves_all_visual_and_geometric_draws():
    cfg = config()
    off = copy.deepcopy(cfg)
    off["dimensions"]["D6"]["enabled"] = False
    for i in range(6):
        assert compare_values(sample(cfg, i), sample(off, i)) == ["D6"]
        assert sample(off, i)["dimensions"]["D6"]["values"]["mass_kg"] == 0.25


def test_cup_height_is_preserved_and_mass_range_is_not_a_force_scale():
    cfg = config()
    for i in range(100):
        values = sample(cfg, i)["dimensions"]
        assert values["D4"]["values"]["position_m"][2] == 0
        assert 0.175 <= values["D6"]["values"]["mass_kg"] <= 0.325
        assert 0 <= values["D3"]["values"]["count"] <= 3


def test_camera_proposals_reproduce_without_changing_other_dimensions():
    original = config()
    guarded = copy.deepcopy(original)
    guarded["dimensions"]["D5"].update(require_free_space=True, proposal_count=128)
    for i in range(6):
        old, new = sample(original, i), sample(guarded, i)
        assert compare_values(new, sample(guarded, i)) == []
        for name in ("D1", "D2", "D3", "D4", "D6"):
            assert old["dimensions"][name] == new["dimensions"][name]
        values = new["dimensions"]["D5"]["values"]
        assert len(values["position_proposals_m"]) == 128
        assert values["position_proposals_m"][0] == values["position_m"]
        assert all(-.05 <= axis <= .05 for point in values["position_proposals_m"] for axis in point)
