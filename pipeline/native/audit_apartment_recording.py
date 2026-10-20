"""Independently check raw measured cup poses and phase contact evidence."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.ue_isaac_pipeline.io import digest_file, read_json, write_json  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--recording", required=True)
p.add_argument("--placement", required=True)
args = p.parse_args()
run = Path(args.recording).resolve(strict=True)
report = read_json(run / "probe-report.json")
contact = read_json(run / "contact-ledger.json")
telemetry = read_json(run / "telemetry.json")
floor_geometry_verified = True
if report.get("geometry_validated_floor_supports"):
    from pxr import Usd, UsdGeom

    support_stage = Usd.Stage.Open(str(run / "Apartment_sim.usda"))
    support_cache = UsdGeom.BBoxCache(0, ["default", "render"])
    for entry in report["geometry_validated_floor_supports"]:
        prim = support_stage.GetPrimAtPath(entry["prim"])
        if not prim or not prim.IsA(UsdGeom.Mesh):
            floor_geometry_verified = False
            continue
        bounds = support_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        low, high = np.asarray(bounds.GetMin()), np.asarray(bounds.GetMax())
        size = high - low
        floor_geometry_verified &= bool(
            size[0] > 1
            and size[1] > 1
            and size[2] < 0.01
            and max(abs(low[2]), abs(high[2])) < 0.08
            and np.max(np.abs(low - entry["bounds_min_m"])) < 1e-6
            and np.max(np.abs(high - entry["bounds_max_m"])) < 1e-6
        )
a = np.load(run / "physics-body-states.npz")
cup = a["cup"].astype(np.float64)
hz = int(a["control_hz"])
if len(cup) < hz + 1 or not np.isfinite(cup).all():
    raise ValueError("Recording has insufficient or invalid cup states")
q = cup[:, 3:]
norm_error = float(np.abs(np.linalg.norm(q, axis=1) - 1).max())
q /= np.linalg.norm(q, axis=1)[:, None]
tilt = np.degrees(np.arccos(np.clip(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1)))
speed = np.linalg.norm(np.diff(cup[:, :3], axis=0) * hz, axis=1)
angular_speed = 2 * np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", q[1:], q[:-1])), 0, 1)) * hz
goal = np.asarray(read_json(args.placement)["cup_origin_m"])
error = cup[-hz:, :3] - goal
clearance_file = Path(args.placement).parent / "desktop-clearance.npz"
safe_tabletop = bool(np.linalg.norm(error[:, :2], axis=1).max() < 0.03)
minimum_clearance = None
if clearance_file.is_file():
    cm = np.load(clearance_file)
    grid = cm["clearance_m"]
    clearances = []
    for xy in cup[-hz:, :2]:
        ij = np.floor((xy - cm["origin_xy_m"]) / cm["resolution_m"]).astype(int)
        clearances.append(
            float(grid[ij[1] : ij[1] + 2, ij[0] : ij[0] + 2].min())
            if 0 <= ij[0] < grid.shape[1] - 1 and 0 <= ij[1] < grid.shape[0] - 1
            else 0
        )
    minimum_clearance = min(clearances)
    safe_tabletop = minimum_clearance >= 0.053
nav = [r for r in telemetry if r.get("phase") == "navigate_apartment_with_cup"]
unexpected = [p for p in contact["pairs"] if p["maximum_unexpected_episode_impulse_Ns"] > 0.0005]
nav_collisions = [
    p
    for p in contact["pairs"]
    if p["kind"] in ("cup_environment", "robot_environment")
    and p.get("phases", {}).get("navigate_apartment_with_cup", {}).get("max_impulse_Ns", 0) > 0.0005
]
desk = "/World/Environment/Source/SM_AI_vol8_03_desk_01"
last_contact_time = max(
    (p["last_time"] for p in contact["pairs"] if p["kind"] == "cup_environment" and desk in p["actors"]),
    default=-1,
)
last_sample_time = telemetry[0]["physics_time"] + (len(cup) - 1) / hz
final_rows = [r for r in telemetry if r["physics_time"] >= last_sample_time - 1.0 - 1e-5]
active_desk_pair = any(
    s.get("active") and desk in s["actors"] and any("/cup_V1" in a for a in s["actors"])
    for s in contact.get("contact_states", [])
)
sleeping_support = False
if report.get("probe_arguments", {}).get("sleep_aware_support") and active_desk_pair and final_rows:
    retained = [
        r for r in final_rows if r.get("desk_contact_evidence") == "native_sleep_active_pair_unchanged_pose"
    ]

    def retained_pose_matches(row):
        previous = np.asarray(row.get("last_native_desk_pose", []), dtype=float)
        current = np.r_[row["cup_position"], row["cup_orientation_wxyz"]]
        return bool(
            len(previous) == 7
            and row.get("native_cup_sleeping")
            and np.linalg.norm(current[:3] - previous[:3]) < 0.002
            and abs(np.dot(current[3:], previous[3:])) > np.cos(np.radians(1) / 2)
        )

    sleeping_support = bool(
        retained
        and all(retained_pose_matches(r) for r in retained)
        and all(
            r.get("desk_contact") is True
            and r.get("desk_contact_evidence")
            in ("fresh_native_impulse", "native_sleep_active_pair_unchanged_pose")
            for r in final_rows
        )
    )
checks = dict(
    registered_floor_support_geometry_verified=bool(floor_geometry_verified),
    source_task_completed=report.get("full_transfer_completed") is True
    and report.get("status") == "completed",
    unit_quaternions=norm_error < 1e-4,
    tabletop_safe_region_one_second=safe_tabletop,
    tabletop_height_one_second=bool(np.abs(error[:, 2]).max() < 0.006),
    cup_upright_one_second=bool(tilt[-hz:].max() < 12),
    cup_settled_one_second=bool(speed[-hz:].max() < 0.025),
    cup_rotation_settled_one_second=bool(angular_speed[-hz:].max() < 0.15),
    exact_desk_contact=(-1e-5 <= last_sample_time - last_contact_time < 0.12) or sleeping_support,
    no_unexpected_robot_environment_contacts=not unexpected and not contact["callback_errors"],
    no_carried_cup_or_hand_obstacle_contacts=bool(nav) and not nav_collisions,
    no_wrong_cup_environment_contact=not report.get("unexpected_cup_environment_contacts", []),
)
result = dict(
    full_transfer_audit_passed=all(checks.values()),
    checks=checks,
    source_npz_sha256=digest_file(run / "physics-body-states.npz"),
    final_cup_position_m=cup[-1, :3].tolist(),
    final_cup_tilt_deg=float(tilt[-1]),
    final_second_maximum_speed_m_s=float(speed[-hz:].max()),
    final_second_maximum_angular_speed_rad_s=float(angular_speed[-hz:].max()),
    final_second_maximum_xy_error_m=float(np.linalg.norm(error[:, :2], axis=1).max()),
    final_second_minimum_tabletop_clearance_m=minimum_clearance,
    final_second_maximum_height_error_m=float(np.abs(error[:, 2]).max()),
    exact_desk_support_evidence="native active pair + native sleeping state + unchanged last contact pose"
    if sleeping_support
    else "recent native contact impulse",
    sleeping_support_verified=sleeping_support,
    navigation_seconds=(nav[-1]["time"] - nav[0]["time"]) if nav else 0,
    navigation_opposed_contact_fraction=sum(bool(r.get("opposing_finger_contact")) for r in nav) / len(nav)
    if nav
    else 0,
    unexpected_robot_environment_pairs=[p["actors"] for p in unexpected],
    note="Raw pose/contact consistency check; no claim of new RL training or hardware calibration",
)
write_json(run / "independent-recording-audit.json", result)
print(result)
