"""Audit completed M2 artifacts, including an independent scalar slip calculation."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_metrics(data, cfg):
    """Scalar cross-check with explicit inverse-quaternion rotation and run counting."""
    t, cup, palm = (data[k] for k in ("time_s", "cup_state", "palm_state"))
    dt = 1 / cfg["physics_hz"]
    assert len(t) == round(cfg["duration_s"] * cfg["physics_hz"])
    assert np.allclose(t, (np.arange(len(t)) + 1) * dt, atol=1e-12, rtol=0)
    names = data["sensor_names"].tolist()
    onset, longest, run, relative, lifts, forces = None, 0, 0, [], [], []
    for i in range(len(t)):
        magnitudes = [float(np.linalg.norm(v.astype(float))) for v in data["finger_forces_N"][i]]
        forces.append(sum(magnitudes))
        thumb = sum(v for name, v in zip(names, magnitudes) if "thumb" in name)
        other = sum(v for name, v in zip(names, magnitudes) if "index" in name or "middle" in name)
        opposed = thumb >= 0.05 and other >= 0.05
        unsupported = np.linalg.norm(data["support_force_N"][i].astype(float)) < 0.05
        lift = float(cup[i, 2]) - float(data["initial_cup_pose"][2])
        lifts.append(lift)
        if onset is None and opposed and unsupported and lift >= 0.001:
            onset = i
        if opposed and unsupported and lift >= 0.05:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
        q = palm[i, 3:7].astype(float)
        q /= np.linalg.norm(q)
        v = cup[i, :3].astype(float) - palm[i, :3].astype(float)
        # Rotation by inverse quaternion, independently of matrix/einsum implementation.
        u, w = -q[1:], q[0]
        relative.append(v + 2 * w * np.cross(u, v) + 2 * np.cross(u, np.cross(u, v)))
    slip = None
    if onset is not None and t[-1] + dt / 2 >= t[onset] + 0.5:
        slip = max(
            float(np.linalg.norm(relative[i] - relative[onset])) * 1000
            for i in range(onset, len(t))
            if t[i] <= t[onset] + 0.5 + 1e-9
        )
    hold, lift = longest * dt, max(lifts) * 1000
    return dict(
        success=bool(lift >= 50 and slip is not None and slip <= 5 and hold + 1e-8 >= 2),
        slip_distance_mm=slip,
        longest_retention_s=hold,
        maximum_lift_mm=lift,
        peak_contact_force_N=max(forces),
        grasp_onset_s=float(t[onset]) if onset is not None else None,
    )


def audit(root):
    root = Path(root)
    cfg = json.loads((root / "resolved-config.json").read_text(encoding="utf8"))
    results = json.loads((root / "results.json").read_text(encoding="utf8"))
    final = json.loads((root / "sweep-report.json").read_text(encoding="utf8"))
    commands = np.load(root / "fixed-commands.npz", allow_pickle=False)
    manifest = json.loads((root / "grid-code-snapshot/source-manifest.json").read_text(encoding="utf8"))
    assert all(sha(root / "grid-code-snapshot" / path) == value for path, value in manifest.items())
    expected = {
        (m, f, s)
        for m in cfg["grid"]["mass_kg"]
        for f in cfg["grid"]["friction"]
        for s in cfg["grid"]["seeds"]
    }
    assert len(results) == 45 and {(r["mass_kg"], r["friction"], r["seed"]) for r in results} == expected
    with (root / "raw.csv").open(encoding="utf8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 45
    max_errors = {
        k: 0.0
        for k in (
            "slip_distance_mm",
            "maximum_lift_mm",
            "longest_retention_s",
            "peak_contact_force_N",
            "grasp_onset_s",
        )
    }
    legs = [
        i
        for i, n in enumerate(commands["joint_names"].tolist())
        if any(part in n for part in ("hip_", "knee_", "ankle_"))
    ]
    assert len(legs) == 12
    fixed = [i for i in range(43) if i not in legs]
    completed, errors, success = 0, [], 0
    jitter_by_seed = {}
    details = []
    for result, row in zip(results, rows):
        ident = result["trial_id"]
        assert row["trial_id"] == ident
        folder = root / ident
        spec = json.loads((folder / "trial.json").read_text(encoding="utf8"))
        assert result == json.loads((folder / "trial-report.json").read_text(encoding="utf8"))
        assert spec["fixed_commands_sha256"] == sha(root / "fixed-commands.npz")
        jitter = np.asarray(spec["cup_xy_jitter_m"])
        expected_jitter = np.random.default_rng(spec["seed"]).uniform(-0.002, 0.002, 2)
        assert np.array_equal(jitter, expected_jitter)
        jitter_by_seed.setdefault(str(spec["seed"]), jitter.tolist())
        assert jitter_by_seed[str(spec["seed"])] == jitter.tolist()
        if result["status"] != "completed":
            errors.append(dict(trial=ident, error=result.get("error")))
            continue
        completed += 1
        assert sha(folder / "measurements.npz") == result["recording_sha256"]
        assert result["worker_sha256"] == manifest["simforge/physics/grasp_trial.py"]
        assert result["leg_policy_sha256"] == sha(cfg["leg_policy"])
        assert result["source_scene_sha256"] == sha(cfg["scene"])
        assert result["source_scene_unchanged"] and result["free_base"]
        assert result["pose_writes_after_initialization"] == 0 and result["upper_and_finger_targets_exact"]
        data = np.load(folder / "measurements.npz", allow_pickle=False)
        assert len(data["body_poses"]) == 600 and data["body_poses"].shape[1:] == (55, 7)
        assert data["finger_forces_N"].shape == (2400, 8, 3)
        for name in data.files:
            if np.issubdtype(data[name].dtype, np.number):
                assert np.isfinite(data[name]).all(), (ident, name)
        indices = np.minimum(np.arange(600), len(commands["targets"]) - 1)
        assert np.array_equal(data["command_indices"], indices)
        assert np.array_equal(data["motor_targets_applied"][:, fixed], commands["targets"][indices][:, fixed])
        expected_initial = commands["initial_cup"][:7].copy()
        expected_initial[:2] += jitter
        assert np.array_equal(data["initial_cup_pose"], expected_initial)
        actual = result["native_parameters"]
        assert abs(actual["mass_kg"] - spec["mass_kg"]) < 1e-6
        target_i = commands["baseline_inertia"].astype(np.float32) * (
            np.float32(spec["mass_kg"]) / np.float32(commands["baseline_mass_kg"])
        )
        assert np.allclose(actual["inertia_tensor_kgm2"], target_i, rtol=1e-5, atol=1e-9)
        assert len(actual["contact_materials"]) == 9
        for props in actual["contact_materials"].values():
            assert np.allclose(np.asarray(props)[:, :2], spec["friction"], rtol=0, atol=1e-6)
        independent = independent_metrics(data, cfg)
        metrics = result["metrics"]
        assert independent["success"] == metrics["success"]
        assert row["success"] == str(metrics["success"])
        for key, value in independent.items():
            if key == "success":
                continue
            if value is None:
                assert metrics[key] is None
            else:
                error = abs(value - metrics[key])
                assert error < 1e-8, (ident, key, value, metrics[key])
                max_errors[key] = max(max_errors[key], error)
        for key in (
            "slip_distance_mm",
            "maximum_lift_mm",
            "longest_retention_s",
            "peak_contact_force_N",
            "drop_time_s",
            "settle_time_s",
        ):
            assert (row[key] == "") if metrics[key] is None else (float(row[key]) == metrics[key])
        success += metrics["success"]
        details.append(dict(trial_id=ident, independent_metrics=independent))
    repeat_report = json.loads((root / "reproducibility.json").read_text(encoding="utf8"))
    reference = root / repeat_report["reference"]
    repeat = root / repeat_report["repeat"]
    arrays_equal, differences = False, []
    if (reference / "measurements.npz").exists() and (repeat / "measurements.npz").exists():
        a, b = (np.load(p / "measurements.npz", allow_pickle=False) for p in (reference, repeat))
        for key in a.files:
            if not np.array_equal(a[key], b[key]):
                delta = (
                    float(np.max(np.abs(a[key].astype(float) - b[key].astype(float))))
                    if np.issubdtype(a[key].dtype, np.number)
                    else None
                )
                differences.append(dict(array=key, maximum_absolute_difference=delta))
        arrays_equal = not differences and a.files == b.files
    assert completed == final["native_trials"] and success == final["successes"]
    for cell in final["cells"]:
        members = [
            r
            for r in results
            if r["mass_kg"] == cell["mass_kg"]
            and r["friction"] == cell["friction"]
            and r["status"] == "completed"
        ]
        assert cell["native_trials"] == len(members)
        assert cell["successes"] == sum(r["metrics"]["success"] for r in members)
    return dict(
        status="passed",
        scope="Artifact integrity and independent numeric audit, separate from performance acceptance",
        completed_trials=completed,
        runtime_errors=errors,
        successes=success,
        all_45_records_present=True,
        csv_exact=True,
        all_native_overrides_verified=True,
        frozen_source_verified=True,
        frozen_non_leg_commands_exact=True,
        independent_metric_maximum_errors=max_errors,
        per_trial=details,
        same_seed_metric_equality=repeat_report["exact_metric_equality"],
        same_seed_all_recorded_arrays_equal=arrays_equal,
        same_seed_array_differences=differences,
        performance_acceptance_passed=final["acceptance_passed"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    args = parser.parse_args()
    root = Path(args.root)
    try:
        report = audit(root)
    except Exception as error:
        report = dict(status="failed", error=repr(error))
        raise
    finally:
        (root / "independent-sweep-audit.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_trial"}, indent=2))


if __name__ == "__main__":
    main()
