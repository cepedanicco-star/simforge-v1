"""M2 scheduler: fixed native commands, explicit trials, raw outcomes and hard gates."""

import argparse
import copy
import csv
import hashlib
import json
import os
import subprocess
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from pipeline.simforge.eval.grasp_metrics import rotation_matrix


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf8")


def scaled_inertia(matrix, new_mass, old_mass):
    if not 0 < min(new_mass, old_mass) or not np.isfinite([new_mass, old_mass]).all():
        raise ValueError("Mass must be positive and finite")
    return np.asarray(matrix, dtype=np.float32) * (np.float32(new_mass) / np.float32(old_mass))


def cup_jitter(seed, amount):
    return np.random.default_rng(seed).uniform(-amount, amount, 2).tolist()


def prepare(config, out):
    target = out / "fixed-commands.npz"
    source = Path(config["source_recording"])
    if target.exists():
        receipt = json.loads((out / "fixed-commands-receipt.json").read_text(encoding="utf8"))
        if sha(target) != receipt["sha256"]:
            raise ValueError("Existing fixed command file changed")
        return target
    data = np.load(source / "physics-body-states.npz", allow_pickle=False)
    report = json.loads((source / "probe-report.json").read_text(encoding="utf8"))
    mass = json.loads((source / "native-mass-properties.json").read_text(encoding="utf8"))
    telemetry = json.loads((source / "telemetry.json").read_text(encoding="utf8"))
    if int(data["control_hz"]) != config["control_hz"]:
        raise ValueError("Source control frequency differs")
    names = data["joint_names"].tolist()
    index = config["replay"]["start_state_index"]
    first = config["replay"]["first_target_index"]
    ends = np.flatnonzero(data["phases"] == config["replay"]["end_before_phase"])
    if not len(ends):
        raise ValueError("Source command phase boundary is missing")
    end = int(ends[0])
    if index != 0 or first != 1 or end <= first:
        raise ValueError("Unsupported approved command initial-state contract")
    observation = data["policy_observations"][index + 1, -91:]
    legs = [
        f"{side}_{part}_joint"
        for part in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
        for side in ("left", "right")
    ]
    arms = [
        f"{side}_{part}_joint"
        for side in ("left", "right")
        for part in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
    ]
    obs_ids = [names.index(name) for name in legs + arms]
    defaults = np.asarray(report["default_joint_positions"])
    error = float(np.abs(observation[10:36] + defaults[obs_ids] - data["joints"][index, obs_ids]).max())
    if error > 1e-6:
        raise ValueError("Next observation does not match the selected initial joint state")
    initial_dq = (data["joints"][index + 1] - data["joints"][index]) * config["control_hz"]
    initial_dq[obs_ids] = observation[36:62]
    pelvis = data["body_names"].tolist().index("pelvis")
    pose = data["states"][index, pelvis]
    initial_root = np.r_[
        pose, telemetry[0]["root_velocity"], rotation_matrix(pose[3:]) @ observation[:3]
    ].astype(np.float32)
    initial_cup = np.r_[data["cup"][index], np.zeros(6)].astype(np.float32)
    np.savez_compressed(
        target,
        targets=data["motor_targets_rad"][first:end],
        effort_limits=data["joint_effort_limits_Nm"][first:end],
        joint_names=data["joint_names"],
        initial_q=data["joints"][index],
        initial_dq=initial_dq,
        initial_root=initial_root,
        initial_cup=initial_cup,
        baseline_mass_kg=np.asarray(mass["cup_mass_kg"]),
        baseline_inertia=np.asarray(mass["cup_inertia_tensor_kgm2"], dtype=np.float32),
        nominal_default_q=defaults.astype(np.float32),
        initial_policy_history=data["policy_observations"][index].reshape(10, 91),
        initial_policy_feedback=observation[-29:],
        body_commands=data["body_commands"][first:end],
    )
    write(
        out / "fixed-commands-receipt.json",
        dict(
            sha256=sha(target),
            source_recording=str(source),
            source_recording_sha256=sha(source / "physics-body-states.npz"),
            source_first_target=first,
            source_end_exclusive=end,
            targets=end - first,
            hold_last_after_sequence=True,
            initial_observation_match_error_rad=error,
            initial_velocity_source="Recorded 26-joint/root angular observation and root linear telemetry; finite difference for waist/fingers; initially resting cup velocity zero",
            joint_drives_sha256=sha(source / "joint-drives-runtime-initial.json"),
        ),
    )
    return target


def run_trial(config, out, commands, ident, friction, mass, seed, jitter=True):
    folder = out / ident
    folder.mkdir(exist_ok=True)
    spec_path = folder / "trial.json"
    spec = dict(
        config=config,
        trial_id=ident,
        friction=friction,
        mass_kg=mass,
        seed=seed,
        cup_xy_jitter_m=cup_jitter(seed, config["jitter"]["cup_xy_m"]) if jitter else [0, 0],
        fixed_commands=str(commands),
        fixed_commands_sha256=sha(commands),
    )
    report_path = folder / "trial-report.json"
    if report_path.exists():
        old_spec = json.loads(spec_path.read_text(encoding="utf8"))
        if old_spec != spec:
            raise ValueError("Existing trial input differs; use a new run, never overwrite trials")
        report = json.loads(report_path.read_text(encoding="utf8"))
        if report["status"] != "completed":
            return report
        if sha(folder / "measurements.npz") != report["recording_sha256"]:
            raise ValueError("Existing trial measurement hash differs")
        print("REUSE VERIFIED TRIAL", ident, flush=True)
        return report
    if spec_path.exists():
        raise FileExistsError(
            "Incomplete previous trial requires inspection, not silent restart: " + str(folder)
        )
    write(spec_path, spec)
    env = dict(os.environ)
    env.update(
        AISIM_PORTABLE_ROOT=config["portable_root"],
        AISIM_H5PY_OVERLAY=config["h5py_overlay"],
        AISIM_ORT_OVERLAY=config["ort_overlay"],
        TEMP=config["temp"],
        TMP=config["temp"],
    )
    package_root = config.get("worker_pythonpath", str(Path(__file__).resolve().parents[2]))
    env["PYTHONPATH"] = (
        package_root + os.pathsep + config["legacy_source"] + os.pathsep + env.get("PYTHONPATH", "")
    )
    command = [config["isaac_python"], "-m", "pipeline.simforge.physics.grasp_trial", "--trial", str(spec_path)]
    with (folder / "console.log").open("wb") as log:
        flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, **flags)
        started, ready = time.monotonic(), None
        while process.poll() is None:
            if report_path.exists():
                # Worker has finished writing its report before printing/closing the app.
                ready = ready or time.monotonic()
                if time.monotonic() - ready > 20:
                    process.terminate()  # This owned Popen child only, after persisted report.
                    break
            if time.monotonic() - started > config["worker_timeout_s"]:
                process.terminate()
                process.wait(timeout=20)
                report = dict(
                    status="error",
                    trial_id=ident,
                    friction=friction,
                    mass_kg=mass,
                    seed=seed,
                    error="owned_worker_timeout",
                    failures=["owned_worker_timeout"],
                )
                write(report_path, report)
                return report
            time.sleep(0.5)
        process.wait(timeout=20)
    if not report_path.exists():
        report = dict(
            status="error",
            trial_id=ident,
            friction=friction,
            mass_kg=mass,
            seed=seed,
            error=f"worker_exit_without_report:{process.returncode}",
            failures=["missing_worker_report"],
        )
        write(report_path, report)
        return report
    return json.loads(report_path.read_text(encoding="utf8"))


def main():
    import yaml

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--preflight-name", default="preflight-v01")
    args = p.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf8"))
    out = Path(config["output"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    commands = prepare(config, out)
    if args.preflight_only:
        c = config["preflight"]
        result = run_trial(
            config,
            out,
            commands,
            args.preflight_name,
            c["friction"],
            c["mass_kg"],
            c["seed"],
            c["jitter_enabled"],
        )
        print(json.dumps(result), flush=True)
        return 0 if result["status"] == "completed" and result["metrics"]["success"] else 1
    preflight_path = out / args.preflight_name / "trial-report.json"
    if not preflight_path.is_file():
        raise ValueError("Run the instrumented native preflight first")
    preflight = json.loads(preflight_path.read_text(encoding="utf8"))
    if (
        preflight["status"] != "completed"
        or preflight["metrics"]["peak_contact_force_N"] <= 0
        or preflight["metrics"]["maximum_lift_mm"] < 50
    ):
        raise ValueError("The preflight did not establish an instrumented physical grasp/lift")
    if (out / "sweep-report.json").exists():
        print("Existing sweep report found; not rerunning completed sweep", flush=True)
        existing = json.loads((out / "sweep-report.json").read_text(encoding="utf8"))
        return 0 if existing["acceptance_passed"] else 1
    frozen = out / "grid-code-snapshot"
    package = Path(__file__).resolve().parents[1]
    if not frozen.exists():
        shutil.copytree(package, frozen / "simforge", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        legacy = frozen / "legacy/ue_isaac_pipeline"
        legacy.mkdir(parents=True)
        for name in ("__init__.py", "unitree_wbc.py"):
            shutil.copy2(Path(config["legacy_source"]) / "ue_isaac_pipeline" / name, legacy / name)
        manifest = {p.relative_to(frozen).as_posix(): sha(p) for p in frozen.rglob("*.py")}
        write(frozen / "source-manifest.json", manifest)
    else:
        manifest = json.loads((frozen / "source-manifest.json").read_text(encoding="utf8"))
        if any(sha(frozen / name) != value for name, value in manifest.items()):
            raise ValueError("Frozen worker source changed")
    frozen_config = copy.deepcopy(config)
    frozen_config["worker_pythonpath"] = str(frozen)
    frozen_config["legacy_source"] = str(frozen / "legacy")
    write(out / "resolved-config.json", frozen_config)
    jobs = []
    for mass in config["grid"]["mass_kg"]:
        for friction in config["grid"]["friction"]:
            for seed in config["grid"]["seeds"]:
                ident = f"mass{mass:.1f}_mu{friction:.1f}_seed{seed:02d}"
                case = copy.deepcopy(frozen_config)
                case["portable_root"] += "-slot" + str(len(jobs) % config["max_parallel_workers"])
                jobs.append((case, ident, friction, mass, seed))
    if len(jobs) != 45:
        raise ValueError("The approved primary design contains exactly 45 trials")
    results = {}
    slot_locks = {case["portable_root"]: threading.Lock() for case, _, _, _, _ in jobs}

    def isolated_trial(case, ident, friction, mass, seed):
        with slot_locks[case["portable_root"]]:
            return run_trial(case, out, commands, ident, friction, mass, seed)

    print("M2 GRID START", len(jobs), "primary trials; fixed commands", sha(commands), flush=True)
    with ThreadPoolExecutor(max_workers=config["max_parallel_workers"]) as pool:
        pending = {
            pool.submit(isolated_trial, case, ident, friction, mass, seed): ident
            for case, ident, friction, mass, seed in jobs
        }
        for future in as_completed(pending):
            ident = pending[future]
            result = future.result()
            results[ident] = result
            print(
                "M2 GRID RESULT",
                len(results),
                "/45",
                ident,
                result["status"],
                result.get("metrics"),
                flush=True,
            )
            write(
                out / "sweep-progress.json",
                dict(
                    completed=len(results),
                    total=45,
                    native_completed=sum(r["status"] == "completed" for r in results.values()),
                    successes=sum(r.get("metrics", {}).get("success", False) for r in results.values()),
                ),
            )
    ordered = [results[ident] for _, ident, _, _, _ in jobs]
    write(out / "results.json", ordered)
    repeat_cfg = config["reproducibility"]
    reference = next(
        r
        for r in ordered
        if (r["friction"], r["mass_kg"], r["seed"])
        == (repeat_cfg["friction"], repeat_cfg["mass_kg"], repeat_cfg["seed"])
    )
    case = next(case for case, ident, _, _, _ in jobs if ident == reference["trial_id"])
    repeat = run_trial(
        case, out, commands, "repro-repeat", reference["friction"], reference["mass_kg"], reference["seed"]
    )
    same = (
        reference["status"] == repeat["status"] == "completed" and reference["metrics"] == repeat["metrics"]
    )
    comparison = dict(
        status="passed" if same else "failed",
        reference=reference["trial_id"],
        repeat="repro-repeat",
        exact_metric_equality=same,
        reference_metrics=reference.get("metrics"),
        repeat_metrics=repeat.get("metrics"),
    )
    write(out / "reproducibility.json", comparison)
    final = summarize(config, out, ordered, comparison)
    print("M2 FINAL", json.dumps(final), flush=True)
    return 0 if final["acceptance_passed"] else 1


def summarize(config, out, results, repro):
    """Persist every outcome and a mass-stratified grid; never infer an untested cutoff."""
    from PIL import Image, ImageDraw, ImageFont

    fields = [
        "trial_id",
        "friction",
        "mass",
        "seed",
        "success",
        "slip_distance_mm",
        "peak_contact_force_N",
        "drop_time_s",
        "settle_time_s",
        "maximum_lift_mm",
        "longest_retention_s",
        "status",
        "failure_reasons",
    ]
    with (out / "raw.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in results:
            m = r.get("metrics", {})
            row = {key: m.get(key) for key in fields}
            row.update(
                trial_id=r["trial_id"],
                friction=r["friction"],
                mass=r["mass_kg"],
                seed=r["seed"],
                status=r["status"],
                failure_reasons="|".join(m.get("failure_reasons", []))
                if m
                else r.get("error", "missing_metrics"),
            )
            writer.writerow(row)
    board = Image.new("RGB", (1100, 850), "#101821")
    pen = ImageDraw.Draw(board)
    font_path = Path(os.environ.get("WINDIR", "")) / "Fonts/consola.ttf"
    font = ImageFont.truetype(str(font_path), 26) if font_path.exists() else ImageFont.load_default()
    pen.text((40, 24), "M2 measured grasp success | 5 seeds per cell", font=font, fill="white")
    cells = []
    for row, mass in enumerate(config["grid"]["mass_kg"]):
        pen.text((26, 210 + row * 185), f"{mass:.1f}kg", font=font, fill="white")
        for col, friction in enumerate(config["grid"]["friction"]):
            matches = [r for r in results if r["mass_kg"] == mass and r["friction"] == friction]
            completed = [r for r in matches if r["status"] == "completed"]
            passed = sum(r["metrics"]["success"] for r in completed)
            rate = passed / len(completed) if completed else None
            x, y = 170 + col * 290, 150 + row * 185
            color = (
                (int(190 * (1 - rate) + 35 * rate), int(55 * (1 - rate) + 155 * rate), 65)
                if rate is not None
                else (70, 70, 70)
            )
            pen.rectangle((x, y, x + 260, y + 155), fill=color)
            pen.text((x + 30, y + 35), f"{passed}/{len(completed)}", font=font, fill="white")
            pen.text(
                (x + 30, y + 85),
                f"{rate * 100:.0f}%" if rate is not None else "UNMEASURED",
                font=font,
                fill="white",
            )
            if row == 0:
                pen.text((x + 50, 100), f"mu={friction:.1f}", font=font, fill="white")
            cells.append(
                dict(
                    mass_kg=mass,
                    friction=friction,
                    successes=passed,
                    native_trials=len(completed),
                    planned_trials=5,
                    success_rate=rate,
                    errors=len(matches) - len(completed),
                )
            )
    pen.text((40, 760), "lift >=50mm | slip <=5mm/0.5s | hold >=2s", font=font, fill="white")
    pen.text((40, 805), "Simulation estimates; no real-material calibration or new RL", fill="#bac8d4")
    board.save(out / "heatmap.png")
    native = [r for r in results if r["status"] == "completed"]
    successes = sum(r["metrics"]["success"] for r in native)
    checks = dict(
        all_45_native_trials_completed=len(native) == 45,
        observed_success_and_failure=0 < successes < len(native),
        same_seed_exact_reproducibility=repro["exact_metric_equality"],
    )
    boundaries = {}
    for mass in config["grid"]["mass_kg"]:
        row = [c for c in cells if c["mass_kg"] == mass]
        fails = [c["friction"] for c in row if c["native_trials"] == 5 and c["successes"] == 0]
        passes = [c["friction"] for c in row if c["successes"] > 0]
        bracket = next(
            ([lo, hi] for lo in sorted(fails, reverse=True) for hi in sorted(passes) if lo < hi), None
        )
        boundaries[str(mass)] = dict(
            tested_cells=row,
            observed_failure_to_success_bracket=bracket,
            universal_slip_cutoff=None,
            reason="Only tested points; five samples each; no universal threshold inferred",
        )
    result = dict(
        status="completed" if all(checks.values()) else "completed_failed_acceptance",
        acceptance_passed=all(checks.values()),
        checks=checks,
        primary_trials=45,
        native_trials=len(native),
        successes=successes,
        failures=len(native) - successes,
        runtime_errors=45 - len(native),
        cells=cells,
        mass_stratified_boundaries=boundaries,
        grasp_predicate="50mm lift + <=5mm slip/first .5s + >=2s retention",
        user_acceptance="pending_manual_review",
        new_rl_training=False,
    )
    write(out / "sweep-report.json", result)
    lines = [
        "# M2 measured friction/mass sweep",
        "",
        f"Native primary trials: {len(native)}/45; successes: {successes}; physical failures: {len(native) - successes}; runtime errors: {45 - len(native)}.",
        f"Acceptance passed: {result['acceptance_passed']}. Exact same-seed metric equality: {repro['exact_metric_equality']}.",
        "",
        "| Mass kg | Friction | Successes/native | Planned |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {c['mass_kg']} | {c['friction']} | {c['successes']}/{c['native_trials']} | 5 |" for c in cells
    ]
    lines += [
        "",
        "Slip boundary: report the mass-specific tested cells/brackets in sweep-report.json. An untested universal friction cutoff is NOT_MEASURED.",
        "If all trials succeeded or all failed, the approved boundary gate failed: stop without changing thresholds or generating a replacement grid.",
        "",
        "Fixed upper/finger sequence and body commands, unchanged official leg feedback, normal gravity/free base. 200Hz filtered normal-force sensors. No new policy training or real-world coefficient calibration.",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf8")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
