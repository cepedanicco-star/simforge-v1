"""Capture already sampled M3 episodes; stop on any native export failure."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.simforge.randomize.sampler import compare_values, sample  # noqa: E402


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--root")
    parser.add_argument("--only-variant", type=int)
    parser.add_argument("--reuse-v01-root", help="Read-only original sampled metadata; do not draw new variants")
    args = parser.parse_args()
    config_path = Path(args.config).resolve(strict=True)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf8"))
    root = Path(args.root or cfg["output"])
    original = json.loads((root / "resolved-config.json").read_text(encoding="utf8"))
    for key in cfg["dimensions"]:
        cfg["dimensions"][key]["enabled"] = original["dimensions"][key]["enabled"]
    write(root / "render-config.json", cfg)
    sensor_config = yaml.safe_load((config_path.parent / cfg["sensor_config"]).read_text(encoding="utf8"))
    env = dict(os.environ)
    runtime = cfg["native_runtime"]
    env.update(
        AISIM_PORTABLE_ROOT=runtime["portable_root"],
        AISIM_H5PY_OVERLAY=runtime["h5py_overlay"],
        TEMP=runtime["temp"],
        TMP=runtime["temp"],
        PYTHONPATH=str(Path(__file__).resolve().parents[2]),
    )
    folders = sorted(root.glob("var_*"))
    if args.only_variant is not None:
        folders = [root / f"var_{args.only_variant:02d}"]
    for folder in folders:
        meta_path = folder / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf8"))
        expected_cfg = copy.deepcopy(cfg)
        if args.reuse_v01_root:
            source_meta = Path(args.reuse_v01_root) / folder.name / "meta.json"
            original_meta = json.loads(source_meta.read_text(encoding="utf8"))
            if compare_values(meta, original_meta):
                raise ValueError("Reused variant differs from immutable original dimensions")
            if meta["reused_source_sha256"] != hashlib.sha256(source_meta.read_bytes()).hexdigest():
                raise ValueError("Original metadata checksum differs from reuse receipt")
            expected_cfg["dimensions"]["D5"]["require_free_space"] = False
        if compare_values(meta, sample(expected_cfg, meta["variant_id"])):
            raise ValueError("Current config differs from saved samples; do not overwrite")
        out = folder / "frames"
        if (out / "export-report.json").exists():
            report = json.loads((out / "export-report.json").read_text(encoding="utf8"))
            if report["status"] != "completed":
                raise ValueError("Incomplete/failed capture retained: " + str(out))
            print("M3 REUSE completed native export", folder.name, flush=True)
            continue
        if out.exists():
            if list(out.iterdir()):
                raise ValueError("Nonempty previous capture without a final report: " + str(out))
            out.rmdir()  # Only this owned, verified-empty placeholder; never recursive.
        sensor = dict(sensor_config)
        sensor.update(
            scene=cfg["scene"],
            output=str(out),
            resolution=cfg["resolution"],
            frames=cfg["frames_per_variant"],
            renderer="PathTracing",
            time_code_start=0,
            randomization=dict(config=str(root / "render-config.json"), meta=str(meta_path)),
        )
        write(folder / "capture-config.json", sensor)
        with (folder / "capture.console.log").open("w", encoding="utf8") as log:
            process = subprocess.Popen(
                [
                    runtime["python"],
                    "-m",
                    "pipeline.simforge.sensors.export",
                    "--config",
                    str(folder / "capture-config.json"),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            started = time.monotonic()
            while process.poll() is None:
                if (out / "export-report.json").exists():
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        process.wait(timeout=20)
                    break
                if time.monotonic() - started > runtime["timeout_s"]:
                    process.terminate()
                    process.wait(timeout=20)
                    raise TimeoutError("Owned native capture timed out: " + folder.name)
                time.sleep(1)
        report = json.loads((out / "export-report.json").read_text(encoding="utf8"))
        if report["status"] != "completed":
            raise RuntimeError(
                "Native M3 gate failed; retain evidence and stop: " + json.dumps(report.get("failures"))
            )
        meta.update(
            status="native_capture_completed",
            applied=json.loads((out / "randomization-applied.json").read_text(encoding="utf8")),
            capture=dict(
                frames=report["frames"],
                source_layers_unchanged=report["source_layers_unchanged"],
                native_export_script_sha256=report["script_sha256"],
                report=str(out / "export-report.json"),
            ),
        )
        write(meta_path, meta)
        print("M3 CAPTURE completed", folder.name, "frames", report["frames"], flush=True)


if __name__ == "__main__":
    main()
