"""Measured CLI orchestration; native engine jobs run in the configured SDK."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from . import __version__


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf8")


def unchanged(report):
    hashes = report.get("source_layer_hashes", {})
    return bool(hashes) and all(Path(p).is_file() and digest(p) == sha for p, sha in hashes.items())


def native_job(config, module, arguments, output_report, log_path):
    runtime = config["native_runtime"]
    env = dict(os.environ, AISIM_PORTABLE_ROOT=runtime["portable_root"],
               AISIM_H5PY_OVERLAY=runtime["h5py_overlay"], TEMP=runtime["temp"], TMP=runtime["temp"],
               PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    started = time.monotonic()
    with Path(log_path).open("w", encoding="utf8") as log:
        process = subprocess.Popen([runtime["python"], "-m", module, *arguments], env=env,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        while process.poll() is None:
            if Path(output_report).is_file():
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired as error:
                    print("Native report exists; owned process shutdown timed out; terminating owned PID", process.pid, flush=True)
                    process.terminate()
                    process.wait(timeout=20)
                    raise RuntimeError("Native shutdown timeout; owned process terminated") from error
                break
            if time.monotonic()-started > runtime["timeout_s"]:
                process.terminate()
                process.wait(timeout=20)
                raise TimeoutError("Native module timeout: " + module)
            time.sleep(.5)
    if not Path(output_report).is_file():
        raise RuntimeError(f"Native module returned {process.returncode} without report; see {log_path}")
    report = read(output_report)
    if report["status"] not in ("PASS", "completed"):
        raise RuntimeError("Native module FAIL: " + str(report.get("failures", report.get("error"))))
    return time.monotonic()-started


def render(config, frames, output):
    import yaml
    output = Path(output)
    report_path = output / "export-report.json"
    if report_path.is_file():
        report = read(report_path)
        if report["status"] == "completed" and report["frames"] == frames and report["scene"] == str(Path(config["scene"]).resolve()) and unchanged(report):
            print("REUSE verified native export", str(output), flush=True)
            return None
        raise ValueError("Existing export differs or failed; use a fresh --out")
    sensor = yaml.safe_load(Path(config["sensor_config"]).read_text(encoding="utf8"))
    sensor.update(scene=config["scene"], frames=frames, output=str(output), renderer="RaytracedLighting")
    sensor_path = output.parent / "sensor-config.json"
    write(sensor_path, sensor)
    return native_job(config, "pipeline.simforge.sensors.export", ["--config", str(sensor_path)], report_path,
                      output.parent / "render.console.log")


def validate(config, output):
    output = Path(output)
    report_path = output / "validation.json"
    if report_path.is_file():
        old = read(report_path)
        if old["status"] == "PASS" and str(Path(old["scene"]).resolve()) == str(Path(config["scene"]).resolve()) and unchanged(old):
            print("REUSE validation8/8", flush=True)
            return old
        raise ValueError("Previous validation failed or source changed; use a fresh --out")
    native_config = output / "native-config.json"
    write(native_config, config)
    structural = output / "structural-validation.json"
    native_job(config, "pipeline.simforge.validate", ["--config", str(native_config), "--out", str(structural)],
               structural, output / "validate.console.log")
    result = read(structural)
    elapsed = render(config, 1, output / "render")
    if elapsed is not None:
        write(output / "timings.json", dict(render_end_to_end_s=elapsed))
    rendered = read(output / "render/export-report.json")
    frame = output / "render/frame_000000"
    from PIL import Image
    import numpy as np
    camera = read(frame / "camera.json")
    checks = [np.asarray(Image.open(frame / name)).size > 0 for name in ("rgb.png", "semantic.png", "instance.png")]
    depth = np.load(frame / "depth.npy", allow_pickle=False)
    checks.append(bool(np.isfinite(depth).any() and (depth > 0).any()))
    principal = max(abs(camera["cx"]-camera["width"]/2), abs(camera["cy"]-camera["height"]/2))
    if not all(checks) or principal > 2 or not rendered["source_layers_unchanged"]:
        raise ValueError("Native four-layer or camera calibration check failed")
    for item in result["checks"]:
        if item["number"] == 6:
            item.update(reason="actual native camera K and world-to-CV extrinsic verified", fx=camera["fx"], fy=camera["fy"],
                        cx=camera["cx"], cy=camera["cy"], principal_error_px=principal)
    result["checks"].append(dict(number=7, name="native four-layer render", status="PASS",
                                 reason="one freshly rendered RGB/depth/semantic/instance frame", frame=str(frame)))
    result["checks"].sort(key=lambda row: row["number"])
    result["status"] = "PASS"
    write(report_path, result)
    for row in result["checks"]:
        print(f"VALIDATE {row['number']} {row['status']} {row['name']}: {row['reason']}", flush=True)
    return result


def randomize(config, variants):
    accepted = read(config["m3_acceptance"])
    if accepted["status"] != "PASS" or variants != len(accepted["variants"]):
        raise ValueError("Requested variants do not match completed M3 cache; explicit new collection is required")
    for index in range(variants):
        folder = Path(config["m3_data"]) / f"var_{index:02d}"
        meta = read(folder / "meta.json")
        report = read(folder / "frames/export-report.json")
        if report["status"] != "completed" or not unchanged(report) or not meta["applied"]["D5"]["camera_safety"]["accepted"]:
            raise ValueError("M3 cache not valid: " + str(folder))
    print(f"RANDOMIZE PASS {variants}/{variants} verified reused native variants; no new draws", flush=True)
    return accepted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command")
    default_config = str(Path(__file__).resolve().parents[2] / "configs/pipeline_office.yaml")
    for name in ("validate", "render", "randomize", "report", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", default=default_config)
        sub.add_argument("--scene", default="apartment")
        sub.add_argument("--out")
        sub.add_argument("--usd", help="Override USD for explicit validation fixtures")
        if name == "render":
            sub.add_argument("--frames", type=int)
        if name == "randomize":
            sub.add_argument("--variants", type=int, default=6)
        if name in ("report", "run"):
            sub.add_argument("--baseline", default="apartment")
        if name == "run":
            sub.add_argument("--resume", action="store_true", help="Resume a failed matching run; preserve its failure receipt")
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    started = time.monotonic()
    output = None
    result = dict(status="running", command=args.command, failures=[])
    try:
        import yaml
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf8"))
        if args.scene not in config["aliases"]:
            raise ValueError("Unknown scene alias: " + args.scene)
        if getattr(args, "baseline", "apartment") != "apartment":
            raise ValueError("Only the measured apartment baseline is configured")
        if args.usd:
            config["scene"] = str(Path(args.usd).resolve(strict=True))
        if args.out:
            config["output"] = str(Path(args.out).resolve())
        output = Path(config["output"])
        output.mkdir(parents=True, exist_ok=True)
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        completed = output / "run-report.json"
        if args.command == "run" and completed.is_file():
            old = read(completed)
            if old["status"] == "PASS" and old["config_sha256"] == config_hash and unchanged(read(output / "validation.json")):
                randomize(config, 6)
                print("RUN PASS: completed matching pipeline reused; no files regenerated", flush=True)
                return 0
            if not args.resume or old["status"] != "FAIL" or read(output / "native-config.json") != config or not unchanged(read(output / "validation.json")):
                raise ValueError("Previous run failed or input changed; explicit --resume or fresh --out required")
            history = output / "attempts"
            history.mkdir(exist_ok=True)
            destination = history / f"run-failure-{len(list(history.glob('run-failure-*')))+1:02d}.json"
            shutil.copyfile(completed, destination)
            print("RESUME matching inputs; previous failure retained", str(destination), flush=True)
        if args.command in ("validate", "run"):
            validate(config, output)
        if args.command in ("render", "run"):
            count = getattr(args, "frames", None) or config["render_frames"]
            elapsed = render(config, count, output / "render")
            if elapsed is not None:
                write(output / "timings.json", dict(render_end_to_end_s=elapsed))
        if args.command in ("randomize", "run"):
            randomize(config, getattr(args, "variants", 6))
        if args.command == "run":
            from .dataset.coco import export
            dataset = output / "dataset"
            if (dataset / "coco-report.json").is_file():
                coco = read(dataset / "coco-report.json")
                instances = read(dataset / "instances.json")
                if coco["status"] != "PASS" or len(instances["images"]) != coco["images"] or len(instances["annotations"]) != coco["annotations"]:
                    raise ValueError("Existing COCO dataset is incomplete")
                for item in coco["sources"]:
                    if digest(Path(item["frame"]) / "rgb.png") != item["rgb_sha256"] or digest(dataset / f"images/{item['image_id']:06d}.png") != item["rgb_sha256"]:
                        raise ValueError("COCO source or copied RGB checksum changed")
                print("REUSE verified COCO dataset; no files regenerated", flush=True)
            else:
                coco = export([config["m1_data"], config["m3_data"]], dataset)
            print(f"COCO PASS images={coco['images']} annotations={coco['annotations']} exact-mask-roundtrips=True", flush=True)
        if args.command in ("report", "run"):
            from .eval.report import generate
            generated = generate(config, output / "reports" / config["scene_name"], time.monotonic()-started)
            print("REPORT PASS", generated["output"], flush=True)
        result.update(status="PASS", elapsed_s=time.monotonic()-started, config_sha256=config_hash)
    except Exception as error:
        result.update(status="FAIL", error=repr(error), elapsed_s=time.monotonic()-started)
        result["failures"].append(dict(error=repr(error), traceback=traceback.format_exc()))
        traceback.print_exc()
    if output is not None:
        write(output / ("run-report.json" if args.command == "run" else f"{args.command}-command-report.json"), result)
    print(args.command.upper(), result["status"], "failures=" + str(len(result["failures"])), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
