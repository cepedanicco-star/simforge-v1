"""Render native measured M2 poses, without running a second physical experiment."""

# ruff: noqa: E402
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", required=True, help="Completed trial directory")
    parser.add_argument("--out", required=True, help="Fresh render directory")
    parser.add_argument(
        "--preview", action="store_true", help="One representative frame, separate fresh directory"
    )
    args = parser.parse_args()
    trial, out = Path(args.trial).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    spec = json.loads((trial / "trial.json").read_text(encoding="utf8"))
    measured = json.loads((trial / "trial-report.json").read_text(encoding="utf8"))
    cfg = spec["config"]
    if (
        measured["status"] != "completed"
        or digest(trial / "measurements.npz") != measured["recording_sha256"]
    ):
        raise ValueError("A complete, hash-verified physical recording is required")
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    if os.environ.get("AISIM_H5PY_OVERLAY"):
        sys.path.insert(0, os.environ["AISIM_H5PY_OVERLAY"])
    handles = []
    if os.name == "nt":
        sdk = Path(sys.executable).parent
        for folder in (sdk / "Library/bin", sdk / "Lib/site-packages/torch/lib"):
            if folder.is_dir():
                os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")
                handles.append(os.add_dll_directory(str(folder)))
    report = dict(
        status="running",
        trial_id=spec["trial_id"],
        measurements_sha256=measured["recording_sha256"],
        native_renderer="Isaac Sim 4.5 RTX",
        physics_resimulated=False,
        playback_speed=2,
        scope="M2 fixed grasp/lift experiment; not full apartment transfer or new RL",
    )
    app = None
    try:
        import numpy as np
        from isaacsim import SimulationApp

        width, height, fps = cfg["demo"]["width"], cfg["demo"]["height"], cfg["demo"]["fps"]
        app = SimulationApp(
            dict(
                headless=True,
                width=width,
                height=height,
                renderer="RaytracedLighting",
                extra_args=["--portable-root", os.environ["AISIM_PORTABLE_ROOT"]],
            )
        )
        print("M2 RENDER app initialized", flush=True)
        import omni.usd
        import omni.timeline
        import omni.replicator.core as rep
        from PIL import Image, ImageDraw, ImageFont
        from pxr import Gf, UsdGeom
        from pipeline.ue_isaac_pipeline.physics_replay import MeasuredPoseBridge, disable_replay_physics

        if not omni.usd.get_context().open_stage(str(trial / "Trial_scene.usda")):
            raise RuntimeError("Native trial stage could not open")
        print("M2 RENDER stage opened", flush=True)
        for _ in range(12):
            app.update()
        stage = omni.usd.get_context().get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        hashes = {
            layer.realPath: digest(layer.realPath)
            for layer in stage.GetUsedLayers()
            if layer.realPath and Path(layer.realPath).is_file()
        }
        disable_replay_physics(stage)
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        timeline.set_current_time(0)
        data = np.load(trial / "measurements.npz", allow_pickle=False)
        paths = ["/World/G1/" + str(name) for name in data["body_names"]]
        paths.append(cfg["contact"]["cup_prim"])
        bridge = MeasuredPoseBridge(stage, paths)
        camera = UsdGeom.Camera.Define(stage, "/World/M2DemoCamera")
        eye, target = (4.35, -1.22, 1.48), (3.60, -1.98, 1.055)
        matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
        UsdGeom.Xformable(camera).MakeMatrixXform().Set(matrix)
        camera.CreateFocalLengthAttr(35)
        camera.CreateHorizontalApertureAttr(36)
        camera.CreateVerticalApertureAttr(36 * height / width)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000))
        product = rep.create.render_product(str(camera.GetPath()), (width, height))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach(product)
        rep.orchestrator.set_capture_on_play(False)
        font = ImageFont.truetype(str(Path(os.environ.get("WINDIR", "")) / "Fonts/consola.ttf"), 20)
        small = ImageFont.truetype(str(Path(os.environ.get("WINDIR", "")) / "Fonts/consola.ttf"), 16)
        frames = out / "frames"
        frames.mkdir()
        source_times = (
            [7.2]
            if args.preview
            else np.minimum(
                np.arange(round(cfg["demo"]["display_seconds"] * fps)) / fps * 2 + 0.02,
                data["frame_time_s"][-1],
            )
        )
        mappings, max_p, max_q = [], 0.0, 0.0
        for number, source_time in enumerate(source_times):
            i = int(np.abs(data["frame_time_s"] - source_time).argmin())
            poses = np.concatenate((data["body_poses"][i], data["cup_poses"][i : i + 1]))
            bridge.apply(poses)
            errors = bridge.verify(poses)
            max_p = max(max_p, errors["position_error_m"])
            max_q = max(max_q, errors["rotation_error_rad"])
            for _ in range(4 if number == 0 else 1):
                rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=True)
            for attempt in range(12):
                pixels = np.asarray(rgb.get_data())[..., :3]
                stats = dict(
                    shape=list(pixels.shape), standard_deviation=float(pixels.std()) if pixels.size else None
                )
                if pixels.shape == (height, width, 3) and pixels.std() >= 2:
                    break
                print("M2 RENDER RGB warmup", number, attempt, stats, flush=True)
                rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=True)
            else:
                report["invalid_rgb"] = stats
                if pixels.shape == (height, width, 3):
                    Image.fromarray(pixels.astype(np.uint8)).save(out / "invalid-rgb.png")
                raise ValueError("Empty or invalid native RGB: " + str(stats))
            image = Image.fromarray(pixels.astype(np.uint8))
            image.save(frames / f"native_{number:04d}.png") if args.preview else None
            draw = ImageDraw.Draw(image)
            local = matrix.GetInverse().Transform(Gf.Vec3d(*map(float, data["cup_poses"][i, :3])))
            if local[2] < -0.01:
                fx = width * 35 / 36
                u, v = width / 2 + fx * local[0] / -local[2], height / 2 - fx * local[1] / -local[2]
                draw.ellipse((u - 10, v - 10, u + 10, v + 10), outline="#8ee9f2", width=2)
                draw.line((u + 12, v - 12, u + 75, v - 75), fill="#8ee9f2", width=2)
                draw.text(
                    (u + 79, v - 91),
                    "cup_V1 origin",
                    font=small,
                    fill="#8ee9f2",
                    stroke_width=2,
                    stroke_fill="#101821",
                )
            draw.rectangle((0, 0, width, 102), fill="#101821")
            metrics = measured["metrics"]
            label = "PASS" if metrics["success"] else "FAIL"
            slip = metrics["slip_distance_mm"]
            slip_text = "UNMEASURED" if slip is None else f"{slip:.2f}mm"
            draw.text(
                (16, 9),
                f"{label} | mu={spec['friction']:.1f} | mass={spec['mass_kg']:.1f}kg | seed={spec['seed']}",
                font=font,
                fill="#70e2a0" if metrics["success"] else "#ffb59d",
            )
            draw.text(
                (16, 36),
                f"slip={slip_text} | lift={metrics['maximum_lift_mm']:.1f}mm | hold={metrics['longest_retention_s']:.2f}s",
                font=font,
                fill="white",
            )
            draw.text(
                (16, 67),
                f"Isaac Sim measured-state replay | 2x | sim t={data['frame_time_s'][i]:.2f}s",
                font=small,
                fill="#bbcbd8",
            )
            sample = int(np.abs(data["time_s"] - data["frame_time_s"][i]).argmin())
            force = float(np.linalg.norm(data["finger_forces_N"][sample], axis=1).sum())
            draw.rectangle((0, height - 38, width, height), fill="#101821")
            draw.text(
                (16, height - 29),
                f"Filtered normal force {force:.2f}N | Gate: lift>=50mm, slip<=5mm, hold>=2s",
                font=small,
                fill="white",
            )
            image.save(frames / f"frame_{number:04d}.png")
            mappings.append(
                dict(
                    frame=number,
                    recorded_index=i,
                    simulation_time_s=float(data["frame_time_s"][i]),
                    pose_errors=errors,
                )
            )
            if number % 25 == 0:
                print("M2 RENDER", spec["trial_id"], number, "/", len(source_times), flush=True)
        if not args.preview:
            subprocess.run(
                [
                    cfg["demo"]["ffmpeg"],
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-n",
                    "-framerate",
                    str(fps),
                    "-i",
                    str(frames / "frame_%04d.png"),
                    "-c:v",
                    "libx264",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(out / "demo.mp4"),
                ],
                check=True,
            )
        unchanged = all(digest(path) == value for path, value in hashes.items())
        if not unchanged:
            raise RuntimeError("A source USD layer changed")
        report.update(
            status="completed",
            frames=len(mappings),
            resolution=[width, height],
            fps=fps,
            camera_eye=eye,
            camera_target=target,
            camera_focal_length_mm=35,
            source_layers_unchanged=unchanged,
            source_layers=len(hashes),
            target_marker="Projection of measured cup origin; not an image detector or occlusion test",
            frame_map=mappings,
            maximum_pose_position_error_m=max_p,
            maximum_pose_rotation_error_rad=max_q,
            trial_metrics=measured["metrics"],
        )
    except Exception as error:
        report.update(status="error", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        (out / "render-report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
