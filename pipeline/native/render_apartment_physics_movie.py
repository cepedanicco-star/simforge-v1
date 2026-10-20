"""Render a measured apartment motion with the saved cam01 spatial trajectory.

Final transfer delivery is gated by a successful source physics report. The
explicit --allow-incomplete option creates a development replay, never a claimed
successful transfer. Physics is disabled only in the replay layer.
"""
# ruff: noqa: E402

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.ue_isaac_pipeline.io import read_json, write_json, digest_file
from pipeline.ue_isaac_pipeline.recording_frames import recording_frames
from pipeline.ue_isaac_pipeline.review_camera import review_camera_eyes
from pipeline.ue_isaac_pipeline.camera_projection import equivalent_native_lens

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--recording", required=True)
p.add_argument("--out", required=True)
p.add_argument("--start-seconds", type=float, default=0)
p.add_argument(
    "--duration",
    type=float,
    help="Optional video duration; selects original states in order and reports the time compression",
)
p.add_argument("--allow-incomplete", action="store_true")
p.add_argument("--max-frames", type=int)
p.add_argument(
    "--preserve-source-fov",
    action="store_true",
    help="Uniformly scale focal/aperture to avoid the native small-focal clamp while preserving UE pinhole field of view",
)
p.add_argument("--layers-every", type=int, default=25)
p.add_argument("--diagnostics", action="store_true")
p.add_argument(
    "--evidence-hud",
    action="store_true",
    help="Native ground-truth boxes and measured control/contact overlay, separate from clean RGB",
)
p.add_argument(
    "--closeups",
    action="store_true",
    help="Additional measured-state diagnostic stills, separate from cam01 video",
)
p.add_argument(
    "--body-review",
    action="store_true",
    help="Separate 960x540 full-body diagnostic movie; cam01 remains unchanged",
)
p.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
p.add_argument("--ffprobe", default=shutil.which("ffprobe"))
args = p.parse_args()
source = Path(args.recording).resolve(strict=True)
out = Path(args.out).resolve()
out.mkdir(parents=True, exist_ok=True)
if (out / "render-report.json").exists():
    raise FileExistsError(out)
task_report = read_json(source / "probe-report.json")
complete = task_report.get("full_transfer_completed") is True and task_report.get("status") == "completed"
if complete and task_report.get("probe_arguments", {}).get("human_forward_transfer"):
    forward_audit = read_json(source / "forward-motion-audit.json")
    if forward_audit.get("status") != "completed" or forward_audit.get("source_npz_sha256") != digest_file(
        source / "physics-body-states.npz"
    ):
        raise RuntimeError("Verified body-forward motion audit required before final capture")
if not complete and not args.allow_incomplete:
    raise RuntimeError(
        "A final transfer video requires a completed source physics transfer; use --allow-incomplete only for labeled development previews"
    )
if not args.ffmpeg or not Path(args.ffprobe).is_file():
    raise RuntimeError("FFmpeg and FFprobe are required")
for key in ("AISIM_H5PY_OVERLAY",):
    if os.environ.get(key):
        sys.path.insert(0, os.environ[key])
handles = []
if os.name == "nt":
    sdk = Path(sys.executable).parent
    for folder in (sdk / "bin", sdk / "Library/bin", sdk / "Lib/site-packages/torch/lib"):
        if folder.is_dir():
            os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")
            handles.append(os.add_dll_directory(str(folder)))
import h5py  # noqa: F401
import numpy as np
from isaacsim import SimulationApp

data = np.load(source / "physics-body-states.npz", allow_pickle=False)
hz, fps = int(data["control_hz"]), 25
if hz != 50 or len(data["states"]) != len(data["cup"]):
    raise ValueError("Expected paired 50 Hz body/cup records")
indices = recording_frames(len(data["states"]), hz, fps, args.start_seconds, args.duration)
if args.max_frames:
    indices = indices[: args.max_frames]
if not len(indices):
    raise ValueError("No measured frames in the requested interval")
duration = len(indices) / fps
source_seconds = (int(indices[-1]) - int(indices[0])) / hz + 1 / fps
if complete and not 20 <= duration <= 35:
    raise ValueError("Final delivery must contain 20–35 seconds of measured motion")
app = SimulationApp(
    dict(
        headless=True,
        width=2048,
        height=1152,
        renderer="RaytracedLighting",
        extra_args=["--portable-root", os.environ["AISIM_PORTABLE_ROOT"]],
    )
)
encoders = []
report = dict(
    status="starting",
    full_transfer_completed=complete,
    development_preview=not complete,
    source_recording=str(source),
    source_npz_sha256=digest_file(source / "physics-body-states.npz"),
    resolution=[2048, 1152],
    fps=fps,
    duration_seconds=duration,
    source_motion_seconds=source_seconds,
    motion_playback_rate=source_seconds / duration,
    temporal_resampling=args.duration is not None,
    source_task_report=task_report,
)
try:
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
    from PIL import Image, ImageDraw, ImageFont
    from isaacsim.core.utils.semantics import add_update_semantics
    from pipeline.ue_isaac_pipeline.physics_replay import MeasuredPoseBridge, disable_replay_physics
    from pipeline.serialization import jsonable as _jsonable
    from pipeline.ue_isaac_pipeline.calibration import calibrate_records, export_calibration

    layer = Sdf.Layer.CreateNew(str(out / "Measured_replay.usdc"))
    layer.subLayerPaths = [Path(os.path.relpath(source / "Apartment_sim.usda", out)).as_posix()]
    layer.timeCodesPerSecond = layer.framesPerSecond = fps
    layer.startTimeCode, layer.endTimeCode = 0, len(indices) - 1
    layer.defaultPrim = "World"
    old_robot = Sdf.CreatePrimInLayer(layer, "/World/Robot")
    old_robot.active = False
    old_robot.referenceList.ClearEditsAndMakeExplicit()
    layer.customLayerData = {
        "aisim:measured_physics_replay": True,
        "aisim:full_transfer_completed": complete,
        "aisim:live_physics": False,
    }
    # Stage metadata does not compose through sublayers. Author the original
    # apartment's SI units on this new root before the renderer sees the stage.
    replay_stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageMetersPerUnit(replay_stage, 1.0)
    UsdGeom.SetStageUpAxis(replay_stage, UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(replay_stage, 1.0)
    layer.Save()
    if not omni.usd.get_context().open_stage(layer.realPath):
        raise RuntimeError("Replay scene could not open")
    stage = omni.usd.get_context().get_stage()
    disable_replay_physics(stage)
    paths = ["/World/G1/" + str(n) for n in data["body_names"]] + ["/World/Environment/Source/cup_V1"]
    bridge = MeasuredPoseBridge(stage, paths)
    original = UsdGeom.Camera(stage.GetPrimAtPath("/World/Cameras/cam01/CameraComponent"))
    times = []
    prim = original.GetPrim()
    while prim and not prim.IsPseudoRoot():
        for attr in prim.GetAttributes():
            if attr.GetName().startswith("xformOp:") or prim == original.GetPrim():
                times.extend(attr.GetTimeSamples())
        prim = prim.GetParent()
    if not times or max(times) <= min(times):
        raise ValueError("cam01 has no saved motion samples")
    source_range = [float(min(times)), float(max(times))]
    # Read every original pose/lens value before authoring a baked capture camera.
    source_camera_rows = []
    source_lenses = []
    lens_gauge_scales = []
    lens_names = (
        "FocalLength",
        "HorizontalAperture",
        "VerticalAperture",
        "HorizontalApertureOffset",
        "VerticalApertureOffset",
    )
    for i in range(len(indices)):
        t = source_range[0] + i / max(len(indices) - 1, 1) * (source_range[1] - source_range[0])
        source_lens = {name: getattr(original, "Get" + name + "Attr")().Get(t) for name in lens_names}
        source_lenses.append(source_lens)
        native_lens, gauge_scale = (
            equivalent_native_lens(source_lens) if args.preserve_source_fov else (source_lens, 1)
        )
        lens_gauge_scales.append(gauge_scale)
        source_camera_rows.append(
            (
                UsdGeom.XformCache(t).GetLocalToWorldTransform(original.GetPrim()),
                native_lens,
            )
        )
    camera = UsdGeom.Camera.Define(stage, "/World/Cameras/cam01_measured_replay")
    camera.GetPrim().SetMetadata("displayName", "cam01 · measured replay")
    camera.SetResetXformStack(True)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100))
    camera.CreateFStopAttr(0)
    camera_op = camera.AddTransformOp()
    camera_op.Set(source_camera_rows[0][0])
    for name, value in source_camera_rows[0][1].items():
        getattr(camera, "Get" + name + "Attr")().Set(value)
    product = rep.create.render_product(str(camera.GetPath()), (2048, 1152))
    annotators = {"rgb": rep.AnnotatorRegistry.get_annotator("rgb")}
    if args.layers_every:
        annotators["camera_params"] = rep.AnnotatorRegistry.get_annotator("CameraParams")
    if args.layers_every or args.diagnostics:
        for actor in stage.GetPrimAtPath("/World/Environment/Source").GetChildren():
            add_update_semantics(actor, actor.GetName())
        add_update_semantics(stage.GetPrimAtPath("/World/G1"), "robot_g1")
        for key, name in (
            ("depth", "distance_to_image_plane"),
            ("semantic", "semantic_segmentation"),
            ("normals", "normals"),
        ):
            annotators[key] = rep.AnnotatorRegistry.get_annotator(name)
    for annotator in annotators.values():
        annotator.attach(product)
    evidence_boxes = None
    if args.evidence_hud:
        evidence_boxes = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
        evidence_boxes.attach(product)
        evidence_telemetry = read_json(source / "telemetry.json")
        evidence_times = np.array([r["time"] for r in evidence_telemetry])
        evidence_rows = []
        hud_font = ImageFont.truetype(str(Path(os.environ.get("WINDIR", "")) / "Fonts/consola.ttf"), 23)
    body_camera_rows = []
    if args.body_review:
        body_map = Path(task_report["probe_arguments"]["transfer_plan"]).parent / "navigation-map.npz"
        pelvis_index = data["body_names"].tolist().index("pelvis")
        body_eyes = review_camera_eyes(data["states"][indices, pelvis_index, :3], np.load(body_map))
        body_camera = UsdGeom.Camera.Define(stage, "/World/Cameras/body_direction_review")
        body_camera.SetResetXformStack(True)
        # Keep this diagnostic lens above the native renderer's small-focal
        # clamp, then use aperture to frame the complete robot comfortably.
        body_camera.CreateFocalLengthAttr(0.50)
        body_camera.CreateHorizontalApertureAttr(1.0)
        body_camera.CreateVerticalApertureAttr(0.5625)
        body_camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100))
        body_camera.CreateFStopAttr(0)
        body_op = body_camera.AddTransformOp()
        body_product = rep.create.render_product(str(body_camera.GetPath()), (960, 540))
        body_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        body_rgb.attach(body_product)
        pelvis_index = data["body_names"].tolist().index("pelvis")
    layers = out / "layers"
    layers.mkdir(exist_ok=True)
    log = open(out / "ffmpeg.log", "wb")
    encoder_sizes = {}

    def encoder(name, size=(2048, 1152)):
        target = out / (name + ".partial.mp4")
        process = subprocess.Popen(
            [
                args.ffmpeg,
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{size[0]}x{size[1]}",
                "-framerate",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(target),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        encoders.append((process, target, out / (name + ".mp4")))
        encoder_sizes[str(out / (name + ".mp4"))] = size
        return process

    rgb_encoder = encoder("apartment-transfer" if complete else "grasp-development")
    diagnostic_encoder = encoder("simulation-layers") if args.diagnostics else None
    body_encoder = encoder("body-forward-review", (960, 540)) if args.body_review else None
    evidence_encoder = encoder("physics-training-evidence") if args.evidence_hud else None
    checks, frame_rows, samples = [], [], []
    for i, record_index in enumerate(indices):
        poses = np.concatenate([data["states"][record_index], data["cup"][record_index][None]])
        bridge.apply(poses)
        error = bridge.verify(poses)
        matrix, lens = source_camera_rows[i]
        camera_op.Set(matrix)
        for name, value in lens.items():
            getattr(camera, "Get" + name + "Attr")().Set(value)
        if args.body_review:
            body_target = data["states"][record_index, pelvis_index, :3].astype(float)
            body_eye = body_eyes[i]
            body_matrix = (
                Gf.Matrix4d()
                .SetLookAt(Gf.Vec3d(*body_eye), Gf.Vec3d(*body_target), Gf.Vec3d(0, 0, 1))
                .GetInverse()
            )
            body_op.Set(body_matrix)
            body_camera_rows.append(body_matrix)
        for _ in range(3 if i == 0 else 1):
            rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=True)
        rgb = np.ascontiguousarray(np.asarray(annotators["rgb"].get_data())[..., :3], dtype=np.uint8)
        if rgb.shape != (1152, 2048, 3):
            raise RuntimeError("RGB dimensions differ from requested output")
        if not 1 < float(rgb.mean()) < 254:
            raise RuntimeError("RGB capture is nearly blank; inspect the saved camera and renderer")
        rgb_encoder.stdin.write(rgb.tobytes())
        if body_encoder is not None:
            body_pixels = np.ascontiguousarray(np.asarray(body_rgb.get_data())[..., :3], dtype=np.uint8)
            if body_pixels.shape != (540, 960, 3) or not 1 < float(body_pixels.mean()) < 254:
                raise ValueError("Invalid full-body diagnostic frame")
            body_encoder.stdin.write(body_pixels.tobytes())
        image = Image.fromarray(rgb)
        if evidence_encoder is not None:
            native = evidence_boxes.get_data()
            boxes = [
                {name: _jsonable(row[name]) for name in native["data"].dtype.names} for row in native["data"]
            ]
            labels = _jsonable(native.get("info", {}).get("idToLabels", {}))
            overlay = image.copy()
            draw = ImageDraw.Draw(overlay)
            for box in boxes:
                label = str(labels.get(str(box["semanticId"]), {}).get("class", ""))
                if not (label.lower() in ("cup_v1", "robot_g1") or "desk_01" in label or "monitor" in label):
                    continue
                xy = [box.get(k, 0) for k in ("x_min", "y_min", "x_max", "y_max")]
                if not (0 <= xy[0] < xy[2] <= 2048 and 0 <= xy[1] < xy[3] <= 1152):
                    continue
                draw.rectangle(xy, outline="#40ffc6", width=3)
                draw.text(
                    (xy[0] + 3, max(0, xy[1] - 20)),
                    "GT: " + label,
                    fill="#40ffc6",
                    stroke_width=1,
                    stroke_fill="black",
                    font=hud_font,
                )
            ti = int(np.argmin(np.abs(evidence_times - (int(record_index) + 1) / hz)))
            measured = evidence_telemetry[ti]
            forces = np.asarray(data["grip_forces_world_N"][record_index])
            force_sum = float(np.linalg.norm(forces, axis=-1).sum())
            command = data["body_commands"][record_index]
            actual_velocity = np.asarray(measured["root_velocity"])
            text = [
                f"PHYSX {int(record_index) / hz:.2f}s | VIDEO {i / fps:.2f}s | playback {source_seconds / duration:.2f}x",
                f"phase: {str(data['phases'][record_index])}",
                f"command vx={command[0]:+.3f} m/s  vy={command[1]:+.3f}  wz={command[2]:+.3f} rad/s",
                f"measured world velocity: {np.array2string(actual_velocity[:3], precision=3)} m/s (telemetry)",
                f"grip sum |F|={force_sum:.2f} N (50Hz) | desk support={measured.get('desk_support_force_N', 0):.2f} N (5Hz)",
                f"opposing finger contact={measured.get('opposing_finger_contact')} | cup tilt={measured.get('cup_tilt_deg', 0):.2f} deg",
                "GT boxes = renderer annotations, not detector predictions; official legs + learned governor + arm IK/contact",
            ]
            draw.rectangle((18, 920, 1740, 1136), fill="#08121e")
            for j, line in enumerate(text):
                draw.text((34, 935 + j * 27), line, fill="white", font=hud_font)
            evidence_encoder.stdin.write(overlay.tobytes())
            if i % 125 == 0:
                overlay.save(out / f"evidence-{i:06d}.jpg", quality=94)
            evidence_rows.append(
                dict(
                    frame=i,
                    record_index=int(record_index),
                    telemetry_index=ti,
                    telemetry_time_s=float(evidence_times[ti]),
                    native_tight_boxes=boxes,
                    id_to_labels=labels,
                    grip_force_sum_N=force_sum,
                    command=command.tolist(),
                )
            )
        values, info = {}, {}
        save_layers = args.layers_every and (i % args.layers_every == 0 or i == len(indices) - 1)
        if args.diagnostics or save_layers:
            values["depth"] = np.asarray(annotators["depth"].get_data()).squeeze().copy()
            seg = annotators["semantic"].get_data()
            values["semantic"] = np.asarray(seg["data"]).squeeze().copy()
            info = seg["info"]
            values["normals"] = np.asarray(annotators["normals"].get_data()).copy()
        if save_layers:
            image.save(layers / f"rgb-{i:06}.png")
            if body_encoder is not None:
                Image.fromarray(body_pixels).save(out / f"body-review-{i:06}.png")
            np.savez_compressed(layers / f"sensors-{i:06}.npz", **values)
            write_json(layers / f"labels-{i:06}.json", info)
            checks.append(dict(frame=i, record_index=int(record_index), **error))
            parameters = _jsonable(annotators["camera_params"].get_data())
            eye = np.asarray(matrix)[3, :3]
            forward = -np.asarray(matrix)[2, :3]
            forward /= np.linalg.norm(forward)
            cup_ids = [
                int(key)
                for key, label in info.get("idToLabels", {}).items()
                if str(label.get("class", "")).lower() == "cup_v1"
            ]
            mask = np.isin(values["semantic"], cup_ids)
            yy, xx = np.where(mask)
            samples.append(
                dict(
                    frame=i,
                    record_index=int(record_index),
                    camera_params=parameters,
                    camera_eye_m=eye.tolist(),
                    camera_look_at_m=(eye + forward).tolist(),
                    cup_visible_pixels=len(xx),
                    bbox_xywh=[
                        int(xx.min()),
                        int(yy.min()),
                        int(xx.max() - xx.min() + 1),
                        int(yy.max() - yy.min() + 1),
                    ]
                    if len(xx)
                    else None,
                    rgb=f"layers/rgb-{i:06}.png",
                    sensors=f"layers/sensors-{i:06}.npz",
                    independent_episode_holdout=False,
                    randomized=False,
                )
            )
            calibrate_records([samples[-1]])
            if args.preserve_source_fov:
                actual_projection = np.asarray(parameters["cameraProjection"]).reshape(4, 4)
                expected_projection = [
                    2 * source_lenses[i]["FocalLength"] / source_lenses[i][name]
                    for name in ("HorizontalAperture", "VerticalAperture")
                ]
                if not np.allclose(
                    actual_projection[[0, 1], [0, 1]], expected_projection, rtol=2e-5, atol=1e-6
                ):
                    raise ValueError(
                        f"Native camera field of view differs from the original UE lens ratio: {actual_projection[[0, 1], [0, 1]].tolist()} vs {expected_projection}"
                    )
        if diagnostic_encoder:
            depth = np.nan_to_num(values["depth"], nan=8.0, posinf=8.0, neginf=0.0)
            d = np.clip(depth / 8 * 255, 0, 255).astype(np.uint8)
            palette = np.random.default_rng(4).integers(
                20, 245, (max(int(values["semantic"].max()) + 1, 2), 3), dtype=np.uint8
            )
            palette[0] = 0
            panels = [
                image,
                Image.fromarray(np.repeat(d[..., None], 3, axis=-1)),
                Image.fromarray(palette[values["semantic"]]),
                Image.fromarray(np.clip((values["normals"][..., :3] + 1) * 127.5, 0, 255).astype(np.uint8)),
            ]
            diagnostic = Image.new("RGB", (2048, 1152))
            pen = ImageDraw.Draw(diagnostic)
            for j, panel in enumerate(panels):
                x, y = (j % 2) * 1024, (j // 2) * 576
                diagnostic.paste(panel.resize((1024, 576)), (x, y))
                pen.text(
                    (x + 12, y + 12),
                    ["RGB", "Depth 0-8 m", "Semantic", "Normals"][j],
                    fill="white",
                    stroke_width=2,
                    stroke_fill="black",
                )
            diagnostic_encoder.stdin.write(diagnostic.tobytes())
        frame_rows.append(
            dict(
                frame=i,
                record_index=int(record_index),
                nominal_control_time_seconds=(int(record_index) + 1) / hz,
                phase=str(data["phases"][record_index]) if "phases" in data else None,
                video_time_seconds=i / fps,
                source_camera_time=source_range[0]
                + i / max(len(indices) - 1, 1) * (source_range[1] - source_range[0]),
            )
        )
        if i % 25 == 0:
            print(f"MEASURED MOVIE {i + 1}/{len(indices)}", flush=True)
            write_json(
                out / "render-progress.json",
                dict(frames=i + 1, total=len(indices), development_preview=not complete),
            )
    for process, temp, target in encoders:
        process.stdin.close()
        if process.wait(timeout=120):
            raise RuntimeError("FFmpeg encoding failed; inspect ffmpeg.log")
        probe = json.loads(
            subprocess.check_output(
                [
                    args.ffprobe,
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,nb_read_frames,r_frame_rate,duration",
                    "-of",
                    "json",
                    str(temp),
                ]
            )
        )["streams"][0]
        if (int(probe["width"]), int(probe["height"]), int(probe["nb_read_frames"])) != (
            *encoder_sizes[str(target)],
            len(indices),
        ) or abs(float(probe["duration"]) - duration) > 0.05:
            raise RuntimeError("Encoded video dimensions/frame count/duration mismatch")
        os.replace(temp, target)
    closeups = []
    if args.closeups and "phases" in data:
        for label, phase_name in [
            ("grasp", "hold"),
            ("carry", "navigate_apartment_with_cup"),
            ("body-right-turn", "turn_right_with_cup"),
            ("body-forward-walk", "navigate_apartment_with_cup"),
            ("body-left-turn", "turn_left_to_desk"),
            ("release", "release_on_desk"),
            ("placed", "verify_placement"),
        ]:
            available = np.flatnonzero(data["phases"] == phase_name)
            if not len(available):
                continue
            ri = int(available[len(available) // 2])
            poses = np.concatenate([data["states"][ri], data["cup"][ri][None]])
            bridge.apply(poses)
            error = bridge.verify(poses)
            target_pos = data["cup"][ri, :3].astype(float) + [0, 0, 0.04]
            eye = target_pos + [0.35, -0.18, 0.25]
            original_focal_length = camera.GetFocalLengthAttr().Get()
            original_apertures = (
                camera.GetHorizontalApertureAttr().Get(),
                camera.GetVerticalApertureAttr().Get(),
            )
            if label.startswith("body-"):
                pelvis_index = data["body_names"].tolist().index("pelvis")
                target_pos = data["states"][ri, pelvis_index, :3].astype(float)
                review_map = (
                    Path(task_report["probe_arguments"]["transfer_plan"]).parent / "navigation-map.npz"
                )
                eye = review_camera_eyes(target_pos[None], np.load(review_map))[0]
                camera.GetFocalLengthAttr().Set(0.50)
                camera.GetHorizontalApertureAttr().Set(1.0)
                camera.GetVerticalApertureAttr().Set(0.5625)
            camera_op.Set(
                Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target_pos), Gf.Vec3d(0, 0, 1)).GetInverse()
            )
            for _ in range(3):
                rep.orchestrator.step(rt_subframes=3, delta_time=0.0, pause_timeline=True)
            pixels = np.asarray(annotators["rgb"].get_data())[..., :3].astype(np.uint8)
            path = out / ("closeup-" + label + ".png")
            Image.fromarray(pixels).save(path)
            closeups.append(
                dict(
                    label=label,
                    path=str(path),
                    record_index=ri,
                    usable=bool(1 < pixels.mean() < 254),
                    **error,
                )
            )
            camera.GetFocalLengthAttr().Set(original_focal_length)
            camera.GetHorizontalApertureAttr().Set(original_apertures[0])
            camera.GetVerticalApertureAttr().Set(original_apertures[1])
    # Persist a separate disabled-physics animation for manual replay inspection.
    for i, record_index in enumerate(indices):
        poses = np.concatenate([data["states"][record_index], data["cup"][record_index][None]])
        bridge.apply(poses, time=i)
        camera_op.Set(source_camera_rows[i][0], i)
        for name, value in source_camera_rows[i][1].items():
            getattr(camera, "Get" + name + "Attr")().Set(value, i)
        if args.body_review:
            body_op.Set(body_camera_rows[i], i)
    bridge.apply(np.concatenate([data["states"][indices[0]], data["cup"][indices[0]][None]]))
    camera_op.Set(source_camera_rows[0][0])
    if args.body_review:
        body_op.Set(body_camera_rows[0])
    stage.GetRootLayer().Save()
    report.update(
        status="completed",
        rendered_frames=len(indices),
        camera_source_time_range=source_range,
        camera_trajectory_preserved=True,
        source_pinhole_field_of_view_preserved=args.preserve_source_fov,
        native_lens_gauge_scales=lens_gauge_scales,
        source_camera_lenses=source_lenses,
        replay_scene=layer.realPath,
        replay_camera=str(camera.GetPath()),
        live_physics_in_replay=False,
        pose_checks=checks,
        videos=[str(t) for _, _, t in encoders if t.name != "body-forward-review.mp4"],
        body_direction_review=dict(
            path=str(out / "body-forward-review.mp4"),
            resolution=[960, 540],
            frames=len(indices),
            camera=str(body_camera.GetPath()),
            same_measured_frame_indices=True,
            viewpoint_rule="Collision-map clear camera and sightline; auxiliary viewpoint may switch",
            navigation_map=str(body_map),
            navigation_map_sha256=digest_file(body_map),
        )
        if args.body_review
        else None,
        diagnostic_closeups=closeups,
    )
    write_json(out / "frame-pairing.json", frame_rows)
    if args.evidence_hud:
        write_json(out / "evidence-annotations.json", evidence_rows)
        report["evidence_hud"] = dict(
            path=str(out / "physics-training-evidence.mp4"),
            native_ground_truth_boxes=True,
            detector_confidence_fabricated=False,
            same_measured_frame_indices=True,
            grip_force_hz=50,
            support_telemetry_hz=5,
            telemetry_interpolation="nearest recorded 5Hz sample; index saved per frame",
        )
    if samples:
        write_json(out / "samples.json", samples)
        report["camera_calibration"] = export_calibration(out)
    write_json(out / "render-report.json", report)
except BaseException:
    import traceback

    report.update(status="failed", error=traceback.format_exc())
    write_json(out / "render-report.json", report)
    raise
finally:
    for process, _, _ in encoders:
        if process.poll() is None:
            process.kill()
    app.close()
