"""Capture a new, session-labeled M1 dataset without changing source USD files."""

# ruff: noqa: E402
import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    return value


def write(path, value):
    Path(path).write_text(json.dumps(plain(value), ensure_ascii=False, indent=2), encoding="utf8")


def related(a, b):
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


def native_boxes(value):
    paths = value["info"]["primPaths"]
    return [dict(prim_path=str(path), **{key: plain(row[key]) for key in row.dtype.names})
            for path, row in zip(paths, value["data"], strict=True)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--frames", type=int)
    p.add_argument("--out")
    args = p.parse_args()
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Use Isaac Sim 4.5 Python 3.10")
    import yaml
    config_path = Path(args.config).resolve(strict=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf8"))
    if args.frames is not None:
        config["frames"] = args.frames
    if args.out:
        config["output"] = args.out
    count = config["frames"]
    if not isinstance(count, int) or not 1 <= count <= 10000:
        raise ValueError("frames must be 1..10000")
    source = Path(config["scene"]).resolve(strict=True)
    out = Path(config["output"]).resolve()
    if out.exists():
        raise FileExistsError("Use a fresh output directory: " + str(out))
    out.mkdir(parents=True)
    report = dict(status="running", requested_frames=count, frames=0, failures=[],
                  scene=str(source), source_sha256=digest(source), config=plain(config),
                  new_behavior_training=False, source_scene_saved=False)
    app, handles = None, []
    if os.environ.get("AISIM_H5PY_OVERLAY"):
        sys.path.insert(0, os.environ["AISIM_H5PY_OVERLAY"])
    if os.name == "nt":
        sdk = Path(sys.executable).parent
        for directory in (sdk / "Library/bin", sdk / "Lib/site-packages/torch/lib"):
            if directory.is_dir():
                os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
                handles.append(os.add_dll_directory(str(directory)))
    try:
        import numpy as np
        from isaacsim import SimulationApp
        width, height = config["resolution"]
        app = SimulationApp(dict(headless=True, width=width, height=height, renderer=config.get("renderer", "RaytracedLighting"),
                                 extra_args=["--portable-root", os.environ["AISIM_PORTABLE_ROOT"]]))
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from PIL import Image, ImageDraw
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics, Semantics
        from .annotators import attach, read
        from .geometry import backproject, camera_contract, native_box_world, require_labels, uint16_ids

        if not omni.usd.get_context().open_stage(str(source)):
            raise RuntimeError("Native USD open failed")
        for _ in range(15):
            app.update()
        stage = omni.usd.get_context().get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        source_hashes = {layer.realPath: digest(layer.realPath) for layer in stage.GetUsedLayers()
                         if layer.realPath and Path(layer.realPath).is_file()}
        if UsdGeom.GetStageMetersPerUnit(stage) != 1:
            raise ValueError("Source root does not declare metre units")
        if not stage.GetPrimAtPath(config["camera"]):
            raise ValueError("Configured camera is missing")
        randomization = None
        if config.get("randomization"):
            from pipeline.simforge.randomize.apply_visual import apply as apply_visual
            from pipeline.simforge.randomize.apply_physics import apply as apply_physics
            randomization_config = json.loads(Path(config["randomization"]["config"]).read_text(encoding="utf8"))
            randomization_sample = json.loads(Path(config["randomization"]["meta"]).read_text(encoding="utf8"))
            randomization = apply_visual(stage, randomization_config, randomization_sample, config["camera"])
            randomization["D6"] = apply_physics(stage, randomization_config, randomization_sample)
            write(out / "randomization-applied.json", randomization)
        objects = json.loads(Path(config["label_plan"]).read_text(encoding="utf8"))["objects"]
        labels = {obj["prepared_prim_path"]: obj["class_name"].lower() for obj in objects}
        labels.update(config["label_overrides"])
        require_labels(labels, config["required_labels"])
        for path, label in labels.items():
            prim = stage.GetPrimAtPath(path)
            if not prim:
                raise ValueError("Label mapping references a missing source object: " + path)
            api = Semantics.SemanticsAPI.Apply(prim, "Semantics")
            api.CreateSemanticTypeAttr().Set("class")
            api.CreateSemanticDataAttr().Set(label)
        write(out / "session-labels.json", dict(labels=labels, source="existing prim mapping plus explicit targets",
                                                unknown_classes_remain_unreviewed=True))
        if randomization is not None:
            stage.GetSessionLayer().Export(str(out / "randomization-opinions.usda"))
            variant_layer = Sdf.Layer.CreateNew(str(out / "Variant_scene.usda"))
            variant_layer.subLayerPaths = ["randomization-opinions.usda", source.as_posix()]
            variant_layer.defaultPrim = "World"
            variant_stage = Usd.Stage.Open(variant_layer)
            UsdGeom.SetStageMetersPerUnit(variant_stage, 1)
            UsdGeom.SetStageUpAxis(variant_stage, "Z")
            UsdPhysics.SetStageKilogramsPerUnit(variant_stage, 1)
            variant_layer.Save()
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        rep.orchestrator.set_capture_on_play(False)
        product = rep.create.render_product(config["camera"], (width, height))
        sensors = attach(product, config["channels"])
        extra_sensors = {}
        if randomization is not None:
            import carb
            settings = carb.settings.get_settings()
            settings.set_bool("/rtx/pathtracing/illuminanceAOV", True)
            settings.set_int("/rtx/pathtracing/spp", 16)
            settings.set_int("/rtx/pathtracing/totalSpp", 64)
            if "PtIlluminance" not in rep.AnnotatorRegistry.get_registered_annotators():
                rep.AnnotatorRegistry.register_annotator_from_aov("PtIlluminance", output_data_type=np.float32, output_channels=4)
            for name in ("PtIlluminance", "PtWorldPos", "PtWorldNormal"):
                extra_sensors[name] = rep.AnnotatorRegistry.get_annotator(name)
                extra_sensors[name].attach(product)
        frames, maximum_box_error, target_depth_samples = [], 0.0, 0
        minimum_depth_consistency = 1.0
        demo = out / "demo"
        demo.mkdir()
        for number in range(count):
            tc = config["time_code_start"] + number * config["time_code_step"]
            timeline.set_current_time(tc / stage.GetTimeCodesPerSecond())
            for _ in range(3 if number == 0 else 1):
                rep.orchestrator.step(rt_subframes=config["rt_subframes"], delta_time=0.0, pause_timeline=True)
            values = read(sensors)
            folder = out / f"frame_{number:06d}"
            folder.mkdir()
            rgb = Image.fromarray(np.asarray(values["rgb"])[..., :3].astype(np.uint8))
            depth = np.asarray(values["distance_to_image_plane"], dtype=np.float32).squeeze()
            semantic = np.asarray(values["semantic_segmentation"]["data"]).squeeze()
            instance = np.asarray(values["instance_segmentation"]["data"]).squeeze()
            sem_png, sem_map = uint16_ids(semantic)
            inst_png, inst_map = uint16_ids(instance)
            rgb.save(folder / "rgb.png")
            Image.fromarray(sem_png).save(folder / "semantic.png")
            Image.fromarray(inst_png).save(folder / "instance.png")
            np.save(folder / "depth.npy", depth)
            np.save(folder / "normals.npy", np.asarray(values["normals"]))
            if extra_sensors:
                illumination = np.asarray(extra_sensors["PtIlluminance"].get_data(), dtype=np.float32)
                if illumination.shape != (height, width, 4) or not np.isfinite(illumination).all() or illumination[..., :3].max() <= 0:
                    raise ValueError("M3 native illuminance AOV is empty or invalid")
                lux = (illumination[..., :3] @ np.asarray([.2126, .7152, .0722],dtype=np.float32)).astype(np.float32)
                positions = np.asarray(extra_sensors["PtWorldPos"].get_data(), dtype=np.float32)
                world_normals = np.asarray(extra_sensors["PtWorldNormal"].get_data(), dtype=np.float32)
                np.save(folder / "illuminance_lux.npy", lux)
                np.save(folder / "world_positions.npy", positions)
                support_path = randomization_config["dimensions"]["D3"]["support_prim"]
                support_ids = [int(key) for key,path in values["instance_segmentation"]["info"]["idToLabels"].items() if isinstance(path,str) and related(path,support_path)]
                cup_xyz = np.asarray(UsdGeom.XformCache(0).GetLocalToWorldTransform(stage.GetPrimAtPath("/World/Environment/Source/cup_V1")).ExtractTranslation())
                patch = np.isin(instance,support_ids) & (np.linalg.norm(positions[...,:2]-cup_xyz[:2],axis=-1)<.35) & (np.abs(positions[...,2]-cup_xyz[2])<.15) & (world_normals[...,2]>.9) & np.isfinite(lux)
                write(folder / "illumination.json", dict(unit="lux",source="Native RTX PtIlluminance AOV",reference_request_lux=randomization["D1"]["reference_illuminance_lux"] if "D1" in randomization else None,whole_scene_uniform_lux_claimed=False,counter_patch_pixels=int(patch.sum()),counter_patch_median_lux=float(np.median(lux[patch])) if patch.any() else None,maximum_lux=float(lux.max()),real_world_calibrated=False))
            camera = camera_contract(values["camera_params"])
            if [camera["width"], camera["height"]] != [width, height]:
                raise ValueError("Native camera resolution differs from RGB")
            center_error = max(abs(camera["cx"] - width / 2), abs(camera["cy"] - height / 2))
            if center_error > config["acceptance"]["max_principal_point_error_px"]:
                raise ValueError("Principal point discrepancy exceeds the configured gate")
            write(folder / "camera.json", camera)
            raw_2d = native_boxes(values["bounding_box_2d_tight"])
            raw_3d = native_boxes(values["bounding_box_3d"])
            write(folder / "native_annotations.json", dict(
                boxes_2d_tight=raw_2d, boxes_2d_loose=native_boxes(values["bounding_box_2d_loose"]),
                boxes_3d=raw_3d, camera_params=values["camera_params"],
                semantic_info=values["semantic_segmentation"]["info"],
                instance_info=values["instance_segmentation"]["info"]))
            sem_labels = values["semantic_segmentation"]["info"]["idToLabels"]
            inst_labels = values["instance_segmentation"]["info"]["idToLabels"]
            records, boxes, boxes3d, depth_count = {}, [], [], 0
            overlay = rgb.copy()
            pen = ImageDraw.Draw(overlay)
            for native_id in np.unique(instance):
                path = inst_labels.get(str(int(native_id)))
                if not isinstance(path, str) or not path.startswith("/"):
                    continue
                mask = instance == native_id
                ys, xs = np.where(mask)
                sem_ids, sizes = np.unique(semantic[mask], return_counts=True)
                label = sem_labels[str(int(sem_ids[np.argmax(sizes)]))]["class"]
                encoded = inst_map[int(native_id)]
                records[str(encoded)] = dict(native_id=int(native_id), prim_path=path, semantic_label=label)
                native = [box for box in raw_2d if related(box["prim_path"], path)]
                if not native:
                    raise ValueError("No native tight box for visible instance: " + path)
                tight = [min(box["x_min"] for box in native), min(box["y_min"] for box in native),
                         max(box["x_max"] for box in native), max(box["y_max"] for box in native)]
                mask_box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                discrepancy = float(np.abs(np.asarray(tight) - mask_box).max())
                maximum_box_error = max(maximum_box_error, discrepancy)
                if discrepancy > config["acceptance"]["max_box_edge_error_px"]:
                    raise ValueError(f"Native box/mask discrepancy {discrepancy}px: {path}")
                box = dict(id=encoded, native_id=int(native_id), semantic_label=label,
                           x_min=int(tight[0]), y_min=int(tight[1]), x_max=int(tight[2]), y_max=int(tight[3]),
                           pixel_bounds="native tight coordinates; mask edge agreement checked",
                           mask_bbox_inclusive=mask_box, maximum_edge_error_px=discrepancy)
                boxes.append(box)
                pen.rectangle(tight, outline="#40ffc6", width=2)
                pen.text((tight[0] + 2, max(0, tight[1] - 14)), f"{encoded}: {label}", fill="white",
                         stroke_width=1, stroke_fill="black")
                for item in raw_3d:
                    if related(item["prim_path"], path):
                        boxes3d.append(dict(id=encoded, semantic_label=label, native_prim_path=item["prim_path"],
                                            **native_box_world(item)))
                if label == "cup_v1":
                    step = config["acceptance"]["depth_sample_stride"]
                    sy, sx = ys[::step], xs[::step]
                    z = depth[sy, sx]
                    valid = np.isfinite(z) & (z > 0)
                    points = backproject(np.c_[sx[valid], sy[valid]], z[valid], camera)
                    bounds = UsdGeom.BBoxCache(Usd.TimeCode(tc), ["default", "render"]).ComputeWorldBound(
                        stage.GetPrimAtPath("/World/Environment/Source/cup_V1")).ComputeAlignedRange()
                    tolerance = config["acceptance"]["depth_target_tolerance_m"]
                    inside = np.all((points >= np.asarray(bounds.GetMin()) - tolerance) &
                                    (points <= np.asarray(bounds.GetMax()) + tolerance), axis=1)
                    if not len(points) or not inside.all():
                        raise ValueError("Cup depth backprojection does not fit the paired USD geometry bounds")
                    minimum_depth_consistency = min(minimum_depth_consistency, float(inside.mean()))
                    depth_count += len(points)
            if not any(record["semantic_label"] == "cup_v1" for record in records.values()):
                raise ValueError("Target cup has no labeled visible pixels")
            if not boxes3d:
                raise ValueError("No 3D boxes for visible instances")
            write(folder / "boxes_2d.json", boxes)
            write(folder / "boxes_3d.json", boxes3d)
            write(folder / "labels.json", dict(instance_ids=records,
                                               native_to_png_instance=inst_map, native_to_png_semantic=sem_map,
                                               semantic_native_labels=sem_labels))
            overlay.save(folder / "check_overlay.png")
            ids, inverse = np.unique(instance, return_inverse=True)
            palette = np.random.default_rng(31).integers(20, 245, (len(ids), 3), dtype=np.uint8)
            palette[ids < 2] = 0
            instances = Image.fromarray(palette[inverse].reshape(height, width, 3))
            gray = np.clip(np.nan_to_num(depth, nan=config["demo"]["depth_far_m"],
                                         posinf=config["demo"]["depth_far_m"]) /
                           config["demo"]["depth_far_m"] * 255, 0, 255).astype(np.uint8)
            board = Image.new("RGB", (1280, 720))
            for i, image in enumerate([rgb, overlay, instances, Image.fromarray(gray).convert("RGB")]):
                x, y = (i % 2) * 640, (i // 2) * 360
                board.paste(image.resize((640, 360)), (x, y))
                ImageDraw.Draw(board).text((x + 10, y + 10), ["Native RGB", "Native tight boxes",
                                            "Instance IDs", "Metric z-depth"][i], fill="white",
                                            stroke_width=2, stroke_fill="black")
            board.save(demo / f"frame_{number:06d}.png")
            if number == 0:
                board.save(out / "demo-preview.png")
            target_depth_samples += depth_count
            frames.append(dict(frame=number, time_code=tc, instances=len(records), boxes_3d=len(boxes3d),
                               principal_point_error_px=center_error, depth_samples=depth_count))
            report["frames"] = len(frames)
            print(f"M1 FRAME {number + 1}/{count}: {len(records)} instances; max bbox error {maximum_box_error}px", flush=True)
        changed = [path for path, sha in source_hashes.items() if digest(path) != sha]
        if changed or digest(source) != report["source_sha256"]:
            raise ValueError("A source USD layer changed")
        command = [config["demo"]["ffmpeg"], "-hide_banner", "-loglevel", "warning", "-n",
                   "-framerate", str(config["demo"]["fps"]), "-i", str(demo / "frame_%06d.png"),
                   "-c:v", "libx264", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                   str(out / "m1-sensor-demo.mp4")]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        report.update(status="completed", frame_audit=frames, session_label_objects=len(labels),
                      source_layers_unchanged=True, source_layer_hashes=source_hashes,
                      maximum_box_edge_error_px=maximum_box_error, target_depth_samples=target_depth_samples,
                      depth_samples_inside_target_bounds_fraction=minimum_depth_consistency,
                      session_layer_saved=randomization is not None, script_sha256=digest(__file__))
        write(out / "resolved-config.json", config)
    except Exception as error:
        report.update(status="failed", error=repr(error))
        report["failures"].append(dict(error=repr(error), traceback=traceback.format_exc()))
        traceback.print_exc()
        raise
    finally:
        write(out / "export-report.json", report)
        print("M1 EXPORT RESULT", report["status"], "frames", report["frames"], flush=True)
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
