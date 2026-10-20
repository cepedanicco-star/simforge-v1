"""Read-only native Isaac Sim 4.5 annotator probe; not the M1 dataset exporter."""

# ruff: noqa: E402
import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

NAMES = (
    "rgb", "distance_to_image_plane", "semantic_segmentation",
    "instance_segmentation", "bounding_box_2d_tight", "bounding_box_2d_loose",
    "bounding_box_3d", "camera_params", "pointcloud", "normals",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    return value


def describe(value):
    """Describe arrays without printing their contents."""
    if isinstance(value, dict):
        return {str(k): describe(v) for k, v in value.items()}
    if hasattr(value, "shape"):
        return dict(type=type(value).__name__, shape=list(value.shape), dtype=str(value.dtype))
    return dict(type=type(value).__name__, length=len(value) if hasattr(value, "__len__") else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--camera", default="/World/Cameras/cam01/CameraComponent")
    p.add_argument("--width", type=int, default=2048)
    p.add_argument("--height", type=int, default=1152)
    args = p.parse_args()
    scene, out = Path(args.scene).resolve(strict=True), Path(args.out).resolve()
    if out.exists():
        raise FileExistsError("Use a new probe output directory: " + str(out))
    out.mkdir(parents=True)
    if not 64 <= min(args.width, args.height) <= max(args.width, args.height) <= 4096:
        raise ValueError("Unsupported probe resolution")
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Use Isaac Sim 4.5 Python 3.10")
    before = digest(scene)
    report = dict(status="running", scene=str(scene), scene_sha256=before,
                  attempted=[], failures=[], preview_only=True, physics_stepped=False)
    app, handles = None, []
    if os.environ.get("AISIM_H5PY_OVERLAY"):
        sys.path.insert(0, os.environ["AISIM_H5PY_OVERLAY"])
    if os.name == "nt":
        sdk = Path(sys.executable).parent
        for folder in (sdk / "Library/bin", sdk / "Lib/site-packages/torch/lib"):
            if folder.is_dir():
                os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")
                handles.append(os.add_dll_directory(str(folder)))
    try:
        import numpy as np
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=True, width=args.width, height=args.height,
                                 renderer="RaytracedLighting",
                                 extra_args=["--portable-root", os.environ["AISIM_PORTABLE_ROOT"]]))
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from PIL import Image, ImageDraw
        from pxr import UsdGeom

        if not omni.usd.get_context().open_stage(str(scene)):
            raise RuntimeError("Native stage open failed")
        for _ in range(15):
            app.update()
        context = omni.usd.get_context()
        stage = context.get_stage()
        # All probe edits go to an anonymous session layer; no source layer is saved.
        stage.SetEditTarget(stage.GetSessionLayer())
        omni.timeline.get_timeline_interface().stop()
        if UsdGeom.GetStageMetersPerUnit(stage) != 1:
            raise ValueError("The composed root must declare metres")
        camera = UsdGeom.Camera(stage.GetPrimAtPath(args.camera))
        if not camera:
            raise ValueError("Missing camera: " + args.camera)
        report["source_layers_before"] = {
            layer.realPath: digest(layer.realPath) for layer in stage.GetUsedLayers()
            if layer.realPath and Path(layer.realPath).is_file()
        }
        report["registry_alias_present"] = hasattr(rep, "annotators")
        registry = getattr(rep, "annotators", rep.AnnotatorRegistry)
        getter = getattr(registry, "get", None) or getattr(registry, "get_annotator", None)
        if not callable(getter):
            raise RuntimeError("No supported annotator getter: " + str(dir(registry)))
        report["registry_getter"] = getter.__name__
        rep.orchestrator.set_capture_on_play(False)
        product = rep.create.render_product(args.camera, (args.width, args.height))
        attached, values = {}, {}
        for requested in NAMES:
            row = dict(requested=requested, status="unavailable", errors=[])
            report["attempted"].append(row)
            # Record exact requested-name failure before testing the known 4.5 alias.
            candidates = [requested, "CameraParams"] if requested == "camera_params" else [requested]
            for actual in candidates:
                try:
                    init = {"colorize": False} if requested in (
                        "semantic_segmentation", "instance_segmentation") else {}
                    ann = getter(actual, init_params=init)
                    ann.attach([product])
                    attached[requested] = ann
                    row.update(status="attached", native_name=actual)
                    break
                except Exception as error:
                    detail = dict(name=actual, error=repr(error))
                    row["errors"].append(detail)
                    print("ANNOTATOR NAME ERROR", requested, detail, flush=True)
            if row["status"] == "unavailable":
                report["failures"].append(dict(requested=requested, phase="attach", errors=row["errors"]))
        rep.orchestrator.step(rt_subframes=8, delta_time=0.0, pause_timeline=True)
        rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=True)
        for row in report["attempted"]:
            name = row["requested"]
            if name not in attached:
                continue
            try:
                value = attached[name].get_data()
                row["returns"] = describe(value)
                if value is None:
                    raise ValueError("Annotator returned None")
                if isinstance(value, np.ndarray) and value.size == 0:
                    raise ValueError("Annotator returned an empty array")
                values[name] = value
                row["status"] = "available"
            except Exception as error:
                row.update(status="error", read_error=repr(error))
                report["failures"].append(dict(requested=name, phase="read", error=repr(error)))
                print("ANNOTATOR READ ERROR", name, repr(error), flush=True)
            print("ANNOTATOR", json.dumps(row, ensure_ascii=False), flush=True)
        if "camera_params" in values:
            native = jsonable(values["camera_params"])
            report["camera_params"] = native
            width, height = native["renderProductResolution"]
            projection = np.asarray(native["cameraProjection"]).reshape(4, 4)
            if not np.allclose(projection[2, :2], 0) or any(native["cameraApertureOffset"]):
                raise ValueError("Non-centred projection needs explicit offset calibration")
            view = np.asarray(native["cameraViewTransform"]).reshape(4, 4)
            world_from_cv = np.linalg.inv(view).T @ np.diag([1, -1, -1, 1])
            calibration = dict(
                fx=float(projection[0, 0] * width / 2), fy=float(projection[1, 1] * height / 2),
                cx=width / 2, cy=height / 2, width=width, height=height,
                extrinsic_world_to_cam=np.linalg.inv(world_from_cv).tolist(),
                depth_unit="m", depth_mode="z", camera_axes="CV: X right, Y down, Z forward",
                meters_per_scene_unit=native["metersPerSceneUnit"],
            )
            report["derived_camera"] = calibration
            print("CAMERA NATIVE", json.dumps(native), flush=True)
            print("CAMERA DERIVED", json.dumps(calibration), flush=True)
        if "distance_to_image_plane" in values:
            depth = np.asarray(values["distance_to_image_plane"]).squeeze()
            finite = depth[np.isfinite(depth) & (depth > 0)]
            if not len(finite):
                raise ValueError("No positive finite depth pixels")
            report["depth_m"] = dict(min=float(finite.min()), max=float(finite.max()),
                                     quantiles=np.quantile(finite, [0.01, 0.5, 0.99]).tolist(),
                                     finite_positive_pixels=len(finite))
            print("DEPTH METRES", report["depth_m"], flush=True)
        for name in ("semantic_segmentation", "instance_segmentation"):
            if name in values:
                value = values[name]
                ids = np.unique(np.asarray(value["data"]).squeeze())
                info = jsonable(value["info"])
                report[name] = dict(unique_id_count=len(ids), ids=ids.tolist(), info=info)
                print("UNIQUE IDS", name, len(ids), "INFO FIELDS", list(info), flush=True)
        if "rgb" in values:
            rgb = Image.fromarray(np.asarray(values["rgb"])[..., :3].astype(np.uint8))
            rgb.save(out / "probe-rgb.png")
            panels = [rgb]
            for name in ("semantic_segmentation", "instance_segmentation"):
                if name in values:
                    array = np.asarray(values[name]["data"]).squeeze()
                    ids, inverse = np.unique(array, return_inverse=True)
                    colors = np.random.default_rng(41).integers(20, 245, (len(ids), 3), dtype=np.uint8)
                    colors[ids == 0] = 0
                    panels.append(Image.fromarray(colors[inverse].reshape(*array.shape, 3)))
            if "distance_to_image_plane" in values:
                array = np.asarray(values["distance_to_image_plane"]).squeeze()
                gray = np.clip(np.nan_to_num(array, nan=10, posinf=10) / 10 * 255, 0, 255).astype(np.uint8)
                panels.append(Image.fromarray(gray).convert("RGB"))
            board = Image.new("RGB", (1280, 720), "#101821")
            labels = ["RGB", "Semantic IDs", "Instance IDs", "Depth z: 0-10 m"]
            for i, panel in enumerate(panels):
                x, y = (i % 2) * 640, (i // 2) * 360
                board.paste(panel.resize((640, 360)), (x, y))
                ImageDraw.Draw(board).text((x + 12, y + 10), labels[i], fill="white", stroke_width=2, stroke_fill="black")
            board.save(out / "probe-preview.png")
        # Check every initially loaded source layer, not just the top-level USD.
        changed = [path for path, sha in report["source_layers_before"].items() if digest(path) != sha]
        report.update(status="completed", attempted_count=len(report["attempted"]),
                      available_count=sum(row["status"] == "available" for row in report["attempted"]),
                      source_layers_unchanged=not changed, changed_source_layers=changed,
                      source_root_unchanged=digest(scene) == before,
                      source_edits_saved=False, script_sha256=digest(__file__))
        if changed or digest(scene) != before:
            raise RuntimeError("A source layer changed during probe")
    except Exception as error:
        report.update(status="failed", error=repr(error))
        traceback.print_exc()
        raise
    finally:
        (out / "probe-report.json").write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
        print("PROBE RESULT", report["status"], "attempted", len(report["attempted"]),
              "failures", len(report["failures"]), flush=True)
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
