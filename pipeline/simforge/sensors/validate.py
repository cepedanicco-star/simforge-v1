"""Independently read persisted M1 pixels, boxes, calibration and source geometry."""

import argparse
import hashlib
import itertools
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Usd, UsdGeom

REQUIRED = ("rgb.png", "depth.npy", "semantic.png", "instance.png", "boxes_2d.json",
            "boxes_3d.json", "camera.json", "labels.json")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(folder, ffprobe):
    folder = Path(folder).resolve(strict=True)
    target = folder / "independent-export-audit.json"
    if target.exists():
        raise FileExistsError(target)
    report = read(folder / "export-report.json")
    if report["status"] != "completed":
        raise ValueError("Capture is not complete")
    stage = Usd.Stage.Open(report["scene"])
    if not stage:
        raise ValueError("Cannot reopen source scene")
    config = report["config"]
    maximum_box, maximum_center, maximum_3d = 0.0, 0.0, 0.0
    depth_points, box_count, box3_count, checked = 0, 0, 0, []
    for number in range(report["requested_frames"]):
        frame = folder / f"frame_{number:06d}"
        for name in REQUIRED:
            path = frame / name
            if not path.is_file() or not path.stat().st_size:
                raise ValueError("Missing/empty required file: " + str(path))
        rgb = Image.open(frame / "rgb.png")
        depth = np.load(frame / "depth.npy", allow_pickle=False)
        semantic = np.asarray(Image.open(frame / "semantic.png"))
        instance = np.asarray(Image.open(frame / "instance.png"))
        labels, camera = read(frame / "labels.json"), read(frame / "camera.json")
        width, height = rgb.size
        if depth.dtype != np.float32 or depth.shape != (height, width):
            raise ValueError("Depth shape/dtype differs from RGB")
        if semantic.shape != depth.shape or instance.shape != depth.shape or instance.max() > 65535:
            raise ValueError("Integer ID channel contract violated")
        if camera["depth_mode"] != "z" or camera["depth_unit"] != "m":
            raise ValueError("Ambiguous depth units/mode")
        center = max(abs(camera["cx"] - width / 2), abs(camera["cy"] - height / 2))
        maximum_center = max(maximum_center, center)
        if center > config["acceptance"]["max_principal_point_error_px"]:
            raise ValueError("Principal point gate failed")
        boxes, boxes3d = read(frame / "boxes_2d.json"), read(frame / "boxes_3d.json")
        if not boxes or not boxes3d:
            raise ValueError("Empty box channel")
        tc = config["time_code_start"] + number * config["time_code_step"]
        bounds = UsdGeom.BBoxCache(Usd.TimeCode(tc), ["default", "render"]).ComputeWorldBound(
            stage.GetPrimAtPath("/World/Environment/Source/cup_V1")).ComputeAlignedRange()
        expected_range = np.asarray([bounds.GetMin(), bounds.GetMax()])
        for box in boxes:
            ys, xs = np.where(instance == box["id"])
            if not len(xs):
                raise ValueError("Saved box points to an absent instance")
            edges = [xs.min(), ys.min(), xs.max(), ys.max()]
            exported = [box[k] for k in ("x_min", "y_min", "x_max", "y_max")]
            error = float(np.abs(np.asarray(edges) - exported).max())
            maximum_box = max(maximum_box, error)
            if error > config["acceptance"]["max_box_edge_error_px"]:
                raise ValueError("Saved box/mask edge discrepancy exceeds gate")
            identity = labels["instance_ids"][str(box["id"])]
            if identity["semantic_label"] != box["semantic_label"]:
                raise ValueError("ID-to-label correspondence differs from box label")
            if identity["semantic_label"] == "cup_v1":
                # Independent matrix solve, rather than the exporter's direct formula.
                stride = config["acceptance"]["depth_sample_stride"]
                uv = np.stack([xs[::stride] + 0.5, ys[::stride] + 0.5, np.ones(len(xs[::stride]))])
                z = depth[ys[::stride], xs[::stride]]
                if not np.all(np.isfinite(z) & (z > 0)):
                    raise ValueError("Target has invalid depth")
                k = np.asarray([[camera["fx"], 0, camera["cx"]], [0, camera["fy"], camera["cy"]], [0, 0, 1]])
                camera_points = np.linalg.solve(k, uv) * z
                world = np.linalg.solve(camera["extrinsic_world_to_cam"], np.vstack([camera_points, np.ones(len(z))]))[:3].T
                tolerance = config["acceptance"]["depth_target_tolerance_m"]
                if not np.all((world >= expected_range[0] - tolerance) & (world <= expected_range[1] + tolerance)):
                    raise ValueError("Persisted depth is inconsistent with paired target geometry")
                depth_points += len(z)
            box_count += 1
        cup_boxes = 0
        for box in boxes3d:
            if box["frame"] != "world" or min(box["extent"]) <= 0:
                raise ValueError("Invalid world-space 3D box")
            if box["semantic_label"] == "cup_v1":
                signs = np.asarray(list(itertools.product([-1, 1], repeat=3)))
                corners = (signs * np.asarray(box["extent"]) / 2) @ np.asarray(box["rotation"]).T + box["center"]
                actual_range = np.asarray([corners.min(0), corners.max(0)])
                error = float(np.abs(actual_range - expected_range).max())
                maximum_3d = max(maximum_3d, error)
                if error > config["acceptance"]["depth_target_tolerance_m"]:
                    raise ValueError("Native target 3D box disagrees with independently loaded USD geometry")
                cup_boxes += 1
            box3_count += 1
        if not cup_boxes:
            raise ValueError("No world-space 3D target box")
        checked.append(dict(frame=number, required_files=len(REQUIRED), cup_3d_boxes=cup_boxes))
    changed = [path for path, value in report["source_layer_hashes"].items() if sha(path) != value]
    if changed:
        raise ValueError("Source layer checksum changed: " + str(changed))
    video = folder / "m1-sensor-demo.mp4"
    stream = json.loads(subprocess.check_output([str(ffprobe), "-v", "error", "-count_frames",
        "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate,duration",
        "-of", "json", str(video)]))["streams"][0]
    if int(stream["nb_read_frames"]) != len(checked) or [int(stream["width"]), int(stream["height"])] != [1280, 720]:
        raise ValueError("Demo video differs from persisted M1 frames")
    result = dict(status="passed", user_acceptance="pending_manual_run_or_review", frames_checked=len(checked),
                  required_files_per_frame=len(REQUIRED), required_files_checked=len(checked)*len(REQUIRED),
                  boxes_2d_checked=box_count, boxes_3d_checked=box3_count,
                  maximum_box_edge_error_px=maximum_box, maximum_principal_point_error_px=maximum_center,
                  target_depth_samples=depth_points, all_sampled_target_depth_inside_bounds=True,
                  maximum_target_3d_box_range_error_m=maximum_3d, source_layers_checked=len(report["source_layer_hashes"]),
                  source_layers_unchanged=True, original_export_report_sha256=sha(folder / "export-report.json"),
                  video=dict(path=str(video), sha256=sha(video), **stream), checked=checked)
    target.write_text(json.dumps(result, indent=2), encoding="utf8")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--ffprobe", required=True)
    args = p.parse_args()
    result = validate(args.dataset, args.ffprobe)
    print({key: value for key, value in result.items() if key != "checked"})
