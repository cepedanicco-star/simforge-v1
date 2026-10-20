"""Pinhole RGB-D calibration with explicit pixel and coordinate conventions."""

from pathlib import Path
import math

from .io import read_json, write_json


def usd_lens_value(millimetres, meters_per_unit=1.0):
    """UsdGeom camera focal/aperture lengths are tenths of a scene unit."""
    if not all(math.isfinite(v) and v > 0 for v in (millimetres, meters_per_unit)):
        raise ValueError("Lens and scene units must be positive")
    return millimetres * 0.01 / meters_per_unit


def calibrate_records(records):
    """Validate native renderer parameters and paired poses before long captures."""
    import numpy as np

    if not records:
        raise ValueError("Cannot calibrate an empty dataset")
    frames = []
    maximum_error = 0.0
    for item in records:
        p = item["camera_params"]
        if p["cameraModel"] != "pinhole" or p["metersPerSceneUnit"] != 1:
            raise ValueError("Expected metre-based pinhole camera")
        width, height = p["renderProductResolution"]
        focal = p["cameraFocalLength"]
        horizontal, vertical = p["cameraAperture"]
        if min(width, height, focal, horizontal, vertical) <= 0 or any(p["cameraApertureOffset"]):
            raise ValueError("Unsupported camera aperture/offset")
        lens_fx, lens_fy = width * focal / horizontal, height * focal / vertical
        projection = np.asarray(p["cameraProjection"]).reshape(4, 4)
        if (
            not np.isfinite(projection).all()
            or not np.isclose(projection[2, 3], -1)
            or not np.isclose(projection[3, 3], 0)
            or not np.allclose(projection[[0, 1, 2, 2], [1, 0, 0, 1]], 0)
            or min(projection[0, 0], projection[1, 1]) <= 0
        ):
            raise ValueError("Unsupported renderer pinhole projection")
        # Use the projection that actually produced the pixels. Kit's effective
        # projection may include uniform zoom absent from nominal lens metadata.
        fx, fy = projection[0, 0] * width / 2, projection[1, 1] * height / 2
        scale_x, scale_y = fx / lens_fx, fy / lens_fy
        if not np.isclose(scale_x, scale_y, rtol=1e-5):
            raise ValueError("Lens aspect ratio disagrees with renderer projection")
        view = np.asarray(p["cameraViewTransform"]).reshape(4, 4)
        world_from_cv = np.linalg.inv(view).T @ np.diag([1, -1, -1, 1])
        error = float(np.linalg.norm(world_from_cv[:3, 3] - item["camera_eye_m"]))
        maximum_error = max(maximum_error, error)
        forward = np.asarray(item["camera_look_at_m"], dtype=float) - item["camera_eye_m"]
        forward /= np.linalg.norm(forward)
        if error > 1e-4 or not np.allclose(world_from_cv[:3, 2], forward, atol=1e-6):
            raise ValueError("Renderer pose differs from paired sample pose")
        frames.append(
            dict(
                frame=item["frame"],
                K=[[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]],
                world_from_cv=world_from_cv.tolist(),
                resolution=[width, height],
                nominal_lens_K=[[lens_fx, 0, width / 2], [0, lens_fy, height / 2], [0, 0, 1]],
                effective_projection_to_nominal_lens_ratio=[scale_x, scale_y],
            )
        )
    result = dict(
        schema_version=1,
        status="completed",
        frames=frames,
        camera_axes="CV: +X right, +Y down, +Z forward; world: metre, Z-up",
        pixel_convention="Array index (u,v) has continuous pixel centre (u+0.5,v+0.5)",
        depth="distance_to_image_plane: positive optical-axis Z in metres; preserve nonfinite no-hit pixels",
        backprojection="p_cv=[(u+0.5-cx)*Z/fx,(v+0.5-cy)*Z/fy,Z,1]; p_world=world_from_cv @ p_cv",
        lens_note="K uses the native effective renderer projection, not nominal focal/aperture metadata. Nominal lens K and their scale ratio are retained explicitly; validate depth backprojection against scene geometry. Raw CameraParams are unchanged.",
        maximum_eye_pairing_error_m=maximum_error,
    )
    return result


def export_calibration(dataset):
    dataset = Path(dataset)
    result = calibrate_records(read_json(dataset / "samples.json"))
    write_json(dataset / "calibration.json", result)
    return dict(
        status="completed",
        frames_checked=len(result["frames"]),
        maximum_eye_pairing_error_m=result["maximum_eye_pairing_error_m"],
    )
