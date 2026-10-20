"""Explicit CV/world and native row-vector USD box conventions."""

import numpy as np


def camera_contract(native):
    width, height = (int(x) for x in native["renderProductResolution"])
    projection = np.asarray(native["cameraProjection"], dtype=float).reshape(4, 4)
    view = np.asarray(native["cameraViewTransform"], dtype=float).reshape(4, 4)
    if native["metersPerSceneUnit"] != 1 or native["cameraModel"] != "pinhole":
        raise ValueError("Expected a metre pinhole camera")
    if any(native["cameraApertureOffset"]) or not np.allclose(projection[2, :2], 0):
        raise ValueError("Non-centred native projection requires explicit offset support")
    if not np.isfinite(projection).all() or min(projection[0, 0], projection[1, 1]) <= 0:
        raise ValueError("Invalid native projection")
    cv_from_world = np.diag([1, -1, -1, 1]) @ view.T
    return dict(
        fx=float(projection[0, 0] * width / 2), fy=float(projection[1, 1] * height / 2),
        cx=width / 2, cy=height / 2, width=width, height=height,
        extrinsic_world_to_cam=cv_from_world.tolist(),
        depth_unit="m", depth_mode="z", pixel_center="(u+0.5,v+0.5)",
        camera_axes="CV: X right, Y down, Z forward; column vectors", world_axes="metre, Z up",
    )


def backproject(uv, z, camera):
    uv, z = np.asarray(uv, dtype=float), np.asarray(z, dtype=float)
    cv = np.c_[(uv[:, 0] + 0.5 - camera["cx"]) * z / camera["fx"],
               (uv[:, 1] + 0.5 - camera["cy"]) * z / camera["fy"], z, np.ones(len(z))]
    return (cv @ np.linalg.inv(camera["extrinsic_world_to_cam"]).T)[:, :3]


def native_box_world(row):
    lo = np.asarray([row[k] for k in ("x_min", "y_min", "z_min")], dtype=float)
    hi = np.asarray([row[k] for k in ("x_max", "y_max", "z_max")], dtype=float)
    transform = np.asarray(row["transform"], dtype=float).reshape(4, 4)
    basis = transform[:3, :3]
    scale = np.linalg.norm(basis, axis=1)
    if min(scale) <= 0:
        raise ValueError("Degenerate native 3D-box basis")
    axes = basis / scale[:, None]
    if not np.allclose(axes @ axes.T, np.eye(3), atol=1e-4):
        raise ValueError("Sheared native box requires an explicit parallelepiped representation")
    center = np.r_[(lo + hi) / 2, 1] @ transform
    return dict(frame="world", center=center[:3].tolist(), extent=((hi - lo) * scale).tolist(),
                rotation=axes.T.tolist(), rotation_convention="columns are world-space box axes",
                parity=float(np.linalg.det(axes)), native_local_min=lo.tolist(), native_local_max=hi.tolist(),
                native_row_transform=transform.tolist())


def uint16_ids(array):
    array = np.asarray(array)
    if array.ndim != 2 or array.dtype.kind not in "ui":
        raise ValueError("Expected a 2D integer ID image")
    ids = np.unique(array)
    if len(ids) > 65536:
        raise ValueError("More than 65536 unique IDs cannot fit a reversible uint16 PNG")
    if int(ids[-1]) <= 65535:
        return array.astype(np.uint16), {int(i): int(i) for i in ids}
    mapping = {int(native): encoded for encoded, native in enumerate(ids)}
    return np.searchsorted(ids, array).astype(np.uint16), mapping


def require_labels(labels, required):
    for path, label in required.items():
        if labels.get(path) != label:
            raise ValueError(f"Required semantic label missing: {path} -> {label}")
