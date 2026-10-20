"""Find cup placement clearance on the exact desk from actual triangle geometry."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pxr import Usd
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.ue_isaac_pipeline.io import write_json  # noqa: E402
from plan_apartment_route import triangles, surface_below  # noqa: E402


def clip_z(polygon, level, above):
    result = []
    previous = polygon[-1]
    for current in polygon:
        old_in = previous[2] >= level if above else previous[2] <= level
        new_in = current[2] >= level if above else current[2] <= level
        if new_in != old_in:
            ratio = (level - previous[2]) / (current[2] - previous[2])
            result.append(previous + ratio * (current - previous))
        if new_in:
            result.append(current)
        previous = current
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--diagnosis", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--preferred-x", type=float, default=-1.48)
    p.add_argument("--preferred-y", type=float, default=-1.30)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(args.diagnosis).read_text(encoding="utf-8"))
    stage = Usd.Stage.Open(args.scene)
    groups = list(triangles(stage))
    desk = data["desk"]
    z = desk["bounds_max"][2]
    low, high = np.array(desk["bounds_min"][:2]), np.array(desk["bounds_max"][:2])
    step = 0.005
    shape = np.ceil((high - low) / step).astype(int) + 1
    obstacle = Image.new("L", tuple(shape), 0)
    draw = ImageDraw.Draw(obstacle)
    contributors = {}
    for name, tri in groups:
        if name.startswith(desk["path"]) or "/cup_V1" in name:
            continue
        selected = tri[
            (tri[:, :, 2].max(1) > z + 0.003)
            & (tri[:, :, 2].min(1) < z + 0.086)
            & (tri[:, :, :2].max(1) >= low).all(1)
            & (tri[:, :, :2].min(1) <= high).all(1)
        ]
        count = 0
        for t in selected:
            polygon = clip_z(t, z + 0.003, True)
            if polygon:
                polygon = clip_z(polygon, z + 0.086, False)
            if len(polygon) >= 3:
                xy = (np.asarray(polygon)[:, :2] - low) / step
                draw.polygon([tuple(v) for v in xy], fill=255)
                count += 1
        if count:
            contributors[name] = count
    mask = np.asarray(obstacle) > 0
    # Padding makes table-edge support part of the clearance calculation.
    clearance = distance_transform_edt(np.pad(~mask, 1))[1:-1, 1:-1] * step
    np.savez_compressed(
        out / "desktop-clearance.npz",
        clearance_m=clearance,
        origin_xy_m=low,
        resolution_m=step,
        cup_radius_m=0.048,
        safety_margin_m=0.005,
    )
    yy, xx = np.indices(mask.shape)
    points = low + np.stack([xx, yy], axis=-1) * step
    preferred = np.array([args.preferred_x, args.preferred_y])
    safe = clearance >= 0.060
    if not safe.any():
        raise RuntimeError("No tabletop location fits the conservative 6 cm cup footprint plus margin")
    costs = np.linalg.norm(points - preferred, axis=-1)
    costs[~safe] = np.inf
    iy, ix = np.unravel_index(np.argmin(costs), costs.shape)
    xy = points[iy, ix]
    support = surface_below([(n, t) for n, t in groups if n.startswith(desk["path"])], xy, z + 0.002)
    if support is None or abs(support["height_m"] - z) > 0.002:
        raise RuntimeError("Candidate lacks the actual desktop triangle support")
    bottom_offset = data["cup"]["bounds_min"][2] - data["cup_origin_m"][2]
    record = dict(
        scene=args.scene,
        desk=desk["path"],
        cup_origin_m=[*xy, support["height_m"] - bottom_offset + 0.001],
        clearance_m=float(clearance[iy, ix]),
        cup_envelope_radius_m=0.048,
        required_clearance_m=0.060,
        grid_resolution_m=step,
        height_band_m=[z + 0.003, z + 0.086],
        obstacle_triangles=contributors,
        hand_swept_volume_validated=False,
        physics_release_validated=False,
    )
    write_json(out / "place-plan.json", record)
    picture = Image.new("RGB", tuple(shape), "#23453b")
    colors = np.asarray(picture).copy()
    colors[mask] = [185, 65, 60]
    colors[safe] = [70, 130, 94]
    picture = Image.fromarray(colors)
    pen = ImageDraw.Draw(picture)
    r = 0.048 / step
    pen.ellipse([ix - r, iy - r, ix + r, iy + r], outline="yellow", width=2)
    picture = picture.transpose(Image.Transpose.FLIP_TOP_BOTTOM).resize(
        (int(shape[0] * 4), int(shape[1] * 4))
    )
    picture.save(out / "desktop-clearance.png")
    print(json.dumps(record, indent=2))
