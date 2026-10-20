"""Plan a body/held-hand route ending before a left turn and forward desk approach."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.ue_isaac_pipeline.io import read_json, write_json  # noqa: E402
from plan_apartment_route import astar  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--desk-x", type=float, default=-1.30)
    p.add_argument("--desk-y", type=float, default=-0.94)
    args = p.parse_args()
    source = Path(args.map).resolve(strict=True)
    a = np.load(source)
    g, origin, res = a["clearance_m"], a["origin_xy"], float(a["resolution_m"])
    yy, xx = np.indices(g.shape)
    offset = np.array([-0.46, -0.17])
    dx, dy = np.rint(offset / res).astype(int)
    valid = (xx + dx >= 0) & (xx + dx < g.shape[1]) & (yy + dy >= 0) & (yy + dy < g.shape[0])
    hc = np.zeros_like(g)
    hc[valid] = g[(yy + dy)[valid], (xx + dx)[valid]]
    free = a["free"] & (g >= 0.50) & (hc >= 0.15)
    start, via, end = np.array([3.47, -0.80]), np.array([-0.50, 0.00]), np.array([-0.65, 0.00])
    cell = lambda xy: tuple(np.rint((xy - origin) / res).astype(int))  # noqa: E731
    pieces = [astar(free, cell(start), cell(via)), astar(free, cell(via), cell(end))]
    if any(not piece for piece in pieces):
        raise RuntimeError("No forward route with a clear final westward approach")
    route = np.asarray(pieces[0] + pieces[1][1:]) * res + origin
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "route-plan.json").exists():
        raise FileExistsError(out)
    np.savez_compressed(
        out / "navigation-map.npz",
        **{k: a[k] for k in a.files if k not in ("free", "route_xy")},
        free=free,
        route_xy=route,
        hand_clearance_m=hc,
    )
    report = read_json(source.parent / "route-plan.json")
    report.update(
        route_xy_m=route.tolist(),
        route_start_xy_m=start.tolist(),
        route_end_xy_m=end.tolist(),
        left_turn_exit_xy_m=(end + [-0.45, -0.15]).tolist(),
        left_turn_exit_tolerance_m=0.40,
        route_length_m=float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum()),
        desk_station=dict(base_xy_m=[args.desk_x, args.desk_y], yaw_deg=-90),
        heading_sequence_deg=[-90, -180, -90],
        motion_sequence=[
            "withdraw",
            "right_turn",
            "walk_forward",
            "left_turn",
            "approach_desk_forward",
            "place",
        ],
        carried_envelope=dict(offset_xy_m=offset.tolist(), radius_m=0.15, reference_heading_deg=-180),
        physically_executed=False,
        note="2.5D planning aid; real heading, carried contacts and complete physics must be validated",
    )
    write_json(out / "route-plan.json", report)
    print(
        dict(
            route_length_m=report["route_length_m"],
            turn_station=end.tolist(),
            desk_station=report["desk_station"],
        )
    )
