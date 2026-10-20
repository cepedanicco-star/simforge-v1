"""0.15m camera clearance against real collision geometry, never aggregate AABBs."""

import numpy as np


def live_query():
    import carb
    import omni.physx
    import omni.usd
    from pxr import UsdPhysics
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No live USD stage; provide explicit trimesh collider surfaces instead")
    scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
    if not scenes:
        raise RuntimeError("No physics scene; refuse an unverified empty query")
    # Query support is off in the source. Enable the readout in the anonymous
    # session only; never step physics or write a source layer to initialize it.
    for scene in scenes:
        attribute = scene.GetAttribute("physxScene:enableSceneQuerySupport")
        if not attribute or not attribute.Set(True):
            raise RuntimeError("Cannot enable native scene-query readout")
    omni.physx.get_physx_interface().force_load_physics_from_usd()
    interface = omni.physx.get_physx_scene_query_interface()

    def overlap(radius, position, callback, any_hit):
        return interface.overlap_sphere(radius, carb.Float3(*map(float, position)), callback, any_hit)

    return overlap


def inspect_position(position, query=None, minimum_clearance_m=0.15, meshes=None):
    if minimum_clearance_m != 0.15:
        raise ValueError("Camera clearance must remain exactly 0.15m")
    point = np.asarray(position, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("Camera position must be a finite world-space point")
    violations = []
    if meshes is not None:
        import trimesh
        if not meshes:
            raise ValueError("Offline surface query requires actual collider meshes")
        for item in meshes:
            _, distance, _ = trimesh.proximity.closest_point_naive(item["mesh"], point[None])
            if not np.isfinite(distance[0]):
                raise ValueError("Invalid offline surface distance")
            if distance[0] <= minimum_clearance_m:
                violations.append(dict(path=item["path"], surface_distance_m=float(distance[0])))
        method = "trimesh.proximity.closest_point_naive: world collider surface"
        count = len(violations)
    else:
        if query is None:
            query = live_query()

        def callback(hit):
            violations.append(dict(path=str(hit.collision), rigid_body=str(hit.rigid_body)))
            return True

        count = int(query(minimum_clearance_m, point.tolist(), callback, False))
        if count != len(violations):
            raise ValueError("Native overlap count/callback mismatch; refuse an unverified free point")
        method = "PhysX overlap_sphere(0.15m, world_position, callback, False)"
    return dict(position_m=point.tolist(), accepted=count == 0, violations=violations,
                overlap_count=count, radius_m=minimum_clearance_m, method=method,
                distance_measurement="zero hits certifies sphere clearance; not an exact nearest-distance readout")


def select_position(base_position, proposals, query=None, minimum_clearance_m=0.15, meshes=None):
    if query is None and meshes is None:
        query = live_query()
    attempts = []
    for delta in proposals:
        row = inspect_position(np.asarray(base_position) + delta, query, minimum_clearance_m, meshes)
        attempts.append(row)
        if row["accepted"]:
            return dict(selected_index=len(attempts)-1, **row, attempts=attempts)
    blockers = sorted({v["path"] for row in attempts for v in row["violations"]})
    raise ValueError(f"No camera position with 0.15m real-geometry clearance after {len(attempts)} attempts; {blockers}")
