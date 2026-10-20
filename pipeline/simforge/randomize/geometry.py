"""Geometry guards for static clutter, independent of random number generation."""

import numpy as np


def triangles(stage, prim):
    from pxr import Usd, UsdGeom

    cache = UsdGeom.XformCache(Usd.TimeCode(0))
    result = []
    for child in Usd.PrimRange(prim):
        if not child.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(child)
        points = np.asarray(mesh.GetPointsAttr().Get(0), dtype=float)
        if not points.size:
            continue
        matrix = np.asarray(cache.GetLocalToWorldTransform(child))
        world = np.c_[points, np.ones(len(points))] @ matrix
        vertices = world[:, :3] / world[:, 3:4]
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(0))
        counts = mesh.GetFaceVertexCountsAttr().Get(0)
        holes = set(mesh.GetHoleIndicesAttr().Get(0) or [])
        cursor, faces = 0, []
        for i, count in enumerate(counts):
            polygon = indices[cursor : cursor + count]
            if i not in holes:
                faces.extend((polygon[0], polygon[j], polygon[j + 1]) for j in range(1, count - 1))
            cursor += count
        if faces:
            result.append(vertices[np.asarray(faces)])
    return np.concatenate(result) if result else np.empty((0, 3, 3))


def support_z(mesh, xy, limits):
    """Highest horizontal surface under a vertical ray within the allowed slab."""
    a, b, c = (mesh[:, i] for i in range(3))
    v0, v1, v2 = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2], np.asarray(xy) - a[:, :2]
    denominator = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    valid = np.abs(denominator) > 1e-12
    denominator = np.where(valid, denominator, 1)
    u = (v2[:, 0] * v1[:, 1] - v2[:, 1] * v1[:, 0]) / denominator
    v = (v0[:, 0] * v2[:, 1] - v0[:, 1] * v2[:, 0]) / denominator
    z = a[:, 2] + u * (b[:, 2] - a[:, 2]) + v * (c[:, 2] - a[:, 2])
    horizontal = np.ptp(mesh[:, :, 2], axis=1) < 1e-5
    valid &= (
        horizontal & (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8) & (z >= limits[0]) & (z <= limits[1])
    )
    return float(z[valid].max()) if valid.any() else None


def mesh_intersects_box(mesh, low, high):
    """Triangle/AABB separating-axis test, including combined wall meshes."""
    if not len(mesh):
        return False
    low, high = np.asarray(low), np.asarray(high)
    mesh = mesh[(mesh.max(1) >= low).all(1) & (mesh.min(1) <= high).all(1)]
    if not len(mesh):
        return False
    center, half = (low + high) / 2, (high - low) / 2
    vertices = mesh - center
    edges = np.roll(vertices, -1, axis=1) - vertices
    axes = [np.cross(edges[:, 0], edges[:, 1])]
    axes += [np.cross(edges[:, edge], axis) for edge in range(3) for axis in np.eye(3)]
    overlap = np.ones(len(mesh), dtype=bool)
    for axis in axes:
        projection = np.einsum("nvc,nc->nv", vertices, axis)
        radius = np.abs(axis) @ half
        overlap &= (projection.min(1) <= radius + 1e-10) & (projection.max(1) >= -radius - 1e-10)
    return bool(overlap.any())
