"""Render measured body poses without allowing a second physics simulation."""

import numpy as np
from pxr import Gf, UsdGeom, UsdPhysics


def disable_replay_physics(stage):
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
        if prim.IsA(UsdPhysics.Joint):
            UsdPhysics.Joint(prim).GetJointEnabledAttr().Set(False)


class MeasuredPoseBridge:
    def __init__(self, stage, paths):
        self.stage, self.paths = stage, list(paths)
        self.ops, self.parents = [], []
        cache = UsdGeom.XformCache()
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            if not prim:
                raise ValueError("Missing measured body " + str(path))
            self.parents.append(cache.GetLocalToWorldTransform(prim.GetParent()).GetInverse())
            self.ops.append(UsdGeom.Xformable(prim).MakeMatrixXform())

    def apply(self, poses, time=None):
        poses = np.asarray(poses)
        if poses.shape != (len(self.paths), 7) or not np.isfinite(poses).all():
            raise ValueError("Every body requires finite world XYZ/WXYZ")
        for op, inverse_parent, pose in zip(self.ops, self.parents, poses):
            q = np.asarray(pose[3:], dtype=np.float64)
            length = np.linalg.norm(q)
            if not 0.9 < length < 1.1:
                raise ValueError("Measured quaternion is invalid")
            q = q / length
            world = Gf.Matrix4d().SetRotate(Gf.Quatd(float(q[0]), Gf.Vec3d(*map(float, q[1:]))))
            world.SetTranslateOnly(Gf.Vec3d(*map(float, pose[:3])))
            matrix = world * inverse_parent
            if time is None:
                op.Set(matrix)
            else:
                op.Set(matrix, time)

    def verify(self, poses, time=0):
        cache = UsdGeom.XformCache(time)
        max_position, max_angle = 0.0, 0.0
        for path, pose in zip(self.paths, poses):
            matrix = cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(path))
            max_position = max(
                max_position, float(np.linalg.norm(np.asarray(matrix.ExtractTranslation()) - pose[:3]))
            )
            q = matrix.ExtractRotationQuat()
            actual = np.array([q.GetReal(), *q.GetImaginary()])
            actual /= np.linalg.norm(actual)
            expected = np.asarray(pose[3:], dtype=np.float64)
            expected /= np.linalg.norm(expected)
            angle = 2 * np.arccos(np.clip(abs(actual @ expected), 0, 1))
            max_angle = max(max_angle, float(angle))
        if max_position > 1e-5 or max_angle > 1e-4:
            raise RuntimeError(f"Measured replay pose mismatch {max_position} m, {max_angle} rad")
        return dict(position_error_m=max_position, rotation_error_rad=max_angle)
