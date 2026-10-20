"""USD joint-frame forward kinematics and bounded multi-start arm IK."""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def pose_matrix(position, wxyz):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(np.asarray(wxyz)[[1, 2, 3, 0]]).as_matrix().T
    matrix[3, :3] = position
    return matrix


def matrix_pose(matrix):
    q = Rotation.from_matrix(np.asarray(matrix[:3, :3]).T).as_quat()
    return np.r_[matrix[3, :3], q[[3, 0, 1, 2]]]


class UsdArmKinematics:
    def __init__(self, asset, base="torso_link", end="right_hand_palm_link"):
        from pxr import Usd, UsdPhysics

        stage = Usd.Stage.Open(str(asset))
        joints = {}
        bodies = {
            p.GetName(): str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)
        }
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            a, b = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if not a or not b:
                continue
            joints[str(b[0])] = (str(a[0]), joint)
        chain = []
        path = bodies[end]
        while path != bodies[base]:
            parent, joint = joints[path]
            chain.append(joint)
            path = parent
        self.chain = []
        self.joint_names, lower, upper = [], [], []
        for joint in reversed(chain):

            def local(position, quaternion):
                q = quaternion.Get()
                return pose_matrix(position.Get(), [q.GetReal(), *q.GetImaginary()])

            a = local(joint.GetLocalPos0Attr(), joint.GetLocalRot0Attr())
            b = local(joint.GetLocalPos1Attr(), joint.GetLocalRot1Attr())
            name, axis = None, None
            if joint.GetPrim().IsA(UsdPhysics.RevoluteJoint):
                revolute = UsdPhysics.RevoluteJoint(joint.GetPrim())
                name = joint.GetPrim().GetName()
                self.joint_names.append(name)
                axis = np.eye(3)["XYZ".index(revolute.GetAxisAttr().Get())]
                lower.append(np.radians(revolute.GetLowerLimitAttr().Get()))
                upper.append(np.radians(revolute.GetUpperLimitAttr().Get()))
            self.chain.append((name, axis, np.linalg.inv(b), a))
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)

    def forward(self, values):
        positions = dict(zip(self.joint_names, values))
        result = np.eye(4)
        for name, axis, inverse_b, a in self.chain:
            rotation = np.eye(4)
            if name:
                rotation[:3, :3] = Rotation.from_rotvec(axis * positions[name]).as_matrix().T
            result = inverse_b @ rotation @ a @ result
        return result

    def solve(self, target, seed, *, starts=8, orientation_weight=0.12, posture_weight=0.0):
        seed = np.clip(np.asarray(seed), self.lower + 1e-6, self.upper - 1e-6)

        def pose_residual(values):
            actual = self.forward(values)
            rotation_error = Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).as_rotvec()
            return np.r_[actual[3, :3] - target[3, :3], orientation_weight * rotation_error]

        def residual(values):
            return (
                np.r_[pose_residual(values), posture_weight * (values - seed)]
                if posture_weight
                else pose_residual(values)
            )

        rng = np.random.default_rng(41)
        solutions = []
        guesses = [seed] + [rng.uniform(self.lower, self.upper) for _ in range(starts - 1)]
        for guess in guesses:
            result = least_squares(
                residual,
                guess,
                bounds=(self.lower, self.upper),
                max_nfev=250,
                xtol=1e-9,
                ftol=1e-9,
                gtol=1e-9,
            )
            error = pose_residual(result.x)
            solutions.append(
                (
                    np.linalg.norm(error),
                    np.linalg.norm(result.x - seed),
                    result.x,
                    np.linalg.norm(error[:3]),
                    np.linalg.norm(error[3:]) / orientation_weight,
                )
            )
        feasible = [s for s in solutions if s[3] < 0.008 and s[4] < 0.12]
        chosen = min(feasible, key=lambda s: s[1]) if feasible else min(solutions, key=lambda s: s[0])
        return chosen[2], dict(
            position_error_m=float(chosen[3]),
            orientation_error_rad=float(chosen[4]),
            feasible=bool(feasible),
            starts=starts,
        )
