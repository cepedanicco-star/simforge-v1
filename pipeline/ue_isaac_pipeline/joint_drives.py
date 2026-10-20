"""Persist measured SI actuator settings in a metre/kg USD inspection stage."""

import math


def capture_joint_drives(robot, phase):
    rows = []
    for i, name in enumerate(robot.joint_names):
        rows.append(
            dict(
                name=name,
                stiffness_Nm_rad=float(robot.data.joint_stiffness[0, i]),
                damping_Nm_s_rad=float(robot.data.joint_damping[0, i]),
                effort_limit_Nm=float(robot.data.joint_effort_limits[0, i]),
                velocity_limit_rad_s=float(robot.data.joint_vel_limits[0, i]),
                armature_kg_m2=float(robot.data.joint_armature[0, i]),
            )
        )
    return dict(
        joints=rows, phase=phase, source="Actual Isaac Lab articulation / configured PhysX drive properties"
    )


def author_joint_drives(stage, rows, positions):
    from pxr import Sdf, UsdPhysics, UsdGeom

    if not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1) or not math.isclose(
        UsdPhysics.GetStageKilogramsPerUnit(stage), 1
    ):
        raise ValueError("Captured SI actuator settings require a metre/kilogram inspection stage")

    joints = {p.GetName(): p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)}
    if set(positions) != {r["name"] for r in rows}:
        raise ValueError("Measured joint positions do not match the captured actuator settings")
    written = []
    for row in rows:
        prim = joints.get(row["name"])
        if prim is None:
            raise ValueError("Missing revolute joint " + row["name"])
        values = [
            row[k]
            for k in (
                "stiffness_Nm_rad",
                "damping_Nm_s_rad",
                "effort_limit_Nm",
                "velocity_limit_rad_s",
                "armature_kg_m2",
            )
        ]
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("Invalid captured joint drive values")
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(row["stiffness_Nm_rad"] * math.pi / 180)
        drive.CreateDampingAttr(row["damping_Nm_s_rad"] * math.pi / 180)
        drive.CreateMaxForceAttr(row["effort_limit_Nm"])
        drive.CreateTargetPositionAttr(math.degrees(float(positions[row["name"]])))
        drive.CreateTargetVelocityAttr(0)
        prim.AddAppliedSchema("PhysxJointAPI")
        prim.CreateAttribute("physxJoint:maxJointVelocity", Sdf.ValueTypeNames.Float).Set(
            math.degrees(row["velocity_limit_rad_s"])
        )
        prim.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float).Set(row["armature_kg_m2"])
        written.append(str(prim.GetPath()))
    return written
