"""Isaac 4.5: finite-state, support and joint-limit smoke checks on the saved scene."""

# ruff: noqa: E402
import argparse
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.ue_isaac_pipeline.io import write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("scene")
parser.add_argument("--out", required=True)
args = parser.parse_args()
report = dict(status="starting", full_contact_penetration_audit=False, trained_task_success=False)
from isaacsim import SimulationApp

app = SimulationApp(
    dict(
        headless=True,
        width=128,
        height=128,
        extra_args=["--portable-root", os.environ["AISIM_PORTABLE_ROOT"]],
    )
)
try:
    import numpy as np
    import omni.physx
    import omni.timeline
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import UsdGeom, UsdPhysics

    context = omni.usd.get_context()
    if not context.open_stage(str(Path(args.scene).resolve(strict=True))):
        raise RuntimeError("Scene open failed")
    for _ in range(10):
        app.update()
    stage = context.get_stage()
    report["environment_collider_prims"] = sum(
        p.HasAPI(UsdPhysics.CollisionAPI)
        for p in stage.Traverse()
        if str(p.GetPath()).startswith("/World/Environment/")
    )
    report["meters_per_unit"] = UsdGeom.GetStageMetersPerUnit(stage)
    report["up_axis"] = str(UsdGeom.GetStageUpAxis(stage))
    world = World(
        stage_units_in_meters=1,
        physics_dt=1 / 120,
        physics_prim_path="/World/PhysicsScene",
        set_defaults=False,
    )
    robot = world.scene.add(SingleArticulation("/World/Robot", name="robot"))
    cup = world.scene.add(SingleRigidPrim("/World/Environment/Source/cup_V1", name="cup"))
    world.reset()
    hold = robot.get_joint_positions()
    limits = np.asarray(robot._articulation_view.get_dof_limits())[0]
    physics = omni.physx.get_physx_simulation_interface()
    omni.timeline.get_timeline_interface().play()
    app.update()
    start = world.current_time
    trajectory = []
    for step in range(2400):
        robot.apply_action(ArticulationAction(joint_positions=hold))
        physics.simulate(1 / 120, world.current_time)
        physics.fetch_results()
        if step % 12 == 0:
            pos, _ = cup.get_world_pose()
            base, _ = robot.get_world_pose()
            q = robot.get_joint_positions()
            velocity = cup.get_linear_velocity()
            if not all(np.isfinite(v).all() for v in (pos, base, q, velocity)):
                raise RuntimeError("Nonfinite simulated state")
            if not 0.85 < pos[2] < 0.98:
                raise RuntimeError("Cup left its cabinet support height")
            if np.max(np.abs(velocity)) > 1:
                raise RuntimeError("Cup velocity exceeded stationary test bound")
            if np.any(q < limits[:, 0] - 0.05) or np.any(q > limits[:, 1] + 0.05):
                raise RuntimeError("Robot exceeded joint limit tolerance")
            trajectory.append(
                dict(
                    t=float(world.current_time),
                    cup_position=pos.tolist(),
                    robot_base=base.tolist(),
                    cup_speed=float(np.linalg.norm(velocity)),
                )
            )
    elapsed = float(world.current_time - start)
    if abs(elapsed - 20) > 0.02:
        raise RuntimeError(f"Physics clock mismatch: {elapsed}")
    report.update(
        status="completed",
        duration_seconds=elapsed,
        physics_steps=2400,
        samples=trajectory,
        cup_mass_kg=float(cup.get_mass()),
        note="Fixed-base robot, stationary cup, local support/finite state/limits only; not navigation or grasp certification",
    )
except BaseException:
    report.update(status="failed", error=traceback.format_exc())
finally:
    write_json(args.out, report)
    app.close()
