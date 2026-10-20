"""One native floating-base fixed-command grasp experiment; no policy retuning."""

# ruff: noqa: E402
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trial", required=True)
    args = p.parse_args()
    spec_path = Path(args.trial).resolve(strict=True)
    spec = json.loads(spec_path.read_text(encoding="utf8"))
    cfg, out = spec["config"], spec_path.parent
    report_path = out / "trial-report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Use Isaac Sim 4.5 Python 3.10")
    for key in ("AISIM_H5PY_OVERLAY", "AISIM_ORT_OVERLAY"):
        if os.environ.get(key):
            sys.path.insert(0, os.environ[key])
    handles = []
    if os.name == "nt":
        sdk = Path(sys.executable).parent
        for folder in (sdk / "Library/bin", sdk / "Lib/site-packages/torch/lib"):
            if folder.is_dir():
                os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")
                handles.append(os.add_dll_directory(str(folder)))
    report = dict(
        status="running",
        trial_id=spec["trial_id"],
        seed=spec["seed"],
        friction=spec["friction"],
        mass_kg=spec["mass_kg"],
        fixed_commands_sha256=spec["fixed_commands_sha256"],
        source_scene_sha256=sha(cfg["scene"]),
        failures=[],
        controller="Frozen upper/finger PD targets and body commands; unchanged official leg state feedback; no retuning",
        worker_sha256=sha(__file__),
    )
    app = None
    started = time.perf_counter()
    try:
        import numpy as np
        from isaaclab.app import AppLauncher

        app = AppLauncher(
            dict(headless=True, device="cpu", kit_args="--portable-root " + os.environ["AISIM_PORTABLE_ROOT"])
        ).app
        import carb
        import torch
        import omni.usd
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
        from isaaclab.sensors import ContactSensor, ContactSensorCfg
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
        from pipeline.simforge.eval.grasp_metrics import evaluate
        from pipeline.ue_isaac_pipeline.unitree_wbc import UnitreeG1WBC
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        policy = ort.InferenceSession(
            cfg["leg_policy"], sess_options=options, providers=["CPUExecutionProvider"]
        )
        report["leg_policy_sha256"] = sha(cfg["leg_policy"])
        baseline_task = json.loads(
            Path(cfg["source_recording"], "probe-report.json").read_text(encoding="utf8")
        )
        if report["leg_policy_sha256"] != baseline_task["policy_sha256"]:
            raise ValueError("Official leg policy differs from the accepted baseline")

        carb.settings.get_settings().set_bool("/isaaclab/render/rtx_sensors", False)
        commands = np.load(spec["fixed_commands"], allow_pickle=False)
        if sha(spec["fixed_commands"]) != spec["fixed_commands_sha256"]:
            raise ValueError("Frozen command file checksum changed")
        layer = Sdf.Layer.CreateNew(str(out / "Trial_scene.usda"))
        layer.subLayerPaths = [Path(os.path.relpath(cfg["scene"], out)).as_posix()]
        layer.defaultPrim = "World"
        stage = Usd.Stage.Open(layer)
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        UsdGeom.SetStageUpAxis(stage, "Z")
        UsdPhysics.SetStageKilogramsPerUnit(stage, 1)
        physics_paths = [prim.GetPath() for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
        for path in physics_paths:
            stage.OverridePrim(path).SetActive(False)
        layer.Save()
        if not omni.usd.get_context().open_stage(str(out / "Trial_scene.usda")):
            raise RuntimeError("Cannot open trial scene")
        for _ in range(5):
            app.update()
        stage = omni.usd.get_context().get_stage()
        cup_path = cfg["contact"]["cup_prim"]
        UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(cup_path)).CreateMassAttr(spec["mass_kg"])
        PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(cup_path)).CreateThresholdAttr(0)
        material_path = "/World/M2/CupFingerMaterial"
        material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=spec["friction"],
            dynamic_friction=spec["friction"],
            restitution=0,
            friction_combine_mode=cfg["contact"]["friction_combine_mode"],
            restitution_combine_mode="average",
        )
        material_cfg.func(material_path, material_cfg)
        material = UsdShade.Material(stage.GetPrimAtPath(material_path))
        bound = []
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if prim.HasAPI(UsdPhysics.CollisionAPI) and (
                path.startswith(cup_path + "/")
                or any(path.startswith("/World/G1/" + body + "/") for body in cfg["contact"]["hand_bodies"])
            ):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    material,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    materialPurpose="physics",
                )
                bound.append(path)
        if not bound:
            raise ValueError("No declared cup/finger colliders matched")
        drives = json.loads(
            Path(cfg["source_recording"], "joint-drives-runtime-initial.json").read_text(encoding="utf8")
        )["joints"]
        drive_map = {row["name"]: row for row in drives}
        names = commands["joint_names"].tolist()
        if set(names) != set(drive_map):
            raise ValueError("Recorded joint set differs from recorded actuator properties")

        def values(field):
            return {name: drive_map[name][field] for name in names}

        actuator = ImplicitActuatorCfg(
            joint_names_expr=names,
            stiffness=values("stiffness_Nm_rad"),
            damping=values("damping_Nm_s_rad"),
            effort_limit_sim=values("effort_limit_Nm"),
            velocity_limit_sim=values("velocity_limit_rad_s"),
            armature=values("armature_kg_m2"),
        )
        robot_cfg = ArticulationCfg(
            prim_path="/World/G1",
            spawn=None,
            actuators={"recorded_drives": actuator},
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={name: float(q) for name, q in zip(names, commands["initial_q"])},
                joint_vel={name: float(v) for name, v in zip(names, commands["initial_dq"])},
            ),
        )
        physics_dt = 1 / cfg["physics_hz"]
        simulation_cfg = sim_utils.SimulationCfg(
            dt=physics_dt,
            device="cpu",
            use_fabric=False,
            physics_prim_path="/World/M2Physics",
            render_interval=4,
        )
        simulation_cfg.physx.bounce_threshold_velocity = 0.01
        simulation_cfg.physx.friction_correlation_distance = 0.00625
        simulation_cfg.physx.enable_enhanced_determinism = cfg["enhanced_determinism"]
        sim = sim_utils.SimulationContext(simulation_cfg)
        robot = Articulation(robot_cfg)
        cup = RigidObject(RigidObjectCfg(prim_path=cup_path, spawn=None))
        sensors = {}
        for body in cfg["contact"]["hand_bodies"]:
            sensors[body] = ContactSensor(
                ContactSensorCfg(
                    prim_path="/World/G1/" + body,
                    filter_prim_paths_expr=cfg["contact"]["filter_prim_paths_expr"],
                    history_length=cfg["contact"]["history_length"],
                    update_period=cfg["contact"]["update_period_s"],
                )
            )
        support = ContactSensor(
            ContactSensorCfg(
                prim_path=cup_path,
                filter_prim_paths_expr=[cfg["contact"]["support_prim"]],
                history_length=1,
                update_period=0,
            )
        )
        layer.Save()
        sim.reset()
        sim.set_simulation_dt(physics_dt=physics_dt, rendering_dt=1 / cfg["control_hz"])
        sim.get_physics_context().set_physics_dt(physics_dt)
        if abs(sim.get_physics_dt() - physics_dt) > 1e-9 or robot.is_fixed_base:
            raise ValueError("Expected native 200Hz floating-base physics")
        if robot.joint_names != names:
            raise ValueError("Native joint ordering differs from frozen commands")
        for field, actual in (
            ("stiffness_Nm_rad", robot.data.joint_stiffness),
            ("damping_Nm_s_rad", robot.data.joint_damping),
            ("effort_limit_Nm", robot.data.joint_effort_limits),
            ("velocity_limit_rad_s", robot.data.joint_vel_limits),
            ("armature_kg_m2", robot.data.joint_armature),
        ):
            expected = np.asarray([drive_map[name][field] for name in names])
            if not np.allclose(actual[0].cpu().numpy(), expected, rtol=1e-6, atol=1e-8):
                raise ValueError("Native actuator properties changed: " + field)
        # Same shape/COM; scale the entire body-local inertia tensor, including products.
        ratio = np.float32(spec["mass_kg"]) / np.float32(commands["baseline_mass_kg"])
        expected_i = commands["baseline_inertia"].astype(np.float32).reshape(1, 9) * ratio
        inertia = torch.as_tensor(expected_i, dtype=torch.float32)
        cup.root_physx_view.set_inertias(inertia, torch.tensor([0], dtype=torch.int32))
        actual_mass = float(cup.root_physx_view.get_masses().reshape(-1)[0])
        actual_i = cup.root_physx_view.get_inertias()[0].cpu().numpy()
        if abs(actual_mass - spec["mass_kg"]) > 1e-6 or not np.allclose(
            actual_i, expected_i[0], rtol=1e-5, atol=1e-9
        ):
            raise ValueError("Native mass/inertia override failed")
        report["native_parameters"] = dict(
            mass_kg=actual_mass,
            inertia_tensor_kgm2=actual_i.reshape(3, 3).tolist(),
            requested_inertia_tensor_kgm2=expected_i.reshape(3, 3).tolist(),
            mass_ratio=float(ratio),
            contact_materials={},
            bound_colliders=bound,
        )
        for body, sensor in sensors.items():
            props = sensor.body_physx_view.get_material_properties()[0].cpu().numpy()
            if not np.allclose(props[:, :2], spec["friction"], atol=1e-6):
                raise ValueError("Finger native friction does not match pair: " + body + str(props.tolist()))
            report["native_parameters"]["contact_materials"][body] = props.tolist()
        props = cup.root_physx_view.get_material_properties()[0].cpu().numpy()
        if not np.allclose(props[:, :2], spec["friction"], atol=1e-6):
            raise ValueError("Cup native friction does not match pair")
        report["native_parameters"]["contact_materials"]["cup"] = props.tolist()
        robot.reset()
        cup.reset()
        robot.write_joint_state_to_sim(
            torch.as_tensor(commands["initial_q"][None], dtype=torch.float32),
            torch.as_tensor(commands["initial_dq"][None], dtype=torch.float32),
        )
        robot.write_root_state_to_sim(torch.as_tensor(commands["initial_root"][None], dtype=torch.float32))
        cup_state = commands["initial_cup"].copy()
        cup_state[:2] += spec["cup_xy_jitter_m"]
        cup.write_root_state_to_sim(torch.as_tensor(cup_state[None], dtype=torch.float32))
        wbc = UnitreeG1WBC(names, commands["nominal_default_q"], policy)
        wbc.history = commands["initial_policy_history"].copy()
        wbc.previous_raw_body_action = commands["initial_policy_feedback"].copy()
        wbc.initialized = True
        # No robot/cup pose writes after this initialization.
        palm_id = robot.body_names.index(cfg["contact"]["hand_side"] + "_hand_palm_link")
        times, cups, palms, roots, forces, supports = [], [], [], [], [], []
        body_frames, cup_frames, frame_times, applied_indices = [], [], [], []
        applied_targets = []
        decimation = cfg["physics_hz"] // cfg["control_hz"]
        if decimation * cfg["control_hz"] != cfg["physics_hz"]:
            raise ValueError("Control/physics clocks must be an integer ratio")
        for step in range(round(cfg["duration_s"] * cfg["physics_hz"])):
            control_index = step // decimation
            command_index = min(control_index, len(commands["targets"]) - 1)
            if step % decimation == 0:
                requested = commands["targets"][command_index].copy()
                balanced, _, _ = wbc.step(
                    robot.data.joint_pos[0].cpu().numpy(),
                    robot.data.joint_vel[0].cpu().numpy(),
                    robot.data.root_ang_vel_b[0].cpu().numpy(),
                    robot.data.projected_gravity_b[0].cpu().numpy(),
                    commands["body_commands"][command_index],
                    arm_targets=requested[wbc.arm_ids],
                    waist_targets={
                        name: float(requested[names.index(name)])
                        for name in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
                    },
                )
                requested[wbc.leg_ids] = balanced[wbc.leg_ids]
                applied_targets.append(requested.copy())
                robot.set_joint_position_target(torch.as_tensor(requested[None], dtype=torch.float32))
                robot.write_joint_effort_limit_to_sim(
                    torch.as_tensor(commands["effort_limits"][command_index][None], dtype=torch.float32)
                )
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(physics_dt)
            cup.update(physics_dt)
            for sensor in [*sensors.values(), support]:
                sensor.update(physics_dt, force_recompute=True)
            cp = cup.data.root_state_w[0].cpu().numpy().copy()
            pp = robot.data.body_state_w[0, palm_id].cpu().numpy().copy()
            rp = robot.data.root_state_w[0].cpu().numpy().copy()
            force = np.asarray(
                [
                    sensor.data.force_matrix_w[0].reshape(-1, 3).sum(0).cpu().numpy()
                    for sensor in sensors.values()
                ]
            )
            sp = support.data.force_matrix_w[0].reshape(-1, 3).sum(0).cpu().numpy().copy()
            if not all(np.isfinite(v).all() for v in (cp, pp, rp, force, sp)):
                raise ValueError("Nonfinite native state/force")
            times.append((step + 1) * physics_dt)
            cups.append(cp)
            palms.append(pp)
            roots.append(rp)
            forces.append(force)
            supports.append(sp)
            if (step + 1) % decimation == 0:
                body_frames.append(robot.data.body_state_w[0, :, :7].cpu().numpy().copy())
                cup_frames.append(cp[:7])
                frame_times.append((step + 1) * physics_dt)
                applied_indices.append(command_index)
            if (step + 1) % cfg["physics_hz"] == 0:
                print(
                    f"M2 {spec['trial_id']} {step + 1}/{round(cfg['duration_s'] * cfg['physics_hz'])} z={cp[2]:.4f} grip={np.linalg.norm(force, axis=1).sum():.3f}N",
                    flush=True,
                )
        arrays = dict(
            time_s=np.asarray(times),
            cup_state=np.asarray(cups),
            palm_state=np.asarray(palms),
            root_state=np.asarray(roots),
            finger_forces_N=np.asarray(forces),
            support_force_N=np.asarray(supports),
            sensor_names=np.asarray(list(sensors)),
            initial_cup_pose=cup_state[:7],
            body_poses=np.asarray(body_frames),
            cup_poses=np.asarray(cup_frames),
            frame_time_s=np.asarray(frame_times),
            body_names=np.asarray(robot.body_names),
            command_indices=np.asarray(applied_indices),
            motor_targets_applied=np.asarray(applied_targets),
        )
        np.savez_compressed(out / "measurements.npz", **arrays)
        fixed_ids = [i for i in range(len(names)) if i not in wbc.leg_ids]
        expected_upper = commands["targets"][
            np.minimum(np.arange(len(applied_targets)), len(commands["targets"]) - 1)
        ][:, fixed_ids]
        if not np.array_equal(arrays["motor_targets_applied"][:, fixed_ids], expected_upper):
            raise ValueError("Upper/finger command sequence changed during a physical comparison")
        report.update(
            status="completed",
            metrics=evaluate(arrays, cfg),
            physics_steps=len(times),
            recording_sha256=sha(out / "measurements.npz"),
            source_scene_unchanged=sha(cfg["scene"]) == report["source_scene_sha256"],
            elapsed_seconds=time.perf_counter() - started,
            free_base=True,
            pose_writes_after_initialization=0,
            upper_and_finger_targets_exact=True,
            maximum_root_xy_displacement_m=float(
                np.linalg.norm(arrays["root_state"][:, :2] - commands["initial_root"][:2], axis=1).max()
            ),
        )
        print("M2 METRICS", json.dumps(report["metrics"]), flush=True)
    except Exception as error:
        report.update(status="error", error=repr(error), elapsed_seconds=time.perf_counter() - started)
        report["failures"].append(dict(error=repr(error), traceback=traceback.format_exc()))
        traceback.print_exc()
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf8")
        print("M2 TRIAL RESULT", report["status"], flush=True)
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
