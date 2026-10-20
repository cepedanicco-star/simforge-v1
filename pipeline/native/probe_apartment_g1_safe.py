# Plan19 derivative of BC wrapper SHA256 84fb15325fe22920234eb31f807fc8c242ede59481158d5901218431ee3ed0ac
"""BC-arm derivative of the frozen G1 apartment driver; same physical success audit.
Original SHA256: 6730e32d386f46c4167b1c581c8dee78a7aeab849e65e39f4d2ee95672c42f0e
Only new BC argument, camera support and arm target hook are added.
"""

# ruff: noqa: E402
import argparse
import importlib.util
import math
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.ue_isaac_pipeline.io import digest_file, write_json, read_json
from pipeline.ue_isaac_pipeline.joint_drives import capture_joint_drives

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scene", required=True)
parser.add_argument("--robot-config", required=True)
parser.add_argument("--asset-root", required=True)
parser.add_argument("--policy", required=True)
parser.add_argument(
    "--walking-policy", help="Optional second vendor ONNX, blended in only after a verified contact lift"
)
parser.add_argument(
    "--cup-asset", help="Equivalent rigid-Xform cup asset, preserving the apartment target path"
)
parser.add_argument("--out", required=True)
parser.add_argument("--seconds", type=float, default=3)
parser.add_argument("--walk-seconds", type=float, default=0)
parser.add_argument("--x", type=float, default=3.82)
parser.add_argument("--y", type=float, default=-1.52)
parser.add_argument("--z", type=float, default=0.80)
parser.add_argument(
    "--hold", action="store_true", help="Hold default joints without walking-network inference"
)
parser.add_argument(
    "--no-cup-view", action="store_true", help="Isolate physics without a cup tensor observer"
)
parser.add_argument(
    "--vendor-hands", action="store_true", help="Use upstream hand actuator settings for a vendor baseline"
)
parser.add_argument("--yaw", type=float, default=-90)
parser.add_argument("--yaw-rate-limit", type=float, default=1.2)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--raise-height", type=float, default=1.35)
parser.add_argument("--grip-yaw", type=float, default=-90)
parser.add_argument("--grip-center-x", type=float, default=0.105)
parser.add_argument("--grip-center-y", type=float, default=0.050)
parser.add_argument("--grip-height", type=float, default=0.050)
parser.add_argument("--hand", choices=("left", "right"), default="right")
parser.add_argument("--arm-integral-gain", type=float, default=0.0)
parser.add_argument(
    "--pregrasp-clearance",
    type=float,
    default=0.0,
    help="Start above the cup and lower the open hand after the floating base settles",
)
parser.add_argument(
    "--forward-carry",
    action="store_true",
    help="Turn toward a short carry target, then use forward-only walking commands",
)
parser.add_argument("--carry-goal-x", type=float, default=2.8)
parser.add_argument("--carry-goal-y", type=float, default=-1.30)
parser.add_argument("--pregrasp-seconds", type=float, default=2.8)
parser.add_argument("--carry-hold-seconds", type=float, default=1.0)
parser.add_argument(
    "--transfer-plan", help="Measured-state follow of a collision-map route, then physical desk placement"
)
parser.add_argument("--placement-plan", help="Desk surface clearance plan JSON")
parser.add_argument("--navigation-policy", help="Newly trained high-level forward/yaw NumPy model")
parser.add_argument("--route-lookahead", type=float, default=0.35)
parser.add_argument("--route-progress-watchdog", action="store_true")
parser.add_argument(
    "--sleep-aware-support",
    action="store_true",
    help="Verify sleeping desk support from native contact events, sleep state and unchanged contact pose",
)
parser.add_argument(
    "--keep-cup-awake",
    action="store_true",
    help="Disable target sleeping during verification so resting desk-support impulses remain observable",
)
parser.add_argument(
    "--feedback-release",
    action="store_true",
    help="Actively open fingers after unloading and verify physical clearance before withdrawing",
)
parser.add_argument("--early-carry-posture", action="store_true")
parser.add_argument("--left-turn-rate-limit", type=float, default=1.0)
parser.add_argument(
    "--neutral-upper-during-transfer",
    action="store_true",
    help="Use nominal upper-body features only while carrying; restore actual features before desk settling",
)
parser.add_argument(
    "--human-forward-transfer",
    action="store_true",
    help="Right turn, forward route, left turn and forward desk approach",
)
parser.add_argument(
    "--lateral-carry",
    action="store_true",
    help="Keep the pickup heading and sidestep toward the carry target after withdrawing",
)
parser.add_argument(
    "--joint-carry",
    action="store_true",
    help="Hold a measured safe arm posture during walking, with physical PD drives",
)
parser.add_argument(
    "--neutral-upper-walking-observation",
    action="store_true",
    help="Ablation: mask upper-arm state only in the walking network input; all physical arm state and motor commands stay real",
)
parser.add_argument(
    "--vendor-foot-friction",
    action="store_true",
    help="Apply the upstream task's 1.0/max foot contact baseline, not a measured apartment coefficient",
)
parser.add_argument(
    "--carry-probe",
    action="store_true",
    help="After a verified grasp, withdraw, retract and turn with the physically held cup",
)
parser.add_argument(
    "--fixed-base-probe",
    action="store_true",
    help="Calibration only: anchor the pelvis to isolate finger/cup contact; never a walking result",
)
parser.add_argument(
    "--opposite-side", action="store_true", help="Approach the cup barrel from the side away from its handle"
)
parser.add_argument(
    "--ready-grasp",
    action="store_true",
    help="Start with the open hand aligned around the cup to isolate grasp contact",
)
parser.add_argument(
    "--start-raised",
    action="store_true",
    help="Initialize the arm above the counter before the first recorded physics step",
)
parser.add_argument(
    "--position-hold", action="store_true", help="Close the base pose loop using measured state"
)
parser.add_argument(
    "--grasp-probe", action="store_true", help="Test real right-hand cup contact and lift in the apartment"
)
parser.add_argument("--bc-checkpoint", required=True)
parser.add_argument("--bc-smoke", action="store_true")
parser.add_argument("--safe-parameters", required=True)
parser.add_argument('--cup-offset-x',type=float,default=0.)
parser.add_argument('--cup-offset-y',type=float,default=0.)
parser.add_argument('--start-offset-y',type=float,default=0.)
parser.add_argument('--scenario-version',choices=('v1','v2'),default='v1')
parser.add_argument('--scenario-preflight-only',action='store_true')
args = parser.parse_args()
NAV_BEGIN, NAV_END, PLACE_BEGIN = 19.0, 34.0, 38.0
ABOVE_END, LOWER_END, RELEASE_END, WITHDRAW_END, TASK_END = 41.0, 43.0, 44.0, 45.0, 46.0
TURN_BEGIN, FACE_DESK_END, ARM_SETTLE_BEGIN = 19.0, NAV_END, NAV_END
if args.human_forward_transfer:
    NAV_BEGIN, NAV_END, FACE_DESK_END, ARM_SETTLE_BEGIN, PLACE_BEGIN = 23.0, 38.0, 42.0, 45.0, 49.0
    ABOVE_END, LOWER_END, RELEASE_END, WITHDRAW_END, TASK_END = 52.0, 54.0, 55.0, 56.0, 57.0
if args.grasp_probe and (args.no_cup_view or not args.position_hold):
    parser.error("--grasp-probe requires cup observation and --position-hold")
if args.carry_probe and (not args.grasp_probe or args.fixed_base_probe):
    parser.error("--carry-probe requires --grasp-probe with a floating base")
if args.grasp_probe and args.hold:
    parser.error("--hold cannot execute a grasp controller")
if args.forward_carry and not args.carry_probe:
    parser.error("--forward-carry requires --carry-probe")
if args.lateral_carry and (not args.carry_probe or args.forward_carry):
    parser.error("--lateral-carry requires --carry-probe and excludes --forward-carry")
if args.transfer_plan and not (
    (args.lateral_carry or args.human_forward_transfer) and args.joint_carry and args.placement_plan
):
    parser.error("--transfer-plan requires lateral or human-forward carry, joint carry and a placement plan")
if args.human_forward_transfer and (not args.transfer_plan or args.lateral_carry or args.forward_carry):
    parser.error("--human-forward-transfer needs --transfer-plan and excludes other carry modes")
out = Path(args.out).resolve()
scenario_roots=[Path(os.environ['SIMFORGE_RUN_ROOT']).resolve()]
if not any(out.is_relative_to(root) for root in scenario_roots):
    parser.error('Requires a fresh isolated --out beneath SIMFORGE_RUN_ROOT; prior artifacts are read-only')
if out.exists() and any(out.iterdir()):
    parser.error('Refuse to append native execution to an existing nonempty episode output')
out.mkdir(parents=True, exist_ok=True)
if (out / "probe-report.json").exists():
    raise FileExistsError(out)
for variable in ("AISIM_ORT_OVERLAY", "AISIM_H5PY_OVERLAY"):
    if os.environ.get(variable):
        sys.path.insert(0, str(Path(os.environ[variable]).resolve(strict=True)))
_dlls = []
if os.name == "nt":
    env_root = Path(sys.executable).resolve().parent
    for directory in (env_root / "bin", env_root / "Library/bin", env_root / "Lib/site-packages/torch/lib"):
        if directory.is_dir():
            os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
            _dlls.append(os.add_dll_directory(str(directory)))
import h5py  # noqa: F401
import numpy as np
from isaaclab.app import AppLauncher

launcher = AppLauncher(
    dict(headless=True, enable_cameras=False, device=args.device, kit_args="--portable-root " + os.environ["AISIM_PORTABLE_ROOT"])
)
app = launcher.app
report = dict(
    status="starting",
    scope="Full apartment; matched vendor walking policy; no grasp claim",
    robot_source_commit="e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc",
    policy_sha256=digest_file(args.policy),
    grasp_completed=False,
    full_transfer_completed=False,
    physics_hz=200,
    control_hz=50,
    probe_arguments=vars(args),
    vendor_leg_pd_targets_not_position_clipped=True,
)
sim = None
try:
    import carb
    import torch
    import omni.kit.app
    import omni.usd
    import onnxruntime as ort
    from pxr import Sdf, UsdGeom, UsdPhysics, PhysxSchema, UsdShade
    from isaacsim.storage.native import get_assets_root_path

    # Kit must load its native runtime before ONNX Runtime's bundled MSVC DLLs.
    # This matches the upstream sim_main -> action-provider import order.
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(Path(args.policy).resolve(strict=True)), sess_options=options, providers=["CPUExecutionProvider"]
    )
    if args.walking_policy:
        from pipeline.ue_isaac_pipeline.unitree_wbc import BlendedPolicySession

        walking_session = ort.InferenceSession(
            str(Path(args.walking_policy).resolve(strict=True)),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        session = BlendedPolicySession(session, walking_session)
        report["walking_policy_sha256"] = digest_file(args.walking_policy)

    settings = carb.settings.get_settings()
    for key, value in {
        "/isaaclab/render/offscreen": False,
        "/isaaclab/render/active_viewport": False,
        "/isaaclab/render/rtx_sensors": False,
        "/physics/fabricUpdateTransformations": True,
    }.items():
        settings.set_bool(key, value)
    asset_root = (
        get_assets_root_path()
        or "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5"
    )
    for key in ("default", "cloud", "nvidia"):
        settings.set_string("/persistent/isaac/asset_root/" + key, asset_root)
    target = out / "Apartment_sim.usda"
    layer = Sdf.Layer.CreateNew(str(target))
    layer.subLayerPaths = [Path(os.path.relpath(Path(args.scene).resolve(), out)).as_posix()]
    for path in ("/World/Robot", "/World/AIWorkflow/TelemetryGraph", "/World/PhysicsScene"):
        prim = Sdf.CreatePrimInLayer(layer, path)
        prim.active = False
    layer.Save()
    if not omni.usd.get_context().open_stage(str(target)):
        raise RuntimeError("Cannot open the material-reviewed apartment")
    for _ in range(5):
        app.update()  # Deliver stage-open notices before constructing physics.
    stage = omni.usd.get_context().get_stage()
    if args.cup_asset:
        cup_prim = stage.GetPrimAtPath("/World/Environment/Source/cup_V1")
        original = UsdGeom.BBoxCache(0, ["default"]).ComputeWorldBound(cup_prim).ComputeAlignedRange()
        local = UsdGeom.Xformable(cup_prim).GetLocalTransformation()
        cup_prim.SetTypeName("Xform")
        cup_prim.GetReferences().SetReferences([Sdf.Reference(Path(args.cup_asset).resolve().as_posix())])
        cup_prim.RemoveAPI(UsdPhysics.CollisionAPI)
        cup_prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        UsdGeom.Xformable(cup_prim).MakeMatrixXform().Set(local)
        equivalent = UsdGeom.BBoxCache(0, ["default"]).ComputeWorldBound(cup_prim).ComputeAlignedRange()
        bounds_error = max(
            np.linalg.norm(np.asarray(original.GetMin()) - np.asarray(equivalent.GetMin())),
            np.linalg.norm(np.asarray(original.GetMax()) - np.asarray(equivalent.GetMax())),
        )
        if bounds_error > 1e-5:
            raise RuntimeError(
                f"Cup hierarchy conversion changed the physical geometry bounds by {bounds_error} m"
            )
        report["cup_hierarchy_conversion"] = dict(
            root="Xform", geometry="Geometry", bounds_error_m=float(bounds_error)
        )
    sys.path.insert(0,str(ROOT))
    from pipeline.simforge.bc.scenario import apply_offsets,start_y
    scenario_offsets=apply_offsets(stage,args.cup_offset_x,args.cup_offset_y)
    args.y=start_y(args.y,args.start_offset_y)
    report['scenario_start_offset_y_m']=args.start_offset_y
    os.environ["PROJECT_ROOT"] = str(Path(args.asset_root).resolve(strict=True))
    spec = importlib.util.spec_from_file_location(
        "unitree_reference_config", str(Path(args.robot_config).resolve(strict=True))
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
    from pipeline.ue_isaac_pipeline.unitree_wbc import UnitreeG1WBC

    cfg = module.G129_CFG_WITH_DEX3_WHOLEBODY.copy()
    cfg.prim_path = "/World/G1"
    cfg.spawn.activate_contact_sensors = args.grasp_probe
    cfg.spawn.rigid_props.retain_accelerations = False
    if args.fixed_base_probe:
        cfg.spawn.articulation_props.fix_root_link = True
    yaw = math.radians(args.yaw)
    cfg.init_state = cfg.init_state.replace(
        pos=(args.x, args.y, args.z), rot=(math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
    )
    # Conservative common ceiling from Unitree Dex3-1's lower actual torque
    # envelope; per-joint motor-type mapping has not been measured here.
    if not args.vendor_hands:
        cfg.actuators["hands"].effort_limit = 0.49
        cfg.actuators["hands"].effort_limit_sim = 0.49
        cfg.actuators["hands"].velocity_limit = 11.0
        cfg.actuators["hands"].velocity_limit_sim = 11.0
        cfg.actuators["hands"].stiffness = 6.0
        cfg.actuators["hands"].damping = 0.2
        cfg.actuators["hands"].armature = 0.0
    scfg = sim_utils.SimulationCfg(
        dt=1 / 200, device=args.device, use_fabric=args.device != "cpu", render_interval=4
    )
    # Replace the imported scene's context with one created by this Lab runtime.
    # There is exactly one active physics scene; the source layer is untouched.
    scfg.physics_prim_path = "/World/ApartmentPhysics"
    if scfg.use_fabric:
        if (
            not omni.kit.app.get_app()
            .get_extension_manager()
            .set_extension_enabled_immediate("omni.physx.fabric", True)
        ):
            raise RuntimeError("GPU physics requires the official PhysX Fabric extension")
    scfg.physx.bounce_threshold_velocity = 0.01
    scfg.physx.friction_correlation_distance = 0.00625
    sim = sim_utils.SimulationContext(scfg)
    # Loading an existing apartment PhysicsScene can retain the SDK's 60 Hz
    # default despite SimulationCfg.dt. Set and verify the effective clock.
    sim.set_simulation_dt(physics_dt=1 / 200, rendering_dt=1 / 50)
    robot = Articulation(cfg)
    if args.vendor_foot_friction:
        foot_material_path = "/World/AIWorkflow/UnitreeFootBaseline"
        foot_material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            friction_combine_mode="max",
            restitution_combine_mode="max",
        )
        foot_material_cfg.func(foot_material_path, foot_material_cfg)
        for side in ("left", "right"):
            foot = stage.GetPrimAtPath(f"/World/G1/{side}_ankle_roll_link/collisions")
            UsdShade.MaterialBindingAPI.Apply(foot).Bind(
                UsdShade.Material(stage.GetPrimAtPath(foot_material_path)),
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )
        report["foot_contact_reference"] = dict(
            static_friction=1.0,
            dynamic_friction=1.0,
            combine_mode="max",
            measured=False,
            source="Unitree move_cylinder_g1_29dof_dex3_hw_env_cfg.py at the pinned commit",
        )
    if args.grasp_probe:
        for prim in stage.Traverse():
            if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                PhysxSchema.PhysxContactReportAPI(prim).CreateThresholdAttr(0.0)
            if prim.HasAPI(UsdPhysics.CollisionAPI) and (
                f"/G1/{args.hand}_hand" in str(prim.GetPath()) or "/cup_V1/" in str(prim.GetPath())
            ):
                contact_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                contact_api.CreateContactOffsetAttr(0.0015)
                contact_api.CreateRestOffsetAttr(0.0)
    if args.keep_cup_awake:
        report["target_contact_observability"] = (
            "Wake target through native PhysX after verified desktop contact only; initial physics, gravity and native contact solving retained, support impulses are not synthesized"
        )
    cup = (
        None
        if args.no_cup_view
        else RigidObject(RigidObjectCfg(prim_path="/World/Environment/Source/cup_V1", spawn=None))
    )
    grip_sensors = {}
    cabinet_sensors = {}
    if args.grasp_probe:
        from pipeline.ue_isaac_pipeline.arm_kinematics import UsdArmKinematics, pose_matrix, matrix_pose
        from isaaclab.sensors import ContactSensor, ContactSensorCfg
        from isaaclab.utils.math import compute_pose_error

        for name in (
            "right_hand_palm_link",
            "right_hand_thumb_0_link",
            "right_hand_index_0_link",
            "right_hand_index_1_link",
            "right_hand_middle_0_link",
            "right_hand_middle_1_link",
            "right_hand_thumb_1_link",
            "right_hand_thumb_2_link",
        ):
            name = name.replace("right_hand", args.hand + "_hand")
            grip_sensors[name] = ContactSensor(
                ContactSensorCfg(
                    prim_path="/World/G1/" + name,
                    filter_prim_paths_expr=["/World/Environment/Source/cup_V1"],
                    history_length=1,
                )
            )
        for name in (
            "pelvis",
            "torso_link",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "right_elbow_link",
            "right_hand_palm_link",
            "right_hand_index_0_link",
            "right_hand_index_1_link",
            "right_hand_middle_0_link",
            "right_hand_middle_1_link",
            "right_hand_thumb_2_link",
        ):
            name = name.replace("right_hand", args.hand + "_hand").replace(
                "right_elbow", args.hand + "_elbow"
            )
            cabinet_sensors[name] = ContactSensor(
                ContactSensorCfg(
                    prim_path="/World/G1/" + name,
                    filter_prim_paths_expr=["/World/Environment/Source/SM_AI_vol8_04_kitchen_furniture_01"],
                    history_length=1,
                )
            )
    if args.grasp_probe:
        from pipeline.ue_isaac_pipeline.contact_ledger import ContactLedger
        from pxr import PhysicsSchemaTools
        from omni.physx import get_physx_simulation_interface

        support_geometry = []
        support_bounds = UsdGeom.BBoxCache(0, ["default", "render"])
        for support in stage.GetPrimAtPath("/World/Environment/Source").GetChildren():
            if (
                not support.IsA(UsdGeom.Mesh)
                or not support.HasAPI(UsdPhysics.CollisionAPI)
                or support.HasAPI(UsdPhysics.RigidBodyAPI)
            ):
                continue
            bounds = support_bounds.ComputeWorldBound(support).ComputeAlignedRange()
            low, high = np.asarray(bounds.GetMin()), np.asarray(bounds.GetMax())
            size = high - low
            if size[0] > 1 and size[1] > 1 and size[2] < 0.01 and max(abs(low[2]), abs(high[2])) < 0.08:
                support_geometry.append(
                    dict(prim=str(support.GetPath()), bounds_min_m=low.tolist(), bounds_max_m=high.tolist())
                )
        report["geometry_validated_floor_supports"] = support_geometry
        contact_ledger = ContactLedger(floor_actor_paths=[entry["prim"] for entry in support_geometry])
        contact_phase = "setup"
        sys.path.insert(0,str(ROOT))
        from pipeline.simforge.bc.contact_resolve import ContactObserver
        setup_hand_cup_observer=ContactObserver()
        PhysxSchema.PhysxContactReportAPI.Apply(
            stage.GetPrimAtPath("/World/Environment/Source/cup_V1")
        ).CreateThresholdAttr(0.0)

        def on_actual_contact(headers, contact_data):
            try:
                contact_ledger.record(
                    headers,
                    contact_data,
                    PhysicsSchemaTools.intToSdfPath,
                    time=sim.current_time,
                    phase=contact_phase,
                )
                setup_hand_cup_observer.record(headers,contact_data,PhysicsSchemaTools.intToSdfPath,
                    time=sim.current_time,phase=contact_phase)
            except Exception as exc:
                contact_ledger.errors.append(str(exc))

        contact_subscription = get_physx_simulation_interface().subscribe_contact_report_events(
            on_actual_contact
        )
    # Session-local native query support, BEFORE original physics reset.
    # Never force-reload PhysX while this live episode is running.
    from pxr import PhysxSchema
    for query_prim in stage.Traverse():
        if query_prim.IsA(UsdPhysics.Scene):
            PhysxSchema.PhysxSceneAPI.Apply(query_prim).CreateEnableSceneQuerySupportAttr(True)
    stage.GetRootLayer().Save()  # Only this new episode's wrapper layer.
    sim.reset()
    # SDK 4.5 rebuilds PhysicsContext during reset and overwrites the clock.
    sim.set_simulation_dt(physics_dt=1 / 200, rendering_dt=1 / 50)
    sim.get_physics_context().set_physics_dt(1 / 200)
    if not math.isclose(sim.get_physics_dt(), 1 / 200, abs_tol=1e-9):
        raise RuntimeError(f"Effective physics dt is {sim.get_physics_dt()}, expected 0.005")
    robot.reset()
    materials, counts = np.unique(
        robot.root_physx_view.get_material_properties()[0].cpu().numpy(), axis=0, return_counts=True
    )
    report["actual_robot_contact_materials"] = [
        dict(values=v.tolist(), shapes=int(n)) for v, n in zip(materials, counts)
    ]
    if args.vendor_foot_friction and not any(
        np.allclose(v[:2], 1.0) and n >= 8 for v, n in zip(materials, counts)
    ):
        raise RuntimeError(
            "Physics backend did not resolve the eight foot spheres to the vendor friction baseline"
        )
    if cup is not None:
        cup.reset()
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.write_root_state_to_sim(robot.data.default_root_state)
    robot.set_joint_position_target(robot.data.default_joint_pos)
    foot_ids = [
        i
        for i, name in enumerate(robot.body_names)
        if name in ("left_ankle_roll_link", "right_ankle_roll_link")
    ]
    if args.grasp_probe:
        palm_id = robot.body_names.index(args.hand + "_hand_palm_link")
        arm_ids = [
            robot.joint_names.index(args.hand + "_" + name + "_joint")
            for name in (
                "shoulder_pitch",
                "shoulder_roll",
                "shoulder_yaw",
                "elbow",
                "wrist_roll",
                "wrist_pitch",
                "wrist_yaw",
            )
        ]
        finger_names = [
            args.hand + "_hand_" + n + "_joint"
            for n in ("index_0", "index_1", "middle_0", "middle_1", "thumb_0", "thumb_1", "thumb_2")
        ]
        finger_ids = [robot.joint_names.index(n) for n in finger_names]
        closed_fingers = np.array([0.30, 0.95, 0.30, 0.95, 0, -0.90, 0], dtype=np.float32)
        if args.hand == "left":
            closed_fingers *= -1
        robot.write_data_to_sim()
        if args.scenario_version=='v2':
            from pipeline.simforge.bc.scenario import measure_before_step,write_scenario
            scenario_before=measure_before_step(stage,robot,cup,cfg.spawn.usd_path,scenario_offsets)
            write_scenario(out/'scenario-before-physics.json',scenario_before)
        sim.step(render=False)
        robot.update(1 / 200)
        cup.update(1 / 200)
        first_palm = robot.data.body_pos_w[0, palm_id].clone()
        first_rotation = robot.data.body_quat_w[0, palm_id].clone()
        arm_command = robot.data.default_joint_pos[0, arm_ids].cpu().numpy().copy()
        full_jacobian_shape = list(robot.root_physx_view.get_jacobians().shape)
        jacobian_offset = full_jacobian_shape[-1] - len(robot.joint_names)
        jacobian_body_offset = len(robot.body_names) - full_jacobian_shape[1]
        if jacobian_offset not in (0, 6) or jacobian_body_offset not in (0, 1):
            raise RuntimeError(f"Unsupported articulation Jacobian layout: {full_jacobian_shape}")
        report.update(
            jacobian_shape=full_jacobian_shape,
            jacobian_joint_offset=jacobian_offset,
            jacobian_body_offset=jacobian_body_offset,
            fixed_base=robot.is_fixed_base,
        )
        cup_start = cup.data.root_pos_w[0].cpu().numpy().copy()
        # Fingers point forward (-world Y), palm faces left (+world X).
        # This keeps the right wrist on the robot's right side of the cup.
        grip_yaw = math.radians(args.grip_yaw)
        desired_rotation = torch.tensor(
            [math.cos(grip_yaw / 2), 0.0, 0.0, math.sin(grip_yaw / 2)], device=robot.device
        )
        if args.opposite_side:
            desired_rotation = torch.tensor(
                [0.0, math.cos(grip_yaw / 2), math.sin(grip_yaw / 2), 0.0], device=robot.device
            )
        if float((first_rotation * desired_rotation).sum()) < 0:
            desired_rotation = -desired_rotation
        arm_model = UsdArmKinematics(cfg.spawn.usd_path, end=args.hand + "_hand_palm_link")
        if arm_model.joint_names != [robot.joint_names[i] for i in arm_ids]:
            raise RuntimeError("Kinematic solver joint names do not match the articulation")
        torso_id = robot.body_names.index("torso_link")
        torso_pose = robot.data.body_state_w[0, torso_id, :7].cpu().numpy()
        torso_matrix = pose_matrix(torso_pose[:3], torso_pose[3:])
        palm_pose = robot.data.body_state_w[0, palm_id, :7].cpu().numpy()
        measured_relative = pose_matrix(palm_pose[:3], palm_pose[3:]) @ np.linalg.inv(torso_matrix)
        fk_error = float(np.max(abs(arm_model.forward(arm_command) - measured_relative)))
        if fk_error > 0.001:
            raise RuntimeError(f"USD kinematics/PhysX frame mismatch {fk_error}")
        grasp_rotation_matrix = pose_matrix([0, 0, 0], desired_rotation.cpu().numpy())[:3, :3]
        final_palm = (
            cup_start
            + [0, 0, args.grip_height]
            - np.array([args.grip_center_x, args.grip_center_y * (-1 if args.hand == "left" else 1), 0])
            @ grasp_rotation_matrix
        )
        initial_palm_target = (
            final_palm + [0, 0, args.pregrasp_clearance]
            if args.ready_grasp
            else [args.x - 0.22, args.y - 0.20, args.raise_height]
        )
        raised_target = pose_matrix(initial_palm_target, desired_rotation.cpu().numpy()) @ np.linalg.inv(
            torso_matrix
        )
        raised_arm_q, raised_ik_report = arm_model.solve(raised_target, arm_command, starts=16)
        report.update(kinematics_matrix_error=fk_error, raised_posture_ik=raised_ik_report)
        if not raised_ik_report["feasible"]:
            raise RuntimeError("No feasible initial raised-hand pose")
        if args.start_raised:
            initial_q = robot.data.default_joint_pos.clone()
            initial_q[0, arm_ids] = torch.tensor(raised_arm_q, dtype=torch.float32, device=robot.device)
            robot.write_joint_state_to_sim(initial_q, robot.data.default_joint_vel)
            robot.set_joint_position_target(initial_q)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(1 / 200)
            cup.update(1 / 200)
            first_palm = robot.data.body_pos_w[0, palm_id].clone()
            first_rotation = robot.data.body_quat_w[0, palm_id].clone()
            arm_command = raised_arm_q.copy()
            report["initial_arm_prepositioned_before_recording"] = True
        if args.scenario_version=='v2':
            from pipeline.simforge.bc.scenario import complete_preflight,write_scenario
            scenario_result=complete_preflight(scenario_before,setup_hand_cup_observer,contact_ledger)
            write_scenario(out/'scenario-v2.json',scenario_result)
            report['scenario_v2']=scenario_result
            if args.scenario_preflight_only:
                report.update(status='completed',scenario_preflight_only=True,bc_inference_calls=0)
                write_json(out/'probe-report.json',report)
                write_json(out/'contact-ledger.json',contact_ledger.export())
                sim.stop()
                app.close();os._exit(0)
            if scenario_result['S2_classification']=='init_infeasible_v2':
                raise RuntimeError('INIT_INFEASIBLE_V2')
            if not scenario_result['G0_pass']:
                raise RuntimeError('SCENARIO_G0_FAILED')
        # Additive contact preparation AFTER the original initializer. Its
        # cup_start/final_palm/16-start IK source block remains unchanged.
        from pipeline.simforge.bc.contact_resolve import run_init_contact_resolve
        from pipeline.simforge.bc.safe_execute import Parameters
        init_parameters=Parameters(**read_json(args.safe_parameters))
        init_metadata=torch.load(args.bc_checkpoint,map_location='cpu',weights_only=False)
        init_result=run_init_contact_resolve(robot,cup,sim,stage,cfg.spawn.usd_path,arm_model,
            arm_ids,finger_ids,palm_id,torso_id,setup_hand_cup_observer,init_parameters,out,
            init_metadata['cup_center_local_m'],initial_physics_steps=2 if args.start_raised else 1)
        report['init_contact_resolve']=init_result
        if init_result['status'] not in ('RESOLVED','NO_CONTACT_DETECTED'):
            report['initialization_contact_failure']=init_result['status']
            raise RuntimeError(init_result['status'])
        first_palm=robot.data.body_pos_w[0,palm_id].clone()
        first_rotation=robot.data.body_quat_w[0,palm_id].clone()
        arm_command=robot.data.joint_pos[0,arm_ids].cpu().numpy().copy()
        initial_arm_q = arm_command.copy()
        arm_servo_integral = np.zeros(7)
        solver_q = raised_arm_q.copy()
        # Measured hand geometry: fingers extend +X, palm faces +Y, spread along Z.
        # World yaw -90 maps local grip center (.095,.055,0) to (.055,-.095,0).
        best_lift = 0.0
        stable_grasp_ticks = 0
        max_stable_grasp_ticks = 0
        stable_carry_ticks = max_stable_carry_ticks = 0
    if args.device == "cpu":
        sim.get_physics_context().enable_fabric(False)
        sim.get_physics_context().set_physx_update_transformations_settings(update_to_usd=True)
    wbc = UnitreeG1WBC(
        robot.joint_names,
        robot.data.default_joint_pos[0].cpu().numpy(),
        session,
    )
    report.update(
        robot_asset=cfg.spawn.usd_path,
        robot_prim=cfg.prim_path,
        joint_names=robot.joint_names,
        body_names=robot.body_names,
        root_has_world_anchor=args.fixed_base_probe,
        grasp_calibration_only=args.fixed_base_probe,
        gravity_disabled=cfg.spawn.rigid_props.disable_gravity,
        default_joint_positions=robot.data.default_joint_pos[0].cpu().tolist(),
        hand_effort_ceiling_Nm=None if args.vendor_hands else 0.49,
        vendor_hand_actuators=args.vendor_hands,
        actual_physics_dt=sim.get_physics_dt(),
        hand_ceiling_source="https://www.unitree.com/mobile/Dex3-1/",
        source_environment=str(Path(args.scene).resolve()),
        network_input=[1, 910],
        network_output=[1, 12],
        walking_policy_active=not args.hold,
        cup_tensor_observer_enabled=not args.no_cup_view,
    )
    metrics = []
    if cup is not None:
        report["actual_cup_mass_kg"] = float(cup.root_physx_view.get_masses().reshape(-1)[0])
    native_masses = robot.root_physx_view.get_masses()[0].cpu().numpy()
    native_inertias = robot.root_physx_view.get_inertias()[0].cpu().numpy()
    if not np.isfinite(native_masses).all() or not np.isfinite(native_inertias).all():
        raise RuntimeError("Native PhysX mass/inertia contains nonfinite values")
    write_json(
        out / "native-mass-properties.json",
        dict(
            source="initialized native PhysX tensors; SI units; no authored property replaced",
            robot_asset=cfg.spawn.usd_path,
            bodies=[
                dict(name=name, mass_kg=float(mass), inertia_tensor_kgm2=inertia.reshape(3, 3).tolist())
                for name, mass, inertia in zip(robot.body_names, native_masses, native_inertias)
            ],
            cup_mass_kg=report.get("actual_cup_mass_kg"),
            cup_inertia_tensor_kgm2=(
                cup.root_physx_view.get_inertias()[0].cpu().numpy().reshape(3, 3).tolist()
                if cup is not None
                else None
            ),
            real_material_calibration=False,
        ),
    )
    write_json(out / "joint-drives-runtime-initial.json", capture_joint_drives(robot, "initial"))
    physics_records = []
    cup_records = []
    joint_records = []
    motor_records, command_records, policy_records, action_records, phase_records, force_records = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    effort_records = []
    initial_root = None
    integral_xy = np.zeros(2)
    motion_time = 0.0
    transfer_settled_ticks = 0
    desk_supported_ticks = 0
    unload_started_step, unload_palm = None, None
    if args.transfer_plan:
        from pipeline.ue_isaac_pipeline.apartment_transfer import PolylineFollower

        transfer_plan = read_json(args.transfer_plan)
        route = np.asarray(transfer_plan["route_xy_m"])
        withdraw_xy = np.asarray(transfer_plan["route_start_xy_m"])
        closest = int(np.argmin(np.linalg.norm(route - withdraw_xy, axis=1)))
        route = np.vstack((withdraw_xy, route[closest + 1 :]))
        if not 0.15 <= args.route_lookahead <= 0.65:
            raise ValueError("Route lookahead must stay within the validated local map scale")
        route_follower = PolylineFollower(route, lookahead=args.route_lookahead, maximum_speed=0.60)
        navigation_map = np.load(Path(args.transfer_plan).parent / "navigation-map.npz")
        place_goal = np.asarray(read_json(args.placement_plan)["cup_origin_m"])
        placement_clearance_map = np.load(Path(args.placement_plan).parent / "desktop-clearance.npz")
        station = np.array([-1.50, -0.88])
        if args.human_forward_transfer:
            station = np.asarray(transfer_plan["desk_station"]["base_xy_m"])
            turn_station = np.asarray(transfer_plan["route_end_xy_m"])
            turn_exit = np.asarray(transfer_plan.get("left_turn_exit_xy_m", turn_station))
            turn_exit_tolerance = float(transfer_plan.get("left_turn_exit_tolerance_m", 0.85))
        route_state = None
        report["transfer_controller"] = (
            "Vendor locomotion policy plus measured-state route following and arm IK; no new RL training"
        )
        report["placement_target_m"] = place_goal.tolist()
        if args.navigation_policy:
            from pipeline.ue_isaac_pipeline.learned_navigation import LearnedForwardGovernor

            navigation_actor = LearnedForwardGovernor(args.navigation_policy)
            report["learned_high_level_navigation"] = dict(
                model=str(Path(args.navigation_policy).resolve()),
                sha256=digest_file(args.navigation_policy),
                training="Supervised geometric-controller imitation",
                new_leg_or_grasp_policy=False,
            )
    minimum_height, maximum_tilt = float("inf"), 0
    with torch.inference_mode():
        for step in range(round((args.seconds + args.walk_seconds) * 50)):
            if args.keep_cup_awake and unload_started_step is not None:
                get_physx_simulation_interface().wake_up(
                    omni.usd.get_context().get_stage_id(),
                    PhysicsSchemaTools.sdfPathToInt(Sdf.Path("/World/Environment/Source/cup_V1")),
                )
                report["target_verification_wake_requests"] = (
                    report.get("target_verification_wake_requests", 0) + 1
                )
            previous_motion_time = motion_time
            seconds = motion_time if args.grasp_probe else step / 50
            if args.walking_policy:
                session.weight = float(np.clip((seconds - 16) / 1.5, 0, 1)) if args.carry_probe else 0.0
            command = [0, 0, 0, 0.8]
            if step >= args.seconds * 50:
                command[0] = -0.25
            if args.position_hold:
                base = robot.data.root_pos_w[0].cpu().numpy()
                orientation = robot.data.root_quat_w[0].cpu().numpy()
                w, x, y, z = orientation
                actual_yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                goal = np.array([args.x, args.y])
                yaw_goal = yaw
                goal[1] += max(0, step / 50 - args.seconds) * 0.20
                if args.grasp_probe:
                    goal[1] = args.y + min(max((seconds - 2.8) / 6, 0), 1) * (-1.56 - args.y)
                if args.carry_probe and seconds >= 16:
                    goal[1] = -1.56 + 0.25 * min((seconds - 16) / 2, 1)
                    yaw_goal = yaw + (-math.pi - yaw) * min(max((seconds - 20) / 3, 0), 1)
                    if args.lateral_carry:
                        yaw_goal = yaw
                        if seconds >= 18:
                            goal = np.array([args.carry_goal_x, args.carry_goal_y])
                    if args.forward_carry:
                        goal = np.array([args.carry_goal_x, args.carry_goal_y])
                        if "carry_arrived" not in locals():
                            carry_arrived = False
                        if np.linalg.norm(goal - base[:2]) < 0.13:
                            carry_arrived = True
                        if "carry_heading" not in locals() or (
                            not carry_arrived and np.linalg.norm(goal - base[:2]) > 0.20
                        ):
                            carry_heading = math.atan2(goal[1] - base[1], goal[0] - base[0])
                        yaw_goal = carry_heading
                    if args.transfer_plan and seconds >= 16:
                        yaw_goal = yaw
                        if seconds < (TURN_BEGIN if args.human_forward_transfer else NAV_BEGIN):
                            goal = np.array([args.x, -1.56]) + (withdraw_xy - [args.x, -1.56]) * min(
                                (seconds - 16)
                                / ((TURN_BEGIN if args.human_forward_transfer else NAV_BEGIN) - 16),
                                1,
                            )
                        elif args.human_forward_transfer and seconds < NAV_BEGIN:
                            goal = withdraw_xy.copy()
                            yaw_goal = -math.pi
                        elif seconds < NAV_END:
                            route_state = route_follower.update(base[:2])
                            goal = route_state["goal"]
                            if args.human_forward_transfer:
                                yaw_goal = (
                                    math.atan2(route_state["velocity"][1], route_state["velocity"][0])
                                    if route_state["endpoint_error"] > 0.12
                                    else -math.pi
                                )
                            ij = np.floor(
                                (base[:2] - navigation_map["origin_xy"]) / navigation_map["resolution_m"]
                            ).astype(int)
                            grid = navigation_map["clearance_m"]
                            if not (0 <= ij[0] < grid.shape[1] - 1 and 0 <= ij[1] < grid.shape[0] - 1):
                                raise RuntimeError("Measured root left navigation map")
                            route_state["obstacle_clearance_m"] = float(
                                grid[ij[1] : ij[1] + 2, ij[0] : ij[0] + 2].min()
                            )
                            if (
                                route_state["obstacle_clearance_m"] < 0.25
                                or route_state["cross_track_error"] > 0.60
                            ):
                                raise RuntimeError("Measured root left the safe navigation area")
                        else:
                            goal = (
                                turn_station.copy()
                                if args.human_forward_transfer and seconds < FACE_DESK_END
                                else station.copy()
                            )
                            if "desk_hold_initialized" not in locals():
                                integral_xy[:] = 0
                                desk_hold_initialized = True
                error_xy = goal - base[:2]
                integral_xy = np.clip(integral_xy + error_xy / 50, -0.3, 0.3)
                velocity = robot.data.root_lin_vel_w[0].cpu().numpy()[:2]
                world_velocity = np.clip(1.8 * error_xy + 0.6 * integral_xy - 0.35 * velocity, -0.45, 0.45)
                if args.grasp_probe:
                    world_velocity = np.clip(
                        2.0 * error_xy + 0.6 * integral_xy - 0.35 * velocity, -0.60, 0.60
                    )
                if args.transfer_plan and NAV_BEGIN <= seconds < NAV_END:
                    world_velocity = route_state["velocity"] - 0.20 * velocity
                if (
                    args.transfer_plan
                    and seconds >= PLACE_BEGIN
                    and np.linalg.norm(error_xy) < 0.04
                    and np.linalg.norm(velocity) < 0.16
                ):
                    world_velocity[:] = 0
                    integral_xy[:] = 0
                c, s = math.cos(actual_yaw), math.sin(actual_yaw)
                angular_z = float(robot.data.root_ang_vel_b[0, 2])
                yaw_error = math.atan2(math.sin(yaw_goal - actual_yaw), math.cos(yaw_goal - actual_yaw))
                command = [
                    float(np.clip(c * world_velocity[0] + s * world_velocity[1], -0.6, 1.0)),
                    float(np.clip(-s * world_velocity[0] + c * world_velocity[1], -0.5, 0.5)),
                    float(
                        np.clip(2 * yaw_error - 0.25 * angular_z, -args.yaw_rate_limit, args.yaw_rate_limit)
                    ),
                    0.8,
                ]
                if args.transfer_plan and seconds >= ARM_SETTLE_BEGIN:
                    crouch_fraction = float(np.clip((seconds - ARM_SETTLE_BEGIN - 1) / 3, 0, 1))
                    crouch_fraction = crouch_fraction * crouch_fraction * (3 - 2 * crouch_fraction)
                    desired_height = 0.8 - 0.10 * crouch_fraction
                    height_request = float(
                        np.clip(
                            desired_height
                            + (desired_height - base[2])
                            - 0.10 * float(robot.data.root_lin_vel_w[0, 2]),
                            0.50,
                            0.80,
                        )
                    )
                    filtered_height = 0.95 * locals().get("filtered_height", 0.8) + 0.05 * height_request
                    command[3] = filtered_height
                if args.carry_probe and args.forward_carry and seconds >= 16:
                    distance = float(np.linalg.norm(goal - base[:2]))
                    actual_forward = math.cos(actual_yaw) * velocity[0] + math.sin(actual_yaw) * velocity[1]
                    forward_velocity_filter = (
                        0.9 * locals().get("forward_velocity_filter", actual_forward) + 0.1 * actual_forward
                    )
                    desired_speed = min(0.30, 0.8 * distance)
                    command[0] = (
                        float(
                            np.clip(
                                0.5 * desired_speed + 0.5 * (desired_speed - forward_velocity_filter), 0, 0.30
                            )
                        )
                        if seconds >= 19 and not carry_arrived and abs(yaw_error) < 0.30
                        else 0.0
                    )
                    command[1] = 0.0
                if args.human_forward_transfer and TURN_BEGIN <= seconds < ARM_SETTLE_BEGIN:
                    docking_reverse = float(np.clip(command[0], -0.08, 0))
                    forward_request = float(np.clip(command[0], 0, 0.5))
                    turning = seconds < NAV_BEGIN or NAV_END <= seconds < FACE_DESK_END
                    lateral_balance = command[1]
                    command[1] = 0.0
                    if NAV_BEGIN <= seconds < NAV_END:
                        alignment = max(0.0, 1.0 - abs(yaw_error) / 0.45)
                        command[0] = float(np.clip(command[0], 0, 0.50)) * alignment
                    elif seconds >= FACE_DESK_END:
                        command[0] = float(np.clip(command[0], -0.25, 0.30))
                        command[1] = float(np.clip(lateral_balance, -0.20, 0.20))
                    else:
                        command[0] = 0.20 if abs(yaw_error) > 0.10 else 0.0
                    command[2] = float(np.clip(command[2], -1.0, 1.0))
                    if args.navigation_policy:
                        prediction = navigation_actor.predict(
                            yaw_error,
                            angular_z,
                            forward_request,
                            turning,
                            measured_forward_speed=float(c * velocity[0] + s * velocity[1]),
                        )
                        command[0], command[2] = map(float, prediction)
                        if turning and abs(yaw_error) > 0.10:
                            command[0] = max(command[0], 0.20 if yaw_error < 0 else 0.14)
                            command[2] = math.copysign(max(abs(command[2]), 0.8), yaw_error)
                            if NAV_END <= seconds < FACE_DESK_END:
                                command[2] = float(
                                    np.clip(command[2], -args.left_turn_rate_limit, args.left_turn_rate_limit)
                                )
                        if seconds >= FACE_DESK_END:
                            command[0] = docking_reverse if docking_reverse < 0 else min(command[0], 0.30)
                    if seconds >= FACE_DESK_END:
                        forward_error = c * error_xy[0] + s * error_xy[1]
                        if abs(forward_error) > 0.10:
                            command[0] = math.copysign(max(abs(command[0]), 0.23), forward_error)
                    if turning and abs(yaw_error) <= 0.10:
                        command[0] = 0.0
            if args.hold:
                target_q = robot.data.default_joint_pos[0].cpu().numpy().copy()
                action = np.zeros(12, dtype=np.float32)
            else:
                desired_arms = None
                closure = 0.0
                if args.grasp_probe:
                    phase = "raise_clear_of_counter"
                    approach = min(seconds / (args.pregrasp_seconds if args.ready_grasp else 2.8), 1.0)
                    smooth = approach * approach * (3 - 2 * approach)
                    above = torch.tensor(final_palm + [0, 0, 0.25], dtype=torch.float32, device=robot.device)
                    raised = torch.tensor(
                        [args.x - 0.22, args.y - 0.20, args.raise_height],
                        dtype=torch.float32,
                        device=robot.device,
                    )
                    desired_palm = first_palm * (1 - smooth) + raised * smooth
                    rotation = torch.nn.functional.normalize(
                        first_rotation * (1 - smooth) + desired_rotation * smooth, dim=-1
                    )
                    if seconds >= 2.8:
                        phase = "walk_with_raised_hand"
                        torso_pose = robot.data.body_state_w[0, torso_id, :7].cpu().numpy()
                        moving_raised = matrix_pose(
                            arm_model.forward(raised_arm_q) @ pose_matrix(torso_pose[:3], torso_pose[3:])
                        )
                        desired_palm = torch.tensor(
                            moving_raised[:3], dtype=torch.float32, device=robot.device
                        )
                        rotation = torch.tensor(moving_raised[3:], dtype=torch.float32, device=robot.device)
                    if seconds >= 8.8:
                        phase = "extend_above"
                        forward = min((seconds - 8.8) / 2.0, 1.0)
                        desired_palm = desired_palm * (1 - forward) + above * forward
                        rotation = desired_rotation
                    if seconds >= 10.8:
                        phase = "lower_open"
                        down = min((seconds - 10.8) / 1.5, 1.0)
                        desired_palm = above - torch.tensor([0, 0, 0.25 * down], device=robot.device)
                        rotation = desired_rotation
                    if seconds >= 12.3:
                        phase = "close"
                        closure = min((seconds - 12.3) / 1.2, 1.0)
                    if seconds >= 13.5:
                        phase = "lift" if seconds < 15.0 else "hold"
                        desired_palm = torch.tensor(
                            final_palm + [0, 0, 0.15 * min((seconds - 13.5) / 1.5, 1)],
                            dtype=torch.float32,
                            device=robot.device,
                        )
                    if args.ready_grasp and seconds < 12.3:
                        phase = "settle_open_grasp"
                        desired_palm = (
                            first_palm * (1 - smooth)
                            + torch.tensor(final_palm, dtype=torch.float32, device=robot.device) * smooth
                            if args.pregrasp_clearance
                            else torch.tensor(final_palm, dtype=torch.float32, device=robot.device)
                        )
                        rotation = desired_rotation
                    if args.carry_probe and seconds >= 16:
                        if "carry_offset_body" not in locals():
                            c, s = math.cos(actual_yaw), math.sin(actual_yaw)
                            initial_offset = robot.data.body_pos_w[0, palm_id, :2].cpu().numpy() - base[:2]
                            carry_offset_body = np.array([[c, s], [-s, c]]) @ initial_offset
                        retract = min(max((seconds - 18) / 2, 0), 1)
                        offset = (
                            carry_offset_body * (1 - retract)
                            + np.array([0.34, 0.22 if args.hand == "left" else -0.22]) * retract
                        )
                        c, s = math.cos(actual_yaw), math.sin(actual_yaw)
                        xy = base[:2] + np.array([[c, -s], [s, c]]) @ offset
                        desired_palm = torch.tensor(
                            [*xy, final_palm[2] + 0.15], dtype=torch.float32, device=robot.device
                        )
                        yaw_delta = actual_yaw - yaw
                        rotation_matrix = pose_matrix(
                            [0, 0, 0], desired_rotation.cpu().numpy()
                        ) @ pose_matrix([0, 0, 0], [math.cos(yaw_delta / 2), 0, 0, math.sin(yaw_delta / 2)])
                        rotation = torch.tensor(
                            matrix_pose(rotation_matrix)[3:], dtype=torch.float32, device=robot.device
                        )
                        phase = (
                            "withdraw_with_cup"
                            if seconds < 18
                            else "retract_carry_pose"
                            if seconds < 20
                            else "turn_with_cup"
                            if seconds < 23
                            else "carry_hold"
                        )
                        if args.forward_carry:
                            phase = (
                                "turn_with_cup"
                                if seconds < 19
                                else "walk_forward_with_cup"
                                if seconds < 23
                                else "carry_hold"
                            )
                        if args.lateral_carry:
                            phase = (
                                "withdraw_with_cup"
                                if seconds < 18
                                else "walk_lateral_with_cup"
                                if seconds < 23
                                else "carry_hold"
                            )
                        if args.joint_carry:
                            if "carry_arm_q" not in locals():
                                carry_arm_q = arm_command.copy()
                            torso_pose = robot.data.body_state_w[0, torso_id, :7].cpu().numpy()
                            carry_world = matrix_pose(
                                arm_model.forward(carry_arm_q) @ pose_matrix(torso_pose[:3], torso_pose[3:])
                            )
                            desired_palm = torch.tensor(
                                carry_world[:3], dtype=torch.float32, device=robot.device
                            )
                            rotation = torch.tensor(carry_world[3:], dtype=torch.float32, device=robot.device)
                    if args.transfer_plan and seconds >= 16:
                        if "grasp_palm_relative" not in locals():
                            measured_palm = robot.data.body_state_w[0, palm_id, :7].cpu().numpy()
                            measured_cup = cup.data.root_state_w[0, :7].cpu().numpy()
                            # Coordinate conversion only: no joint or force attaches the cup to the hand.
                            grasp_palm_relative = pose_matrix(
                                measured_palm[:3], measured_palm[3:]
                            ) @ np.linalg.inv(pose_matrix(measured_cup[:3], measured_cup[3:]))
                        phase = (
                            "withdraw_with_cup"
                            if seconds < (TURN_BEGIN if args.human_forward_transfer else NAV_BEGIN)
                            else "turn_right_with_cup"
                            if seconds < NAV_BEGIN
                            else "navigate_apartment_with_cup"
                            if seconds < NAV_END
                            else "turn_left_to_desk"
                            if seconds < FACE_DESK_END
                            else "approach_desk_forward"
                            if seconds < ARM_SETTLE_BEGIN
                            else "settle_at_desk"
                            if seconds < PLACE_BEGIN
                            else "move_above_desk"
                            if seconds < ABOVE_END
                            else "lower_to_desk"
                            if seconds < LOWER_END
                            else "release_on_desk"
                            if seconds < RELEASE_END
                            else "withdraw_hand"
                            if seconds < WITHDRAW_END
                            else "verify_placement"
                        )
                        if ARM_SETTLE_BEGIN <= seconds < PLACE_BEGIN:
                            if "settle_palm_height" not in locals():
                                robot.write_joint_damping_to_sim(
                                    torch.tensor([[6.0, 6.0, 4.0, 4.0, 2.0, 2.0, 2.0]], device=robot.device),
                                    joint_ids=arm_ids,
                                )
                                report["placement_arm_damping_Nm_s_per_rad"] = [6, 6, 4, 4, 2, 2, 2]
                                settle_palm_height = float(robot.data.body_pos_w[0, palm_id, 2])
                                settle_initial_palm = (
                                    robot.data.body_state_w[0, palm_id, :7].cpu().numpy().copy()
                                )
                                # Restore the calibrated grasp wrist before reaching. A transient
                                # in-hand cup attitude is not a reliable wrist calibration.
                                settle_upright_rotation = desired_rotation.clone()
                                report["settle_upright_palm_rotation_wxyz"] = (
                                    settle_upright_rotation.cpu().tolist()
                                )
                                settle_initial_base = base[:2].copy()
                                solver_q = arm_command.copy()
                            c, s = math.cos(actual_yaw), math.sin(actual_yaw)
                            stabilized_xy = base[:2] + np.array([[c, -s], [s, c]]) @ carry_offset_body
                            settle_blend = min(seconds - ARM_SETTLE_BEGIN, 1)
                            settle_blend = settle_blend * settle_blend * (3 - 2 * settle_blend)
                            from_xy = settle_initial_palm[:2] + base[:2] - settle_initial_base
                            desired_palm = torch.tensor(
                                [
                                    *(from_xy * (1 - settle_blend) + stabilized_xy * settle_blend),
                                    settle_palm_height,
                                ],
                                dtype=torch.float32,
                                device=robot.device,
                            )
                            from_r = torch.tensor(
                                settle_initial_palm[3:], dtype=torch.float32, device=robot.device
                            )
                            to_r = (
                                settle_upright_rotation
                                if float((from_r * settle_upright_rotation).sum()) >= 0
                                else -settle_upright_rotation
                            )
                            rotation = torch.nn.functional.normalize(
                                from_r * (1 - settle_blend) + to_r * settle_blend, dim=-1
                            )
                        if seconds >= PLACE_BEGIN:
                            if "place_start_palm" not in locals():
                                pp = robot.data.body_state_w[0, palm_id, :7].cpu().numpy()
                                place_start_palm = pp.copy()
                                cp_now = cup.data.root_state_w[0, :7].cpu().numpy()
                                grasp_palm_relative = pose_matrix(pp[:3], pp[3:]) @ np.linalg.inv(
                                    pose_matrix(cp_now[:3], cp_now[3:])
                                )
                                solver_q = arm_command.copy()
                            elevated = place_goal + [0, 0, 0.10]
                            lowering = min(max((seconds - ABOVE_END) / 2, 0), 1)
                            target_cup = elevated - [0, 0, 0.1008 * lowering]
                            target_hand = matrix_pose(
                                grasp_palm_relative @ pose_matrix(target_cup, [1, 0, 0, 0])
                            )
                            blend = min((seconds - PLACE_BEGIN) / 3, 1)
                            blend = blend * blend * (3 - 2 * blend)
                            desired_palm = torch.tensor(
                                place_start_palm[:3] * (1 - blend) + target_hand[:3] * blend,
                                dtype=torch.float32,
                                device=robot.device,
                            )
                            rq = target_hand[3:]
                            if np.dot(rq, place_start_palm[3:]) < 0:
                                rq = -rq
                            rotation = torch.nn.functional.normalize(
                                torch.tensor(
                                    place_start_palm[3:] * (1 - blend) + rq * blend,
                                    dtype=torch.float32,
                                    device=robot.device,
                                ),
                                dim=-1,
                            )
                            closure = 1 - min(max((seconds - LOWER_END) / 0.35, 0), 1)
                            if seconds >= RELEASE_END:
                                withdraw = min(seconds - RELEASE_END, 1)
                                desired_palm += torch.tensor(
                                    [0, 0.12 * withdraw, 0.10 * withdraw], device=robot.device
                                )
                        if unload_palm is not None:
                            unload_elapsed = (step - unload_started_step) / 50
                            closure = float(
                                np.clip(
                                    1 - (unload_elapsed - 0.25) / (0.8 if args.feedback_release else 0.20),
                                    0,
                                    1,
                                )
                            )
                            desired_palm = torch.tensor(
                                unload_palm[:3], dtype=torch.float32, device=robot.device
                            )
                            rotation = torch.tensor(unload_palm[3:], dtype=torch.float32, device=robot.device)
                            # Clear the desktop before the opened fingers sweep backwards.
                            # Torque has already ramped down before this upward separation.
                            release_clearance = 0.015 * float(np.clip((unload_elapsed - 0.25) / 0.20, 0, 1))
                            desired_palm[2] += release_clearance
                            if args.feedback_release:
                                desired_palm[2] = float(unload_palm[2])
                                fingers_actual = robot.data.joint_pos[0, finger_ids].cpu().numpy()
                                palm_actual = robot.data.body_state_w[0, palm_id, :7].cpu().numpy()
                                opening_ready = bool(np.max(np.abs(fingers_actual)) < 0.20)
                                if opening_ready and closure < 0.01 and "release_open_step" not in locals():
                                    release_open_step = step
                                if "release_open_step" in locals():
                                    desired_palm[2] += 0.025 * float(
                                        np.clip((step - release_open_step) / 17.5, 0, 1)
                                    )
                                height_ready = bool(palm_actual[2] > unload_palm[2] + 0.012)
                                if opening_ready:
                                    rotation = torch.tensor(
                                        palm_actual[3:], dtype=torch.float32, device=robot.device
                                    )
                                if (
                                    unload_elapsed > 0.55
                                    and opening_ready
                                    and height_ready
                                    and "clear_withdraw_step" not in locals()
                                ):
                                    clear_withdraw_step = step
                                    clear_withdraw_pose = palm_actual.copy()
                                    report["release_feedback_clearance_verified"] = dict(
                                        time_s=step / 50,
                                        palm_height_increase_m=float(palm_actual[2] - unload_palm[2]),
                                        maximum_open_finger_error_rad=float(np.max(np.abs(fingers_actual))),
                                    )
                                if "clear_withdraw_step" in locals():
                                    withdraw = min((step - clear_withdraw_step) / 65, 1)
                                    desired_palm = torch.tensor(
                                        clear_withdraw_pose[:3]
                                        + np.array([0, 0.08 * withdraw, 0.05 * withdraw]),
                                        dtype=torch.float32,
                                        device=robot.device,
                                    )
                            elif unload_elapsed >= 0.55:
                                withdraw = min(unload_elapsed - 0.55, 1)
                                desired_palm += torch.tensor(
                                    [0, 0.12 * withdraw, 0.10 * withdraw], device=robot.device
                                )
                    pos_error, rot_error = compute_pose_error(
                        robot.data.body_pos_w[:, palm_id],
                        robot.data.body_quat_w[:, palm_id],
                        desired_palm.unsqueeze(0),
                        rotation.unsqueeze(0),
                        rot_error_type="axis_angle",
                    )
                    current = robot.data.joint_pos[0, arm_ids].cpu().numpy()
                    if (
                        args.joint_carry
                        and args.carry_probe
                        and seconds >= 16
                        and not (args.transfer_plan and seconds >= ARM_SETTLE_BEGIN)
                    ):
                        requested = carry_arm_q
                    elif seconds < 2.65 and not (args.ready_grasp and args.pregrasp_clearance):
                        requested = initial_arm_q * (1 - smooth) + raised_arm_q * smooth
                    elif 2.8 <= seconds < 8.8 and not args.ready_grasp:
                        requested = raised_arm_q
                    else:
                        if step % 5 == 0 or (args.transfer_plan and seconds >= ARM_SETTLE_BEGIN):
                            torso_pose = robot.data.body_state_w[0, torso_id, :7].cpu().numpy()
                            target_relative = pose_matrix(
                                desired_palm.cpu().numpy(), rotation.cpu().numpy()
                            ) @ np.linalg.inv(pose_matrix(torso_pose[:3], torso_pose[3:]))
                            solver_q, solver_report = arm_model.solve(
                                target_relative,
                                solver_q,
                                starts=1,
                                orientation_weight=0.03
                                if args.feedback_release and unload_palm is not None and closure < 0.01
                                else 0.12,
                                posture_weight=0.002
                                if args.transfer_plan and seconds >= ARM_SETTLE_BEGIN
                                else 0,
                            )
                        requested = solver_q
                    arm_rate = 0.04 if args.transfer_plan and seconds >= ARM_SETTLE_BEGIN else 0.008
                    arm_command += np.clip(requested - arm_command, -arm_rate, arm_rate)
                    arm_limits = robot.data.joint_pos_limits[0, arm_ids].cpu().numpy()
                    arm_command = np.clip(arm_command, arm_limits[:, 0], arm_limits[:, 1])
                    requested = arm_command.copy()
                    compensation = robot.root_physx_view.get_gravity_compensation_forces()[0].cpu().numpy()
                    gravity_offset = len(compensation) - len(robot.joint_names)
                    if gravity_offset not in (0, 6):
                        raise RuntimeError("Gravity force layout does not match articulation")
                    stiffness = robot.data.joint_stiffness[0, arm_ids].cpu().numpy()
                    if (stiffness <= 0).any():
                        raise RuntimeError("Gravity feedforward needs positive drive stiffness")
                    gravity_arm = compensation[np.asarray(arm_ids) + gravity_offset]
                    requested += np.clip(gravity_arm / stiffness, -0.25, 0.25)
                    arm_servo_integral = np.clip(
                        arm_servo_integral + args.arm_integral_gain * (arm_command - current) / 50,
                        -0.25,
                        0.25,
                    )
                    requested += arm_servo_integral
                    requested = np.clip(requested, arm_limits[:, 0], arm_limits[:, 1])
                    desired_arms = robot.data.default_joint_pos[0, wbc.arm_ids].cpu().numpy().copy()
                    desired_arms[0:7] = requested if args.hand == "left" else desired_arms[0:7]
                    desired_arms[7:] = requested if args.hand == "right" else desired_arms[7:]
                    if args.transfer_plan and args.hand == "left":
                        # Raise the unused hand above the chair before approaching the desk.
                        idle_blend = float(
                            np.clip(
                                (seconds - (16 if args.human_forward_transfer else NAV_END - 3)) / 2,
                                0,
                                1,
                            )
                        )
                        idle_goal = (
                            np.array([-0.4, 0, 0, 1.0, 0, 0, 0])
                            if args.human_forward_transfer
                            else np.array([1.2, -0.3, 0, 2.0, 0, 1.0, 0])
                        )
                        if args.human_forward_transfer:
                            lift_idle = float(
                                np.clip(
                                    (seconds - NAV_BEGIN - (0 if args.early_carry_posture else 4))
                                    / (3 if args.early_carry_posture else 6),
                                    0,
                                    1,
                                )
                            )
                            idle_goal = (
                                idle_goal * (1 - lift_idle)
                                + np.array(
                                    [
                                        -0.372309,
                                        -0.494941,
                                        -0.797240,
                                        -0.509505,
                                        1.269598,
                                        -0.745902,
                                        0.783234,
                                    ]
                                )
                                * lift_idle
                            )
                            chair_clearance_blend = float(np.clip((seconds - FACE_DESK_END) / 3, 0, 1))
                            idle_goal = (
                                idle_goal * (1 - chair_clearance_blend)
                                + np.array(
                                    [
                                        -0.725509,
                                        -0.712858,
                                        -0.698228,
                                        -0.510140,
                                        0.965471,
                                        -0.527561,
                                        0.369421,
                                    ]
                                )
                                * chair_clearance_blend
                            )
                        desired_arms[7:] = desired_arms[7:] * (1 - idle_blend) + idle_goal * idle_blend
                upper_observation_scale = (
                    1 - float(np.clip((seconds - 16) / 1.5, 0, 1))
                    if args.neutral_upper_walking_observation and args.carry_probe
                    else 1.0
                )
                if args.neutral_upper_during_transfer and args.human_forward_transfer:
                    fade_in = float(np.clip((seconds - 16) / 1.5, 0, 1))
                    restore = float(np.clip((seconds - FACE_DESK_END) / 1.5, 0, 1))
                    upper_observation_scale = 1 - fade_in * (1 - restore)
                    report["upper_body_policy_observation"] = (
                        "Nominal upper features during carry, actual arm physics/actuation retained; actual features restored before desk settling"
                    )
                target_q, observation, action = wbc.step(
                    robot.data.joint_pos[0].cpu().numpy(),
                    robot.data.joint_vel[0].cpu().numpy(),
                    robot.data.root_ang_vel_b[0].cpu().numpy(),
                    robot.data.projected_gravity_b[0].cpu().numpy(),
                    command,
                    desired_arms,
                    upper_observation_scale=upper_observation_scale,
                )
                if args.grasp_probe:
                    target_q[finger_ids] = closed_fingers * closure
                    thumb_closure = (
                        closure
                        if args.transfer_plan and (seconds >= LOWER_END or unload_started_step is not None)
                        else min(closure * 1.5, 1.0)
                    )
                    target_q[finger_ids[4:]] = closed_fingers[4:] * thumb_closure
                    if args.transfer_plan and unload_started_step is not None:
                        unloading = min((step - unload_started_step) / 12.5, 1)
                        finger_ceiling = 0.49 * (1 - unloading) + 0.015 * unloading
                        if args.feedback_release:
                            elapsed = (step - unload_started_step) / 50
                            finger_ceiling += (0.15 - 0.015) * float(np.clip((elapsed - 0.25) / 0.25, 0, 1))
                        robot.write_joint_effort_limit_to_sim(
                            torch.full((1, len(finger_ids)), finger_ceiling, device=robot.device),
                            joint_ids=finger_ids,
                        )
                if step == 0:
                    write_json(out / "initial-observation.json", observation.reshape(10, 91).tolist())
            # Match the vendor provider: PD angle targets can exceed joint limits
            # to request balancing torque. PhysX still enforces actual joint
            # limits and the actuator's torque/velocity limits. Clipping these
            # targets again severely reduced the ankle torques in earlier probes.
            if args.grasp_probe:
                contact_phase = phase
            # New policy path only: original file and success/contact gates remain frozen.
            if args.grasp_probe:
                if "_bc_actor" not in globals():
                    sys.path.insert(0, str(ROOT))
                    from pipeline.simforge.bc.safe_execute import ArmPolicy
                    _bc_actor = ArmPolicy(args.bc_checkpoint, robot, sim, stage, out / "bc-observations", cfg.spawn.usd_path, args.safe_parameters)
                    report["bc_policy"] = dict(checkpoint=args.bc_checkpoint,
                        checkpoint_sha256=digest_file(args.bc_checkpoint), trained_arm_actions=True, K=8, h=2, policy_version="privileged-state BC",
                        vendor_leg_policy_unchanged=True, original_phase_control_retained=True, finger_closure_alignment_gate_added=True,
                        execution_nature="BC + privileged grasp servo + safety filter; not pure BC",
                        observation="50D privileged-state BC; measured cup/palm/joints/fingers/phase/physical clock, zero render products")
                # Track Cup translation only in grasp phases, preserving calibrated
                # palm/grasp-center geometry and the original pregrasp height path.
                from pipeline.simforge.bc.safe_execute import GRASP_PHASES
                safe_goal = desired_palm.cpu().numpy().copy()
                if phase in GRASP_PHASES:
                    safe_goal += cup.data.root_pos_w[0].cpu().numpy() - cup_start
                target_q[_bc_actor.joint_ids] = _bc_actor.safe_targets(
                    robot, step, seconds, phase, cup, safe_goal,
                    rotation.cpu().numpy(), arm_model, args.hand)
                from pipeline.simforge.bc.safe_execute import gate_finger_targets
                target_q[finger_ids]=gate_finger_targets(target_q[finger_ids],phase,_bc_actor.finger_close_allowed)
                report["bc_inference_calls"] = _bc_actor.calls
                if args.bc_smoke:
                    report.update(status="completed", smoke_only=True, bc_inference_calls=_bc_actor.calls)
                    write_json(out / "probe-report.json", report)
                    write_json(out / "smoke-decision.json", dict(completed=True, calls=_bc_actor.calls, step=step))
                    app.close()
                    os._exit(0)
            for _ in range(4):
                robot.set_joint_position_target(
                    torch.tensor(target_q, dtype=torch.float32, device=robot.device).unsqueeze(0)
                )
                robot.write_data_to_sim()
                sim.step(render=False)
                robot.update(1 / 200)
                if cup is not None:
                    cup.update(1 / 200)
                for sensor in grip_sensors.values():
                    sensor.update(1 / 200, force_recompute=True)
                for sensor in cabinet_sensors.values():
                    sensor.update(1 / 200, force_recompute=True)
            root = robot.data.root_pos_w[0].cpu().numpy()
            if args.grasp_probe and contact_ledger.errors:
                raise RuntimeError("Contact audit failed: " + contact_ledger.errors[0])
            if args.human_forward_transfer and step % 50 == 0:
                collision = next(
                    (
                        pair
                        for pair in contact_ledger.pairs.values()
                        if pair["kind"] == "robot_environment"
                        and pair.get("maximum_unexpected_episode_impulse_Ns", 0) > 0.0005
                    ),
                    None,
                )
                if collision:
                    raise RuntimeError("Unexpected robot/environment contact: " + str(collision["actors"]))
            if initial_root is None:
                initial_root = root.copy()
            gravity = robot.data.projected_gravity_b[0].cpu().numpy()
            tilt = float(np.degrees(np.arccos(np.clip(-gravity[2], -1, 1))))
            minimum_height, maximum_tilt = min(minimum_height, float(root[2])), max(maximum_tilt, tilt)
            grasp_state = {}
            if args.grasp_probe:
                forces = {
                    name: sensor.data.force_matrix_w[0].reshape(-1, 3).sum(0).cpu().numpy()
                    for name, sensor in grip_sensors.items()
                }
                thumb = sum(v for k, v in forces.items() if "thumb" in k)
                fingers = sum(v for k, v in forces.items() if "index" in k or "middle" in k)
                a, b = np.linalg.norm(thumb), np.linalg.norm(fingers)
                opposing = bool(a > 0.25 and b > 0.25 and np.dot(thumb, fingers) / (a * b + 1e-8) < -0.1)
                lift = float(cup.data.root_pos_w[0, 2]) - cup_start[2]
                best_lift = max(best_lift, lift)
                cup_q = cup.data.root_quat_w[0].cpu().numpy()
                cup_tilt = math.degrees(
                    math.acos(float(np.clip(1 - 2 * (cup_q[1] ** 2 + cup_q[2] ** 2), -1, 1)))
                )
                stable_grasp_ticks = (
                    stable_grasp_ticks + 1 if lift > 0.10 and opposing and cup_tilt < 20 else 0
                )
                max_stable_grasp_ticks = max(max_stable_grasp_ticks, stable_grasp_ticks)
                stable_carry_ticks = (
                    stable_carry_ticks + 1
                    if stable_grasp_ticks > 0 and np.linalg.norm(root[:2] - initial_root[:2]) > 0.20
                    else 0
                )
                max_stable_carry_ticks = max(max_stable_carry_ticks, stable_carry_ticks)
                grasp_state = dict(
                    phase=phase,
                    cup_lift_m=lift,
                    cup_tilt_deg=cup_tilt,
                    opposing_finger_contact=opposing,
                    finger_forces_N={k: v.tolist() for k, v in forces.items()},
                    hand_net_forces_N={
                        k: s.data.net_forces_w[0].reshape(-1, 3).sum(0).cpu().tolist()
                        for k, s in grip_sensors.items()
                    },
                    finger_joint_position=robot.data.joint_pos[0, finger_ids].cpu().tolist(),
                    palm_position=robot.data.body_pos_w[0, palm_id].cpu().tolist(),
                    torso_pose=robot.data.body_state_w[0, torso_id, :7].cpu().tolist(),
                    idle_palm_position=robot.data.body_pos_w[
                        0, robot.body_names.index("right_hand_palm_link")
                    ]
                    .cpu()
                    .tolist()
                    if args.hand == "left"
                    else None,
                    desired_palm=desired_palm.cpu().tolist(),
                    closure=closure,
                    arm_joint_position=robot.data.joint_pos[0, arm_ids].cpu().tolist(),
                    arm_joint_command=arm_command.tolist(),
                    arm_drive_target=requested.tolist(),
                    arm_gravity_feedforward_Nm=gravity_arm.tolist(),
                    arm_servo_integral_rad=arm_servo_integral.tolist(),
                    arm_ik=solver_report if "solver_report" in locals() else None,
                    motion_time=motion_time,
                    route_state={
                        k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in route_state.items()
                    }
                    if args.transfer_plan and route_state is not None
                    else None,
                    walking_policy_weight=session.weight if args.walking_policy else None,
                    orientation_error_rad=float(rot_error.norm()),
                    cabinet_forces_N={
                        k: s.data.force_matrix_w[0].reshape(-1, 3).sum(0).cpu().tolist()
                        for k, s in cabinet_sensors.items()
                    },
                )
                # Wait for physical readiness before lowering, closing or lifting.
                next_motion_time = motion_time + 1 / 50
                position_error = float(
                    np.linalg.norm(
                        robot.data.body_pos_w[0, palm_id].cpu().numpy() - desired_palm.cpu().numpy()
                    )
                )
                base_error = float(np.linalg.norm(root[:2] - np.array([args.x, -1.56])))
                guards = [
                    (
                        args.pregrasp_seconds if args.ready_grasp else 2.8,
                        position_error < (0.008 if args.ready_grasp else 0.03)
                        and float(rot_error.norm()) < (0.03 if args.ready_grasp else 0.35)
                        and tilt < 10,
                    ),
                    (8.8, position_error < 0.03 and base_error < 0.06 and tilt < 6),
                    (
                        10.8,
                        position_error < 0.02
                        and float(rot_error.norm()) < 0.15
                        and base_error < 0.06
                        and tilt < 8,
                    ),
                    (
                        12.3,
                        position_error < 0.010
                        and float(rot_error.norm()) < 0.20
                        and base_error < 0.08
                        and cup_tilt < 12,
                    ),
                    (13.5, opposing and cup_tilt < 20),
                ]
                if args.carry_probe:
                    guards += (
                        [
                            (16, stable_grasp_ticks >= 10),
                            (19, abs(yaw_error) < 0.15 and cup_tilt < 20),
                            (
                                23,
                                np.linalg.norm(root[:2] - goal) < 0.15
                                and np.linalg.norm(robot.data.root_lin_vel_w[0, :2].cpu().numpy()) < 0.15
                                and opposing,
                            ),
                        ]
                        if args.forward_carry
                        else [
                            (16, stable_grasp_ticks >= 10),
                            (18, np.linalg.norm(root[:2] - goal) < 0.06 and cup_tilt < 20),
                            (20, position_error < 0.02 and cup_tilt < 20),
                            (23, abs(yaw_error) < 0.10 and tilt < 10 and opposing),
                        ]
                    )
                    if args.lateral_carry:
                        guards = [g for g in guards if g[0] < 16] + [
                            (16, stable_grasp_ticks >= 10),
                            (
                                18,
                                np.linalg.norm(root[:2] - np.array([args.x, -1.31])) < 0.07 and cup_tilt < 20,
                            ),
                            (
                                23,
                                np.linalg.norm(root[:2] - goal) < 0.12
                                and np.linalg.norm(robot.data.root_lin_vel_w[0, :2].cpu().numpy()) < 0.15
                                and opposing,
                            ),
                        ]
                if args.transfer_plan:
                    cp = cup.data.root_pos_w[0].cpu().numpy()
                    cup_speed = float(cup.data.root_lin_vel_w[0].norm())
                    desk_contact = contact_ledger.recent_cup_contact(
                        "/World/Environment/Source/SM_AI_vol8_03_desk_01", sim.current_time
                    )
                    desk_contact_evidence = "fresh_native_impulse" if desk_contact else "none"
                    native_cup_sleeping = False
                    if args.sleep_aware_support and unload_started_step is not None:
                        native_cup_sleeping = bool(
                            get_physx_simulation_interface().is_sleeping(
                                omni.usd.get_context().get_stage_id(),
                                PhysicsSchemaTools.sdfPathToInt(Sdf.Path("/World/Environment/Source/cup_V1")),
                            )
                        )
                        pose = cup.data.root_state_w[0, :7].cpu().numpy().copy()
                        if contact_ledger.recent_cup_contact(
                            "/World/Environment/Source/SM_AI_vol8_03_desk_01",
                            sim.current_time,
                            maximum_age=0.025,
                        ):
                            last_native_desk_pose = pose.copy()
                        if not desk_contact and native_cup_sleeping and "last_native_desk_pose" in locals():
                            unchanged = np.linalg.norm(pose[:3] - last_native_desk_pose[:3]) < 0.002 and abs(
                                np.dot(pose[3:], last_native_desk_pose[3:])
                            ) > np.cos(np.radians(1) / 2)
                            if unchanged and contact_ledger.active_cup_contact(
                                "/World/Environment/Source/SM_AI_vol8_03_desk_01"
                            ):
                                desk_contact = True
                                desk_contact_evidence = "native_sleep_active_pair_unchanged_pose"
                                report["sleep_retained_support_samples"] = (
                                    report.get("sleep_retained_support_samples", 0) + 1
                                )
                    desk_support_force = contact_ledger.recent_cup_support_force(
                        "/World/Environment/Source/SM_AI_vol8_03_desk_01", sim.current_time
                    )
                    place_error = cp - place_goal
                    ij = np.floor(
                        (cp[:2] - placement_clearance_map["origin_xy_m"])
                        / placement_clearance_map["resolution_m"]
                    ).astype(int)
                    g = placement_clearance_map["clearance_m"]
                    actual_placement_clearance = (
                        float(g[ij[1] : ij[1] + 2, ij[0] : ij[0] + 2].min())
                        if 0 <= ij[0] < g.shape[1] - 1 and 0 <= ij[1] < g.shape[0] - 1
                        else 0.0
                    )
                    supported_placement = bool(
                        desk_contact
                        and actual_placement_clearance >= 0.053
                        and abs(place_error[2]) < 0.006
                        and cup_tilt < 12
                    )
                    if (
                        seconds >= ABOVE_END
                        and supported_placement
                        and position_error < 0.02
                        and unload_started_step is None
                    ):
                        unload_started_step = step
                        unload_palm = robot.data.body_state_w[0, palm_id, :7].cpu().numpy().copy()
                        report["placement_soft_release"] = dict(
                            start_physics_time=sim.current_time,
                            minimum_finger_torque_Nm=0.015,
                            ramp_seconds=0.25,
                            opened_hand_clearance_lift_m=0.015,
                            clearance_lift_start_seconds=0.25,
                            clearance_lift_duration_seconds=0.20,
                        )
                    desk_supported_ticks = (
                        desk_supported_ticks + 1
                        if supported_placement
                        and cup_speed < 0.05
                        and cup_tilt < 8
                        and not opposing
                        and float(cup.data.root_ang_vel_w[0].norm()) < 0.3
                        else 0
                    )
                    placed = bool(
                        supported_placement
                        and cup_speed < 0.025
                        and np.linalg.norm(robot.data.body_pos_w[0, palm_id].cpu().numpy() - cp) > 0.14
                        and a < 0.25
                        and b < 0.25
                        and float(cup.data.root_ang_vel_w[0].norm()) < 0.15
                    )
                    transfer_settled_ticks = (
                        transfer_settled_ticks + 1 if seconds >= WITHDRAW_END and placed else 0
                    )
                    grasp_state.update(
                        desk_contact=desk_contact,
                        desk_contact_evidence=desk_contact_evidence,
                        native_cup_sleeping=native_cup_sleeping,
                        last_native_desk_pose=last_native_desk_pose.tolist()
                        if "last_native_desk_pose" in locals()
                        else None,
                        desk_support_force_N=desk_support_force,
                        desk_supported_seconds=desk_supported_ticks / 50,
                        supported_placement=supported_placement,
                        placement_error_m=place_error.tolist(),
                        actual_placement_clearance_m=actual_placement_clearance,
                        cup_speed_m_s=cup_speed,
                        placement_settled_seconds=transfer_settled_ticks / 50,
                    )
                    if args.human_forward_transfer:
                        dock_stable = (
                            seconds >= FACE_DESK_END
                            and np.linalg.norm(root[:2] - station) < 0.07
                            and np.linalg.norm(robot.data.root_lin_vel_w[0, :2].cpu().numpy()) < 0.16
                        )
                        dock_stable_ticks = locals().get("dock_stable_ticks", 0) + 1 if dock_stable else 0
                    turn_guards = (
                        [
                            (TURN_BEGIN, np.linalg.norm(root[:2] - withdraw_xy) < 0.07 and opposing),
                            (
                                NAV_BEGIN,
                                abs(yaw_error) < 0.10
                                and np.linalg.norm(root[:2] - withdraw_xy) < 0.55
                                and opposing
                                and cup_tilt < 15,
                            ),
                            (
                                FACE_DESK_END,
                                abs(yaw_error) < 0.10
                                and np.linalg.norm(root[:2] - turn_exit) < turn_exit_tolerance
                                and opposing
                                and cup_tilt < 25,
                            ),
                            (
                                ARM_SETTLE_BEGIN,
                                dock_stable_ticks >= 25 and opposing and cup_tilt < 25,
                            ),
                        ]
                        if args.human_forward_transfer
                        else [(NAV_BEGIN, np.linalg.norm(root[:2] - withdraw_xy) < 0.07 and opposing)]
                    )
                    guards = (
                        [g for g in guards if g[0] < 18]
                        + turn_guards
                        + [
                            (
                                NAV_END,
                                route_state is not None
                                and route_state["endpoint_error"] < 0.10
                                and np.linalg.norm(robot.data.root_lin_vel_w[0, :2].cpu().numpy()) < 0.16
                                and opposing
                                and cup_tilt < 10,
                            ),
                            (
                                PLACE_BEGIN,
                                np.linalg.norm(root[:2] - station) < 0.07
                                and tilt < 10
                                and opposing
                                and cup_tilt < 10
                                and root[2] < 0.70
                                and abs(yaw_error) < 0.10
                                and abs(float(robot.data.root_lin_vel_w[0, 2])) < 0.08,
                            ),
                            (ABOVE_END, position_error < 0.018 and float(rot_error.norm()) < 0.14),
                            (
                                LOWER_END,
                                supported_placement
                                and (
                                    position_error < 0.018
                                    or (args.feedback_release and "clear_withdraw_step" in locals())
                                )
                                and desk_supported_ticks >= 2
                                and closure < 0.01,
                            ),
                            (RELEASE_END, supported_placement and cup_speed < 0.05),
                            (TASK_END, transfer_settled_ticks >= 50),
                        ]
                    )
                    if seconds >= 18:
                        forbidden = [
                            p
                            for p in contact_ledger.pairs.values()
                            if p["kind"] == "cup_environment"
                            and p["phases"].get("navigate_apartment_with_cup", {}).get("max_impulse_Ns", 0)
                            > 0.0005
                        ]
                        if forbidden:
                            raise RuntimeError(
                                "Cup contacted the apartment while navigating: " + str(forbidden[0]["actors"])
                            )
                for boundary, ready in guards:
                    if motion_time < boundary <= next_motion_time and not ready:
                        next_motion_time = boundary - 1e-7
                if args.transfer_plan and PLACE_BEGIN < motion_time < LOWER_END:
                    if position_error > 0.025 or float(rot_error.norm()) > 0.14:
                        next_motion_time = motion_time
                motion_time = next_motion_time
                if args.ready_grasp and args.pregrasp_seconds <= motion_time < 12.3:
                    motion_time = 12.3
                if args.human_forward_transfer and seconds >= TURN_BEGIN:
                    phase_stalled_ticks = (
                        locals().get("phase_stalled_ticks", 0) + 1
                        if motion_time <= previous_motion_time + 1e-9
                        else 0
                    )
                    if (
                        args.route_progress_watchdog
                        and phase == "navigate_apartment_with_cup"
                        and route_state
                    ):
                        remaining = float(route_state["remaining"])
                        best_remaining = locals().get("watchdog_best_remaining", remaining)
                        if remaining < best_remaining - 0.04:
                            phase_stalled_ticks = 0
                            watchdog_best_remaining = remaining
                            report["navigation_watchdog_progress_resets"] = (
                                report.get("navigation_watchdog_progress_resets", 0) + 1
                            )
                        elif "watchdog_best_remaining" not in locals():
                            watchdog_best_remaining = remaining
                    if phase_stalled_ticks > 500:
                        raise RuntimeError("Forward-transfer phase guard stalled for ten seconds: " + phase)
            physics_records.append(
                np.concatenate(
                    [robot.data.body_pos_w[0].cpu().numpy(), robot.data.body_quat_w[0].cpu().numpy()], axis=-1
                )
            )
            joint_records.append(robot.data.joint_pos[0].cpu().numpy().copy())
            motor_records.append(np.asarray(target_q).copy())
            effort_records.append(robot.data.joint_effort_limits[0].cpu().numpy().copy())
            command_records.append(np.asarray(command).copy())
            policy_records.append(observation.reshape(-1).copy() if not args.hold else np.zeros(910))
            action_records.append(action.copy())
            phase_records.append(phase if args.grasp_probe else "locomotion_probe")
            if args.grasp_probe:
                force_records.append(np.stack([forces[k] for k in grip_sensors]))
            if cup is not None:
                cup_records.append(cup.data.root_state_w[0, :7].cpu().numpy().copy())
            if step % 10 == 0 or step < 10:
                metrics.append(
                    dict(
                        time=(step + 1) / 50,
                        root_position=root.tolist(),
                        tilt_deg=tilt,
                        cup_position=cup.data.root_pos_w[0].cpu().tolist() if cup is not None else None,
                        cup_orientation_wxyz=cup.data.root_quat_w[0].cpu().tolist()
                        if cup is not None
                        else None,
                        command=command,
                        raw_leg_action=action.tolist(),
                        feet_positions=robot.data.body_pos_w[0, foot_ids].cpu().tolist(),
                        root_velocity=robot.data.root_lin_vel_w[0].cpu().tolist(),
                        root_orientation_wxyz=robot.data.root_quat_w[0].cpu().tolist(),
                        physics_time=sim.current_time,
                        **grasp_state,
                    )
                )
                write_json(out / "telemetry.json", metrics)
            if not np.isfinite(root).all() or root[2] < 0.5 or tilt > 55:
                raise RuntimeError(
                    f"G1 balance failed at control step {step}, height {root[2]:.3f}, tilt {tilt:.1f}"
                )
            cup_tilt_limit = 60 if args.transfer_plan and unload_started_step is not None else 25
            if args.grasp_probe and cup_tilt > cup_tilt_limit:
                raise RuntimeError(f"Cup tipped to {cup_tilt:.1f} degrees during {phase}")
            if (
                args.transfer_plan
                and unload_started_step is not None
                and float(cup.data.root_pos_w[0, 2]) < place_goal[2] - 0.03
            ):
                raise RuntimeError("Cup dropped below the desktop during release")
            if (
                args.grasp_probe
                and motion_time
                > (
                    TASK_END
                    if args.transfer_plan
                    else 23 + args.carry_hold_seconds
                    if args.carry_probe
                    else 16
                )
                and max_stable_grasp_ticks >= 25
            ):
                break
            if step % 50 == 0:
                if args.grasp_probe:
                    write_json(out / "contact-ledger.json", contact_ledger.export())
                report.update(
                    status="running",
                    steps=step + 1,
                    minimum_base_height_m=minimum_height,
                    maximum_tilt_deg=maximum_tilt,
                )
                if args.grasp_probe:
                    report["grasp_completed"] = bool(max_stable_grasp_ticks >= 25)
                write_json(out / "probe-report.json", report)
                detail = (
                    f", phase {phase}, cup tilt {cup_tilt:.2f}, motion {motion_time:.2f}, command {[round(float(v), 3) for v in command[:3]]}"
                    if args.grasp_probe
                    else ""
                )
                print(f"G1 {step / 50:.1f}s: root {root.tolist()}, tilt {tilt:.2f}" + detail, flush=True)
    report.update(
        status="completed",
        physics_seconds=(step + 1) / 50,
        minimum_base_height_m=minimum_height,
        maximum_tilt_deg=maximum_tilt,
        root_displacement_m=(root - initial_root).tolist(),
        scene=str(target),
        standing_probe_passed=bool(
            not args.hold
            and maximum_tilt < 15
            and max(
                np.linalg.norm(np.asarray(m["root_position"])[:2] - initial_root[:2])
                for m in metrics
                if m["time"] <= args.seconds
            )
            < 0.12
        ),
        base_pose_feedback_enabled=args.position_hold,
        physics_stepping_completed=True,
    )
    if args.grasp_probe:
        report.update(
            grasp_completed=bool(max_stable_grasp_ticks >= 25),
            maximum_cup_lift_m=best_lift,
            longest_stable_contact_lift_seconds=max_stable_grasp_ticks / 50,
            scope="Real contact grasp probe in the full apartment; no completed transfer claim",
            carry_probe_passed=bool(
                args.carry_probe
                and motion_time >= 23 + args.carry_hold_seconds
                and stable_grasp_ticks >= 25
                and maximum_tilt < 15
                and np.linalg.norm(root[:2] - goal) < 0.15
            ),
            carry_contact_passed=bool(args.carry_probe and max_stable_carry_ticks >= 25),
            longest_stable_displaced_carry_seconds=max_stable_carry_ticks / 50,
        )
        if args.transfer_plan:
            navigation_rows = [m for m in metrics if m["phase"] == "navigate_apartment_with_cup"]
            navigation_completed = bool(any(m["motion_time"] >= NAV_END for m in metrics))
            report.update(
                full_transfer_completed=bool(motion_time >= TASK_END and transfer_settled_ticks >= 50),
                navigation_completed=navigation_completed,
                carry_probe_passed=navigation_completed and maximum_tilt < 15,
                scope="Physical apartment pickup, policy locomotion and contact-validated desk release",
                placement_settled_seconds=transfer_settled_ticks / 50,
                final_placement_error_m=place_error.tolist(),
                final_cup_tilt_deg=cup_tilt,
                final_cup_speed_m_s=cup_speed,
                last_phase=phase,
                motion_time=motion_time,
                navigation_maximum_cup_tilt_deg=max(
                    (m["cup_tilt_deg"] for m in navigation_rows), default=None
                ),
                navigation_minimum_obstacle_clearance_m=min(
                    (m["route_state"]["obstacle_clearance_m"] for m in navigation_rows), default=None
                ),
            )
            if not report["full_transfer_completed"]:
                report.update(
                    status="incomplete",
                    termination_reason="Physics duration limit reached before validated release/settle",
                )
except BaseException:
    report.update(status="failed", error=traceback.format_exc(), standing_probe_passed=False)
    print(report["error"], flush=True)
finally:
    if "contact_ledger" in locals():
        contact_audit = contact_ledger.export()
        bad_cup_contacts = []
        if args.transfer_plan:
            for pair in contact_audit["pairs"]:
                if pair["kind"] != "cup_environment":
                    continue
                actor = next(a for a in pair["actors"] if "/cup_V1" not in a)
                for contact_stage, values in pair["phases"].items():
                    allowed = (
                        contact_stage == "setup"
                        or (
                            actor.endswith("/SM_AI_vol8_04_kitchen_furniture_01")
                            and contact_stage
                            in {
                                "settle_open_grasp",
                                "close",
                                "lift",
                                "hold",
                                "lower_open",
                                "extend_above",
                                "raise_clear_of_counter",
                                "walk_with_raised_hand",
                            }
                        )
                        or (
                            actor.endswith("/SM_AI_vol8_03_desk_01")
                            and contact_stage
                            in {"lower_to_desk", "release_on_desk", "withdraw_hand", "verify_placement"}
                        )
                    )
                    if not allowed and values["max_impulse_Ns"] > 0.0005:
                        bad_cup_contacts.append(dict(actor=actor, phase=contact_stage, **values))
        report["unexpected_cup_environment_contacts"] = bad_cup_contacts
        write_json(out / "contact-ledger.json", contact_audit)
        report["unexpected_robot_environment_pairs"] = contact_audit["unexpected_robot_environment_pairs"]
        report["external_contact_check_passed"] = bool(
            contact_audit["pairs"]
            and not contact_audit["callback_errors"]
            and not contact_audit["unexpected_robot_environment_pairs"]
            and not bad_cup_contacts
        )
        if report.get("carry_probe_passed") and not report["external_contact_check_passed"]:
            report["carry_navigation_pose_passed"] = True
            report["carry_probe_passed"] = False
        if report.get("full_transfer_completed") and not report["external_contact_check_passed"]:
            report["full_transfer_pose_completed"] = True
            report["full_transfer_completed"] = False
    if "minimum_height" in locals():
        report.update(minimum_base_height_m=minimum_height, maximum_tilt_deg=maximum_tilt)
    if "best_lift" in locals():
        report.update(
            maximum_cup_lift_m=best_lift, longest_stable_contact_lift_seconds=max_stable_grasp_ticks / 50
        )
    if "physics_records" in locals() and physics_records:
        write_json(
            out / "joint-drives-runtime.json",
            capture_joint_drives(robot, phase if args.grasp_probe else "probe"),
        )
    write_json(out / "probe-report.json", report)
    if "metrics" in locals():
        write_json(out / "telemetry.json", metrics)
    if "physics_records" in locals() and physics_records:
        np.savez_compressed(
            out / "physics-body-states.npz",
            states=np.asarray(physics_records),
            motor_targets_rad=np.asarray(motor_records, dtype=np.float32),
            joint_effort_limits_Nm=np.asarray(effort_records, dtype=np.float32),
            body_commands=np.asarray(command_records, dtype=np.float32),
            policy_observations=np.asarray(policy_records, dtype=np.float32),
            leg_actions=np.asarray(action_records, dtype=np.float32),
            phases=np.asarray(phase_records),
            grip_forces_world_N=np.asarray(force_records, dtype=np.float32),
            grip_sensor_names=np.asarray(list(grip_sensors)),
            body_names=np.asarray(robot.body_names),
            control_hz=50,
            physics_hz=200,
            joints=np.asarray(joint_records),
            joint_names=np.asarray(robot.joint_names),
            cup=np.asarray(cup_records),
        )
    if sim is not None:
        sim.stop()
    app.close()
