"""Capture the accepted apartment under the common UE/Isaac material-test fixture."""
# ruff: noqa: E402
import argparse
import json
import math
import os
from pathlib import Path
import sys
import traceback

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", required=True)
parser.add_argument("--ue-acceptance", required=True)
parser.add_argument("--graycard-config")
args = parser.parse_args()
config = json.loads(Path(args.config).read_text(encoding="utf8"))
ue_acceptance = json.loads(Path(args.ue_acceptance).read_text(encoding="utf8"))
if ue_acceptance["status"] != "PASS" or ue_acceptance["passed"] != 6:
    raise ValueError("All six UE pairs must pass before Isaac startup")
out = Path(config["output"]) / "isaac"
out.mkdir(exist_ok=False)
handles = []
sdk = Path(sys.executable).parent
for folder in (sdk / "Library/bin", sdk / "Lib/site-packages/torch/lib"):
    os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")
    handles.append(os.add_dll_directory(str(folder)))
if os.environ.get("AISIM_H5PY_OVERLAY"):
    sys.path.insert(0, os.environ["AISIM_H5PY_OVERLAY"])
    import h5py  # noqa: F401
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.ue_isaac_pipeline.camera_projection import equivalent_native_lens
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.simforge.materials.camera_pairing import angle_degrees, require_camera_pairing
from pipeline.simforge.sensors.geometry import camera_contract
from pipeline.simforge.sensors.export import plain
from isaacsim import SimulationApp

app = SimulationApp(dict(headless=True, width=2048, height=1152, renderer="RaytracedLighting",
                         extra_args=["--portable-root", os.environ["AISIM_PORTABLE_ROOT"]]))


def write(name, value):
    (out / name).write_text(json.dumps(plain(value), indent=2), encoding="utf8")


try:
    import carb
    import numpy as np
    import omni.replicator.core as rep
    import omni.usd
    from PIL import Image
    from pxr import Gf, UsdGeom, UsdLux, UsdShade
    from isaacsim.core.utils.semantics import add_update_semantics, remove_all_semantics

    context = omni.usd.get_context()
    if not context.open_stage(config["source_scene"]):
        raise RuntimeError("Apartment stage could not open")
    stage = context.get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    disabled = []
    for prim in list(stage.Traverse()):
        if "Light" in prim.GetTypeName():
            prim.SetActive(False)
            disabled.append(str(prim.GetPath()))
    for path in ("/World/Robot", "/World/G1", "/World/AIWorkflow/TelemetryGraph"):
        prim = stage.GetPrimAtPath(path)
        if prim:
            prim.SetActive(False)
    actual_lights = []
    for fixture in config["lights"]:
        p, y, _ = np.radians(fixture["ue_rotation_pitch_yaw_roll"])
        direction = Gf.Vec3d(math.cos(p)*math.cos(y), -math.cos(p)*math.sin(y), math.sin(p))
        light = UsdLux.DistantLight.Define(stage, "/World/MaterialFixture/" + fixture["name"])
        light.CreateIntensityAttr(fixture["illuminance_lux"])
        light.CreateAngleAttr(0)
        light.CreateNormalizeAttr(True)
        light.CreateColorAttr(Gf.Vec3f(1))
        UsdLux.ShadowAPI.Apply(light.GetPrim()).CreateShadowEnableAttr(False)
        UsdGeom.Xformable(light).AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0,0,-1),direction).GetQuat()))
        world = UsdGeom.XformCache(0).GetLocalToWorldTransform(light.GetPrim())
        actual = world.TransformDir(Gf.Vec3d(0,0,-1)).GetNormalized()
        # Quatf light orientation has float32 precision; record actual axes too.
        error = angle_degrees(list(direction), list(actual))
        if error > 1e-4:
            raise ValueError("Native fixture direction mismatch: " + str(error))
        actual_lights.append(dict(name=fixture["name"], forward_usd=list(actual), direction_error_degrees=error,
            intensity=light.GetIntensityAttr().Get(), normalize=light.GetNormalizeAttr().Get(),
            shadows=UsdLux.ShadowAPI(light.GetPrim()).GetShadowEnableAttr().Get()))
    settings = carb.settings.get_settings()
    desired = {"/rtx/post/histogram/enabled": False, "/rtx/post/tonemap/op": 6,
               "/rtx/post/tonemap/filmIso": 100.0, "/rtx/post/tonemap/fNumber": 4.0,
               "/rtx/post/tonemap/exposureTime": 1/16, "/rtx/post/dof/enabled": False,
               "/rtx/post/motionblur/enabled": False, "/rtx/post/lensFlares/enabled": False}
    previous = {k: settings.get(k) for k in desired}
    for k, value in desired.items():
        settings.set(k, value)
    remove_all_semantics(stage.GetPrimAtPath("/World"), recursive=True)
    materials = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if not mat:
                continue
            key = str(mat.GetPath())
            if key not in materials:
                materials[key] = dict(label="material_"+str(len(materials)),
                                       source_asset=mat.GetPrim().GetAssetInfo().get("unreal",{}).get("contentPath"),
                                       shader_path=key+"/SurfaceShader", meshes=[])
            materials[key]["meshes"].append(str(prim.GetPath()))
            add_update_semantics(prim, materials[key]["label"])
    write("material-map.json", materials)
    cam = UsdGeom.Camera.Define(stage, "/World/MaterialFixture/Camera")
    cam.CreateClippingRangeAttr(Gf.Vec2f(.01,100))
    cam.CreateFStopAttr(0)
    transform = cam.AddTransformOp()
    product = rep.create.render_product(str(cam.GetPath()), (2048,1152))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    semantic = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize":False})
    camera_params = rep.AnnotatorRegistry.get_annotator("camera_params")
    for annotator in (rgb, semantic, camera_params):
        annotator.attach(product)
    calibration = None
    if args.graycard_config:
        from pipeline.simforge.materials.graycard_runtime import calibrate
        calibration = calibrate(stage, cam, transform, rgb, semantic, camera_params, rep, settings, args.graycard_config)
        desired["/rtx/post/tonemap/exposureTime"] = calibration["locked_exposure_seconds"]
    records = []
    native_pairing = []
    for view in config["views"]:
        lens, scale = equivalent_native_lens(view["lens"])
        m = np.asarray(view["usd_camera_matrix"], dtype=np.float64)
        transform.Set(Gf.Matrix4d(*m.ravel().tolist()))
        for name, value in lens.items():
            getattr(cam, "Get"+name+"Attr")().Set(value)
        for _ in range(8):
            rep.orchestrator.step(rt_subframes=8, delta_time=0, pause_timeline=True)
        native = camera_params.get_data()
        intrinsics = camera_contract(native)
        measured_matrix = np.asarray(UsdGeom.XformCache(0).GetLocalToWorldTransform(cam.GetPrim()))
        if not np.allclose(measured_matrix, view["usd_camera_matrix"], rtol=0, atol=1e-10):
            raise ValueError("Actual USD camera transform changed")
        native_inverse_view = np.linalg.inv(np.asarray(native["cameraViewTransform"]).reshape(4,4))
        view_error = float(np.max(np.abs(native_inverse_view-measured_matrix)))
        if view_error > 1e-5:
            raise ValueError("Native camera view disagrees with actual prim: " + str(view_error))
        hfov = math.degrees(2*math.atan(intrinsics["width"]/(2*intrinsics["fx"])))
        vfov = math.degrees(2*math.atan(intrinsics["height"]/(2*intrinsics["fy"])))
        observed = dict(position_cm=(measured_matrix[3,:3]*[100,-100,100]).tolist(),
            forward=(-measured_matrix[2,:3]*[1,-1,1]).tolist(), up=(measured_matrix[1,:3]*[1,-1,1]).tolist(),
            horizontal_fov_degrees=hfov, vertical_fov_degrees=vfov)
        pairing = require_camera_pairing(view, observed)
        native_pairing.append(dict(check=pairing, actual_observed=observed, actual_usd_camera_matrix=measured_matrix.tolist(),
                                   native_inverse_view_max_error=view_error, intrinsics=intrinsics))
        image = np.asarray(rgb.get_data())[...,:3]
        if image.shape != (1152,2048,3):
            raise ValueError("Wrong RGB dimensions")
        stem = f"cam01-{view['source_frame']:03d}"
        Image.fromarray(image.astype(np.uint8)).save(out / (stem+".png"))
        seg = semantic.get_data()
        np.save(out / (stem+"-material-ids.npy"), np.asarray(seg["data"]).squeeze())
        write(stem+"-material-labels.json", seg["info"]["idToLabels"])
        write(stem+"-camera-native.json", {k: v.tolist() if isinstance(v,np.ndarray) else v for k,v in native.items()})
        records.append(dict(source_frame=view["source_frame"], camera_matrix=view["usd_camera_matrix"],
                             source_lens=view["lens"], native_authored_lens=lens, equivalent_lens_scale=scale,
                             mean_rgb=float(image.mean()), native_K=[[intrinsics["fx"],0,intrinsics["cx"]],[0,intrinsics["fy"],intrinsics["cy"]],[0,0,1]], actual_usd_camera_matrix=measured_matrix.tolist(), native_projection=native["cameraProjection"].tolist()))
        print("MATERIAL ISAAC", len(records), "/6", flush=True)
    write("native-pairing-acceptance.json", dict(status="PASS", passed=len(native_pairing), checks=native_pairing,
                                               shared_function="pipeline.simforge.materials.camera_pairing.camera_pairing"))
    write("capture-report.json", dict(status="PASS", records=records, resolution=[2048,1152],
          disabled_lights=disabled, fixture=config["lights"], actual_lights=actual_lights, graycard_calibration=calibration, settings_before=previous,
          settings_after={k:settings.get(k) for k in desired}, source_saved=False,
          lighting_note="Common white directional fixture, normalized zero-angle USD lights; no measured photometric ground truth"))
    print("MATERIAL ISAAC PASS", flush=True)
except Exception:
    write("failure.json", dict(error=traceback.format_exc()))
    raise
finally:
    app.close()
