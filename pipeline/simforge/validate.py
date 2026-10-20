"""Native, read-only structural validation. The CLI adds the real one-frame check."""
# ruff: noqa: E402
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf8"))
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("Refuse to overwrite a validation report")
    out.parent.mkdir(parents=True, exist_ok=True)
    result = dict(status="running", checks=[], failures=[], physics_stepped=False, source_saved=False)
    app, handles = None, []
    started = time.monotonic()
    try:
        import numpy as np
        for directory in (Path(sys.executable).parent / "Library/bin", Path(sys.executable).parent / "Lib/site-packages/torch/lib"):
            os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
            handles.append(os.add_dll_directory(str(directory)))
        if os.environ.get("AISIM_H5PY_OVERLAY"):
            sys.path.insert(0, os.environ["AISIM_H5PY_OVERLAY"])
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=True, renderer="RaytracedLighting",
                                 extra_args=["--portable-root", os.environ["AISIM_PORTABLE_ROOT"]]))
        import omni.usd
        import omni.timeline
        import omni.physx
        from omni.physx.bindings._physx import PhysxPropertyQueryMode, PhysxPropertyQueryResult
        from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils, PhysicsSchemaTools, Semantics

        def check(number, name, passed, reason, **details):
            row = dict(number=number, name=name, status="PASS" if passed else "FAIL", reason=reason, **details)
            result["checks"].append(row)
            if not passed:
                result["failures"].append(dict(check=number, reason=reason, **details))
            print(f"VALIDATE {number} {row['status']} {name}: {reason}", flush=True)

        source = Path(config["scene"]).resolve(strict=True)
        if not omni.usd.get_context().open_stage(str(source)):
            raise ValueError("Native USD cannot load " + str(source))
        for _ in range(15):
            app.update()
        stage = omni.usd.get_context().get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        omni.timeline.get_timeline_interface().stop()
        dependencies = UsdUtils.ComputeAllDependencies(str(source))[2]
        unresolved = [str(path) for path in dependencies]
        unloaded = [str(p.GetPath()) for p in stage.Traverse() if p.HasAuthoredPayloads() and not p.IsLoaded()]
        check(1, "USD dependencies", not unresolved and not unloaded, "native stage loaded; unresolved references listed",
              unresolved=unresolved, unloaded=unloaded)
        hashes = {layer.realPath: hashlib.sha256(Path(layer.realPath).read_bytes()).hexdigest()
                  for layer in stage.GetUsedLayers() if layer.realPath and Path(layer.realPath).is_file()}
        check(2, "metre units", UsdGeom.GetStageMetersPerUnit(stage) == 1., "metersPerUnit=" + str(UsdGeom.GetStageMetersPerUnit(stage)))
        robot = stage.GetPrimAtPath(config["robot_root"])
        joints = [str(p.GetName()) for p in Usd.PrimRange(robot) if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint)] if robot else []
        hands = [name for name in joints if name.startswith(("left_hand_", "right_hand_"))]
        body = [name for name in joints if name not in hands]
        check(3, "robot joints", len(body) == 29 and len(hands) == 14 and len(set(joints)) == 43,
              f"G1 body={len(body)}/29; Dex3 hands={len(hands)}/14; total={len(joints)}/43", joint_names=joints)
        interact = []
        collision_errors = []
        for item in config["interactive_objects"]:
            prim = stage.GetPrimAtPath(item["path"])
            colliders = [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.CollisionAPI)
                         and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()] if prim else []
            bad = []
            if item["role"] == "dynamic":
                for p in colliders:
                    if p.IsA(UsdGeom.Mesh):
                        approx = UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get()
                        if approx not in ("convexHull", "convexDecomposition", "sdf"):
                            bad.append(dict(path=str(p.GetPath()), approximation=str(approx)))
            row = dict(path=item["path"], role=item["role"], colliders=len(colliders), invalid_dynamic_visual_reuse=bad)
            interact.append(row)
            if not colliders or bad:
                collision_errors.append(row)
        geometry_count = sum(p.IsA(UsdGeom.Gprim) for p in stage.Traverse())
        collider_count = sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in stage.Traverse())
        check(4, "interactive collisions", not collision_errors, "Static triangles allowed; dynamic meshes require cooked convex/SDF geometry",
              objects=interact, invalid=collision_errors, geometry_count=geometry_count, collider_count=collider_count,
              collider_to_geometry_ratio=collider_count / geometry_count if geometry_count else None)
        materials = []
        invalid_materials = []
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.MaterialAPI):
                material = UsdPhysics.MaterialAPI(prim)
                values = [material.GetStaticFrictionAttr().Get(), material.GetDynamicFrictionAttr().Get()]
                row = dict(path=str(prim.GetPath()), static_friction=values[0], dynamic_friction=values[1])
                materials.append(row)
                if any(v is None or not math.isfinite(v) or not .1 <= v <= 1.5 for v in values):
                    invalid_materials.append(row)
        # USD mass=0 means automatic mass, not zero physical mass. Query actual
        # parsed PhysX bodies, including the official robot's four sensor frames.
        omni.physx.get_physx_interface().force_load_physics_from_usd()
        interface = omni.physx.get_physx_property_query_interface()
        masses, query_errors = [], []
        for prim in stage.Traverse():
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI) or not UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get():
                continue
            path = str(prim.GetPath())
            answers, finished = [], []

            def received(answer):
                answers.append(dict(result=int(answer.result), mass_kg=float(answer.mass)))

            interface.query_prim(omni.usd.get_context().get_stage_id(), PhysicsSchemaTools.sdfPathToInt(prim.GetPath()),
                                 query_mode=PhysxPropertyQueryMode.QUERY_RIGID_BODY_WITH_COLLIDERS,
                                 timeout_ms=5000, finished_fn=lambda: finished.append(True), rigid_body_fn=received)
            deadline = time.monotonic() + 6
            while not finished and time.monotonic() < deadline:
                app.update()
            if not answers or any(a["result"] != int(PhysxPropertyQueryResult.VALID) or not math.isfinite(a["mass_kg"]) or a["mass_kg"] <= 0 for a in answers):
                query_errors.append(dict(path=path, answers=answers, callback_finished=bool(finished)))
            masses.append(dict(path=path, authored_mass=UsdPhysics.MassAPI(prim).GetMassAttr().Get(), native_answers=answers))
        check(5, "mass and friction", not query_errors and not invalid_materials and bool(masses),
              "native computed mass >0; material friction in[.1,1.5]", masses=masses, invalid_mass=query_errors,
              material_count=len(materials), invalid_materials=invalid_materials)
        camera = UsdGeom.Camera(stage.GetPrimAtPath(config["camera"]))
        matrix = np.asarray(UsdGeom.XformCache(0).GetLocalToWorldTransform(camera.GetPrim())) if camera else np.zeros((4, 4))
        camera_ok = bool(camera) and np.isfinite(matrix).all() and abs(np.linalg.det(matrix)) > 1e-8
        if camera:
            camera_ok = camera_ok and camera.GetFocalLengthAttr().Get(0) > 0 and camera.GetHorizontalApertureAttr().Get(0) > 0
        check(6, "camera pose/lens", bool(camera_ok), "native rendered K will be verified by CLI one-frame check", camera=config["camera"])
        missing = []
        for item in config["interactive_objects"]:
            prim = stage.GetPrimAtPath(item["path"])
            labels = []
            if prim:
                for schema in prim.GetAppliedSchemas():
                    if schema.startswith("SemanticsAPI:"):
                        api = Semantics.SemanticsAPI(prim, schema.split(":", 1)[1])
                        if api.GetSemanticTypeAttr().Get() == "class":
                            labels.append(api.GetSemanticDataAttr().Get())
            if item["label"] not in labels:
                missing.append(dict(path=item["path"], expected=item["label"], found=labels))
        check(8, "critical semantics", not missing, "cup/desk/robot labels checked on source before collection", missing=missing)
        changed = [p for p, h in hashes.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h]
        if changed:
            raise ValueError("Source files changed: " + str(changed))
        result.update(status="PASS" if not result["failures"] else "FAIL", source_layer_hashes=hashes,
                      source_layers_unchanged=True, scene=str(source), elapsed_s=time.monotonic()-started)
    except Exception as error:
        result.update(status="FAIL", error=repr(error))
        result["failures"].append(dict(error=repr(error), traceback=traceback.format_exc()))
        traceback.print_exc()
    finally:
        out.write_text(json.dumps(result, indent=2), encoding="utf8")
        print("STRUCTURAL RESULT", result["status"], flush=True)
        if app is not None:
            app.close()
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
