"""Audit composed USD physics; do not replace real parameters with name guesses."""

import argparse
import hashlib
import sys
from pathlib import Path

import jsonschema
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.ue_isaac_pipeline.io import digest_file, read_json, write_json  # noqa: E402


def audit(scene, schema, out, plan=None):
    scene = Path(scene).resolve(strict=True)
    stage = Usd.Stage.Open(str(scene))
    if not stage:
        raise ValueError(scene)
    length = UsdGeom.GetStageMetersPerUnit(stage)
    kg = UsdPhysics.GetStageKilogramsPerUnit(stage)
    plan_rows = read_json(plan).get("objects", []) if plan else []
    mapping = {r["prepared_prim_path"]: r for r in plan_rows}
    rows, warnings = [], []
    for prim in stage.Traverse():
        rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        if not (rigid or collision or prim.IsA(UsdGeom.Mesh)):
            continue
        ancestor = prim.GetParent()
        owner = prim if rigid else None
        while ancestor and not owner:
            if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                owner = ancestor
            ancestor = ancestor.GetParent()
        if owner and owner != prim:
            continue  # Listed as a collider of its rigid-body owner.
        colliders = []
        for child in Usd.PrimRange(prim):
            if child != prim and child.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            if not child.HasAPI(UsdPhysics.CollisionAPI):
                continue
            api = UsdPhysics.CollisionAPI(child)
            if not api.GetCollisionEnabledAttr().Get():
                continue
            approximation = (
                UsdPhysics.MeshCollisionAPI(child).GetApproximationAttr().Get()
                if child.HasAPI(UsdPhysics.MeshCollisionAPI)
                else None
            )
            mat, _ = UsdShade.MaterialBindingAPI(child).ComputeBoundMaterial(materialPurpose="physics")
            physics = UsdPhysics.MaterialAPI(mat.GetPrim()) if mat else None
            material = {}
            if physics and physics:
                for name, getter in [
                    ("static_friction", "GetStaticFrictionAttr"),
                    ("dynamic_friction", "GetDynamicFrictionAttr"),
                    ("restitution", "GetRestitutionAttr"),
                ]:
                    value = getattr(physics, getter)().Get()
                    if value is not None:
                        material[name] = float(value)
            colliders.append(
                dict(
                    prim_path=str(child.GetPath()),
                    shape=child.GetTypeName(),
                    approximation=approximation or "schema_default",
                    material_path=str(mat.GetPath()) if mat else None,
                    material_parameters=material,
                )
            )
        path = str(prim.GetPath())
        source = mapping.get(path, {})
        robot = path.startswith("/World/G1/") or path == "/World/G1"
        role = (
            "articulated"
            if robot and rigid
            else "dynamic"
            if rigid
            else "static"
            if colliders
            else "visual_only"
        )
        approximations = {c["approximation"] for c in colliders}
        collider = next(iter(approximations)) if len(approximations) == 1 else "mesh"
        # The supplied V1 enum omits decomposition/primitive/mixed shapes. Preserve
        # the exact per-shape truth in collider_details, without claiming a hull.
        if not colliders:
            collider = "none"
        elif collider not in {"convexHull", "sdf", "boundingCube", "boundingSphere", "mesh"}:
            collider = "mesh"
        row = dict(
            object_id=source.get("object_id", "usd-" + hashlib.sha256(path.encode()).hexdigest()[:20]),
            prim_path=path,
            role=role,
            collider=collider,
            collider_details=colliders,
            reviewed=False,
            material_class=source.get("class_name", "unclassified"),
            parameter_provenance="composed USD values; not measured apartment material pairs",
            preserve_official_robot_properties=robot,
        )
        if rigid:
            body = UsdPhysics.RigidBodyAPI(prim)
            row["kinematic"] = bool(body.GetKinematicEnabledAttr().Get())
            row["rigid_body_enabled"] = bool(body.GetRigidBodyEnabledAttr().Get())
            mass = UsdPhysics.MassAPI(prim)
            m = mass.GetMassAttr().Get() if mass else None
            inertia = mass.GetDiagonalInertiaAttr().Get() if mass else None
            if m is not None and m > 0:
                row["mass_kg"] = float(m) * kg
            else:
                row["mass_status"] = "runtime_computed_requires_native_audit"
            if inertia is not None and all(v > 0 for v in inertia):
                row["inertia_diagonal_kgm2"] = [float(v) * kg * length**2 for v in inertia]
            else:
                row["inertia_status"] = "runtime_computed_from_collision_geometry_requires_native_audit"
            row["mass_source_opinions"] = (
                [str(s.layer.identifier) for s in mass.GetMassAttr().GetPropertyStack()] if mass else []
            )
            if not colliders:
                warnings.append(path + ": rigid body has no enabled collider")
        if colliders:
            materials = [c["material_parameters"] for c in colliders]
            for field in ("static_friction", "dynamic_friction", "restitution"):
                values = {m[field] for m in materials if field in m}
                if len(values) == 1 and all(field in m for m in materials):
                    row[field] = values.pop()
        if path.endswith("/cup_V1"):
            row["mass_calibration"] = (
                "0.25 kg nominal empty ceramic mug estimate authorized by user; not measured"
            )
        rows.append(row)
    result = dict(
        schema_version=1,
        scope=dict(
            scene=str(scene),
            source_inventory_complete=False,
            friction_reference="Authored PhysX nominal defaults/contact assumptions; no CRC or real-pair calibration asserted.",
            notes="Composed USD audit, static masses optional; schema collider=mesh may contain decomposition or primitive shapes detailed per collider. Runtime-computed mass/inertia need native PhysX evidence.",
        ),
        objects=rows,
        stage_units=dict(meters_per_unit=length, kilograms_per_unit=kg),
        scene_sha256=digest_file(scene),
        warnings=warnings,
        status="composed_usd_audited_native_validation_separate",
        sources=[
            "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/rigid_bodies.html",
            "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/collision.html",
        ],
    )
    jsonschema.Draft202012Validator(read_json(schema)).validate(result)
    write_json(out, result)
    return dict(
        objects=len(rows),
        rigid_bodies=sum(r["role"] in ("dynamic", "articulated") for r in rows),
        colliders=sum(len(r["collider_details"]) for r in rows),
        warnings=warnings,
        schema_valid=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--plan")
    print(audit(**vars(p.parse_args())))
