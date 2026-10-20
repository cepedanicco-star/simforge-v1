"""OpenUSD inventory, composition, authoring and conservative structural validation.

No Isaac or network import is needed. Input layers are never modified.
"""

import hashlib
import math
from pathlib import Path

from .io import digest_file, read_json, write_json
from .provenance import evidence, is_verified, validate_profile
from .rules import classify


def _pxr():
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    except ImportError as e:
        raise RuntimeError("Install the [usd] extra outside Isaac Sim, or use Isaac Sim's Python") from e
    return Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def open_scene(path):
    _, _, Usd, _, _, _ = _pxr()
    path = Path(path).resolve(strict=True)
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    if not stage or not stage.GetDefaultPrim():
        raise ValueError("Scene needs a defaultPrim containing the scene and material hierarchy")
    return stage


def _meshes(prim):
    _, _, Usd, UsdGeom, _, _ = _pxr()
    return [
        p
        for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
        if p.IsA(UsdGeom.Gprim) and not p.IsA(UsdGeom.PointInstancer)
    ]


def _supported_collider(prim):
    from pxr import UsdGeom

    if prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(prim)
        if not len(mesh.GetPointsAttr().Get() or []) or not len(mesh.GetFaceVertexIndicesAttr().Get() or []):
            return False
    return any(
        prim.IsA(kind)
        for kind in (
            UsdGeom.Mesh,
            UsdGeom.Cube,
            UsdGeom.Sphere,
            UsdGeom.Capsule,
            UsdGeom.Cylinder,
            UsdGeom.Cone,
        )
    )


def inventory(path, manifest=None):
    _, _, Usd, UsdGeom, UsdPhysics, _ = _pxr()
    stage = open_scene(path)
    root = stage.GetDefaultPrim()
    manifest = manifest or {}
    if manifest and manifest.get("schema_version") != 1:
        raise ValueError("Manifest schema_version must be 1")
    entries = manifest.get("objects", [])
    if not isinstance(entries, list):
        raise ValueError("Manifest objects must be an array")
    complete = manifest.get("scope", {}).get("source_inventory_complete", False)
    if type(complete) is not bool:
        raise ValueError("source_inventory_complete must be boolean")
    objects = []
    issues = []
    candidates = {}
    for entry in entries:
        if not isinstance(entry, dict) or type(entry.get("reviewed", False)) is not bool:
            raise ValueError("Manifest objects need boolean reviewed flags")
        src_path = entry.get("prim_path") or entry.get("usd_prim_path")
        if not src_path or src_path in candidates:
            raise ValueError("Manifest needs unique prim_path entries")
        p = stage.GetPrimAtPath(src_path)
        if not p or not p.GetPath().HasPrefix(root.GetPath()):
            raise ValueError(f"Manifest prim not inside defaultPrim: {src_path}")
        if not _meshes(p):
            raise ValueError(f"Manifest object contains no supported geometry: {src_path}")
        candidates[src_path] = entry
    for a in candidates:
        for b in candidates:
            if a != b and stage.GetPrimAtPath(b).GetPath().HasPrefix(stage.GetPrimAtPath(a).GetPath()):
                raise ValueError(f"Overlapping manifest object roots: {a}, {b}")
    scope = (manifest or {}).get("scope", {})
    if not scope.get("source_inventory_complete", False):
        issues.append("source_scope_unverified: USD traversal cannot prove unloaded UE cells were exported")
    for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if p.IsA(UsdGeom.PointInstancer):
            count = len(UsdGeom.PointInstancer(p).GetProtoIndicesAttr().Get() or [])
            issues.append(f"point_instancer_requires_bake:{p.GetPath()}:{count}")
        if p.IsA(UsdGeom.Gprim) and not p.IsA(UsdGeom.PointInstancer):
            if any(p.GetPath().HasPrefix(stage.GetPrimAtPath(c).GetPath()) for c in candidates):
                continue
            parent = p.GetParent()
            owner = parent if parent.IsA(UsdGeom.Xform) and parent != root else p
            # Ascend to an authored rigid/articulation root if this is already a simulation asset.
            walk = owner
            while walk and walk != root.GetParent():
                if walk.HasAPI(UsdPhysics.RigidBodyAPI) or walk.HasAPI(UsdPhysics.ArticulationRootAPI):
                    owner = walk
                if walk == root:
                    break
                walk = walk.GetParent()
            key = str(owner.GetPath())
            if not any(owner.GetPath().HasPrefix(stage.GetPrimAtPath(c).GetPath()) for c in candidates):
                # Do not swallow explicit roots with an inferred parent.
                if any(stage.GetPrimAtPath(c).GetPath().HasPrefix(owner.GetPath()) for c in candidates):
                    key = str(p.GetPath())
                candidates.setdefault(key, {})
    ids = set()
    for src_path, entry in sorted(candidates.items()):
        p = stage.GetPrimAtPath(src_path)
        # Explicit parents win; this removes inferred child entries of source rigid roots.
        if any(
            src_path != other and p.GetPath().HasPrefix(stage.GetPrimAtPath(other).GetPath())
            for other in candidates
        ):
            continue
        ident = entry.get("object_id") or p.GetCustomDataByKey("aisim:object_id")
        if not ident:
            ident = "usd-" + hashlib.sha256(src_path.encode()).hexdigest()[:16]
        if not isinstance(ident, str) or ident in ids:
            raise ValueError(f"Invalid or duplicate object_id: {ident}")
        ids.add(ident)
        explicit = None
        if entry.get("class_name"):
            explicit = entry
        elif p.GetCustomDataByKey("aisim:class"):
            explicit = dict(
                class_name=p.GetCustomDataByKey("aisim:class"),
                role=p.GetCustomDataByKey("aisim:role") or "static",
                source="usd_metadata",
                reviewed=bool(p.GetCustomDataByKey("aisim:reviewed")),
            )
        result = classify(src_path + " " + entry.get("asset_path", ""), explicit)
        body_prims = [
            x for x in Usd.PrimRange(p, Usd.TraverseInstanceProxies()) if x.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        rigid = [
            str(x.GetPath()) for x in body_prims if UsdPhysics.RigidBodyAPI(x).GetRigidBodyEnabledAttr().Get()
        ]
        articulation = any(
            x.HasAPI(UsdPhysics.ArticulationRootAPI) for x in Usd.PrimRange(p, Usd.TraverseInstanceProxies())
        )
        if body_prims or articulation:
            result["role"] = "preserve"
        objects.append(
            dict(
                object_id=ident,
                prim_path=src_path,
                **result,
                source_entry=entry,
                geometry_count=len(_meshes(p)),
                authored_rigid_roots=rigid,
                authored_physics=_authored_physics(p),
                authored_articulation=articulation,
                instance=p.IsInstance(),
            )
        )
    if not objects:
        issues.append("no_mesh_objects_found")
    from pxr import UsdUtils

    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(Path(path).resolve()))
    dependency_paths = set(str(layer.realPath) for layer in layers if layer.realPath)
    dependency_paths.update(str(asset) for asset in assets)
    dependencies = {}
    for dep in sorted(dependency_paths):
        local = Path(dep)
        if local.is_file():
            dependencies[str(local.resolve())] = digest_file(local)
        else:
            issues.append(f"nonlocal_dependency_not_hashed:{dep}")
    issues.extend(f"unresolved_source_dependency:{dep}" for dep in unresolved)
    return dict(
        schema_version=1,
        source_path=str(Path(path).resolve()),
        source_sha256=digest_file(path),
        source_dependencies=dependencies,
        default_prim=str(root.GetPath()),
        meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage),
        up_axis=str(UsdGeom.GetStageUpAxis(stage)),
        scope=scope,
        objects=objects,
        issues=issues,
    )


def _authored_physics(prim):
    """Capture actual source opinions per prim/material without averaging different contact surfaces."""
    _, _, Usd, _, UsdPhysics, UsdShade = _pxr()
    records = []
    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        values = {}
        if p.HasAPI(UsdPhysics.MassAPI):
            mass = UsdPhysics.MassAPI(p)
            for key, attr in (
                ("mass", mass.GetMassAttr()),
                ("density", mass.GetDensityAttr()),
                ("diagonalInertia", mass.GetDiagonalInertiaAttr()),
            ):
                if attr.HasAuthoredValueOpinion():
                    value = attr.Get()
                    values[key] = list(value) if hasattr(value, "__len__") else value
        if p.HasAPI(UsdPhysics.CollisionAPI):
            material, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial("physics")
            if material and material.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
                api = UsdPhysics.MaterialAPI(material.GetPrim())
                values["material_path"] = str(material.GetPath())
                for name, attr in (
                    ("static_friction", api.GetStaticFrictionAttr()),
                    ("dynamic_friction", api.GetDynamicFrictionAttr()),
                    ("restitution", api.GetRestitutionAttr()),
                ):
                    if attr.HasAuthoredValueOpinion():
                        values[name] = attr.Get()
        if values:
            records.append(
                dict(prim_path=str(p.GetPath()), values=values, method="source_authored", measured=False)
            )
    return records


def build_plan(inv, registry):
    profiles = registry.get("profiles", {})
    for profile in profiles.values():
        validate_profile(profile)
    plan = dict(
        schema_version=1,
        source_path=inv["source_path"],
        source_sha256=inv["source_sha256"],
        source_dependencies=inv["source_dependencies"],
        objects=[],
        issues=list(inv["issues"]),
        runtime_validation="not_run",
    )
    for obj in inv["objects"]:
        item = dict(obj)
        entry = obj["source_entry"]
        profile_id = entry.get("physics_profile") or registry.get("default_preview_profile")
        profile = entry.get("physics") or profiles.get(profile_id, {})
        if profile:
            validate_profile(profile)
        item["parameters"] = profile
        item["profile_id"] = profile_id
        item["parameter_review_required"] = any(not is_verified(v) for v in profile.values())
        if obj["review_required"]:
            plan["issues"].append(f"semantic_review:{obj['object_id']}")
        if obj["class_name"] == "unknown":
            plan["issues"].append(f"unknown_object:{obj['object_id']}")
        if obj["role"] == "articulated":
            plan["issues"].append(f"joint_topology_required:{obj['object_id']}")
        if obj["role"] == "preserve":
            src = entry.get("authored_physics_evidence", {})
            trusted = src.get("method") in {"official_asset", "manufacturer", "measured"}
            if not trusted or not src.get("source_url") or not src.get("applies_to"):
                plan["issues"].append(f"authored_physics_source_unverified:{obj['object_id']}")
            # Existing physics is left intact, irrespective of default preview profiles.
            item["parameters"] = {}
        elif obj["role"] != "visual_only":
            if not profile:
                plan["issues"].append(f"physical_material_missing:{obj['object_id']}")
            if item["parameter_review_required"]:
                plan["issues"].append(f"provisional_parameters:{obj['object_id']}")
            if obj["role"] == "dynamic" and "mass_kg" not in profile:
                plan["issues"].append(f"dynamic_mass_missing:{obj['object_id']}")
        plan["objects"].append(item)
    plan["issues"] = sorted(set(plan["issues"]))
    plan["data_verified"] = not plan["issues"]
    plan["training_ready"] = False
    return plan


def _map_path(src, default_prim):
    if src == default_prim:
        return "/World/Environment/Source"
    if not src.startswith(default_prim + "/"):
        raise ValueError(f"Prim is outside source defaultPrim: {src}")
    return "/World/Environment/Source" + src[len(default_prim) :]


def _labels(prim, label):
    try:
        from pxr import UsdSemantics

        UsdSemantics.LabelsAPI.Apply(prim, "class").CreateLabelsAttr([label])
    except ImportError:
        # Compatibility with older Isaac USD builds. Never pretend an arbitrary custom attr is a label.
        from pxr import Semantics

        api = Semantics.SemanticsAPI.Apply(prim, "Semantics")
        api.CreateSemanticTypeAttr().Set("class")
        api.CreateSemanticDataAttr().Set(label)


def prepare(source, output, *, manifest=None, registry=None, allow_provisional=False):
    Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade = _pxr()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = output / "prepared.usda"
    if target.exists():
        raise FileExistsError("Use a new output directory; existing prepared scene will not be overwritten")
    manifest_data = read_json(manifest) if manifest else {}
    registry_data = read_json(registry) if registry else {}
    inv = inventory(source, manifest_data)
    plan = build_plan(inv, registry_data)
    plan["config_inputs"] = dict(
        manifest_sha256=digest_file(manifest) if manifest else None,
        registry_sha256=digest_file(registry) if registry else None,
    )
    write_json(output / "inventory.json", inv)
    write_json(output / "plan.json", plan)
    if plan["issues"] and not allow_provisional:
        raise ValueError(
            "Strict preparation blocked; inspect plan.json. Use a preview preset or resolve evidence"
        )
    if any(x.startswith("dynamic_mass_missing:") for x in plan["issues"]):
        raise ValueError("No mass for dynamic object; supply applicable parameter evidence before authoring")
    src = open_scene(source)
    if (
        not math.isfinite(inv["meters_per_unit"])
        or inv["meters_per_unit"] <= 0
        or inv["up_axis"] not in ("Y", "Z")
    ):
        raise ValueError("Invalid USD units/up-axis")
    has_physics = any(
        p.HasAPI(UsdPhysics.RigidBodyAPI) or p.HasAPI(UsdPhysics.CollisionAPI) for p in src.Traverse()
    )
    if has_physics and (abs(inv["meters_per_unit"] - 1) > 1e-8 or inv["up_axis"] != "Z"):
        raise ValueError(
            "Authored physics with non-meter/Z-up units requires a physics-aware asset conversion first"
        )
    kg = UsdPhysics.GetStageKilogramsPerUnit(src)
    if abs(kg - 1) > 1e-8:
        raise ValueError("V1 requires kilogramsPerUnit=1; convert authored masses before preparing")
    stage = Usd.Stage.CreateNew(str(target))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    env = UsdGeom.Xform.Define(stage, "/World/Environment")
    import os

    relative_source = Path(os.path.relpath(Path(source).resolve(), output)).as_posix()
    reference = UsdGeom.Xform.Define(stage, "/World/Environment/Source")
    reference.GetPrim().GetReferences().AddReference(relative_source)
    x = UsdGeom.XformCommonAPI(env)
    if abs(inv["meters_per_unit"] - 1) > 1e-8:
        x.SetScale(Gf.Vec3f(inv["meters_per_unit"]))
    if inv["up_axis"] == "Y":
        x.SetRotate(Gf.Vec3f(90, 0, 0))
    # Ignore imported simulation contexts; there is exactly one managed PhysicsScene.
    for p in list(stage.Traverse()):
        if p.IsA(UsdPhysics.Scene) and str(p.GetPath()).startswith("/World/Environment/"):
            p.SetActive(False)
    physics = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics.CreateGravityMagnitudeAttr(9.80665)
    # Standard gravity is a nominal reference, not a local gravity measurement.
    physics.GetPrim().SetCustomDataByKey(
        "aisim:gravity_source", "https://www.nist.gov/system/files/documents/srd/jpcrd574.pdf"
    )
    for idx, obj in enumerate(plan["objects"]):
        mapped = _map_path(obj["prim_path"], inv["default_prim"])
        obj["prepared_prim_path"] = mapped
        p = stage.GetPrimAtPath(mapped)
        if not p:
            raise ValueError(f"Composition did not map {obj['prim_path']}")
        # Author the instance root first; child instance proxies cannot be edited directly.
        if p.IsInstance():
            p.SetInstanceable(False)
            p = stage.GetPrimAtPath(mapped)
        p.SetCustomDataByKey("aisim:object_id", obj["object_id"])
        p.SetCustomDataByKey("aisim:class_source", obj["class_source"])
        _labels(p, obj["class_name"])
        role = obj["role"]
        if role in ("preserve", "visual_only"):
            continue
        if role == "dynamic":
            if "mass_kg" not in obj["parameters"]:
                raise ValueError(f"No mass for dynamic object {obj['object_id']}; cannot simulate it safely")
            if any(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in Usd.PrimRange(p) if prim != p):
                raise ValueError("Refusing to author nested rigid bodies")
            UsdPhysics.RigidBodyAPI.Apply(p).CreateRigidBodyEnabledAttr(True)
            UsdPhysics.MassAPI.Apply(p).CreateMassAttr(obj["parameters"]["mass_kg"]["value"])
        geoms = _meshes(p)
        for mesh in geoms:
            if not _supported_collider(mesh):
                plan["issues"].append(f"unsupported_collision_geometry:{mesh.GetPath()}")
                continue
            if mesh.IsInstanceProxy():
                plan["issues"].append(f"nested_instance_requires_bake:{mesh.GetPath()}")
                continue
            UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr(True)
            if mesh.IsA(UsdGeom.Mesh):
                approximation = "convexDecomposition" if role == "dynamic" else "none"
                UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr(approximation)
        if role == "articulated":
            p.SetCustomDataByKey("aisim:interaction_status", "static_until_joint_topology_reviewed")
        params = obj["parameters"]
        if params:
            mat = UsdShade.Material.Define(stage, f"/World/AIWorkflow/PhysicsMaterials/M{idx:05d}")
            api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
            api.CreateStaticFrictionAttr(params["static_friction"]["value"])
            api.CreateDynamicFrictionAttr(params["dynamic_friction"]["value"])
            api.CreateRestitutionAttr(params["restitution"]["value"])
            for geom in geoms:
                if geom.IsInstanceProxy():
                    continue
                # Existing physical material bindings take precedence; visual material is untouched.
                existing, _ = UsdShade.MaterialBindingAPI(geom).ComputeBoundMaterial("physics")
                if existing and existing.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
                    plan["issues"].append(f"existing_material_preserved:{geom.GetPath()}")
                else:
                    UsdShade.MaterialBindingAPI.Apply(geom).Bind(mat, materialPurpose="physics")
    stage.GetRootLayer().Save()
    plan["issues"] = sorted(set(plan["issues"]))
    plan["data_verified"] = not plan["issues"]
    plan["prepared_path"] = str(target)
    plan["prepared_sha256"] = digest_file(target)
    report = validate_scene(target)
    for warning in report["warnings"]:
        plan["issues"].append("structural_warning:" + warning)
    plan["issues"] = sorted(set(plan["issues"]))
    plan["data_verified"] = not plan["issues"] and report["structural_passed"]
    write_json(output / "plan.json", plan)
    report.update(
        data_verified=plan["data_verified"],
        training_ready=False,
        runtime_validation="not_run",
        parameter_policy="provisional_allowed" if allow_provisional else "strict",
        review_items=plan["issues"],
    )
    write_json(output / "validation.json", report)
    return dict(scene=str(target), plan=str(output / "plan.json"), validation=report)


def validate_scene(path):
    _, _, _, UsdGeom, UsdPhysics, UsdShade = _pxr()
    from pxr import UsdUtils

    stage = open_scene(path)
    errors = []
    warnings = []
    _, _, unresolved = UsdUtils.ComputeAllDependencies(str(Path(path).resolve()))
    for dep in unresolved:
        errors.append(f"unresolved_asset:{dep}")
    if UsdGeom.GetStageUpAxis(stage) != "Z" or abs(UsdGeom.GetStageMetersPerUnit(stage) - 1) > 1e-8:
        errors.append("stage_must_be_z_up_meters")
    scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if len(scenes) != 1:
        errors.append(f"expected_one_physics_scene:found_{len(scenes)}")
    mesh_count = 0
    collider_count = 0
    body_count = 0
    empty_mesh_count = 0
    for p in stage.Traverse():
        for rel in p.GetRelationships():
            for target in rel.GetTargets():
                if not stage.GetPrimAtPath(target.GetPrimPath()):
                    errors.append(f"unresolved_relationship:{p.GetPath()}:{rel.GetName()}:{target}")
        if p.IsA(UsdGeom.Gprim):
            mesh_count += 1
            if p.IsA(UsdGeom.Mesh) and not len(UsdGeom.Mesh(p).GetPointsAttr().Get() or []):
                empty_mesh_count += 1
            if p.HasAPI(UsdPhysics.CollisionAPI):
                collider_count += 1
                if not _supported_collider(p):
                    errors.append(f"unsupported_collider_type:{p.GetPath()}")
                mat, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial("physics")
                if not mat or not mat.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
                    warnings.append(f"no_explicit_physics_material:{p.GetPath()}")
        if p.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get():
            body_count += 1
            colliders = [c for c in _meshes(p) if c.HasAPI(UsdPhysics.CollisionAPI)]
            if not colliders:
                errors.append(f"rigid_body_without_collider:{p.GetPath()}")
            mass = UsdPhysics.MassAPI(p).GetMassAttr()
            if mass and mass.HasAuthoredValueOpinion() and (not math.isfinite(mass.Get()) or mass.Get() < 0):
                errors.append(f"invalid_mass:{p.GetPath()}")
            ancestor = p.GetParent()
            while ancestor and not ancestor.IsPseudoRoot():
                if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                    errors.append(f"nested_rigid_body:{p.GetPath()}")
                    break
                ancestor = ancestor.GetParent()
        if p.HasAPI(UsdPhysics.MaterialAPI):
            api = UsdPhysics.MaterialAPI(p)
            for key, attr in (
                ("static_friction", api.GetStaticFrictionAttr()),
                ("dynamic_friction", api.GetDynamicFrictionAttr()),
                ("restitution", api.GetRestitutionAttr()),
            ):
                value = attr.Get()
                if value is not None and (
                    not math.isfinite(value) or value < 0 or (key == "restitution" and value > 1)
                ):
                    errors.append(f"invalid_physics_material:{p.GetPath()}:{key}")
    if not mesh_count or not collider_count:
        errors.append("scene_needs_geometry_and_colliders")
    return dict(
        schema_version=1,
        structural_passed=not errors,
        errors=sorted(set(errors)),
        warnings=sorted(set(warnings)),
        geometry_count=mesh_count,
        empty_mesh_count=empty_mesh_count,
        collider_count=collider_count,
        rigid_body_count=body_count,
        runtime_validation="not_run",
        training_ready=False,
    )


def make_demo(output):
    """Original, synthetic test scene; its numbers are deliberately NOT claimed as measured data."""
    Gf, _, Usd, UsdGeom, _, _ = _pxr()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene = output / "demo.usda"
    if scene.exists():
        raise FileExistsError("Demo already exists; choose a new output directory")
    stage = Usd.Stage.CreateNew(str(scene))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    specs = [
        ("Floor", [0, 0, -0.1], [7, 7, 0.1], "floor", "static"),
        ("Box", [2, 0, 1.2], [0.2, 0.2, 0.2], "box", "dynamic"),
        ("Wall", [0, 7, 1.5], [7, 0.1, 1.5], "wall", "static"),
    ]
    entries = []
    for name, position, scale, label, role in specs:
        root = UsdGeom.Xform.Define(stage, "/World/" + name)
        UsdGeom.XformCommonAPI(root).SetTranslate(Gf.Vec3d(*position))
        shape = UsdGeom.Cube.Define(stage, f"/World/{name}/Geometry")
        shape.CreateSizeAttr(2)
        shape.AddScaleOp().Set(Gf.Vec3f(*scale))
        shape.CreateDisplayColorAttr([Gf.Vec3f(0.4, 0.55, 0.7)])
        entries.append(
            dict(
                object_id="demo-" + name.lower(),
                prim_path="/World/" + name,
                class_name=label,
                role=role,
                reviewed=True,
                source="synthetic_demo",
            )
        )
    stage.GetRootLayer().Save()
    manifest = dict(
        schema_version=1,
        scope=dict(source_inventory_complete=True, description="synthetic fixture"),
        objects=entries,
    )
    note = "Synthetic preview value, not a measurement and not a universal material constant"
    src = "https://openusd.org/release/api/class_usd_physics_material_a_p_i.html"
    profile = {
        k: evidence(v, method="example", source_url=src, note=note, unit=unit)
        for k, v, unit in [
            ("static_friction", 0.6, "1"),
            ("dynamic_friction", 0.5, "1"),
            ("restitution", 0.0, "1"),
            ("mass_kg", 1.0, "kg"),
        ]
    }
    write_json(output / "manifest.json", manifest)
    write_json(
        output / "profiles.json",
        dict(schema_version=1, default_preview_profile="synthetic", profiles={"synthetic": profile}),
    )
    return dict(
        source=str(scene), manifest=str(output / "manifest.json"), registry=str(output / "profiles.json")
    )
