"""Create local task assets with a spawner-safe rigid Xform / Mesh hierarchy."""

import os
from pathlib import Path


def extract_cup(source, target, source_prim="/World/Environment/Source/cup_V1"):
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    source, target = Path(source).resolve(strict=True), Path(target).resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(target))
    root = UsdGeom.Xform.Define(stage, "/Cup").GetPrim()
    geometry = stage.DefinePrim("/Cup/Geometry")
    geometry.GetReferences().AddReference(
        Path(os.path.relpath(source, target.parent)).as_posix(), source_prim
    )
    if geometry.GetTypeName() != "Mesh":
        raise ValueError("Expected the exact cup mesh, not an empty placeholder")
    geometry.SetInstanceable(False)
    geometry.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray).Set([])
    mass = UsdPhysics.MassAPI(geometry).GetMassAttr().Get()
    if mass is None or mass <= 0:
        raise ValueError("Prepared cup needs an explicit positive mass")
    # Isaac Lab spawns files below an Xform. A Mesh as the file root loses its
    # geometry type under that stronger Xform opinion, and silently has no shape.
    geometry.RemoveAPI(UsdPhysics.RigidBodyAPI)
    geometry.RemoveAPI(UsdPhysics.MassAPI)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(mass)
    root.SetCustomDataByKey("aisim:mass_source", source.as_posix() + "#" + source_prim)
    material = UsdShade.Material.Define(stage, "/Cup/TrainingPhysicsMaterial")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(0.5)
    physics.CreateDynamicFrictionAttr(0.5)
    physics.CreateRestitutionAttr(0)
    UsdShade.MaterialBindingAPI.Apply(geometry).Bind(material, materialPurpose="physics")
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    stage.GetRootLayer().Save()
    bounds = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_]).ComputeWorldBound(root).ComputeAlignedRange()
    return dict(
        asset=str(target),
        root=str(root.GetPath()),
        geometry=str(geometry.GetPath()),
        bounds_min=list(bounds.GetMin()),
        bounds_max=list(bounds.GetMax()),
    )
