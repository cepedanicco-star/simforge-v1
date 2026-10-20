"""Author one sampled physical material/mass opinion layer; no draws or stepping."""

import json
from pathlib import Path

import numpy as np


def principal_inertia(tensor):
    """Return positive eigenvalues and a proper body-from-principal rotation."""
    matrix = np.asarray(tensor, dtype=float)
    matrix = (matrix + matrix.T) / 2
    values, axes = np.linalg.eigh(matrix)
    if values.min() <= 0 or not np.isfinite(values).all():
        raise ValueError("Cup inertia must be positive definite")
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    return values, axes


def apply(stage, config, sample):
    from pxr import Gf, UsdPhysics, UsdShade, PhysxSchema

    dimension = sample["dimensions"]["D6"]
    if not dimension["enabled"]:
        return dict(enabled=False, source_physics_preserved=True)
    values = dimension["values"]
    source = json.loads(
        (Path(config["source_recording"]) / "native-mass-properties.json").read_text(encoding="utf8")
    )
    tensor = np.asarray(source["cup_inertia_tensor_kgm2"]) * (values["mass_kg"] / source["cup_mass_kg"])
    diagonal, axes = principal_inertia(tensor)
    cup_path = config["dimensions"]["D4"]["target_prim"]
    cup = stage.GetPrimAtPath(cup_path)
    if not cup or not cup.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError("Missing dynamic cup: " + cup_path)
    mass = UsdPhysics.MassAPI.Apply(cup)
    mass.CreateMassAttr(values["mass_kg"])
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, diagonal)))
    rotation = Gf.Matrix3d(*map(float, axes.T.ravel())).ExtractRotation().GetQuat()
    mass.CreatePrincipalAxesAttr(
        Gf.Quatf(float(rotation.GetReal()), Gf.Vec3f(*map(float, rotation.GetImaginary())))
    )
    material = UsdShade.Material.Define(stage, "/World/SimForgeRandomization/ContactMaterial")
    physical = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physical.CreateStaticFrictionAttr(values["friction"])
    physical.CreateDynamicFrictionAttr(values["friction"])
    physical.CreateRestitutionAttr(0)
    PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim()).CreateFrictionCombineModeAttr("average")
    paths = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        hand = path.startswith("/World/G1/left_hand_")
        if prim.HasAPI(UsdPhysics.CollisionAPI) and (
            path == cup_path or path.startswith(cup_path + "/") or hand
        ):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material, materialPurpose="physics", bindingStrength=UsdShade.Tokens.strongerThanDescendants
            )
            paths.append(path)
    if not paths or abs(mass.GetMassAttr().Get() - values["mass_kg"]) > 1e-6:
        raise ValueError("Cup mass/contact override did not author")
    q = mass.GetPrincipalAxesAttr().Get()
    authored_rotation = np.asarray(Gf.Matrix3d(q)).T
    reconstructed = authored_rotation @ np.diag(mass.GetDiagonalInertiaAttr().Get()) @ authored_rotation.T
    if not np.allclose(reconstructed, tensor, rtol=1e-5, atol=1e-9):
        raise ValueError("USD principal inertia does not reconstruct the scaled full tensor")
    return dict(
        enabled=True,
        mass_kg=float(mass.GetMassAttr().Get()),
        friction=float(physical.GetStaticFrictionAttr().Get()),
        full_inertia_tensor_kgm2=tensor.tolist(),
        native_usd_reconstructed_inertia_kgm2=reconstructed.tolist(),
        inertia_scale=values["mass_kg"] / source["cup_mass_kg"],
        bound_colliders=paths,
        scope="Authored/read-back USD physics properties; behavior is not evaluated by this render",
    )
