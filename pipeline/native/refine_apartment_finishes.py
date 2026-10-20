"""Refine inspected apartment finishes and author real MDL window transmission.

Appearance values are declared finish choices, not measured material parameters.
The original geometry, collision and friction properties are preserved.
"""

import argparse
import os
import sys
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.ue_isaac_pipeline.io import read_json, write_json  # noqa: E402


def finish(name):
    n = name.lower()
    if "glass_10" in n:
        return "clear_window_glass", None, None
    if "concrete_01" in n:
        return "satin_floor", 0.035, (0.36, 0.52)
    if "ceiling_wall" in n:
        return "smooth_painted_wall", 0.012, (0.56, 0.70)
    if "wood_american_walnut_02" in n:
        return None  # Previously accepted computer desk finish.
    if "kitchen_props_001" in n:
        return "utensil_and_ceramic_atlas", 0.035, (0.24, 0.42)
    if "sink_v1" in n:
        return "sink_basin", 0.025, (0.23, 0.36)
    if any(x in n for x in ("ceramic", "porcelain")):
        return "glazed_ceramic", 0.025, (0.15, 0.27)
    if (
        any(x in n for x in ("wooden_chair", "wood_black_ash", "wood_black_koto", "desktop_chair_01"))
        or n == "mi_ai_vol8_04_window"
    ):
        return "satin_wood_furniture", 0.035, (0.37, 0.55)
    if "steel_pan" in n:
        return "steel_cookware", 0.025, (0.24, 0.40)
    if any(x in n for x in ("chrome", "metal", "aluminium", "steel", "towelhooks", "doorhandle", "pipes")):
        return "metal_finish", 0.025, (0.20, 0.38)
    return None


def refine(scene, audit, out, sky=None):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    target = out / "Apartment_materials.usda"
    if target.exists():
        raise FileExistsError(target)
    layer = Sdf.Layer.CreateNew(str(target))
    layer.subLayerPaths = [Path(os.path.relpath(Path(scene).resolve(), out)).as_posix()]
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(875)
    stage.SetFramesPerSecond(25)
    stage.SetTimeCodesPerSecond(25)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    records = []
    for row in read_json(audit)["materials"]:
        setting = finish(row["name"])
        record = dict(name=row["name"], prim=row["path"], changes=[], calibrated=False)
        records.append(record)
        if setting is None or not row["surface_shader"]:
            continue
        label, strength, bounds = setting
        surface = UsdShade.Shader(stage.GetPrimAtPath(row["surface_shader"]))
        record["finish_choice"] = label
        if label == "clear_window_glass":
            material = UsdShade.Material(stage.GetPrimAtPath(row["path"]))
            glass = UsdShade.Shader.Define(stage, row["path"] + "/ClearWindowGlass")
            glass.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
            glass.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
            glass.CreateOutput("out", Sdf.ValueTypeNames.Token)
            for name, kind, value in [
                ("glass_color", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(0.99, 0.995, 1)),
                ("glass_ior", Sdf.ValueTypeNames.Float, 1.52),
                ("thin_walled", Sdf.ValueTypeNames.Bool, True),
                ("frosting_roughness", Sdf.ValueTypeNames.Float, 0.015),
            ]:
                glass.CreateInput(name, kind).Set(value)
            material.CreateSurfaceOutput("mdl").ConnectToSource(glass.ConnectableAPI(), "out")
            record["changes"].append("MDL OmniGlass thin-wall transmission; PreviewSurface fallback retained")
            continue
        normal = surface.GetInput("normal")
        source = normal.GetConnectedSource() if normal else None
        if source:
            texture = UsdShade.Shader(source[0].GetPrim())
            if texture.GetIdAttr().Get() == "UsdUVTexture":
                old_scale = texture.GetInput("scale").Get() or Gf.Vec4f(2, 2, 2, 1)
                old_bias = texture.GetInput("bias").Get() or Gf.Vec4f(-1, -1, -1, 0)
                sx, sy = (1 if old_scale[i] >= 0 else -1 for i in (0, 1))
                scale = Gf.Vec4f(2 * strength * sx, 2 * strength * sy, 2 * strength, 1)
                bias = Gf.Vec4f(-strength * sx, -strength * sy, 1 - 2 * strength, 0)
                texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(scale)
                texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(bias)
                texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
                record.update(
                    normal_scale_before=list(old_scale),
                    normal_bias_before=list(old_bias),
                    normal_scale_after=list(scale),
                    normal_bias_after=list(bias),
                )
                record["changes"].append("blend decoded tangent normal toward neutral, including Z")
        elif normal:
            normal.Set(Gf.Vec3f(0, 0, 1))
        roughness = surface.GetInput("roughness")
        source = roughness.GetConnectedSource() if roughness else None
        lo, hi = bounds
        if source:
            texture = UsdShade.Shader(source[0].GetPrim())
            if texture.GetIdAttr().Get() == "UsdUVTexture":
                texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
                    Gf.Vec4f(hi - lo, hi - lo, hi - lo, 1)
                )
                texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(lo, lo, lo, 0))
                texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
        else:
            surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set((lo + hi) / 2)
        record["roughness_interval"] = [lo, hi]
        record["changes"].append("bound roughness variation to declared finish interval")
        if label == "sink_basin":
            metallic = surface.CreateInput("metallic", Sdf.ValueTypeNames.Float)
            metallic.DisconnectSource()
            metallic.Set(1.0)
            base = surface.GetInput("diffuseColor")
            connected = base.GetConnectedSource() if base else None
            if connected:
                texture = UsdShade.Shader(connected[0].GetPrim())
                if texture.GetIdAttr().Get() == "UsdUVTexture":
                    texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(0.15, 0.15, 0.15, 1))
                    texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(0.50, 0.52, 0.54, 0))
            record["changes"].append(
                "stainless basin finish choice: metallic=1; silver reflectance with retained texture variation"
            )
    window_restoration = None
    if sky:
        # Main glazing is a separate original actor named Glass, not a child
        # of the detailed window frame. Its material was converted above.
        root = "/World/Environment/Source/Glass"
        original_glass = stage.GetPrimAtPath(root)
        if not original_glass or not original_glass.IsA(UsdGeom.Mesh):
            raise ValueError("Expected original main-window Glass mesh")
        dome = UsdLux.DomeLight.Define(stage, "/World/Lighting/RestoredExteriorSky")
        dome.CreateTextureFileAttr(Sdf.AssetPath(Path(sky).resolve().as_posix()))
        dome.CreateTextureFormatAttr("latlong")
        dome.CreateIntensityAttr(1000.0)
        dome.CreateExposureAttr(0)
        dome.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(True)
        dome.GetPrim().CreateAttribute("ghostFactor", Sdf.ValueTypeNames.Float).Set(0.0)
        window_restoration = dict(
            glazing=root,
            original_glazing_and_collision_preserved=True,
            sky_intensity=1000.0,
            photometric_calibration=False,
            sky_asset=str(Path(sky).resolve()),
            note="Original Glass mesh, frame and exterior facade retained; MDL transmission plus brighter sky lighting. Intensity is a visual daylight reference, not measured illuminance.",
        )
    layer.Save()
    write_json(
        out / "material-changes.json",
        dict(
            status="completed",
            scene=str(target),
            instances_reviewed=len(records),
            adjusted_instances=sum(bool(r["changes"]) for r in records),
            records=records,
            real_surface_calibration=False,
            original_geometry_and_physics_preserved=True,
            main_window_restoration=window_restoration,
            glass_source="https://docs.omniverse.nvidia.com/materials-and-rendering/latest/templates/OmniGlass.html",
            note="User-requested appearance refinement; intervals are finish choices, not measurements. Mixed utensil/ceramic atlas retained as one source material.",
        ),
    )
    return target


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--audit", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sky")
    args = p.parse_args()
    print(refine(args.scene, args.audit, args.out, args.sky))
