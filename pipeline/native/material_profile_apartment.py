"""Seven apartment-specific appearance repairs in fresh opinion layers; not measured BRDF calibration."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdShade

from .preview_safe import compose_scene


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    recovery = root / "audit"
    recovery.mkdir()
    original = Usd.Stage.Open(args.source)
    if not original:
        raise ValueError("Source stage cannot open")
    desk_prefix = "/World/Environment/Source/SM_AI_vol8_03_desk_01/Section0/UnrealMaterial"
    # Author a new desk opinion; the source layer and textures are read-only.
    layer = Sdf.Layer.CreateAnonymous()
    desk = Usd.Stage.Open(layer)
    for node, scale, bias in [("normalTexture", (.1,.1,.1,1), (0,0,.9,0)),
                              ("roughnessTexture", (1,1,1,1), (0,0,0,0))]:
        sh = UsdShade.Shader(desk.OverridePrim(desk_prefix+"/"+node))
        sh.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale))
        sh.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*bias))
    layer.Export(str(root / "Desk_material_override.usda"))
    fixes_path = root / "Material_fixes.usda"
    assert not fixes_path.exists()
    edit = Usd.Stage.CreateNew(str(fixes_path))
    edit.GetRootLayer().subLayerPaths = [(root / "Desk_material_override.usda").resolve().as_posix()]
    changes = []
    def sh(path):
        return UsdShade.Shader(edit.OverridePrim(str(path)))
    def const(shader, name, kind, value):
        inp = shader.CreateInput(name, kind)
        inp.DisconnectSource()
        inp.Set(value)
    raw_cache, materials = {}, []
    for prim in original.Traverse():
        if str(prim.GetPath()).startswith("/World/Environment/Source") and prim.IsA(UsdShade.Material):
            material = UsdShade.Material(prim)
            surface = material.ComputeSurfaceSource()[0]
            if surface and surface.GetIdAttr().Get() == "UsdPreviewSurface":
                materials.append((material, surface, prim.GetAssetInfo().get("name", "")))
    fixed_constants, glass = [], []
    for material, surface, asset in materials:
        path = str(surface.GetPath())
        identifier = material.GetPrim().GetAssetInfo().get("identifier")
        raw_bad = False
        if identifier and identifier.resolvedPath:
            file = identifier.resolvedPath
            if file not in raw_cache:
                raw = Usd.Stage.Open(file)
                raw_cache[file] = any(
                    p.IsA(UsdShade.Shader) and UsdShade.Shader(p).GetInput("normal")
                    and not UsdShade.Shader(p).GetInput("normal").GetConnectedSource()
                    and UsdShade.Shader(p).GetInput("normal").Get() is not None
                    and np.linalg.norm(np.asarray(UsdShade.Shader(p).GetInput("normal").Get())[:2]) > 1e-4
                    for p in raw.Traverse())
            raw_bad = raw_cache[file]
        n = surface.GetInput("normal")
        current_bad = bool(n and not n.GetConnectedSource() and n.Get() is not None
                           and np.linalg.norm(np.asarray(n.Get())[:2]) > 1e-4)
        if raw_bad or current_bad:
            const(sh(path), "normal", Sdf.ValueTypeNames.Normal3f, Gf.Vec3f(0,0,1))
            fixed_constants.append(dict(shader=path, asset=asset, current_was_bad=current_bad, source_asset_was_bad=raw_bad))
        if asset in ("MI_AI_vol8_04_glass_01", "MI_AI_vol8_04_glass_03", "MI_AI_vol8_03_glass_01"):
            m = UsdShade.Material(edit.OverridePrim(material.GetPath()))
            mdl = UsdShade.Shader.Define(edit, str(material.GetPath())+"/V2ClearGlass")
            mdl.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
            mdl.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
            mdl.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.99,.995,1))
            mdl.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.52)
            mdl.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)
            mdl.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(.02)
            mdl.CreateOutput("out", Sdf.ValueTypeNames.Token)
            m.CreateSurfaceOutput("mdl").ConnectToSource(mdl.ConnectableAPI(), "out")
            glass.append(dict(material=str(material.GetPath()), shader=str(mdl.GetPath()), asset=asset))
        if asset == "MI_AI_vol8_04_bottle_V1_fluid":
            target = sh(path)
            const(target,"diffuseColor",Sdf.ValueTypeNames.Color3f,Gf.Vec3f(.63,.30,.12))
            const(target,"opacity",Sdf.ValueTypeNames.Float,.5)
            const(target,"metallic",Sdf.ValueTypeNames.Float,0.)
            const(target,"normal",Sdf.ValueTypeNames.Normal3f,Gf.Vec3f(0,0,1))
            changes.append(dict(item="fluid", shader=path, opacity=.5, metallic=0., note="Preview alpha approximation, not a measured liquid BRDF"))
        if path.startswith("/World/Environment/Source/SM_AI_vol8_03_stand_lamp_01/Section0/"):
            target = sh(path)
            for name, kind, value in [("diffuseColor",Sdf.ValueTypeNames.Color3f,Gf.Vec3f(.92,.92,.93)),
                                      ("metallic",Sdf.ValueTypeNames.Float,0.),("roughness",Sdf.ValueTypeNames.Float,.4),
                                      ("normal",Sdf.ValueTypeNames.Normal3f,Gf.Vec3f(0,0,1))]:
                const(target,name,kind,value)
            changes.append(dict(item="lamp_white_paint",shader=path,diffuse=[.92,.92,.93],metallic=0,roughness=.4))
        if asset == "MI_AI_vol8_03_fabric_linen_dirt_05" and "bed_fabric" in path:
            const(sh(path),"diffuseColor",Sdf.ValueTypeNames.Color3f,Gf.Vec3f(.85,.82,.78))
            const(sh(path),"opacity",Sdf.ValueTypeNames.Float,1.)
            changes.append(dict(item="bed_white_linen",shader=path,diffuse=[.85,.82,.78],approximation="Opaque diffuse compensation, no claim of simulated SSS; normal/roughness weave retained"))
        if "SM_AI_vol8_02_tv_01/" in path:
            target = sh(path)
            color = (.08,.08,.10) if "/Section0/" in path else (.012,.018,.03)
            const(target,"diffuseColor",Sdf.ValueTypeNames.Color3f,Gf.Vec3f(*color))
            if "/Section0/" not in path:
                const(target,"emissiveColor",Sdf.ValueTypeNames.Color3f,Gf.Vec3f(.002,.004,.012))
            changes.append(dict(item="tv",shader=path,diffuse=list(color),dark_blue_emission="/Section0/" not in path))
    # The table's node belongs only to this desk subset, never shared chairs.
    transform = UsdShade.Shader.Define(edit, desk_prefix+"/V2UVTiling")
    transform.CreateIdAttr("UsdTransform2d")
    transform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(2,2))
    transform.CreateInput("rotation", Sdf.ValueTypeNames.Float).Set(0.)
    transform.CreateInput("translation", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(0,0))
    reader = UsdShade.Shader(original.GetPrimAtPath(desk_prefix+"/PrimvarReader"))
    transform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),"result")
    transform.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    for node in ["diffuseTexture", "normalTexture", "roughnessTexture"]:
        sh(desk_prefix+"/"+node).CreateInput("st",Sdf.ValueTypeNames.Float2).ConnectToSource(transform.ConnectableAPI(),"result")
    cooker = "/World/Environment/Source/SM_AI_vol8_04_induction_cooker_001/UnrealMaterial/SurfaceShader"
    const(sh(cooker), "normal", Sdf.ValueTypeNames.Normal3f, Gf.Vec3f(0,0,1))
    changes.append(dict(item="induction_smooth",shader=cooker,normal_strength=0.,note="Flat normal selected per plan; no anomalous negative-Z decode"))
    edit.GetRootLayer().Save()
    scene = compose_scene(args.source, str(fixes_path), str(root / "Apartment_materials_v2_composed.usda"))
    composed = Usd.Stage.Open(scene)
    bad = []
    for prim in composed.Traverse():
        if not str(prim.GetPath()).startswith("/World/Environment/Source"):
            continue
        shader = UsdShade.Shader(prim)
        normal = shader.GetInput("normal") if shader else None
        if normal and not normal.GetConnectedSource() and normal.Get() is not None and np.linalg.norm(np.asarray(normal.Get())[:2])>1e-4:
            bad.append(str(prim.GetPath()))
    assert not bad, bad
    for entry in glass:
        source = UsdShade.Material(composed.GetPrimAtPath(entry["material"])).GetSurfaceOutput("mdl").GetConnectedSource()
        assert source and str(source[0].GetPath()) == entry["shader"]
    rough = UsdShade.Shader(composed.GetPrimAtPath(desk_prefix+"/roughnessTexture"))
    values = np.asarray(Image.open(rough.GetInput("file").Get().resolvedPath).convert("RGB"),dtype=float)[...,0]/255
    scale, bias = rough.GetInput("scale").Get(), rough.GetInput("bias").Get()
    values = values*scale[0]+bias[0]
    assert .30 <= values.mean() <= .35 and values.std() >= .03
    assert np.allclose(UsdShade.Shader(composed.GetPrimAtPath(desk_prefix+"/normalTexture")).GetInput("scale").Get(),[.1,.1,.1,1])
    # Inspect only the source layers actually loaded by this input stage.
    source_files = sorted({Path(layer.realPath) for layer in original.GetUsedLayers()
                           if layer.realPath and Path(layer.realPath).suffix.lower()
                           in {".usd", ".usda", ".usdc"}})
    raw_bad_files = []
    for path in source_files:
        stage = Usd.Stage.Open(str(path))
        for prim in stage.Traverse():
            shader = UsdShade.Shader(prim)
            n = shader.GetInput("normal") if shader else None
            if n and not n.GetConnectedSource() and n.Get() is not None and np.linalg.norm(np.asarray(n.Get())[:2])>1e-4:
                raw_bad_files.append(dict(file=str(path),shader=str(prim.GetPath()),value=list(n.Get())))
    report = dict(status="PARAMETERS_PASS_PENDING_RENDER", source=args.source, composed_scene=scene,
                  material_layer=str(fixes_path), constant_overrides=fixed_constants,
                  effective_bad_normal_count=len(bad), raw_bad_files=raw_bad_files,
                  raw_source_note="Loaded source layers remain unchanged; the effective composed stage is the acceptance scope. Source inspection is limited to layers loaded by the input stage.",
                  glass=glass, changes=changes, desk_tiling=[2,2], normal_strength=.1,
                  roughness_mean=float(values.mean()),roughness_std=float(values.std()),
                  preserved_preview=None)
    write(recovery / "material-fixes-report.json", report)
    print(json.dumps(dict(status=report["status"],constant_overrides=len(fixed_constants),
                         effective_bad_normals=len(bad),raw_bad_source_files=len(raw_bad_files),
                         glass_bindings=len(glass),roughness_mean=values.mean(),roughness_std=values.std()),indent=2))


if __name__ == "__main__":
    main()
