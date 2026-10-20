"""Author a desk-only opinion layer; never rewrite a source material/texture."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source, out = Path(args.source).resolve(strict=True), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    stage = Usd.Stage.Open(str(source))
    desk = stage.GetPrimAtPath("/World/Environment/Source/SM_AI_vol8_03_desk_01")
    assert desk and desk.IsA(UsdGeom.Mesh)
    candidates = []
    for prim in Usd.PrimRange(desk):
        if not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Subset)):
            continue
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not material:
            continue
        surface = material.ComputeSurfaceSource()[0]
        assert surface and surface.GetIdAttr().Get() == "UsdPreviewSurface"
        diffuse = surface.GetInput("diffuseColor").GetConnectedSource()
        if diffuse and "wood_american_walnut" in str(diffuse[0].GetInput("file").Get()):
            candidates.append((prim, material, surface))
    assert len(candidates) == 1, [str(c[0].GetPath()) for c in candidates]
    prim, material, shader = candidates[0]
    normal = UsdShade.Shader(shader.GetInput("normal").GetConnectedSource()[0].GetPrim())
    rough = UsdShade.Shader(shader.GetInput("roughness").GetConnectedSource()[0].GetPrim())
    assert normal.GetIdAttr().Get() == rough.GetIdAttr().Get() == "UsdUVTexture"
    users = []
    for p in stage.Traverse():
        if p.IsA(UsdGeom.Mesh) or p.IsA(UsdGeom.Subset):
            bound, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()
            if bound and bound.GetPath() == material.GetPath():
                users.append(str(p.GetPath()))
    assert users == [str(prim.GetPath())], users
    nscale = np.asarray(normal.GetInput("scale").Get() or (1, 1, 1, 1), dtype=float)
    nbias = np.asarray(normal.GetInput("bias").Get() or (0, 0, 0, 0), dtype=float)
    # Blend the currently evaluated tangent normal toward flat (0,0,1).
    # The renderer renormalizes it. Preserve existing map decode rather than
    # introducing a second, unrelated normal conversion in this desk-only edit.
    ns, nb = nscale.copy(), nbias.copy()
    ns[:3] *= .1
    nb[:3] = nb[:3] * .1 + [0, 0, .9]
    rscale = np.asarray(rough.GetInput("scale").Get() or (1, 1, 1, 1), dtype=float)
    rbias = np.asarray(rough.GetInput("bias").Get() or (0, 0, 0, 0), dtype=float)
    texture = Path(rough.GetInput("file").Get().resolvedPath)
    values = np.asarray(Image.open(texture).convert("RGBA"), dtype=float)[..., 0] / 255
    before = values * rscale[0] + rbias[0]
    rs, rb = rscale.copy(), rbias.copy()
    rs[:3] *= .1
    rb[:3] = rb[:3] * .1 + .9 * float(before.mean())
    after = values * rs[0] + rb[0]
    np.testing.assert_allclose(after.mean(), before.mean(), atol=1e-12)
    np.testing.assert_allclose(after.std(), before.std() * .1, atol=1e-12)
    layer = Sdf.Layer.CreateNew(str(out / "Desk_material_override.usda"))
    edit = Usd.Stage.Open(layer)
    for path, scale, bias in [(str(normal.GetPath()), ns, nb), (str(rough.GetPath()), rs, rb)]:
        sh = UsdShade.Shader(edit.OverridePrim(path))
        sh.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*map(float, scale)))
        sh.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*map(float, bias)))
    layer.Save()
    composed = Usd.Stage.CreateNew(str(out / "Apartment_materials_v2.usda"))
    composed.GetRootLayer().subLayerPaths = [layer.realPath, source.as_posix()]
    UsdGeom.SetStageMetersPerUnit(composed, 1)
    UsdGeom.SetStageUpAxis(composed, UsdGeom.Tokens.z)
    composed.SetTimeCodesPerSecond(stage.GetTimeCodesPerSecond())
    composed.SetFramesPerSecond(stage.GetFramesPerSecond())
    composed.SetStartTimeCode(stage.GetStartTimeCode())
    composed.SetEndTimeCode(stage.GetEndTimeCode())
    composed.SetDefaultPrim(composed.GetPrimAtPath("/World"))
    composed.GetRootLayer().Save()
    texture_inputs = []
    for p in Usd.PrimRange(material.GetPrim()):
        sh = UsdShade.Shader(p)
        if not sh:
            continue
        inp = sh.GetInput("file")
        if inp and inp.Get() and inp.Get().resolvedPath:
            file = Path(inp.Get().resolvedPath)
            texture_inputs.append(dict(path=str(file), sha256=sha(file)))
    # No authored attributes outside these four shader inputs.
    authored = []
    def visit(path):
        obj = layer.GetObjectAtPath(path)
        if isinstance(obj, Sdf.AttributeSpec):
            authored.append(str(path))
    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    assert len(authored) == 4
    report = dict(status="AUTHORED_PENDING_NATIVE_PREVIEW", source_scene=str(source), source_sha256=sha(source),
                  mesh=str(desk.GetPath()), tabletop_subset=str(prim.GetPath()), material=str(material.GetPath()),
                  material_users=users, opinion_layer=layer.realPath, scene=composed.GetRootLayer().realPath,
                  normal_strength_before=1., normal_strength_after=.1,
                  normal_method="Current evaluated tangent normal N -> 0.1*N + 0.9*(0,0,1), before shader normalization; no source texture change",
                  normal_scale_before=nscale.tolist(), normal_bias_before=nbias.tolist(),
                  normal_scale_after=ns.tolist(), normal_bias_after=nb.tolist(),
                  roughness_mean_before=float(before.mean()), roughness_mean_after=float(after.mean()),
                  roughness_std_before=float(before.std()), roughness_std_after=float(after.std()),
                  roughness_variance_factor=.01, roughness_texel_note="Linear raw R channel, texel statistics, not UV weighted",
                  basecolor_unchanged=True, clearcoat_added=False, changed_attributes=authored,
                  source_textures=texture_inputs)
    (out / "desk-change.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
