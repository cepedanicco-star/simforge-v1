"""Audit every apartment material binding, including constant and texture inputs.

Reports encoded-data errors and appearance diagnostics. Texture statistics and
material names are not measurements of the real apartment's surface finish.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pxr import Usd, UsdGeom, UsdShade

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.ue_isaac_pipeline.io import digest_file, write_json  # noqa: E402


def family(name):
    name = name.lower()
    groups = [
        ("bristles", ("bristles",)),
        ("metal", ("towelhooks", "doorhandle", "drain", "showerhandle", "pipes", "knife", "sink_v1")),
        ("plastic", ("toothbrush", "toiletbrush", "shampoo", "toothpaste", "deo_", "hairspray", "pen_")),
        ("liquid", ("water", "fluid")),
        ("organic", ("food", "cork", "soapblock", "wax")),
        ("stone", ("marble",)),
        ("wood", ("cabinet", "kitchen_furniture", "doorframe", "m_door", "beams")),
        ("paper", ("albums", "folder", "box_", "bottle_v1_label", "pictures_v", "paint_")),
        ("glass", ("glass", "mirror", "window")),
        ("textile", ("fabric", "cloth", "carpet", "cotton", "curtain", "towel", "sofa", "pillow", "rug")),
        ("wood", ("wood", "walnut", "oak", "parquet")),
        ("ceramic", ("ceramic", "porcelain", "tile", "bathsink", "bathtub", "toilet", "mug", "cup")),
        ("metal", ("metal", "steel", "chrome", "radiator", "alumin", "copper", "brass")),
        ("plaster_concrete", ("concrete", "wall", "ceiling", "plaster", "stucco")),
        ("plastic", ("plastic", "keyboard", "mouse", "monitor", "screen", "tv")),
        ("paper", ("paper", "book", "cardboard")),
    ]
    return next((label for label, terms in groups if any(term in name for term in terms)), "unclassified")


def texture_info(shader, channel):
    value = shader.GetInput(channel)
    if not value:
        return dict(kind="schema_default")
    source = value.GetConnectedSource()
    if not source:
        v = value.Get()
        return dict(kind="constant", value=list(v) if hasattr(v, "__len__") and not isinstance(v, str) else v)
    texture = UsdShade.Shader(source[0].GetPrim())
    if texture.GetIdAttr().Get() != "UsdUVTexture":
        return dict(kind="shader", id=str(texture.GetIdAttr().Get()), path=str(texture.GetPath()))
    asset = texture.GetInput("file").Get()
    path = Path(asset.resolvedPath) if asset and asset.resolvedPath else None
    result = dict(
        kind="texture",
        path=str(path) if path else str(asset),
        output=str(source[1]),
        shader_path=str(texture.GetPath()),
        color_space=texture.GetInput("sourceColorSpace").Get(),
        issues=[],
    )
    if path is None or not path.is_file():
        result["issues"].append("missing_texture")
        return result
    result["sha256"] = digest_file(path)
    with Image.open(path) as image:
        result["resolution"] = list(image.size)
        image = image.convert("RGB")
        image.thumbnail((384, 384))
        data = np.asarray(image).astype(np.float32) / 255
    scale = texture.GetInput("scale").Get()
    bias = texture.GetInput("bias").Get()
    scale = np.array(scale if scale is not None else (1, 1, 1, 1))
    bias = np.array(bias if bias is not None else (0, 0, 0, 0))
    result.update(scale=scale.tolist(), bias=bias.tolist())
    decoded = data * scale[:3] + bias[:3]
    if channel == "normal":
        norm = np.linalg.norm(decoded, axis=-1)
        angles = np.degrees(np.arctan2(np.linalg.norm(decoded[..., :2], axis=-1), decoded[..., 2]))
        result.update(
            angle_deg_percentiles=np.percentile(angles, [5, 50, 95, 99]).tolist(),
            negative_z_fraction=float((decoded[..., 2] < 0).mean()),
            norm_percentiles=np.percentile(norm, [1, 50, 99]).tolist(),
            xy_strength=float(scale[0] / 2),
        )
        if (norm < 1e-5).any():
            result["issues"].append("zero_normal_vectors")
        if result["negative_z_fraction"] > 0.01:
            result["issues"].append("normal_negative_z_needs_source_check")
    else:
        component = {"r": 0, "g": 1, "b": 2}.get(str(source[1]), 0)
        values = decoded[..., component]
        result.update(
            mean=float(values.mean()),
            percentiles=np.percentile(values, [1, 5, 50, 95, 99]).tolist(),
            near_zero_fraction=float((values < 0.02).mean()),
            near_one_fraction=float((values > 0.98).mean()),
        )
        if (values < 0).any() or (values > 1).any():
            result["issues"].append("roughness_outside_zero_one")
    if result["color_space"] != "raw":
        result["issues"].append("noncolor_input_not_raw")
    return result


def audit(scene, plan_path, out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    stage = Usd.Stage.Open(str(Path(scene).resolve()))
    root = stage.GetPrimAtPath("/World/Environment/Source")
    if not root:
        raise ValueError("Apartment environment prim is missing")
    materials, object_rows = {}, []
    for obj in plan["objects"]:
        prim = stage.GetPrimAtPath(obj["prepared_prim_path"])
        bound, mesh_count, populated_mesh_count = set(), 0, 0
        if prim:
            for child in Usd.PrimRange(prim):
                if not (child.IsA(UsdGeom.Mesh) or child.IsA(UsdGeom.Subset)):
                    continue
                mesh_count += int(child.IsA(UsdGeom.Mesh))
                if child.IsA(UsdGeom.Mesh):
                    populated_mesh_count += int(bool(UsdGeom.Mesh(child).GetPointsAttr().Get()))
                material, _ = UsdShade.MaterialBindingAPI(child).ComputeBoundMaterial()
                if not material:
                    continue
                mpath = str(material.GetPath())
                bound.add(mpath)
                if mpath in materials:
                    materials[mpath]["objects"].append(obj["prepared_prim_path"])
                    continue
                info = material.GetPrim().GetAssetInfo()
                unreal = info.get("unreal", {}).get("contentPath", "")
                name = unreal.split(".")[-1] if unreal else info.get("name", str(material.GetPath()))
                surface = next(
                    (
                        UsdShade.Shader(p)
                        for p in Usd.PrimRange(material.GetPrim())
                        if p.IsA(UsdShade.Shader)
                        and UsdShade.Shader(p).GetIdAttr().Get() == "UsdPreviewSurface"
                    ),
                    None,
                )
                materials[mpath] = dict(
                    path=mpath,
                    surface_shader=str(surface.GetPath()) if surface else None,
                    source_material=unreal,
                    name=name,
                    family_candidate=family(name),
                    calibrated_against_real_surface=False,
                    objects=[obj["prepared_prim_path"]],
                    normal=texture_info(surface, "normal") if surface else dict(kind="missing_surface"),
                    roughness=texture_info(surface, "roughness") if surface else dict(kind="missing_surface"),
                )
        object_rows.append(
            dict(
                path=obj["prepared_prim_path"],
                mesh_count=mesh_count,
                populated_mesh_count=populated_mesh_count,
                materials=sorted(bound),
                material_status="bound"
                if bound
                else "empty_geometry"
                if not populated_mesh_count
                else "unbound",
            )
        )
    rows = list(materials.values())
    for row in rows:
        row["objects"] = sorted(set(row["objects"]))
    report = dict(
        status="completed",
        scene=str(Path(scene).resolve()),
        object_records=len(object_rows),
        bound_material_instances=len(rows),
        unique_source_materials=len({r["name"] for r in rows}),
        objects=object_rows,
        materials=rows,
        real_material_calibration=False,
        evidence_scope="All mapped apartment objects and their bindings; appearance statistics and source encoding, not real-world measurements",
    )
    write_json(out / "materials-audit.json", report)
    unique = {row["name"]: row for row in rows}
    font = ImageFont.load_default(size=14)
    for page, start in enumerate(range(0, len(unique), 12)):
        canvas = Image.new("RGB", (960, 12 * 106), "#18222c")
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(list(unique.values())[start : start + 12]):
            y = index * 106
            draw.text((10, y + 4), row["name"][:90], font=font, fill="white")
            for column, channel in enumerate(("normal", "roughness")):
                data = row[channel]
                path = Path(data.get("path", ""))
                if path.is_file():
                    with Image.open(path) as image:
                        image = image.convert("RGB").resize((88, 76))
                        canvas.paste(image, (10 + column * 100, y + 25))
            normal, rough = row["normal"], row["roughness"]
            draw.text(
                (225, y + 32),
                f"Family hypothesis: {row['family_candidate']} | normal P50/P95: {normal.get('angle_deg_percentiles', [0, 0, 0, 0])[1:3]}",
                font=font,
                fill="#c5d4e5",
            )
            draw.text(
                (225, y + 56),
                f"Roughness mean: {rough.get('mean', rough.get('value'))} | normal XY: {normal.get('xy_strength')} | real calibrated: NO",
                font=font,
                fill="#c5d4e5",
            )
        canvas.save(out / f"materials-contact-{page:02d}.png")
    return {k: v for k, v in report.items() if k not in ("objects", "materials")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.scene, args.plan, args.out), ensure_ascii=True, indent=2))
