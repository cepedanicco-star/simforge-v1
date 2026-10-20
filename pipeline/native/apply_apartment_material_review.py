"""Author reversible appearance corrections; preserve previously accepted finishes.

The angle and roughness bounds here are declared look-development priors, not
measured properties. No physical friction or robot asset is modified.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Gf, Sdf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.ue_isaac_pipeline.io import write_json  # noqa: E402

PROTECTED = {
    "MI_AI_vol8_03_wood_american_walnut_02",
    "MI_AI_vol10_02_concrete_01",
    "MI_AI_vol8_04_Table_metal",
}
MEDIAN_ANGLE_PRIOR = {
    "glass": 2,
    "textile": 12,
    "wood": 6,
    "ceramic": 4,
    "metal": 5,
    "plaster_concrete": 4,
    "plastic": 5,
    "paper": 5,
    "bristles": 18,
    "organic": 20,
    "stone": 5,
    "unclassified": 10,
}
ROUGHNESS_FLOOR_PRIOR = {
    "textile": (0.45, 0.60),
    "bristles": (0.45, 0.55),
    "wood": (0.08, 0.24),
    "plastic": (0.06, 0.12),
    "paper": (0.10, 0.25),
    "stone": (0.04, 0.18),
    "ceramic": (0.04, 0.16),
}


def attr(layer, path, name, kind, value):
    prim = Sdf.CreatePrimInLayer(layer, path)
    prim.specifier = Sdf.SpecifierOver
    field = prim.attributes.get(name) or Sdf.AttributeSpec(prim, name, kind)
    field.default = value


def normal_angles(path, scale, bias):
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((384, 384))
        data = np.asarray(image).astype(float) / 255
    decoded = data * np.asarray(scale[:3]) + np.asarray(bias[:3])
    angles = np.degrees(np.arctan2(np.linalg.norm(decoded[..., :2], axis=-1), decoded[..., 2]))
    return np.percentile(angles, [5, 50, 95, 99]).tolist()


def apply(scene, audit, camera_source, out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    target = out / "Apartment_materials.usda"
    if target.exists():
        raise FileExistsError(target)
    layer = Sdf.Layer.CreateNew(str(target))
    layer.subLayerPaths = [Path(os.path.relpath(Path(scene).resolve(), out)).as_posix()]
    rows = json.loads(Path(audit).read_text(encoding="utf-8"))["materials"]
    ledger = []
    for row in rows:
        record = dict(
            material=row["name"],
            prim=row["path"],
            family_candidate=row["family_candidate"],
            calibrated=False,
            changes=[],
            status="retained_after_review",
        )
        if row["name"] in PROTECTED:
            record["status"] = "preserved_user_accepted_finish"
            ledger.append(record)
            continue
        n = row["normal"]
        if n["kind"] == "constant":
            value = np.asarray(n["value"], dtype=float)
            if len(value) == 3 and np.linalg.norm(value[:2]) > 0.02:
                # An untextured tangent-space normal should follow geometry.
                # UE's baked constants here tilt flat glass/ceramic/cabinets.
                attr(
                    layer,
                    row["surface_shader"],
                    "inputs:normal",
                    Sdf.ValueTypeNames.Normal3f,
                    Gf.Vec3f(0, 0, 1),
                )
                record["changes"].append("neutralize_untextured_constant_tangent_normal")
                record["constant_normal_before"] = value.tolist()
                record["constant_normal_after"] = [0, 0, 1]
        if n["kind"] == "texture" and Path(n["path"]).is_file():
            scale, bias = list(n["scale"]), list(n["bias"])
            if scale == [1, 1, 1, 1] and bias == [0, 0, 0, 0] and Path(n["path"]).suffix.lower() == ".png":
                scale, bias = [2, 2, 2, 1], [-1, -1, -1, 0]
                record["changes"].append("repair_unsigned_normal_decode")
            before = normal_angles(n["path"], scale, bias)
            goal = MEDIAN_ANGLE_PRIOR.get(row["family_candidate"])
            # A mostly flat normal with a few steep bevel pixels is preserved.
            # Broad large slopes are reduced; hard-edge bake detail is retained.
            if goal and before[1] > max(goal * 1.25, 8):
                factor = max(0.05, min(1, math.tan(math.radians(goal)) / math.tan(math.radians(before[1]))))
                for axis in (0, 1):
                    scale[axis] *= factor
                    bias[axis] *= factor
                record["changes"].append("reduce_broad_normal_slope_using_declared_finish_prior")
                record["relative_xy_multiplier"] = factor
            if row["family_candidate"] == "plaster_concrete" and before[2] > 6:
                # Smooth painted walls have no baked metal/stone bevels to preserve.
                # Cap broad upper-tail bump contrast after reviewing the actual room.
                current = normal_angles(n["path"], scale, bias)
                factor = min(1, math.tan(math.radians(6)) / math.tan(math.radians(current[2])))
                for axis in (0, 1):
                    scale[axis] *= factor
                    bias[axis] *= factor
                record["changes"].append("reduce_wall_bump_upper_tail_after_render_review")
            if record["changes"]:
                attr(layer, n["shader_path"], "inputs:scale", Sdf.ValueTypeNames.Float4, Gf.Vec4f(*scale))
                attr(layer, n["shader_path"], "inputs:bias", Sdf.ValueTypeNames.Float4, Gf.Vec4f(*bias))
                attr(layer, n["shader_path"], "inputs:sourceColorSpace", Sdf.ValueTypeNames.Token, "raw")
                record.update(
                    normal_before_deg=before, normal_after_deg=normal_angles(n["path"], scale, bias)
                )
        rough = row["roughness"]
        value = rough.get("mean", rough.get("value"))
        floor = ROUGHNESS_FLOOR_PRIOR.get(row["family_candidate"])
        if floor and isinstance(value, (int, float)) and value < floor[0]:
            minimum = floor[1]
            if rough["kind"] == "texture":
                scale = np.asarray(rough["scale"], dtype=float)
                bias = np.asarray(rough["bias"], dtype=float)
                scale[:3] *= 1 - minimum
                bias[:3] = bias[:3] * (1 - minimum) + minimum
                attr(layer, rough["shader_path"], "inputs:scale", Sdf.ValueTypeNames.Float4, Gf.Vec4f(*scale))
                attr(layer, rough["shader_path"], "inputs:bias", Sdf.ValueTypeNames.Float4, Gf.Vec4f(*bias))
            elif rough["kind"] == "constant":
                attr(
                    layer,
                    row["surface_shader"],
                    "inputs:roughness",
                    Sdf.ValueTypeNames.Float,
                    minimum + value * (1 - minimum),
                )
            record["changes"].append("replace_extreme_gloss_using_declared_finish_prior")
            record.update(roughness_before_mean=value, roughness_after_mean=minimum + value * (1 - minimum))
        if row["name"] == "WorldGridMaterial":
            record["placeholder_checker_material_requires_assignment"] = True
        if record["changes"]:
            record["status"] = "adjusted_provisional_finish"
        ledger.append(record)
    camera = Sdf.CreatePrimInLayer(layer, "/World/Cameras")
    camera.specifier = Sdf.SpecifierOver
    camera.referenceList.explicitItems = [
        Sdf.Reference(
            Path(os.path.relpath(Path(camera_source).resolve(), out)).as_posix(),
            "/Root",
            Sdf.LayerOffset(0, 3),
        )
    ]
    layer.startTimeCode, layer.endTimeCode = 0, 750
    layer.timeCodesPerSecond = layer.framesPerSecond = 25
    layer.customLayerData = {
        "aisim:appearance_review": "provisional_finish_priors_not_measured",
        "aisim:task_complete": False,
    }
    layer.Save()
    report = dict(
        status="completed",
        scene=str(target),
        instances_reviewed=len(rows),
        source_materials_reviewed=len({x["name"] for x in rows}),
        adjusted_instances=sum(bool(x["changes"]) for x in ledger),
        protected_materials=sorted(PROTECTED),
        records=ledger,
        real_surface_calibration=False,
        camera_retime=dict(
            source_frames=250, source_fps=25, duration_seconds=30, output_frames=750, output_fps=25
        ),
        sources=[
            "https://openusd.org/dev/spec_usdpreviewsurface.html",
            "https://dev.epicgames.com/documentation/en-us/unreal-engine/physically-based-materials-in-unreal-engine",
        ],
        note="Finish priors are authored visual choices, not numerical measurements from those sources. Lighting, base color, metallic, texture UV, geometry, physical friction and the three accepted finishes are preserved.",
    )
    write_json(out / "material-changes.json", report)
    return {key: value for key, value in report.items() if key not in ("records", "sources")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene")
    parser.add_argument("--audit", required=True)
    parser.add_argument("--camera-source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(apply(args.scene, args.audit, args.camera_source, args.out), indent=2))
