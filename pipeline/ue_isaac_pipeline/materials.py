"""Audit baked PreviewSurface maps and repair unsigned normal decoding in a copy."""

from pathlib import Path

from .io import digest_file, write_json

NORMAL_SPEC = "https://openusd.org/dev/spec_usdpreviewsurface.html"


def audit_and_repair(directory, report_path, *, repair=False):
    """Use only on a disposable/portable export copy; preserve roughness values.

    Missing scale/bias on an 8-bit PNG is a representation error. Existing
    non-identity decoding is retained for review rather than overwritten.
    This does not establish that artistic maps match measured real materials.
    """
    import numpy as np
    from PIL import Image
    from pxr import Gf, Sdf, Usd, UsdShade

    records = []
    for path in sorted(Path(directory).rglob("*.usda")):
        # UE material assets are separate small layers; avoid opening mesh layers.
        if "Materials" not in path.parts:
            continue
        before = digest_file(path)
        stage = Usd.Stage.Open(str(path))
        changed = False
        for prim in stage.Traverse():
            surface = UsdShade.Shader(prim)
            if not surface or surface.GetIdAttr().Get() != "UsdPreviewSurface":
                continue
            for channel in ("normal", "roughness"):
                surface_input = surface.GetInput(channel)
                if not surface_input:
                    continue
                source = surface_input.GetConnectedSource()
                if not source:
                    continue
                texture = UsdShade.Shader(source[0].GetPrim())
                if texture.GetIdAttr().Get() != "UsdUVTexture":
                    continue
                asset = texture.GetInput("file").Get()
                image_path = Path(asset.resolvedPath or path.parent / asset.path)
                entry = dict(
                    layer=str(path),
                    shader=str(texture.GetPath()),
                    channel=channel,
                    texture=str(image_path),
                    source_sha256=before,
                    issues=[],
                    changes=[],
                )
                with Image.open(image_path) as image:
                    entry.update(mode=image.mode, size=list(image.size))
                    values = np.asarray(image)
                    entry["byte_range"] = [int(values.min()), int(values.max())]
                    if channel == "roughness":
                        entry["mean_roughness"] = (
                            float(values[..., 0].mean() / 255)
                            if values.ndim == 3
                            else float(values.mean() / 255)
                        )
                    is_byte_rgb = image.mode in ("RGB", "RGBA") and values.dtype == np.uint8
                color = texture.GetInput("sourceColorSpace")
                entry["color_space"] = color.Get() if color else None
                if channel == "normal":
                    scale = texture.GetInput("scale")
                    bias = texture.GetInput("bias")
                    scale_value = scale.Get() if scale else None
                    bias_value = bias.Get() if bias else None
                    identity = (scale_value is None or tuple(scale_value) == (1, 1, 1, 1)) and (
                        bias_value is None or tuple(bias_value) == (0, 0, 0, 0)
                    )
                    if is_byte_rgb and identity:
                        entry["issues"].append("unsigned_8bit_normal_missing_signed_decode")
                        if repair:
                            texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
                            texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(
                                Gf.Vec4f(-1, -1, -1, 0)
                            )
                            texture.CreateInput("fallback", Sdf.ValueTypeNames.Float4).Set(
                                Gf.Vec4f(0.5, 0.5, 1, 1)
                            )
                            entry["changes"].append("decode_0_1_to_minus1_plus1")
                            changed = True
                    elif not is_byte_rgb:
                        entry["issues"].append("normal_encoding_needs_manual_review")
                if entry["color_space"] != "raw":
                    entry["issues"].append("noncolor_data_not_explicit_raw")
                    if repair:
                        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
                        entry["changes"].append("set_raw_color_space")
                        changed = True
                records.append(entry)
        if changed:
            stage.GetRootLayer().Save()
        after = digest_file(path)
        for record in records:
            if record["layer"] == str(path):
                record["result_sha256"] = after
    report = dict(
        schema_version=1,
        mode="repair_copy" if repair else "audit",
        specification=NORMAL_SPEC,
        changed_textures=sum(bool(r["changes"]) for r in records),
        records=records,
        physical_appearance_calibrated=False,
        note="Roughness magnitudes, normal strength and green-channel orientation preserved; real appearance requires reference photographs/measurements.",
    )
    write_json(report_path, report)
    return report
