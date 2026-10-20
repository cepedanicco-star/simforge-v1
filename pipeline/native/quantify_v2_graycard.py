"""Compare frozen UE/native baseline with exposure-only native rerendering."""
# ruff: noqa: E402
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.simforge.materials.camera_pairing import camera_pairing
from pipeline.simforge.materials.graycard import srgb_to_linear
from pipeline.simforge.materials.metrics import delta_e2000, srgb_to_lab, ssim_map


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image(path):
    with Image.open(path) as im:
        assert im.size == (2048, 1152)
        return np.asarray(im.convert("RGB"))


def normalize(rgb, factor):
    linear = srgb_to_linear(rgb) * factor
    clipped = float(np.any((linear < 0) | (linear > 1), axis=2).mean())
    linear = np.clip(linear, 0, 1)
    encoded = np.where(linear <= .0031308, linear * 12.92,
                       1.055 * linear ** (1 / 2.4) - .055)
    return np.clip(np.rint(encoded * 255), 0, 255).astype(np.uint8), clipped


def measure(ue, native, mask):
    lab_u, lab_n = srgb_to_lab(ue[mask]), srgb_to_lab(native[mask])
    differences = np.concatenate([
        delta_e2000(lab_u[i:i+262144], lab_n[i:i+262144])
        for i in range(0, len(lab_u), 262144)
    ])
    result = dict(delta_e2000=float(differences.mean()),
                  delta_e2000_p95=float(np.percentile(differences, 95)),
                  ssim_full_frame=float(ssim_map(ue, native)[5:-5, 5:-5].mean()))
    assert all(np.isfinite(v) for v in result.values())
    return result, differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--baseline", required=True)
    args = parser.parse_args()
    root, baseline = Path(args.root), Path(args.baseline)
    assert not (root / "metrics.json").exists()
    config = read(root / "fixture-config.json")
    calibration = read(root / "calibration/calibration-report.json")
    assert calibration["status"] in ("EXPOSURE_ALIGNED", "FALLBACK_LINEAR_NORMALIZATION")
    assert calibration["acceptance"]["gray_pass"]
    native = read(root / "isaac/capture-report.json")
    ue = read(baseline / "ue/capture-report.json")
    gate = read(root / "isaac/native-pairing-acceptance.json")
    base_native = read(baseline / "isaac/capture-report.json")
    base_metrics = read(baseline / "metrics.json")
    assert native["status"] == ue["status"] == gate["status"] == "PASS"
    assert gate["passed"] == len(native["records"]) == len(config["views"]) == 6
    materials = read(root / "isaac/material-map.json")
    assert materials == read(baseline / "isaac/material-map.json")
    label_map = {v["label"]: dict(usd_material=k, **v) for k, v in materials.items()}
    # Preflight all six pairs before reporting any metric.
    for view, ur, nr, ng, br, bm in zip(config["views"], ue["records"], native["records"],
                                       gate["checks"], base_native["records"], base_metrics["pairs"], strict=True):
        frame = view["source_frame"]
        assert frame == ur["source_frame"] == nr["source_frame"] == br["source_frame"] == bm["source_frame"]
        assert camera_pairing(view, ur)["status"] == camera_pairing(view, ng["actual_observed"])["status"] == "PASS"
        np.testing.assert_allclose(nr["native_K"], br["native_K"], rtol=0, atol=1e-10)
        np.testing.assert_allclose(ng["actual_usd_camera_matrix"], view["usd_camera_matrix"], rtol=0, atol=1e-10)
        assert digest(baseline / "ue" / f"cam01-{frame:03d}.png") == bm["ue_sha256"]
        assert digest(baseline / "isaac" / f"cam01-{frame:03d}.png") == bm["isaac_sha256"]
        name = f"cam01-{frame:03d}-material-labels.json"
        new_labels = read(root / "isaac" / name)
        old_labels = read(baseline / "isaac" / name)
        ids_name = f"cam01-{frame:03d}-material-ids.npy"
        new_ids, old_ids = np.load(root / "isaac" / ids_name), np.load(baseline / "isaac" / ids_name)
        # Replicator numeric IDs are allocated per session, not stable material IDs.
        # Compare every named material's actual pixel region, not integer allocation.
        for label in label_map:
            new_region = np.isin(new_ids, [int(k) for k, v in new_labels.items() if v.get("class") == label])
            old_region = np.isin(old_ids, [int(k) for k, v in old_labels.items() if v.get("class") == label])
            assert np.array_equal(new_region, old_region), (frame, label, int(np.sum(new_region != old_region)))
        for path in (baseline / "ue" / f"cam01-{frame:03d}.png", root / "isaac" / f"cam01-{frame:03d}.png"):
            image(path)
    best = min(calibration["trials"], key=lambda t: abs(t["exposure_seconds"] - calibration["locked_exposure_seconds"]))
    factors = dict(ue=.18 / calibration["reference_ue"]["gray18"]["linear_luminance_mean"],
                   isaac=.18 / best["cards"]["gray18"]["linear_luminance_mean"])
    fallback = calibration["status"] == "FALLBACK_LINEAR_NORMALIZATION"
    rows, totals, sheets = [], {}, []
    font = ImageFont.load_default(size=23)
    for view, bm in zip(config["views"], base_metrics["pairs"], strict=True):
        frame = view["source_frame"]
        u = image(baseline / "ue" / f"cam01-{frame:03d}.png")
        before = image(baseline / "isaac" / f"cam01-{frame:03d}.png")
        after = image(root / "isaac" / f"cam01-{frame:03d}.png")
        ids = np.load(root / "isaac" / f"cam01-{frame:03d}-material-ids.npy")
        labels = read(root / "isaac" / f"cam01-{frame:03d}-material-labels.json")
        mask = np.isin(ids, [int(k) for k, v in labels.items() if v.get("class") in label_map])
        assert mask.any() and np.array_equal(mask, np.asarray(Image.open(baseline / f"cam01-{frame:03d}-valid-mask.png")) > 0)
        after_metrics, differences = measure(u, after, mask)
        row = dict(frame=frame, masked_pixels=int(mask.sum()),
                   before_delta_e2000=bm["delta_e2000"], before_ssim_full_frame=bm["ssim_full_frame"],
                   after_delta_e2000=after_metrics["delta_e2000"], after_ssim_full_frame=after_metrics["ssim_full_frame"],
                   after_delta_e2000_p95=after_metrics["delta_e2000_p95"],
                   native_sha256=digest(root / "isaac" / f"cam01-{frame:03d}.png"))
        if fallback:
            un, uc = normalize(u, factors["ue"])
            an, ac = normalize(after, factors["isaac"])
            normalized, _ = measure(un, an, mask)
            row["normalized_diagnostic"] = dict(**normalized, ue_clipped_fraction=uc, isaac_clipped_fraction=ac)
        rows.append(row)
        diff_map = np.full(mask.shape, np.nan, dtype=np.float32)
        diff_map[mask] = differences
        np.save(root / f"cam01-{frame:03d}-after-delta-e2000.npy", diff_map)
        for key, label in labels.items():
            info = label_map.get(label.get("class"))
            if not info:
                continue
            selected = ids == int(key)
            count = int(selected.sum())
            if count:
                name = info["source_asset"] or info["usd_material"]
                state = totals.setdefault(name, dict(material=name, pixels=0, delta_sum=0.))
                state["pixels"] += count
                state["delta_sum"] += float(diff_map[selected].sum(dtype=np.float64))
        sheet = Image.new("RGB", (2048, 1024), "#111827")
        draw = ImageDraw.Draw(sheet)
        for i, (im, label) in enumerate([(u, "Frozen UE reference"), (before, "Isaac before"), (after, "Isaac exposure-only after")]):
            x = (i % 2) * 1024
            y = (i // 2) * 512
            sheet.paste(Image.fromarray(im).resize((906, 510)), (x, y))
            draw.rectangle((x, y, x+1024, y+31), fill="#111827")
            draw.text((x+10, y+2), label, font=font, fill="white")
        draw.text((1044, 550), f"cam01 {frame} | RAW native renders", font=font, fill="white")
        draw.text((1044, 605), f"DE: {row['before_delta_e2000']:.3f} -> {row['after_delta_e2000']:.3f}", font=font, fill="white")
        draw.text((1044, 660), f"Full SSIM: {row['before_ssim_full_frame']:.4f} -> {row['after_ssim_full_frame']:.4f}", font=font, fill="white")
        draw.text((1044, 730), "No post-render brightening/warping", font=font, fill="#b8cae3")
        target = root / f"comparison-{frame:03d}.png"
        sheet.save(target)
        sheets.append(target)
        print("V2 QUANTIFIED", frame, row["after_delta_e2000"], row["after_ssim_full_frame"], flush=True)
    per_material = [dict(material=v["material"], pixels=v["pixels"], mean_delta_e2000=v["delta_sum"] / v["pixels"],
                         reference_band="<5" if v["delta_sum"] / v["pixels"] < 5 else ">=5") for v in totals.values()]
    per_material.sort(key=lambda r: r["mean_delta_e2000"], reverse=True)
    before_mean = float(np.mean([r["before_delta_e2000"] for r in rows]))
    after_mean = float(np.mean([r["after_delta_e2000"] for r in rows]))
    result = dict(status="MEASURED", valid_pairs=6, rows=rows, calibration_status=calibration["status"],
                  before_mean_delta_e2000=before_mean, after_mean_delta_e2000=after_mean,
                  delta_e_reduction=before_mean-after_mean, delta_e_reduction_fraction=(before_mean-after_mean)/before_mean,
                  before_mean_ssim_full_frame=float(np.mean([r["before_ssim_full_frame"] for r in rows])),
                  after_mean_ssim_full_frame=float(np.mean([r["after_ssim_full_frame"] for r in rows])),
                  method=dict(delta="CIEDE2000 D65 assumed display-sRGB, identical frozen material-label mask",
                              ssim="Whole frame at native resolution, 11x11 Gaussian sigma1.5 BT601 luma; valid rectangle [5:-5,5:-5]. No irregular mask SSIM.",
                              fitting="Only native engine exposureTime changed; raw PNGs unchanged after rendering",
                              scope="92 mesh-level materials; subset-only/unlabelled regions excluded from DeltaE. Cross-engine lighting, BRDF approximations, normals and tonemapping remain confounders.",
                              normalized_diagnostic_factors=factors if fallback else None,
                              normalized_diagnostic="If present, one engine-wide card-derived factor in display-linear RGB, reencoded and clipped; diagnostic only, not a physically aligned render.",
                              roughness_reference=str(baseline / "all-scene-roughness-alignment.json")))
    write(root / "metrics.json", result)
    write(root / "per-material-metrics.json", per_material)
    with (root / "six-frame-comparison.csv").open("w", newline="", encoding="utf8") as stream:
        fields = [k for k in rows[0] if k != "normalized_diagnostic"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    contact = Image.new("RGB", (1536, 1152))
    for i, path in enumerate(sheets):
        with Image.open(path) as im:
            contact.paste(im.resize((768, 384)), ((i % 2) * 768, (i // 2) * 384))
    contact.save(root / "six-pair-contact-sheet.png")
    print(json.dumps({k:v for k,v in result.items() if k not in ("rows", "method")}, indent=2))


if __name__ == "__main__":
    main()
