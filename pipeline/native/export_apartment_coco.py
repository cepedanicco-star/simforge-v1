"""Export lossless instance masks to COCO, preserving renderer provenance."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.ue_isaac_pipeline.io import read_json, write_json  # noqa: E402


def mask_rle(mask):
    flat = np.asarray(mask, dtype=np.uint8).ravel(order="F")
    boundaries = np.flatnonzero(np.diff(flat)) + 1
    runs = np.diff(np.r_[0, boundaries, flat.size]).tolist()
    if flat[0]:
        runs.insert(0, 0)
    return dict(size=list(mask.shape), counts=runs)


def export(dataset):
    dataset = Path(dataset).resolve(strict=True)
    samples = read_json(dataset / "samples.json")
    frames, classes = [], set()
    for sample in samples:
        n = sample["frame"]
        info = read_json(dataset / "annotations" / f"frame_{n:06d}.json")
        instance = np.load(dataset / "layers/instance" / f"frame_{n:06d}.npy").squeeze()
        semantic = np.load(dataset / "layers/semantic" / f"frame_{n:06d}.npy").squeeze()
        labels = info["semantic"]["idToLabels"]
        annotations = []
        for ident in np.unique(instance):
            if ident in (0, 1):
                continue
            mask = instance == ident
            ids, counts = np.unique(semantic[mask], return_counts=True)
            label = labels[str(int(ids[np.argmax(counts)]))].get("class", "unknown").lower()
            if label in ("background", "unlabelled"):
                continue
            path = info["instance"]["idToLabels"].get(str(int(ident)))
            if not isinstance(path, str) or not path.startswith("/"):
                raise ValueError(f"Missing instance identity: {n}/{ident}")
            ys, xs = np.where(mask)
            classes.add(label)
            annotations.append(
                dict(
                    category_name=label,
                    renderer_instance_id=int(ident),
                    prim_path=path,
                    bbox=[
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max() - xs.min() + 1),
                        int(ys.max() - ys.min() + 1),
                    ],
                    segmentation=mask_rle(mask),
                    area=int(mask.sum()),
                    iscrowd=0,
                )
            )
        frames.append((sample, instance.shape, annotations))
    categories = [
        dict(
            id=i + 1,
            name=name,
            supercategory="apartment",
            label_status="user_identified" if name == "cup_v1" else "source_or_name_rule_unreviewed",
        )
        for i, name in enumerate(sorted(classes))
    ]
    ids = {c["name"]: c["id"] for c in categories}
    images, annotations, next_ann = [], [], 1
    for sample, shape, anns in frames:
        n, split = sample["frame"], sample["split"]
        source = dataset / "layers/rgb" / f"frame_{n:06d}.png"
        relative = f"{split}/frame_{n:06d}.png"
        target = dataset / "images" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.link(source, target)  # Same volume: no duplicate RGB payload.
        images.append(
            dict(
                id=n + 1,
                file_name=relative,
                width=shape[1],
                height=shape[0],
                split=split,
                randomization_seed=sample["seed"],
            )
        )
        for ann in anns:
            ann.update(id=next_ann, image_id=n + 1, category_id=ids[ann.pop("category_name")])
            annotations.append(ann)
            next_ann += 1
    common = dict(
        info=dict(
            description="Updated UE apartment Replicator instances; labels are not vision predictions",
            version="1",
            depth_unit="m",
            label_calibration="Only cup target explicitly identified; remaining categories unreviewed rules",
        ),
        licenses=[],
        categories=categories,
    )
    directory = dataset / "coco_annotations"
    write_json(directory / "instances.json", dict(**common, images=images, annotations=annotations))
    for split in ("train", "val", "test"):
        subset = [im for im in images if im["split"] == split]
        image_ids = {im["id"] for im in subset}
        write_json(
            directory / f"instances_{split}.json",
            dict(**common, images=subset, annotations=[a for a in annotations if a["image_id"] in image_ids]),
        )
    write_json(directory / "categories.json", categories)
    (dataset / "README.md").write_text(
        "# Updated apartment synthetic data\n\n"
        "RGB: images/{train,val,test}; COCO: coco_annotations/instances*.json. "
        "Lossless Fortran-order uncompressed COCO RLE masks and pixel xywh boxes derive from native instance IDs. "
        "No detector confidence is fabricated. Unknown classes remain explicit and unreviewed.\n\n"
        "Depth: layers/depth/*.npy, distance to image plane in metres. Normals: layers/normals/*.npy. "
        "Semantic/instance integer arrays and mappings: layers/{semantic,instance}, annotations/*.json. "
        "samples.json includes RNG seeds, source scene hash, camera matrices/lenses, physical units, "
        "tight/loose native boxes and randomization values. This is randomized perception data, not an RL rollout.\n\n"
        "```python\nfrom torchvision.datasets import CocoDetection\n"
        "ds = CocoDetection(root='images', annFile='coco_annotations/instances_train.json')\n"
        "rgb, targets = ds[0]\n```\n\n"
        "Train/validation/test use distinct seeds and camera sectors. "
        "Synthetic held-out scores do not establish real-world generalization.\n",
        encoding="utf-8",
    )
    canvas = Image.new("RGB", (1536, 864), "#14212e")
    pen = ImageDraw.Draw(canvas)
    extremes = [
        ("dark", min(range(len(samples)), key=lambda i: samples[i].get("light_scale", 1))),
        ("bright", max(range(len(samples)), key=lambda i: samples[i].get("light_scale", 1))),
        ("warm", min(range(len(samples)), key=lambda i: samples[i].get("light_temperature_K", 5000))),
        ("cool", max(range(len(samples)), key=lambda i: samples[i].get("light_temperature_K", 5000))),
        ("smooth", min(range(len(samples)), key=lambda i: samples[i].get("roughness_offset", 0))),
        ("rough", max(range(len(samples)), key=lambda i: samples[i].get("roughness_offset", 0))),
    ]
    for k, (label, n) in enumerate(extremes):
        frame = samples[n]["frame"]
        with Image.open(dataset / "layers/rgb" / f"frame_{frame:06d}.png") as im:
            canvas.paste(im.convert("RGB").resize((512, 288)), ((k % 3) * 512, (k // 3) * 432))
        pen.text(
            ((k % 3) * 512 + 8, (k // 3) * 432 + 298),
            f"{label}: frame={frame} split={samples[n]['split']} seed={samples[n]['seed']}",
            fill="white",
        )
        pen.text(
            ((k % 3) * 512 + 8, (k // 3) * 432 + 322),
            f"light={samples[n].get('light_scale', 1):.3f} T={samples[n].get('light_temperature_K', 5000):.0f}K roughness delta={samples[n].get('roughness_offset', 0):+.3f}",
            fill="white",
        )
    canvas.save(dataset / "randomization-contact-sheet.jpg", quality=93)
    result = dict(
        status="completed",
        images=len(images),
        annotations=len(annotations),
        categories=len(categories),
        splits={s: sum(i["split"] == s for i in images) for s in ("train", "val", "test")},
        torchvision_load_verified=False,
        segmentation_format="COCO uncompressed RLE",
    )
    write_json(dataset / "coco-export-report.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset")
    print(export(p.parse_args().dataset))
