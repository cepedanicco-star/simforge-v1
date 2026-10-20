"""COCO instances from verified M1-format masks; no inferred rectangles as masks."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image


def encode_rle(mask):
    flat = np.asarray(mask, dtype=np.uint8).ravel(order="F")
    if not flat.size:
        raise ValueError("COCO masks must have positive image dimensions")
    boundaries = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.r_[0, boundaries, flat.size]).tolist()
    if flat[0]:
        counts.insert(0, 0)
    return dict(size=list(mask.shape), counts=counts)


def decode_rle(rle):
    count = sum(rle["counts"])
    if count != rle["size"][0] * rle["size"][1]:
        raise ValueError("RLE does not span the source image")
    values = np.repeat(np.arange(len(rle["counts"])) % 2, rle["counts"])
    return values.reshape(rle["size"], order="F").astype(bool)


def export(sources, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    frames = []
    for source in sources:
        source = Path(source)
        folders = sorted(source.glob("frame_*")) if (source / "export-report.json").is_file() else sorted(source.glob("var_*/frames/frame_*"))
        if not folders:
            raise ValueError("No complete frame directories: " + str(source))
        frames.extend(folders)
    labels = sorted({box["semantic_label"] for frame in frames
                     for box in json.loads((frame / "boxes_2d.json").read_text(encoding="utf8"))})
    categories = {label: index + 1 for index, label in enumerate(labels)}
    result = dict(info=dict(description="Verified native Isaac Sim 4.5 instance masks; uncompressed COCO RLE"),
                  images=[], annotations=[], categories=[dict(id=i, name=name, supercategory="scene_object") for name, i in categories.items()])
    receipts = []
    for image_id, frame in enumerate(frames, 1):
        image_path = frame / "rgb.png"
        with Image.open(image_path) as image:
            width, height = image.size
        with Image.open(frame / "instance.png") as image:
            ids = np.asarray(image)
        boxes = json.loads((frame / "boxes_2d.json").read_text(encoding="utf8"))
        filename = f"images/{image_id:06d}.png"
        shutil.copyfile(image_path, output / filename)
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if hashlib.sha256((output / filename).read_bytes()).hexdigest() != digest:
            raise ValueError("Copied training RGB differs from source")
        result["images"].append(dict(id=image_id, file_name=filename, width=width, height=height))
        for box in boxes:
            mask = ids == int(box["id"])
            ys, xs = np.where(mask)
            if not len(xs):
                raise ValueError("Native box has no corresponding instance mask")
            rle = encode_rle(mask)
            if not np.array_equal(decode_rle(rle), mask):
                raise ValueError("COCO RLE roundtrip changed the mask")
            coordinates = [int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)]
            if max(abs(int(box[key]) - value) for key, value in zip(
                    ("x_min", "y_min", "x_max", "y_max"), [xs.min(), ys.min(), xs.max(), ys.max()])) > 5:
                raise ValueError("COCO box differs from verified native box by >5px")
            result["annotations"].append(dict(id=len(result["annotations"])+1, image_id=image_id,
                category_id=categories[box["semantic_label"]], bbox=coordinates, area=int(mask.sum()),
                segmentation=rle, iscrowd=0))
        receipts.append(dict(image_id=image_id, frame=str(frame), rgb_sha256=digest))
    (output / "instances.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf8")
    report = dict(status="PASS", images=len(frames), annotations=len(result["annotations"]), categories=len(categories),
                  all_RLE_roundtrips_equal=True, sources=receipts, format="uncompressed COCO RLE in Fortran order")
    (output / "coco-report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.out), indent=2))


if __name__ == "__main__":
    main()
