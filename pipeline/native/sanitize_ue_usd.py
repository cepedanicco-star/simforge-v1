"""Copy an Epic USD export and clear engine-only shader asset addresses.

Preserves baked UsdPreviewSurface materials and files. /Game and /Engine paths
in info:unreal:sourceAsset are UE package identifiers, not portable dependencies.
Only that exact shader property is changed; missing textures remain errors.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from pxr import Sdf, UsdUtils

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.ue_isaac_pipeline.io import digest_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source, output = Path(args.source).resolve(strict=True), Path(args.out).resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Output must be outside the source export")
    shutil.copytree(source, output)
    changes = []
    for file in output.rglob("*.usda"):
        layer = Sdf.Layer.FindOrOpen(str(file))
        paths = []
        layer.Traverse(Sdf.Path.absoluteRootPath, paths.append)
        original_hash = digest_file(file)
        removed = []
        remapped = []

        def relocate(asset):
            normalized = asset.replace("\\", "/")
            prefix = source.as_posix() + "/"
            if normalized.lower().startswith(prefix.lower()):
                destination = output / normalized[len(prefix):]
                rewritten = Path(os.path.relpath(destination, file.parent)).as_posix()
                remapped.append(dict(original=asset, portable=rewritten))
                return rewritten
            return asset

        # Epic may author absolute paths into its export. A directory copy alone
        # would still load the original, unsanitized material/mesh layers.
        UsdUtils.ModifyAssetPaths(layer, relocate)
        for path in paths:
            spec = layer.GetObjectAtPath(path)
            if isinstance(spec, Sdf.AttributeSpec) and spec.name == "info:unreal:sourceAsset":
                asset = spec.default
                if isinstance(asset, Sdf.AssetPath) and asset.path.startswith(("/Game/", "/Engine/")):
                    removed.append(dict(property=str(path), unreal_package=asset.path))
                    spec.owner.RemoveProperty(spec)
        if removed or remapped:
            layer.Save()
            changes.append(dict(layer=str(file.relative_to(output)), source_sha256=original_hash,
                                removed=removed, relocated=remapped))
    write_json(output / "portability-report.json", dict(changes=changes, source_export=str(source)))
    print(f"Cleared Unreal shader package references from {len(changes)} copied layers")


if __name__ == "__main__":
    main()
