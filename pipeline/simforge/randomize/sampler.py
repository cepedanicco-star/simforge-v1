"""Pure episode sampler: seeded dictionaries, without USD or Isaac imports."""

import argparse
import copy
import csv
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np


def sample(config, variant_id):
    seed = config["global_seed"] + variant_id
    result = dict(variant_id=variant_id, seed=seed, dimensions={})
    for number in range(1, 7):
        name = f"D{number}"
        cfg = config["dimensions"][name]
        if cfg["timing"] != "episode_start":
            raise ValueError("This protocol samples every dimension once at episode start")
        rng = np.random.default_rng(np.random.SeedSequence([seed, number]))
        row = dict(enabled=bool(cfg["enabled"]), timing=cfg["timing"], distribution=cfg["distribution"])

        def uniform(key, size=None):
            limits = cfg[key]
            if len(limits) != 2 or limits[0] > limits[1]:
                raise ValueError("Invalid sampling range: " + name + "." + key)
            value = rng.uniform(*limits, size=size)
            return value.tolist() if size is not None else float(value)

        if not row["enabled"]:
            row["applied"] = False
            row["values"] = copy.deepcopy(cfg["center"])
            if name == "D6":
                row["values"] = dict(
                    friction=cfg["nominal_friction"], mass_kg=cfg["nominal_mass_kg"], mass_scale=1.0
                )
        elif name == "D1":
            row["values"] = dict(
                illuminance_lux=uniform("illuminance_lux"), temperature_K=uniform("temperature_K")
            )
        elif name == "D2":
            row["values"] = dict(roughness=uniform("roughness"), hue_degrees=uniform("hue_degrees"))
        elif name == "D3":
            count = int(rng.integers(cfg["count"][0], cfg["count"][1] + 1))
            row["values"] = dict(
                count=count,
                proposal_xy_unit=rng.uniform(0, 1, (cfg["proposal_count"], 2)).tolist(),
                colors_rgb=rng.uniform(0.1, 0.9, (count, 3)).tolist(),
            )
        elif name == "D4":
            if cfg["axes"] != ["x", "y"] or cfg["z_rule"] != "preserve_support_height":
                raise ValueError("Cup jitter must preserve support height")
            row["values"] = dict(
                position_m=uniform("position_m", 2) + [0.0], yaw_degrees=uniform("yaw_degrees")
            )
        elif name == "D5":
            row["values"] = dict(
                position_m=uniform("position_m", 3),
                euler_rad=uniform("euler_rad", 3),
                fov_scale=uniform("fov_scale"),
            )
            if cfg.get("require_free_space") and "proposal_count" in cfg:
                count = cfg["proposal_count"]
                if not isinstance(count, int) or count < 1:
                    raise ValueError("D5 proposal_count must be positive")
                row["values"]["position_proposals_m"] = [row["values"]["position_m"]] + [
                    uniform("position_m", 3) for _ in range(count - 1)]
        else:
            scale = uniform("mass_scale")
            row["values"] = dict(
                friction=uniform("friction"), mass_scale=scale, mass_kg=cfg["nominal_mass_kg"] * scale
            )
        result["dimensions"][name] = row
    return result


def compare_values(first, second):
    """Compare sampled physical/visual values, never timestamps or run identifiers."""
    if (first["seed"], first["variant_id"]) != (second["seed"], second["variant_id"]):
        return ["seed_or_variant_id"]
    return [name for name in first["dimensions"] if first["dimensions"][name] != second["dimensions"][name]]


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--variants", type=int)
    parser.add_argument("--out")
    parser.add_argument("--disable", choices=[f"D{i}" for i in range(1, 7)])
    args = parser.parse_args()
    config_path = Path(args.config).resolve(strict=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf8"))
    if args.disable:
        config["dimensions"][args.disable]["enabled"] = False
    count = args.variants or config["variants"]
    if not 1 <= count <= 10000:
        raise ValueError("variants must be 1..10000")
    out = Path(args.out or config["output"])
    out.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config_path.parent, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=config_path.parent))
    rows = []
    for index in range(count):
        meta = sample(config, index)
        meta.update(
            sampled_at=datetime.now(timezone.utc).isoformat(),
            git_commit=commit,
            git_dirty=dirty,
            sampler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            config_path=str(config_path),
            status="sampled_not_rendered",
        )
        folder = out / f"var_{index:02d}"
        folder.mkdir()
        (folder / "frames").mkdir()
        (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf8")
        row = dict(variant_id=index, seed=meta["seed"])
        row.update(
            {
                name: json.dumps(value["values"], separators=(",", ":"))
                for name, value in meta["dimensions"].items()
            }
        )
        rows.append(row)
        compact = {
                name: {k: v for k, v in value["values"].items() if k not in (
                    "proposal_xy_unit", "colors_rgb", "position_proposals_m")}
            for name, value in meta["dimensions"].items()
        }
        print("M3 SAMPLE", index, "seed", meta["seed"], json.dumps(compact), flush=True)
    with (out / "samples.csv").open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["variant_id", "seed"] + [f"D{i}" for i in range(1, 7)])
        writer.writeheader()
        writer.writerows(rows)
    (out / "resolved-config.json").write_text(json.dumps(config, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
