"""Attach native engine mass distributions to the separately audited USD manifest."""

import argparse
import sys
from pathlib import Path

import jsonschema
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.ue_isaac_pipeline.io import read_json, write_json, digest_file  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--manifest", required=True)
p.add_argument("--native", required=True)
p.add_argument("--settle", required=True)
p.add_argument("--schema", required=True)
p.add_argument("--out", required=True)
args = p.parse_args()
data = read_json(args.manifest)
native = read_json(args.native)
settle = read_json(args.settle)
bodies = {"/World/G1/" + b["name"]: b for b in native["bodies"]}
bodies["/World/Environment/Source/cup_V1"] = dict(
    mass_kg=native["cup_mass_kg"], inertia_tensor_kgm2=native["cup_inertia_tensor_kgm2"]
)
count = 0
for row in data["objects"]:
    body = bodies.get(row["prim_path"])
    if body is None:
        continue
    inertia = np.asarray(body["inertia_tensor_kgm2"], dtype=float)
    if not np.isfinite(inertia).all() or not np.allclose(inertia, inertia.T, atol=1e-7):
        raise ValueError("Invalid native inertia tensor")
    values, axes = np.linalg.eigh((inertia + inertia.T) / 2)
    if np.min(values) <= 0 or not np.isfinite(body["mass_kg"]) or body["mass_kg"] <= 0:
        raise ValueError("Non-positive physical mass/inertia")
    row.update(
        mass_kg=body["mass_kg"],
        inertia_tensor_kgm2=inertia.tolist(),
        inertia_diagonal_kgm2=values.tolist(),
        inertia_principal_axes_columns_local=axes.tolist(),
        mass_status="native_physx_initialized",
        inertia_status="native_physx_initialized",
        native_evidence=str(Path(args.native).resolve()),
        reviewed=True,
        reviewed_scope="USD/native configuration consistency; not real-material calibration",
    )
    count += 1
data.update(
    native_mass_inertia_verified_objects=count,
    native_evidence_sha256=digest_file(args.native),
    cup_settle_test=dict(
        report=str(Path(args.settle).resolve()),
        sha256=digest_file(args.settle),
        status=settle["status"],
        checks=settle.get("checks"),
        all_scene_penetrations_certified=settle["all_scene_penetrations_certified"],
    ),
    status="composed_usd_and_native_mass_distribution_audited",
)
jsonschema.Draft202012Validator(read_json(args.schema)).validate(data)
write_json(args.out, data)
print(dict(schema_valid=True, native_objects=count, static_masses_not_invented=True))
