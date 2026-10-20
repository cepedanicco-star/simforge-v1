"""Validate and export RGB-D intrinsics/extrinsics for an existing capture."""

# ruff: noqa: E402
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.ue_isaac_pipeline.calibration import export_calibration

print(export_calibration(sys.argv[1]))
