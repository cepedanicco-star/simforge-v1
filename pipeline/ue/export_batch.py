"""Unreal -ExecutePythonScript entry point. AISIM_EXPORT_DIR selects a new directory."""

import os
import sys
from pathlib import Path

import unreal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_scene import export_scene

try:
    export_scene(os.environ["AISIM_EXPORT_DIR"])
except Exception:
    import traceback

    unreal.log_error(traceback.format_exc())
    raise
