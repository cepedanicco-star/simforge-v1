"""Versioned render/run presets with explicit timing and no hidden fallbacks."""

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .io import read_json


@dataclass
class Preset:
    schema_version: int = 1
    width: int = 2048
    height: int = 1152
    fps: int = 30
    duration_seconds: int = 25
    physics_hz: int = 200
    rt_subframes: int = 4
    headless: bool = True
    preview_light: str = "auto"
    device: str = "cpu"
    allow_provisional: bool = True
    robot_path: str = "/World/Humanoid"
    robot_position: list = field(default_factory=lambda: [0.0, 0.0, 1.05])
    camera_position: list = field(default_factory=lambda: [5.0, -6.0, 3.0])
    camera_target: list = field(default_factory=lambda: [0.0, 0.0, 1.0])
    robot_min_height: float = 0.55
    max_body_speed: float = 20.0
    max_body_distance: float = 1000.0
    camera_clipping: list = field(default_factory=lambda: [0.05, 1000.0])
    depth_preview_far_m: float = 15.0
    layers: list = field(default_factory=lambda: ["rgb", "depth", "semantic", "instance", "normals"])
    commands: list = field(default_factory=lambda: [{"until_seconds": 25.0, "velocity": [0.0, 0.0, 0.0]}])

    def validate(self):
        if self.schema_version != 1:
            raise ValueError("Unsupported preset schema_version")
        for name in ("width", "height", "fps", "duration_seconds", "physics_hz", "rt_subframes"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if (self.width, self.height) != (2048, 1152):
            raise ValueError("V1 delivery preset requires 2048x1152")
        if not 20 <= self.duration_seconds <= 35:
            raise ValueError("duration_seconds must be between 20 and 35")
        if self.fps not in (24, 25, 30, 50, 60) or not self.fps <= self.physics_hz <= 1000:
            raise ValueError("Invalid fps / physics_hz")
        if not 1 <= self.rt_subframes <= 64 or self.device not in ("cpu", "cuda"):
            raise ValueError("Invalid renderer subframes / physics device")
        if type(self.allow_provisional) is not bool or type(self.headless) is not bool:
            raise ValueError("allow_provisional/headless must be boolean")
        if self.preview_light not in ("auto", "indoor", "off"):
            raise ValueError("preview_light must be auto, indoor or off")
        if not self.robot_path.startswith("/World/") or not all(
            c.isalnum() or c in "_/" for c in self.robot_path
        ):
            raise ValueError("robot_path must be a prim path under /World")
        for name in ("robot_position", "camera_position", "camera_target"):
            v = getattr(self, name)
            if len(v) != 3 or not all(isinstance(x, (float, int)) and math.isfinite(x) for x in v):
                raise ValueError(f"Invalid {name}")
        if self.camera_position == self.camera_target:
            raise ValueError("Camera eye and target must differ")
        if not self.layers or not set(self.layers) <= {"rgb", "depth", "semantic", "instance", "normals"}:
            raise ValueError("Unsupported layers")
        if "rgb" not in self.layers or len(self.layers) != len(set(self.layers)):
            raise ValueError("One RGB layer is required; duplicate layers are not allowed")
        for name in ("robot_min_height", "max_body_speed", "max_body_distance", "depth_preview_far_m"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"Invalid {name}")
        if (
            len(self.camera_clipping) != 2
            or not all(math.isfinite(v) for v in self.camera_clipping)
            or not 0 < self.camera_clipping[0] < self.camera_clipping[1]
        ):
            raise ValueError("Invalid camera clipping range")
        end = 0.0
        if not self.commands:
            raise ValueError("A command schedule is required")
        for c in self.commands:
            if set(c) != {"until_seconds", "velocity"}:
                raise ValueError("Command needs until_seconds and velocity")
            if not math.isfinite(c["until_seconds"]) or c["until_seconds"] <= end:
                raise ValueError("Command times must increase")
            end = c["until_seconds"]
            if len(c["velocity"]) != 3 or not all(math.isfinite(x) for x in c["velocity"]):
                raise ValueError("Velocity is [forward_m_s, lateral_m_s, yaw_rad_s]")
        if end < self.duration_seconds:
            raise ValueError("Command schedule must cover the entire video")
        return self

    def to_dict(self):
        return asdict(self)


def load_preset(path):
    values = read_json(Path(path))
    unknown = set(values) - set(Preset.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown preset fields: {sorted(unknown)}")
    return Preset(**values).validate()


def frame_ticks(frame, physics_hz, fps):
    """Integer tick boundaries: exact total time even for 200 Hz physics / 30 fps video."""
    if frame < 0:
        raise ValueError("Negative frame index")

    # Integer round-half-up, deliberately independent of floating point/banker's rounding.
    def boundary(n):
        return (2 * n * physics_hz + fps) // (2 * fps)

    return boundary(frame + 1) - boundary(frame)


def command_at(preset, seconds):
    for cmd in preset.commands:
        if seconds < cmd["until_seconds"]:
            return cmd["velocity"]
    return preset.commands[-1]["velocity"]
