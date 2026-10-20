"""Equivalent pinhole lens gauges that avoid Kit's small-focal clamp."""

import math


def equivalent_native_lens(source, minimum_focal=0.5):
    names = ("FocalLength", "HorizontalAperture", "VerticalAperture")
    if any(not math.isfinite(source[n]) or source[n] <= 0 for n in names):
        raise ValueError("Positive finite pinhole lens parameters required")
    scale = max(1.0, minimum_focal / source["FocalLength"])
    values = {name: value * scale for name, value in source.items()}
    return values, scale
