"""Select original measured states for a transparent video time mapping."""

import numpy as np


def recording_frames(count, control_hz=50, fps=25, start_seconds=0, duration=None):
    if control_hz <= 0 or fps <= 0 or start_seconds < 0:
        raise ValueError("Positive rates and a nonnegative start time are required")
    start = round(start_seconds * control_hz)
    if count <= start:
        raise ValueError("No measured states remain after the requested start")
    if duration is None:
        if control_hz % fps:
            raise ValueError("Realtime capture requires an integer rate ratio")
        indices = np.arange(start, count, control_hz // fps)
    else:
        frames = round(duration * fps)
        if frames < 2 or frames > count - start:
            raise ValueError("Requested duration requires invalid or duplicated measured states")
        indices = np.rint(np.linspace(start, count - 1, frames)).astype(int)
    return indices
