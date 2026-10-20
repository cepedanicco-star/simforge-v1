"""A small trained high-level forward/turn controller, independent of leg policy.

Training is supervised imitation of a geometric velocity teacher. This is not a
new whole-body RL policy, visual navigation or proof of scene generalization.
"""

import numpy as np


def teacher(features, blended_turns=False):
    x = np.asarray(features)
    error, rate, requested, turn = x[..., 0] * np.pi, x[..., 1], x[..., 2], x[..., 3]
    forward = requested * np.maximum(0, 1 - np.abs(error) / 0.45)
    creep = np.where(error < 0, 0.20, 0.14) * (np.abs(error) > 0.10)
    forward = np.where(turn > 0.5, creep, forward)
    if blended_turns:
        # The vendor gait can stand still under pure yaw commands. Keep a small
        # forward arc during intermediate path alignment, rather than sending
        # zero forward velocity until an unreachable heading gate is met.
        arc = np.minimum(requested, 0.30) * np.clip((np.abs(error) - 0.25) / 0.20, 0, 1)
        forward = np.where((turn <= 0.5) & (requested > 0.02), np.maximum(forward, arc), forward)
    yaw = np.clip(2 * error - 0.25 * rate, -1, 1)
    yaw = np.where(
        (turn > 0.5) & (np.abs(error) > 0.10),
        np.sign(error) * np.maximum(np.abs(yaw), 0.8),
        yaw,
    )
    if x.shape[-1] == 5:
        measured = x[..., 4]
        stalled = np.clip((0.08 - measured) / 0.08, 0, 1)
        boost = np.minimum(requested, 0.45) * stalled
        forward = np.where(
            (turn <= 0.5) & (requested > 0.12) & (np.abs(error) < 0.9), np.maximum(forward, boost), forward
        )
        yaw_stalled = (turn <= 0.5) & (requested > 0.02) & (np.abs(error) > 0.15) & (np.abs(rate) < 0.08)
        yaw = np.where(yaw_stalled, np.sign(error) * np.maximum(np.abs(yaw), 0.8), yaw)
    return np.stack([forward, yaw], axis=-1)


class LearnedForwardGovernor:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as a:
            self.weights = [
                (a[f"weight_{i}"].astype(np.float64), a[f"bias_{i}"].astype(np.float64)) for i in range(3)
            ]
        self.feature_count = self.weights[0][0].shape[1]
        if self.feature_count not in (4, 5) or [w.shape for w, _ in self.weights] != [
            (64, self.feature_count),
            (64, 64),
            (2, 64),
        ]:
            raise ValueError("Unexpected navigation model architecture")
        if [b.shape for _, b in self.weights] != [(64,), (64,), (2,)] or any(
            not np.isfinite(a).all() for pair in self.weights for a in pair
        ):
            raise ValueError("Invalid navigation model parameters")

    def predict(self, yaw_error, angular_z, requested_speed, turning, measured_forward_speed=0.0):
        features = [yaw_error / np.pi, angular_z, requested_speed, float(turning)]
        if self.feature_count == 5:
            features.append(measured_forward_speed)
        x = np.array(features, dtype=np.float64)
        if not np.isfinite(x).all() or abs(yaw_error) > np.pi + 1e-5:
            raise ValueError("Invalid high-level navigation observation")
        for i, (w, b) in enumerate(self.weights):
            x = x @ w.T + b
            if i < 2:
                x = np.maximum(x, 0)
        return np.clip(x, [0, -1], [0.25 if turning else 0.5, 1])
