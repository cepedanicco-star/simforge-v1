"""Measured-state path following helpers for the apartment cup task."""

import numpy as np


class PolylineFollower:
    def __init__(self, points, lookahead=0.25, maximum_speed=0.45):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError("A finite XY path with at least two points is required")
        keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6]
        self.points = points[keep]
        if len(self.points) < 2:
            raise ValueError("Path must have nonzero length")
        self.delta = np.diff(self.points, axis=0)
        self.lengths = np.linalg.norm(self.delta, axis=1)
        self.arc = np.r_[0.0, np.cumsum(self.lengths)]
        self.progress, self.lookahead, self.maximum_speed = 0.0, lookahead, maximum_speed

    def sample(self, distance):
        distance = float(np.clip(distance, 0, self.arc[-1]))
        i = min(np.searchsorted(self.arc, distance, side="right") - 1, len(self.lengths) - 1)
        ratio = (distance - self.arc[i]) / self.lengths[i]
        return self.points[i] + ratio * self.delta[i]

    def update(self, point):
        point = np.asarray(point, dtype=np.float64)
        projection = np.clip(
            np.einsum("ij,ij->i", point - self.points[:-1], self.delta) / self.lengths**2, 0, 1
        )
        nearest = self.points[:-1] + projection[:, None] * self.delta
        arc = self.arc[:-1] + projection * self.lengths
        distances = np.linalg.norm(point - nearest, axis=1)
        distances[arc < self.progress - 0.10] = np.inf
        i = int(np.argmin(distances))
        if not np.isfinite(distances[i]):
            raise RuntimeError("Cannot project measured position onto the remaining route")
        self.progress = max(self.progress, float(arc[i]))
        goal = self.sample(self.progress + self.lookahead)
        velocity = 2.0 * (goal - point)
        speed = np.linalg.norm(velocity)
        if speed > self.maximum_speed:
            velocity *= self.maximum_speed / speed
        return dict(
            goal=goal,
            velocity=velocity,
            cross_track_error=float(distances[i]),
            remaining=float(self.arc[-1] - self.progress),
            endpoint_error=float(np.linalg.norm(point - self.points[-1])),
        )
