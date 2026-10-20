"""Choose auxiliary review viewpoints with a conservative 2-D clear sightline."""

import numpy as np


def review_camera_eyes(positions, navigation_map):
    positions = np.asarray(positions, dtype=float)
    grid = navigation_map["clearance_m"]
    origin = np.asarray(navigation_map["origin_xy"], dtype=float)
    resolution = float(navigation_map["resolution_m"])
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
        raise ValueError("Expected finite world XYZ body positions")

    def clearance(points):
        cells = np.floor((points - origin) / resolution).astype(int)
        valid = (cells[:, 0] >= 0) & (cells[:, 1] >= 0)
        valid &= (cells[:, 0] < grid.shape[1] - 1) & (cells[:, 1] < grid.shape[0] - 1)
        values = np.zeros(len(points))
        x, y = cells[valid].T
        values[valid] = np.minimum.reduce([grid[y, x], grid[y + 1, x], grid[y, x + 1], grid[y + 1, x + 1]])
        return values

    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    directions = np.c_[np.sin(angles), np.cos(angles)]
    eyes, previous_offset = [], np.array([0.0, 1.5])
    for position in positions:
        candidates = []
        for radius in (1.5, 1.35, 1.7):
            for direction in directions:
                offset = direction * radius
                eye_xy = position[:2] + offset
                eye_clearance = float(clearance(eye_xy[None])[0])
                if eye_clearance < 0.15:
                    continue
                ray = position[:2] + np.linspace(0, 1, 48)[:, None] * offset
                if clearance(ray).min() < 0.025:
                    continue
                score = np.linalg.norm(offset - previous_offset) + 0.15 * (1 - direction[1])
                score += 0.03 / max(eye_clearance, 0.15) + abs(radius - 1.5)
                candidates.append((float(score), eye_xy, offset))
        if not candidates:
            raise ValueError(f"No clear auxiliary camera viewpoint near {position.tolist()}")
        _, xy, previous_offset = min(candidates, key=lambda row: row[0])
        eyes.append([*xy, position[2] + 0.7])
    return np.asarray(eyes)
