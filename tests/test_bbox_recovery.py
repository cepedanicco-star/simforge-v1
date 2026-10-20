import numpy as np
import pytest

from pipeline.simforge.sensors.annotators import wait_for_bbox
from pipeline.simforge.sensors.pass_alignment import verify


class SequenceSensor:
    def __init__(self, sizes):
        self.sizes = iter(sizes)

    def get_data(self):
        return {"data": np.zeros(next(self.sizes))}


def test_ten_empty_buffers_fail_without_returning_frame():
    records, steps = [], []
    with pytest.raises(ValueError, match="10 consecutive"):
        wait_for_bbox({"bounding_box_3d": SequenceSensor([0] * 10)},
                      lambda: steps.append(1), records.append)
    assert len(steps) == len(records) == 10
    assert all(row["is_none"] is False and row["size"] == 0 for row in records)


def test_last_allowed_buffer_can_recover():
    records = []
    values = wait_for_bbox({"bounding_box_3d": SequenceSensor([0] * 9 + [2, 2])},
                           lambda: None, records.append)
    assert len(records) == 10 and values["bounding_box_3d"]["data"].size == 2


def test_alignment_requires_exact_k_and_strict_pose_tolerance():
    camera = dict(fx=800., fy=800., cx=640., cy=360., width=1280, height=720,
                  extrinsic_world_to_cam=np.eye(4).tolist())
    poses = {"/World/cup": np.eye(4).tolist()}
    assert verify(camera, camera, poses, poses)["passed"]
    with pytest.raises(ValueError, match="alignment failed"):
        verify(camera, dict(camera, fx=np.nextafter(800., 801.)), poses, poses)
    moved = np.eye(4)
    moved[3, 0] = 1e-4
    with pytest.raises(ValueError, match="alignment failed"):
        verify(camera, camera, poses, {"/World/cup": moved.tolist()})
    with pytest.raises(ValueError, match="alignment failed"):
        verify(camera, camera, poses, {})
