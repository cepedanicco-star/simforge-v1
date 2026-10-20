# SPDX-License-Identifier: Apache-2.0
# Observation/action contract adapted from Unitree Robotics, Copyright 2025.
"""Named-joint adapter for Unitree's 910-observation / 12-action G1 policy.

Observation/action contract adapted from Unitree Robotics (2025), Apache-2.0:
https://github.com/unitreerobotics/unitree_sim_isaaclab/blob/e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc/action_provider/action_provider_wh_dds.py
The upstream policy and robot assets remain external and retain their licenses.
"""

import numpy as np

LEGS = [
    f"{side}_{part}_joint"
    for part in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    for side in ("left", "right")
]
ARMS = [
    f"{side}_{part}_joint"
    for side in ("left", "right")
    for part in (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    )
]
BODY_HISTORY_ORDER = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]


class BlendedPolicySession:
    """Blend two verified vendor policy outputs without resetting observation history."""

    def __init__(self, primary, secondary):
        self.primary, self.secondary = primary, secondary
        self.weight = 0.0
        for session in (primary, secondary):
            if (
                len(session.get_inputs()) != 1
                or session.get_inputs()[0].shape != [1, 910]
                or len(session.get_outputs()) != 1
                or session.get_outputs()[0].shape != [1, 12]
            ):
                raise ValueError("Both policies must match the same 910/12 observation contract")

    def get_inputs(self):
        return self.primary.get_inputs()

    def get_outputs(self):
        return self.primary.get_outputs()

    def run(self, output_names, inputs):
        weight = float(self.weight)
        if not np.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("Policy blend weight must be in [0,1]")
        if weight == 0:
            return self.primary.run(output_names, inputs)
        secondary_inputs = {self.secondary.get_inputs()[0].name: inputs[self.primary.get_inputs()[0].name]}
        secondary = self.secondary.run(None, secondary_inputs)
        if weight == 1:
            return secondary
        primary = self.primary.run(output_names, inputs)
        return [(1 - weight) * np.asarray(primary[0]) + weight * np.asarray(secondary[0])]


class UnitreeG1WBC:
    def __init__(self, joint_names, default_positions, session):
        self.joint_names = list(joint_names)
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("Joint names must be unique")
        missing = set(BODY_HISTORY_ORDER) - set(self.joint_names)
        if missing:
            raise ValueError(f"Asset does not match the 29-body-joint G1 policy: {sorted(missing)}")
        self.leg_ids = [self.joint_names.index(name) for name in LEGS]
        self.arm_ids = [self.joint_names.index(name) for name in ARMS]
        self.obs_ids = self.leg_ids + self.arm_ids
        self.history_ids = [self.joint_names.index(name) for name in BODY_HISTORY_ORDER]
        self.default = np.asarray(default_positions, dtype=np.float32).copy()
        if self.default.shape != (len(self.joint_names),):
            raise ValueError("Default position vector does not match the asset")
        self.session = session
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if (
            len(inputs) != 1
            or inputs[0].shape != [1, 910]
            or len(outputs) != 1
            or outputs[0].shape != [1, 12]
        ):
            raise ValueError("Expected exact upstream ONNX input [1,910], output [1,12]")
        self.input_name = inputs[0].name
        self.reset()

    def reset(self):
        self.history = np.zeros((10, 91), dtype=np.float32)
        self.previous_raw_body_action = np.zeros(29, dtype=np.float32)
        self.initialized = False

    def step(
        self,
        position,
        velocity,
        angular_velocity_body,
        gravity_body,
        command,
        arm_targets=None,
        *,
        upper_observation_scale=1.0,
        waist_targets=None,
    ):
        q, dq = np.asarray(position, dtype=np.float32), np.asarray(velocity, dtype=np.float32)
        angular = np.asarray(angular_velocity_body, dtype=np.float32)
        gravity, command = np.asarray(gravity_body, dtype=np.float32), np.asarray(command, dtype=np.float32)
        if (
            q.shape != self.default.shape
            or dq.shape != q.shape
            or angular.shape != (3,)
            or gravity.shape != (3,)
            or command.shape != (4,)
        ):
            raise ValueError("Observation fields do not match the verified G1 contract")
        if not np.isfinite(upper_observation_scale) or not 0 <= upper_observation_scale <= 1:
            raise ValueError("Upper observation scale must be in [0,1]")
        relative = q[self.obs_ids] - self.default[self.obs_ids]
        observed_velocity = dq[self.obs_ids].copy()
        feedback = self.previous_raw_body_action.copy()
        relative[12:] *= upper_observation_scale
        observed_velocity[12:] *= upper_observation_scale
        for i, name in enumerate(BODY_HISTORY_ORDER):
            if name in ARMS:
                feedback[i] *= upper_observation_scale
        obs = np.concatenate(
            (
                angular,
                gravity,
                command,
                relative,
                observed_velocity,
                feedback,
            )
        )
        if obs.shape != (91,) or not np.isfinite(obs).all():
            raise ValueError("Invalid WBC observation")
        if not self.initialized:
            self.history[:] = obs
            self.initialized = True
        else:
            self.history[:-1] = self.history[1:]
            self.history[-1] = obs
        inputs = np.clip(self.history.reshape(1, 910), -100, 100)
        raw = np.asarray(self.session.run(None, {self.input_name: inputs})[0], dtype=np.float32)
        if raw.shape != (1, 12) or not np.isfinite(raw).all():
            raise ValueError("Invalid WBC action")
        target = self.default.copy()
        raw_all = self.default.copy()
        for name, value in (waist_targets or {}).items():
            if name not in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint") or not np.isfinite(
                value
            ):
                raise ValueError("Waist targets must name finite waist joint angles")
            index = self.joint_names.index(name)
            target[index] = raw_all[index] = value
        if arm_targets is not None:
            arms = np.asarray(arm_targets, dtype=np.float32)
            if arms.shape != (14,) or not np.isfinite(arms).all():
                raise ValueError("Upper-body controller must supply named 14-joint targets")
            target[self.arm_ids] = arms
            raw_all[self.arm_ids] = arms
        raw_all[self.leg_ids] = raw[0]
        self.previous_raw_body_action = raw_all[self.history_ids].copy()
        target[self.leg_ids] = np.clip(raw[0], -100, 100) * 0.25 + self.default[self.leg_ids]
        return target, inputs.copy(), raw[0].copy()
