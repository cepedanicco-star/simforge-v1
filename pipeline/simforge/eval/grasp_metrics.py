"""Fixed M2 grasp predicate; undefined measurements stay null, never zero-filled."""

import numpy as np


def rotation_matrix(q):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    values = [
        1 - 2 * (y * y + z * z),
        2 * (x * y - z * w),
        2 * (x * z + y * w),
        2 * (x * y + z * w),
        1 - 2 * (x * x + z * z),
        2 * (y * z - x * w),
        2 * (x * z - y * w),
        2 * (y * z + x * w),
        1 - 2 * (x * x + y * y),
    ]
    return np.stack(values, axis=-1).reshape(q.shape[:-1] + (3, 3))


def longest_true_seconds(mask, dt):
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8)
    changes = np.flatnonzero(np.diff(padded))
    return float(np.max(changes[1::2] - changes[::2]) * dt) if len(changes) else 0.0


def evaluate(data, config):
    """Success iff lift>=50mm, first 0.5s slip<=5mm, continuous airborne grasp>=2s.

    Grasp onset is the first opposed cup/finger contact while off its original
    support. Retention counts samples at >=50mm with opposed contact and no
    cabinet support. The same origin/frame and thresholds apply to every trial.
    """
    thresholds = config["metrics"]
    expected = {"lift_m": 0.05, "max_slip_mm": 5, "slip_window_s": 0.5, "hold_s": 2}
    if any(thresholds[key] != value for key, value in expected.items()):
        raise ValueError("Changing the approved M2 success predicate requires an explicit new protocol")
    times, cup, palm = (np.asarray(data[key], dtype=float) for key in ("time_s", "cup_state", "palm_state"))
    forces = np.asarray(data["finger_forces_N"], dtype=float)
    support = np.linalg.norm(np.asarray(data["support_force_N"], dtype=float), axis=-1)
    if len(times) < 2 or not all(np.isfinite(v).all() for v in (times, cup, palm, forces, support)):
        raise ValueError("Invalid/incomplete physical measurement arrays")
    dt = float(np.median(np.diff(times)))
    if not np.allclose(np.diff(times), 1 / config["physics_hz"], rtol=1e-7, atol=1e-10):
        raise ValueError("Physical samples must have the declared uniform timestep")
    if any(np.any(np.linalg.norm(p[:, 3:7], axis=-1) < 0.99) for p in (cup, palm)):
        raise ValueError("Invalid rigid-body orientation")
    names = [str(x) for x in data["sensor_names"]]
    magnitudes = np.linalg.norm(forces, axis=-1)
    thumb = [i for i, name in enumerate(names) if "thumb" in name]
    other = [i for i, name in enumerate(names) if "index" in name or "middle" in name]
    if not thumb or not other:
        raise ValueError("The same declared thumb/index/middle sensors are required")
    opposed = (magnitudes[:, thumb].sum(1) >= thresholds["minimum_opposed_force_N"]) & (
        magnitudes[:, other].sum(1) >= thresholds["minimum_opposed_force_N"]
    )
    initial_z = float(np.asarray(data["initial_cup_pose"])[2])
    lift = cup[:, 2] - initial_z
    unsupported = support < thresholds["maximum_support_force_N"]
    established = opposed & unsupported & (lift >= thresholds["takeoff_m"])
    onset = int(np.flatnonzero(established)[0]) if established.any() else None
    slip = None
    if onset is not None:
        window = (times >= times[onset]) & (times <= times[onset] + thresholds["slip_window_s"] + 1e-9)
        if times[-1] + dt / 2 >= times[onset] + thresholds["slip_window_s"]:
            # Cup origin expressed in the measured palm frame, including palm rotation.
            relative = np.einsum(
                "nij,nj->ni", np.swapaxes(rotation_matrix(palm[:, 3:7]), 1, 2), cup[:, :3] - palm[:, :3]
            )
            slip = float(np.linalg.norm(relative[window] - relative[onset], axis=1).max() * 1000)
    retained = opposed & unsupported & (lift >= thresholds["lift_m"])
    hold = longest_true_seconds(retained, dt)
    takeoffs = np.flatnonzero((lift >= thresholds["takeoff_m"]) & unsupported)
    takeoff = int(takeoffs[0]) if len(takeoffs) else None
    drop, settle = None, None
    if takeoff is not None:
        returns = np.flatnonzero((np.arange(len(times)) > takeoff) & ~unsupported)
        if len(returns):
            landing = int(returns[0])
            drop = float(times[landing] - times[takeoff])
            stationary = (np.linalg.norm(cup[:, 7:10], axis=-1) <= thresholds["settle_linear_speed_m_s"]) & (
                np.linalg.norm(cup[:, 10:13], axis=-1) <= thresholds["settle_angular_speed_rad_s"]
            )
            required = int(np.ceil(thresholds["settle_window_s"] / dt - 1e-8))
            for i in range(landing, len(times) - required + 1):
                if stationary[i : i + required].all():
                    settle = float(times[i + required - 1] + dt - times[landing])
                    break
    success = bool(
        float(lift.max()) >= thresholds["lift_m"]
        and slip is not None
        and slip <= thresholds["max_slip_mm"]
        and hold + 1e-8 >= thresholds["hold_s"]
    )
    reasons = []
    if float(lift.max()) < thresholds["lift_m"]:
        reasons.append("lift_below_50mm")
    if slip is None:
        reasons.append("no_complete_established_grasp_slip_window")
    elif slip > thresholds["max_slip_mm"]:
        reasons.append("slip_above_5mm")
    if hold + 1e-8 < thresholds["hold_s"]:
        reasons.append("retention_below_2s")
    return dict(
        success=success,
        slip_distance_mm=slip,
        peak_contact_force_N=float(magnitudes.sum(1).max()),
        drop_time_s=drop if not success else None,
        settle_time_s=settle,
        maximum_lift_mm=float(lift.max() * 1000),
        longest_retention_s=hold,
        grasp_onset_s=float(times[onset]) if onset is not None else None,
        failure_reasons=reasons,
        force_definition=config["contact"]["force_definition"],
    )
