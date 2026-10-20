"""Verify measured right/left turns and body-forward apartment walking."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.ue_isaac_pipeline.io import read_json, write_json, digest_file  # noqa: E402


def audit(recording, out):
    recording, out = Path(recording).resolve(strict=True), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(recording / "physics-body-states.npz", allow_pickle=False)
    report = read_json(recording / "probe-report.json")
    root = data["states"][:, data["body_names"].tolist().index("pelvis")].astype(float)
    q = root[:, 3:] / np.linalg.norm(root[:, 3:], axis=1)[:, None]
    w, x, y, z = q.T
    yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    phases, command = data["phases"], data["body_commands"]
    turns = {}
    for name in ["turn_right_with_cup", "turn_left_to_desk"]:
        indices = np.flatnonzero(phases == name)
        turns[name] = (
            dict(samples=len(indices), angle_deg=float(np.degrees(yaw[indices[-1]] - yaw[indices[0]])))
            if len(indices)
            else dict(samples=0, angle_deg=None)
        )
    window = int(data["control_hz"]) // 2
    nav = (phases[:-window] == "navigate_apartment_with_cup") & (
        phases[window:] == "navigate_apartment_with_cup"
    )
    displacement = root[window:, :2] - root[:-window, :2]
    length = np.linalg.norm(displacement, axis=1)
    nav &= length > 0.12 * window / int(data["control_hz"])
    heading = (yaw[window:] + yaw[:-window]) / 2
    facing = np.c_[np.cos(heading), np.sin(heading)]
    alignment = (
        np.einsum("ij,ij->i", displacement[nav], facing[nav]) / length[nav] if nav.any() else np.array([])
    )
    navigation_commands = command[phases == "navigate_apartment_with_cup"]
    right, left = (turns[k]["angle_deg"] for k in turns)
    final_error = float(abs(np.arctan2(np.sin(yaw[-1] + np.pi / 2), np.cos(yaw[-1] + np.pi / 2))))
    checks = dict(
        full_physical_transfer_completed=report.get("status") == "completed"
        and report.get("full_transfer_completed") is True,
        right_turn_measured=right is not None and -120 <= right <= -60,
        left_turn_measured=left is not None and 60 <= left <= 120,
        forward_translation_measured=bool(
            len(alignment) and np.mean(alignment > 0.8) >= 0.85 and np.median(alignment) >= 0.90
        ),
        no_commanded_crab_walking=bool(
            len(navigation_commands)
            and np.max(np.abs(navigation_commands[:, 1])) < 1e-7
            and np.min(navigation_commands[:, 0]) >= -1e-7
        ),
        final_facing_desk=final_error < np.radians(10),
        no_unexpected_environment_contacts=report.get("external_contact_check_passed") is True,
    )
    checks = {key: bool(value) for key, value in checks.items()}
    result = dict(
        status="completed" if all(checks.values()) else "failed",
        checks=checks,
        source_recording=str(recording),
        source_npz_sha256=digest_file(recording / "physics-body-states.npz"),
        turns=turns,
        moving_half_second_windows=len(alignment),
        forward_alignment_median=float(np.median(alignment)) if len(alignment) else None,
        forward_aligned_window_fraction=float(np.mean(alignment > 0.8)) if len(alignment) else None,
        final_desk_heading_error_deg=float(np.degrees(final_error)),
        maximum_navigation_lateral_command_m_s=float(np.max(np.abs(navigation_commands[:, 1])))
        if len(navigation_commands)
        else None,
        learned_high_level_policy=report.get("learned_high_level_navigation"),
        note="Half-second measured displacement vs body heading for speeds above 0.12 m/s; small desk docking balance corrections are recorded separately. No new full-body RL claim.",
    )
    write_json(out / "forward-motion-audit.json", result)
    write_json(recording / "forward-motion-audit.json", result)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(root[:, 0], root[:, 1], color="#246a9c", label="Measured pelvis")
    every = np.arange(0, len(root), 50)
    axes[0].quiver(
        root[every, 0],
        root[every, 1],
        np.cos(yaw[every]),
        np.sin(yaw[every]),
        angles="xy",
        scale_units="xy",
        scale=5,
        width=0.004,
        color="#d28a24",
    )
    axes[0].plot(data["cup"][:, 0], data["cup"][:, 1], color="#57905a", alpha=0.7, label="Measured cup")
    axes[0].set(
        xlabel="World X (m)", ylabel="World Y (m)", title="Position and actual body-forward direction"
    )
    axes[0].axis("equal")
    axes[0].grid(alpha=0.2)
    axes[0].legend()
    time = np.arange(len(root)) / int(data["control_hz"])
    axes[1].plot(time, np.degrees(yaw), color="#246a9c")
    for name, color in [("turn_right_with_cup", "#e2bd75"), ("turn_left_to_desk", "#7db58b")]:
        ids = np.flatnonzero(phases == name)
        if len(ids):
            axes[1].axvspan(time[ids[0]], time[ids[-1]], color=color, alpha=0.3, label=name)
    axes[1].set(
        xlabel="Physical time (s)",
        ylabel="Unwrapped body yaw (degrees)",
        title="Measured turns; original pickup heading is -90 degrees",
    )
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "heading-and-route.png", dpi=160)
    plt.close(fig)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recording", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    result = audit(args.recording, args.out)
    print(result["status"], result["checks"])
    if result["status"] != "completed":
        raise SystemExit(1)
