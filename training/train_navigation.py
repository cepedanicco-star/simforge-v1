"""Train and evaluate a high-level heading/forward imitation policy on CPU.

This learns a geometric controller's state-to-velocity map. Physical G1/cup
evaluation remains a separate mandatory step in the full apartment.
"""

import argparse
import copy
import os
import sys
from pathlib import Path

sdk = Path(sys.executable).parent
handles = []
if os.name == "nt":
    for directory in [sdk / "Library/bin", sdk / "Lib/site-packages/torch/lib"]:
        if directory.exists():
            os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
            handles.append(os.add_dll_directory(str(directory)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.ue_isaac_pipeline.learned_navigation import LearnedForwardGovernor, teacher  # noqa: E402
from pipeline.ue_isaac_pipeline.io import write_json, digest_file  # noqa: E402


def states(seed, count, blended_turns=False, velocity_feedback=False):
    rng = np.random.default_rng(seed)
    x = rng.uniform([-1, -1.5, 0, 0], [1, 1.5, 0.55, 1], (count, 4)).astype(np.float32)
    x[:, 3] = rng.integers(0, 2, count)
    # Emphasize final alignment and the forward/turn transition.
    x[: count // 2, 0] *= 0.20
    if velocity_feedback:
        measured = rng.uniform(-0.15, 0.65, (count, 1)).astype(np.float32)
        measured[: count // 3] = rng.uniform(-0.03, 0.03, (count // 3, 1))
        x = np.c_[x, measured]
    return x, teacher(x, blended_turns=blended_turns).astype(np.float32)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--blended-turns", action="store_true")
    p.add_argument("--velocity-feedback", action="store_true")
    args = p.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "training-report.json").exists():
        raise FileExistsError(out)
    torch.set_num_threads(4)
    torch.manual_seed(418)
    train_x, train_y = states(418, 60000, args.blended_turns, args.velocity_feedback)
    val_x, val_y = states(419, 12000, args.blended_turns, args.velocity_feedback)
    tx, ty = torch.from_numpy(train_x), torch.from_numpy(train_y)
    vx, vy = torch.from_numpy(val_x), torch.from_numpy(val_y)
    model = torch.nn.Sequential(
        torch.nn.Linear(train_x.shape[1], 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    curve = []
    best_score, best_epoch, best_state, best_optimizer = float("inf"), 0, None, None
    for epoch in range(args.epochs):
        order = torch.randperm(len(tx))
        for indices in order.split(2048):
            predicted = model(tx[indices])
            loss = ((predicted - ty[indices]) ** 2 * torch.tensor([10.0, 1.0])).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                error = (model(vx) - vy).abs().mean(0).tolist()
                score = float(((model(vx) - vy) ** 2 * torch.tensor([10.0, 1.0])).mean())
            if score < best_score:
                best_score, best_epoch = score, epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                best_optimizer = copy.deepcopy(optimizer.state_dict())
            curve.append(
                dict(epoch=epoch + 1, validation_forward_mae_m_s=error[0], validation_yaw_mae_rad_s=error[1])
            )
            print(curve[-1], flush=True)
    model.load_state_dict(best_state)
    optimizer.load_state_dict(best_optimizer)
    arrays = {}
    for i, layer in enumerate([model[0], model[2], model[4]]):
        arrays[f"weight_{i}"] = layer.weight.detach().numpy()
        arrays[f"bias_{i}"] = layer.bias.detach().numpy()
    filename = out / "forward-governor.npz"
    np.savez_compressed(filename, **arrays)
    torch.save(
        dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epochs=args.epochs),
        out / "checkpoint.pt",
    )
    actor = LearnedForwardGovernor(filename)
    predictions = np.asarray(
        [
            actor.predict(float(x[0]) * np.pi, x[1], x[2], x[3] > 0.5, x[4] if len(x) == 5 else 0)
            for x in val_x
        ]
    )
    error = np.abs(predictions - val_y)
    with torch.no_grad():
        float32_reference = model(vx).numpy()
        expected = model.double()(vx.double()).numpy()
        arithmetic_difference = float(np.max(np.abs(float32_reference - expected)))
    expected = np.clip(expected, [0, -1], np.c_[np.where(val_x[:, 3] > 0.5, 0.25, 0.5), np.ones(len(val_x))])
    export_error = float(np.max(np.abs(expected - predictions)))
    result = dict(
        status="completed",
        training="Supervised behavior cloning of geometric velocity teacher",
        teacher_version="pd_with_measured_forward_velocity_recovery_v04"
        if args.velocity_feedback
        else "pd_with_navigation_forward_arc_030_v04"
        if args.blended_turns
        else "pd_with_phase_turn_creep_v03",
        features=[
            "wrapped_yaw_error/pi",
            "measured_yaw_rate_rad_s",
            "requested_forward_speed_m_s",
            "turn_mode",
        ]
        + (["measured_forward_speed_m_s"] if args.velocity_feedback else []),
        outputs=["forward_speed_m_s", "yaw_rate_rad_s"],
        train_samples=len(tx),
        validation_samples=len(vx),
        train_seed=418,
        validation_seed=419,
        epochs=args.epochs,
        best_epoch=best_epoch,
        checkpoint_selection="Independent synthetic-state validation weighted MSE, before physical evaluation",
        validation_mae=error.mean(0).tolist(),
        numpy_torch_export_max_error=export_error,
        inference_arithmetic="float64 NumPy and float64 Torch reference using exported float32-trained weights",
        float32_training_reference_max_deviation=arithmetic_difference,
        architecture=[train_x.shape[1], 64, 64, 2],
        model=str(filename),
        model_sha256=digest_file(filename),
        physical_task_evaluation_completed=False,
        new_leg_or_grasp_policy_trained=False,
        scene_generalization_evaluated=False,
        curve=curve,
        source_sha256=digest_file(__file__),
    )
    if error[:, 0].mean() > 0.015 or error[:, 1].mean() > 0.04 or export_error > 2e-6:
        result["status"] = "failed_validation"
    write_json(out / "training-report.json", result)
    print(result["status"], result["validation_mae"], flush=True)
    if result["status"] != "completed":
        raise SystemExit(1)
