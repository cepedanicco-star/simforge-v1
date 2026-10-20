"""Source-backed pipeline metrics; absent measurements remain null."""
import csv
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess


def load(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def generate(config, output, elapsed_s):
    import yaml
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh report directory; completed reports are immutable")
    output.mkdir(parents=True, exist_ok=True)
    baseline = yaml.safe_load(Path(config["baseline_config"]).read_text(encoding="utf8"))["baseline"]
    m2 = load(Path(config["m2_data"]) / "sweep-report.json")
    # Strict success count is sourced from the actual 45 recorded trials.
    import csv as csv_reader
    with (Path(config["m2_data"]) / "raw.csv").open(encoding="utf-8-sig", newline="") as stream:
        trials = list(csv_reader.DictReader(stream))
    wins = sum(str(row["success"]).lower() in ("true", "1") for row in trials)
    grasp = 100 * wins / len(trials)
    coco = load(Path(config["output"]) / "dataset/coco-report.json")
    m3 = load(config["m3_acceptance"])
    validation = load(Path(config["output"]) / "validation.json")
    regression_file = Path(config["output"]) / "regression.json"
    if not regression_file.is_file():
        raise ValueError('Run pytest and write actual passed/total counts to the local regression receipt before M4 reporting')
    regression = load(regression_file)
    if not isinstance(regression.get('passed'),int) or regression.get('total')!=regression['passed']:
        raise ValueError('M4 requires a complete passing local regression receipt')
    timings = load(Path(config["output"]) / "timings.json")
    render_ms = timings.get("render_end_to_end_s")
    rendered_frames = load(Path(config["output"]) / "render/export-report.json")["frames"]
    render_ms = render_ms * 1000 / rendered_frames if render_ms is not None else None
    values = [
        ("scene_integration_hours", None, "h", None, "NOT_MEASURED: no timed source integration experiment"),
        ("automatic_collision_generation_percent", None, "%", None, "NOT_MEASURED: this module validates existing colliders"),
        ("grasp_success_rate_percent", grasp, "%", baseline["grasp_success_rate_percent"]["value"], "M2 same 45-trial grid; not a new policy or full-transfer evaluation"),
        ("universal_critical_friction", None, "coefficient", baseline["slip_friction_boundary"]["value"], "NOT_IDENTIFIED: M2 non-monotonic grid"),
        ("four_layer_frame_export_ms", render_ms, "ms", None, "End-to-end one-frame export including engine startup, not GPU-only latency"),
        ("simulation_real_time_factor", None, "x", None, "NOT_MEASURED: this CLI did not step physics"),
        ("dataset_frames", coco["images"], "frames", None, "20 original M1 plus 6 reused-seed M3 native images; copied, not regenerated"),
        ("randomized_variants", len(m3["variants"]), "variants", None, "M3 completed native variants"),
        ("regression_tests_passed", regression["passed"], "tests", None, "Executed test evidence, not inferred from files"),
        ("regression_tests_total", regression["total"], "tests", None, "Executed test evidence"),
        ("pose_verified_frames", None, "frames", None, "NOT_MEASURED in this CLI; historical full-transfer audit remains separate"),
        ("validation_checks_passed", sum(row["status"] == "PASS" for row in validation["checks"]), "checks", None, "Current native positive validation"),
    ]
    with (output / "metrics.csv").open("w", encoding="utf8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["name", "value", "unit", "baseline_value", "delta", "status", "scope"])
        for name, value, unit, old, note in values:
            delta = value-old if value is not None and old is not None else None
            writer.writerow([name, "null" if value is None else value, unit, "null" if old is None else old,
                             "null" if delta is None else delta, "NOT_MEASURED" if value is None else "MEASURED", note])
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True).strip()
    repo = Path(__file__).resolve().parents[3]
    git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo))
    env = dict(python=platform.python_version(), engine=importlib.metadata.version("isaacsim"), gpu_and_driver=gpu,
               git_commit=git, git_dirty=dirty, elapsed_s=elapsed_s, scene=config["scene"], new_policy_training=False,
               M2_report_status=m2.get("status"))
    (output / "env.json").write_text(json.dumps(env, indent=2), encoding="utf8")
    text = ("# Pipeline metrics\n\n"
            f"与 baseline apartment 相比，抓取成功率同为 {grasp:.6f}%（{wins}/{len(trials)}），差值0；两列引用同一M2严格网格，不代表一次新性能提升实验。\n\n"
            f"与 baseline 相比，本次增加可供COCO读取的{coco['images']}张图、{coco['annotations']}个真实实例掩码，并串联了6个原生随机化变体；缺少可比旧测量的delta保留null。\n\n"
            "与 baseline 相比，当前仍没有新的行为泛化成功率、实时倍率或接入耗时测量，全部写null/NOT_MEASURED；全任务20回合评测属于下一模块。\n\n"
            "临界摩擦无法从本非单调网格识别（null）。原生单帧端到端耗时包含启动，不是纯渲染器性能。详见metrics.csv的scope和env.json。\n")
    (output / "summary.md").write_text(text, encoding="utf8")
    return dict(status="PASS", metrics=len(values), images=coco["images"], annotations=coco["annotations"], output=str(output))
