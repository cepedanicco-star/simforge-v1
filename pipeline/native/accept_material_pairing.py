"""Independently re-run the exact live UE camera criterion after all six writes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.simforge.materials.camera_pairing import camera_pairing  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", help="Fresh audit path; never replace an earlier receipt")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf8"))
    root = Path(config["output"])
    output = Path(args.out) if args.out else root / "camera-pairing-acceptance.json"
    if output.exists():
        raise FileExistsError(output)
    ue = Path(config.get("ue_output", str(root / "ue")))
    checks = []
    try:
        if (ue / "failure.json").exists():
            raise ValueError("UE live gate failed; see " + str(ue / "failure.json"))
        report = json.loads((ue / "capture-report.json").read_text(encoding="utf8"))
        assert report["status"] == "PASS" and len(report["records"]) == len(config["views"]) == 6
        for view, observed in zip(config["views"], report["records"], strict=True):
            assert view["source_frame"] == observed["source_frame"]
            check = camera_pairing(view, observed)
            checks.append(check)
            for timing in ("before-render", "before-write"):
                saved = json.loads((ue / f"gate-{view['source_frame']:03d}-{timing}.json").read_text(encoding="utf8"))
                assert saved["status"] == "PASS"
                if timing == "before-write":
                    assert saved == observed["pre_write_gate"]
                    # UE Python and audit Python may round the last ulp of sum
                    # differently near acos(1). All five user limits still apply
                    # independently to both results; compare raw axes exactly.
                    for key in ("actual_forward", "actual_up", "expected_forward", "expected_up", "limits", "violations", "status"):
                        assert saved[key] == check[key]
                    check["live_vs_audit_rounding_difference"] = {
                        key: abs(saved[key]-check[key]) for key in check["limits"]}
                    assert all(saved[key] <= limit for key, limit in check["limits"].items())
            assert check["status"] == "PASS"
            image = ue / observed["image"]
            with Image.open(image) as picture:
                assert picture.size == (2048, 1152)
                picture.verify()
            check["image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
        fixture = json.loads((ue / "fixture.json").read_text(encoding="utf8"))
        assert fixture["lighting"] == config["lights"]
        assert len(fixture["actual_lights"]) == len(config["lights"]) == 3
        for measured, requested in zip(fixture["actual_lights"], config["lights"], strict=True):
            assert measured["direction_error_degrees"] <= 1e-4
            assert measured["intensity"] == requested["illuminance_lux"]
            assert measured["cast_shadows"] is False
        result = dict(status="PASS", passed=6, expected=6, checks=checks, shared_function="pipeline.simforge.materials.camera_pairing.camera_pairing",
                      delta_e2000=None, ssim=None, next="Isaac native capture now permitted")
    except Exception as error:
        result = dict(status="FAIL_STOPPED", checks=checks, error=repr(error), delta_e2000=None, ssim=None)
    output.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
