"""Supervise only this material worker, checking reports rather than SDK exit alone."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ue-acceptance", required=True)
    parser.add_argument("--native-python", required=True)
    parser.add_argument("--graycard-config")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf8"))
    root = Path(cfg["output"])
    report = root / "isaac/capture-report.json"
    failure = root / "isaac/failure.json"
    command = [args.native_python, str(Path(__file__).with_name("capture_material_isaac.py").resolve()),
               "--config", args.config, "--ue-acceptance", args.ue_acceptance]
    if args.graycard_config:
        command += ["--graycard-config", args.graycard_config]
    if (root / "isaac").exists():
        raise FileExistsError(root / "isaac")
    start = time.monotonic()
    ready = None
    terminated = False
    with (root / "isaac-capture.console.log").open("wb") as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            while process.poll() is None:
                if report.exists() or failure.exists():
                    ready = ready or time.monotonic()
                    if time.monotonic()-ready > 20:
                        process.terminate()
                        terminated = True
                        break
                if time.monotonic()-start > 900:
                    raise TimeoutError("Owned native material worker timed out")
                time.sleep(.5)
            process.wait(timeout=15)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)
    result = json.loads(report.read_text(encoding="utf8")) if report.exists() else {}
    success = result.get("status") == "PASS" and len(result.get("records", [])) == 6 and not failure.exists()
    for frame in [0,50,100,150,200,249]:
        for suffix in [".png", "-material-ids.npy", "-material-labels.json", "-camera-native.json"]:
            success = success and (root / "isaac" / (f"cam01-{frame:03d}"+suffix)).is_file()
    receipt = dict(status="PASS" if success else "FAIL_STOPPED", owned_pid=process.pid,
                   exit_code=process.returncode, shutdown_terminated_after_artifacts=terminated,
                   elapsed_seconds=time.monotonic()-start)
    (root / "isaac-supervisor.json").write_text(json.dumps(receipt, indent=2), encoding="utf8")
    print(json.dumps(receipt, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
