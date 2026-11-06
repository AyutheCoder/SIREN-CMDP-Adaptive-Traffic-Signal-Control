"""Resumable job queue for the camera-ready experiments.

    python scripts/camera_ready_queue.py [--part runs|mixed_eval|latency]

Every unit of work writes its own output file under results/camera_ready/units/
and is skipped if that file already exists, so the queue can be stopped and
restarted (laptop sleep, closed session) without losing finished runs. While
it runs it asks Windows not to sleep; that request ends with the process.
"""

from __future__ import annotations

import argparse
import ctypes
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "camera_ready" / "units"
PY = sys.executable


def keep_awake() -> None:
    if sys.platform == "win32":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def evaluate(name: str, seed: int, entry: list, flow=700, ev=30, regime="balanced") -> None:
    out = OUT / f"{name}_{regime}_f{flow}_ev{ev}_s{seed}.json"
    if out.exists():
        return
    cmd = [PY, "scripts/evaluate.py", "--seeds", "1", "--seed-start", str(seed),
           "--horizon", "3600", "--flow", str(flow), "--ev-rate", str(ev),
           "--regime", regime, "--out", str(out)] + entry
    log = out.with_suffix(".log")
    with log.open("w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode
    print(f"{'ok ' if rc == 0 and out.exists() else 'FAIL'} {out.name}", flush=True)


ENTRY = {
    "transmitter_preemption": ["--baselines", "max_pressure_preempt:oracle"],
    "siren_preemption": ["--baselines", "max_pressure_preempt"],
    "actuated": ["--baselines", "actuated"],
    "siren_cmdp": ["--baselines", "none", "--corridor", "siren_cmdp=models/siren_cmdp.h5:adapt"],
    "siren_arc": ["--baselines", "none", "--corridor", "siren_arc=models/siren_arc.h5"],
    "siren_arc_mixed": ["--baselines", "none",
                        "--corridor", "siren_arc_mixed=models/siren_arc_mixed.h5"],
}


def part_runs() -> None:
    # 1. The extra-seed runs the interrupted session did not finish.
    for s in range(50015, 50030):
        evaluate("transmitter_preemption", s, ENTRY["transmitter_preemption"])
    for s in range(50025, 50030):
        evaluate("siren_cmdp", s, ENTRY["siren_cmdp"])
    # 2. The last stress-test scenario.
    stress = REPO / "results" / "camera_ready" / "shield_stress_random_flood.json"
    if not stress.exists():
        subprocess.run([PY, "scripts/shield_stress.py", "--seeds", "6",
                        "--scenarios", "random_flood", "--out", str(stress)], cwd=REPO,
                       stdout=(stress.with_suffix(".log")).open("w"), stderr=subprocess.STDOUT)
        print("ok stress random_flood", flush=True)
    # 3. Other traffic levels and emergency arrival rates.
    for flow, ev in ((500, 30), (800, 30), (700, 15), (700, 60)):
        for name in ("siren_arc", "siren_preemption", "actuated"):
            for s in range(50000, 50004):
                evaluate(name, s, ENTRY[name], flow=flow, ev=ev)


def part_mixed_eval() -> None:
    for regime in ("asymmetric", "peak", "surge"):
        for s in range(50000, 50006):
            evaluate("siren_arc_mixed", s, ENTRY["siren_arc_mixed"], regime=regime)
    for s in range(50000, 50012):
        evaluate("siren_arc_mixed", s, ENTRY["siren_arc_mixed"])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--part", choices=["runs", "mixed_eval"], default="runs")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    keep_awake()
    {"runs": part_runs, "mixed_eval": part_mixed_eval}[args.part]()
    print("PART_DONE", args.part, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
