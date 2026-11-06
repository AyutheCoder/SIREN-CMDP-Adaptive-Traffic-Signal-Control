"""Robustness check over 30 seeds: medians, 95 % CIs and paired tests.

    python scripts/seeds30_analysis.py

Seeds 50000-50011 come from results/evaluation_balanced.json (Table 2). Seeds
50012-50029 come from the camera-ready runs: the per-run lines the evaluation
printed (an interrupted session lost the summary files, but every finished run
had already printed its ordinary delay and emergency stopped delay) plus the
per-seed unit files written by scripts/camera_ready_queue.py.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from atsc.evaluation.metrics import paired_bootstrap_pvalue  # noqa: E402

CR = REPO / "results" / "camera_ready"
KEYS = {"delay": "mean_time_loss_s", "ev_stopped": "ev_mean_waiting_time_s"}
UNIT_NAME = {"max_pressure_preempt_oracle": "transmitter_preemption",
             "siren_cmdp": "siren_cmdp"}
CTRLS = ["fixed_time", "actuated", "max_pressure", "max_pressure_preempt",
         "max_pressure_preempt_oracle", "base_arc", "base_cmdp", "siren_cmdp",
         "siren_arc"]
LINE = re.compile(r"\[(\w+)\] seed=(\d+)\s+delay=\s*([\d.]+)s\s+ev_stopped=\s*([\d.]+)s")


def collect() -> dict:
    data = {c: {} for c in CTRLS}
    base = json.loads((REPO / "results" / "evaluation_balanced.json").read_text())["results"]
    for c in CTRLS:
        for r in base[c]["runs"]:
            data[c][r["seed"]] = (r[KEYS["delay"]], r[KEYS["ev_stopped"]])
    for log in ("extra_seeds_baselines.log", "extra_seeds_ours.log"):
        for m in LINE.finditer((CR / log).read_text(encoding="utf-8", errors="ignore")):
            name, seed = m.group(1), int(m.group(2))
            if name in data:
                data[name][seed] = (float(m.group(3)), float(m.group(4)))
    for c, unit in UNIT_NAME.items():
        for f in (CR / "units").glob(f"{unit}_balanced_f700_ev30_s*.json"):
            run = next(iter(json.loads(f.read_text())["results"].values()))["runs"][0]
            data[c][run["seed"]] = (run[KEYS["delay"]], run[KEYS["ev_stopped"]])
    return data


def median_ci(x: np.ndarray, n_boot: int = 20000, seed: int = 0):
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(n_boot, x.size), replace=True), axis=1)
    return float(np.median(x)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def main() -> int:
    data = collect()
    seeds = sorted(set.intersection(*(set(v) for v in data.values())))
    out = {"n_seeds": len(seeds), "seeds": seeds, "controllers": {}}
    ours = {k: np.array([data["siren_arc"][s][i] for s in seeds]) for i, k in enumerate(KEYS)}
    for c in CTRLS:
        row = {}
        for i, k in enumerate(KEYS):
            x = np.array([data[c][s][i] for s in seeds])
            m, lo, hi = median_ci(x)
            row[k] = {"median": m, "ci95": [lo, hi]}
            if c != "siren_arc":
                row[k]["p_vs_siren_arc"] = float(paired_bootstrap_pvalue(
                    ours[k], x, n_boot=20000)["p_value"])
        out["controllers"][c] = row
    missing = {c: sorted(set(range(50000, 50030)) - set(v)) for c, v in data.items()}
    out["missing"] = {c: m for c, m in missing.items() if m}
    (CR / "seeds30.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("seeds:", len(seeds), "missing:", out["missing"])
    for c, row in out["controllers"].items():
        print(f"{c:30s}", "  ".join(
            f"{k}={v['median']:.2f} [{v['ci95'][0]:.2f},{v['ci95'][1]:.2f}]"
            + (f" p={v['p_vs_siren_arc']:.4f}" if "p_vs_siren_arc" in v else "")
            for k, v in row.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
