"""Compare the policy retrained on mixed traffic with the original one.

    python scripts/mixed_summary.py

Original SIREN-ARC and SUMO actuated come from results/evaluation_{regime}.json
(same seeds); the retrained policy from the camera-ready unit files.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys_path = REPO
import sys  # noqa: E402
sys.path.insert(0, str(REPO))
from atsc.evaluation.metrics import paired_bootstrap_pvalue  # noqa: E402

UNITS = REPO / "results" / "camera_ready" / "units"
K = {"delay": "mean_time_loss_s", "ev_stopped": "ev_mean_waiting_time_s"}


def mixed_runs(regime):
    out = {}
    for f in UNITS.glob(f"siren_arc_mixed_{regime}_f700_ev30_s*.json"):
        r = next(iter(json.loads(f.read_text())["results"].values()))["runs"][0]
        out[r["seed"]] = r
    return out


def main():
    res = {}
    for regime in ("asymmetric", "peak", "surge", "balanced"):
        base = json.loads((REPO / "results" / f"evaluation_{regime}.json").read_text())["results"]
        mixed = mixed_runs(regime)
        seeds = sorted(mixed)
        row = {"seeds": seeds}
        for name, runs in (("siren_arc", base["siren_arc"]["runs"]),
                           ("actuated", base["actuated"]["runs"]),
                           ("preemption", base["max_pressure_preempt"]["runs"])):
            by = {r["seed"]: r for r in runs}
            row[name] = {k: float(np.median([by[s][v] for s in seeds])) for k, v in K.items()}
            row[name]["teleports_med"] = float(np.median([by[s].get("teleports", 0) for s in seeds]))
        row["mixed"] = {k: float(np.median([mixed[s][v] for s in seeds])) for k, v in K.items()}
        row["mixed"]["teleports_med"] = float(np.median([mixed[s].get("teleports", 0) for s in seeds]))
        by = {r["seed"]: r for r in base["siren_arc"]["runs"]}
        for k, v in K.items():
            a = np.array([mixed[s][v] for s in seeds])
            b = np.array([by[s][v] for s in seeds])
            row[f"p_{k}_mixed_vs_original"] = float(
                paired_bootstrap_pvalue(a, b, n_boot=20000)["p_value"])
        res[regime] = row
        print(regime, json.dumps({k: v for k, v in row.items() if k != "seeds"}))
    (REPO / "results" / "camera_ready" / "mixed.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
