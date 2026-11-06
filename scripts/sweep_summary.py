"""Summarise the traffic-level / emergency-rate sweep (camera-ready units).

    python scripts/sweep_summary.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
UNITS = REPO / "results" / "camera_ready" / "units"
CONDS = [(700, 30), (500, 30), (800, 30), (700, 15), (700, 60)]
NAMES = ["siren_arc", "siren_preemption", "actuated"]
BASE_KEY = {"siren_arc": "siren_arc", "siren_preemption": "max_pressure_preempt",
            "actuated": "actuated"}


def runs(name, flow, ev):
    if (flow, ev) == (700, 30):  # the main experiment, same seeds
        base = json.loads((REPO / "results" / "evaluation_balanced.json").read_text())
        return [r for r in base["results"][BASE_KEY[name]]["runs"]
                if r["seed"] in range(50000, 50004)]
    out = []
    for f in sorted(UNITS.glob(f"{name}_balanced_f{flow}_ev{ev}_s*.json")):
        out.append(next(iter(json.loads(f.read_text())["results"].values()))["runs"][0])
    return out


def main():
    res = {}
    for flow, ev in CONDS:
        key = f"flow{flow}_ev{ev}"
        res[key] = {}
        for n in NAMES:
            rs = runs(n, flow, ev)
            res[key][n] = {
                "n": len(rs),
                "delay": float(np.median([r["mean_time_loss_s"] for r in rs])),
                "ev_stopped": float(np.median([r["ev_mean_waiting_time_s"] for r in rs])),
                "jammed": int(sum(bool(r.get("jammed", False)) for r in rs)),
            }
            print(f"{key:14s} {n:18s}", res[key][n])
    (REPO / "results" / "camera_ready" / "sweep.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
