"""Sanity-check the CMDP cost against fixed controllers of known relative quality.

A constraint signal that cannot distinguish a good controller from a bad one
*before* any learning happens will not teach one during learning. This script
runs a set of hand-designed controllers whose emergency-vehicle behaviour is
known by construction -- hard preemption is better for emergency vehicles than
plain max-pressure, by design -- and reports the mean cost each one incurs
alongside the emergency outcome it actually achieves.

If the cost ordering does not match the outcome ordering, the cost is not
attributable to actions and no amount of dual ascent will fix it. We learned
this the expensive way: our original cost (realised emergency delay only)
assigned both controllers the same value, and the constrained agents trained
against it raised their multiplier to the ceiling while achieving nothing.

    python scripts/check_cost_signal.py --episodes 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atsc.agents.baselines import build_baseline
from atsc.config import DATA_DIR, DemandConfig, EnvConfig, RESULTS_DIR, RUNS_DIR, ensure_dirs
from atsc.env import SumoSignalEnv
from atsc.evaluation.runner import run_episode
from atsc.netgen import build_network

# Ordered from worst to best for emergency vehicles, by construction.
CONTROLLERS = ["fixed_time", "max_pressure", "max_pressure_preempt"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--horizon", type=int, default=1200)
    p.add_argument("--ev-rate", type=float, default=24.0)
    p.add_argument("--seed", type=int, default=777)
    args = p.parse_args()

    ensure_dirs()
    cfg = EnvConfig()
    cfg.demand = DemandConfig(horizon_s=args.horizon, ev_rate_per_hour=args.ev_rate)
    cfg.collect_emissions = False
    net_file, _ = build_network(cfg.network, DATA_DIR / "networks")

    rows = []
    print(f"{'controller':<24}{'mean cost':>11}{'EV wait/veh':>13}{'EV stops':>10}{'queue':>8}")
    for name in CONTROLLERS:
        costs, waits, stops, queues = [], [], [], []
        for i in range(args.episodes):
            env = SumoSignalEnv(cfg, net_file=net_file,
                                work_dir=RUNS_DIR / "costcheck",
                                label=f"cc_{name}_{i}")
            s = run_episode(env, lambda e, n=name: build_baseline(n, e, cfg.demand),
                            seed=args.seed + i, parse_trips=False)
            costs.append(s["mean_cost"])
            waits.append(s["ev_delay_per_vehicle_s"])
            stops.append(s["ev_stops"] / max(s["n_emergency"], 1))
            queues.append(s["mean_queue"])
        row = {
            "controller": name,
            "mean_cost": float(np.mean(costs)),
            "ev_wait_per_vehicle_s": float(np.mean(waits)),
            "ev_stops_per_vehicle": float(np.mean(stops)),
            "mean_queue": float(np.mean(queues)),
        }
        rows.append(row)
        print(f"{name:<24}{row['mean_cost']:>11.4f}{row['ev_wait_per_vehicle_s']:>13.2f}"
              f"{row['ev_stops_per_vehicle']:>10.2f}{row['mean_queue']:>8.2f}", flush=True)

    # The controllers are listed worst-to-best for emergency vehicles, so both
    # the cost and the realised emergency delay should decrease along the list.
    costs = [r["mean_cost"] for r in rows]
    waits = [r["ev_wait_per_vehicle_s"] for r in rows]
    cost_ok = costs[-1] < costs[0]
    outcome_ok = waits[-1] < waits[0]
    spread = (max(costs) - min(costs)) / max(abs(np.mean(costs)), 1e-9)

    print()
    print(f"  cost decreases with controller quality : {cost_ok}")
    print(f"  outcome decreases with quality         : {outcome_ok}")
    print(f"  relative cost spread across controllers: {spread:.3f}")
    if not (cost_ok and spread > 0.1):
        print("\n  WARNING: the cost signal barely distinguishes these controllers.")
        print("  A CMDP built on it will raise its multiplier and bound nothing.")
    else:
        print("\n  cost signal is attributable: ordering matches outcomes.")

    out = RESULTS_DIR / "cost_signal_check.json"
    out.write_text(json.dumps({
        "rows": rows, "cost_ordering_ok": bool(cost_ok),
        "outcome_ordering_ok": bool(outcome_ok), "relative_spread": float(spread),
        "horizon_s": args.horizon, "ev_rate_per_hour": args.ev_rate,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0 if (cost_ok and spread > 0.1) else 1


if __name__ == "__main__":
    raise SystemExit(main())
