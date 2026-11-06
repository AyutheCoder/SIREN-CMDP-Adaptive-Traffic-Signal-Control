"""Sweep the emergency/background exchange rate and trace the operating frontier.

The corridor scheduler has exactly one normative parameter: ``ev_value_ratio``
(gamma), the number of background vehicle-seconds one priority-weighted
emergency vehicle-second is worth. Everything else in the decision rule is
derived from the traffic state.

That makes the sweep over gamma the operating-point figure of the whole method:
an operator does not tune a detection threshold, they state an exchange rate and
read off where it lands on the delay / emergency-delay frontier. This script
produces that curve.

    python scripts/value_sweep.py --seeds 8 --gammas 1,10,25,50,100,200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from atsc.agents.baselines import CorridorPolicy, LearnedPolicy
from atsc.agents.dueling_ddqn import LagrangianDuelingDDQN
from atsc.config import (
    AgentConfig, DATA_DIR, DemandConfig, EnvConfig, MODELS_DIR, RESULTS_DIR, RUNS_DIR, ensure_dirs)
from atsc.control.corridor import CorridorScheduler
from atsc.evaluation.runner import evaluate_controller
from atsc.netgen import build_network
from atsc.sensing.sensor import AcousticSensor, SensorErrorModel


def _f(v, w=6, d=2):
    """Format a metric that may be absent.

    An aggregate key is None when a run produced nothing to average -- a
    horizon too short for any vehicle to complete, or a condition in which
    none arrived. Formatting that directly raises TypeError and loses the
    whole sweep at the print statement, after the compute has been done.
    """
    return ("{:%d.%df}" % (w, d)).format(v) if isinstance(v, (int, float)) else " " * (w - 3) + "n/a"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=str(MODELS_DIR / "siren_arc.h5"))
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--seed-start", type=int, default=300000,
                   help="disjoint from the evaluation seeds (50000..), so the "
                        "operating point an operator reads off this curve is "
                        "not chosen on the data the headline table reports")
    p.add_argument("--horizon", type=int, default=3600)
    p.add_argument("--ev-rate", type=float, default=30.0)
    p.add_argument("--regime", default="balanced")
    p.add_argument("--gammas", default="0,1,5,10,25,50,100,200,400")
    p.add_argument("--out", default=str(RESULTS_DIR / "value_sweep.json"))
    args = p.parse_args()

    ensure_dirs()
    cfg = EnvConfig()
    cfg.demand = DemandConfig(horizon_s=args.horizon, regime=args.regime,
                              ev_rate_per_hour=args.ev_rate)
    cfg.collect_emissions = True
    seeds = [args.seed_start + i for i in range(args.seeds)]

    net_file, _ = build_network(cfg.network, DATA_DIR / "networks")
    em_path = MODELS_DIR / "sensor_error_model.json"
    error_model = SensorErrorModel.load(em_path) if em_path.exists() else None

    probe_env_cfg = cfg
    from atsc.env import SumoSignalEnv
    probe = SumoSignalEnv(probe_env_cfg, net_file=net_file,
                          work_dir=RUNS_DIR / "probe", label="probe_gamma")
    probe.reset(seed=seeds[0])
    state_dim, n_actions = probe.observation_dim, probe.n_actions
    probe.close()

    agent = LagrangianDuelingDDQN(state_dim, n_actions, AgentConfig())
    agent.load(Path(args.checkpoint))

    gammas = [float(g) for g in args.gammas.split(",") if g.strip()]
    rows = []
    references = {}

    def summarise(agg, extra=None):
        row = {
            "delay_mean": agg.get("mean_time_loss_s_mean"),
            "delay_median": agg.get("mean_time_loss_s_median"),
            "delay_ci": agg.get("mean_time_loss_s_ci95"),
            "ev_delay_mean": agg.get("ev_mean_waiting_time_s_mean"),
            "ev_delay_median": agg.get("ev_mean_waiting_time_s_median"),
            "ev_delay_ci": agg.get("ev_mean_waiting_time_s_ci95"),
            "ev_stops": agg.get("ev_mean_stops_mean"),
            "clean_pass": agg.get("ev_zero_stop_rate_mean"),
            "throughput": agg.get("throughput_mean"),
            "preemptions": agg.get("n_preemptions_mean"),
            "max_wait": agg.get("max_movement_wait_s_max"),
            "teleports": agg.get("teleports_mean"),
            "jam_rate": agg.get("jam_rate"),
            "n_jammed": agg.get("n_jammed"),
        }
        row.update(extra or {})
        return row

    # Fixed alternatives on the *same* seeds. A frontier with nothing on it is
    # a curve; a frontier with the alternatives plotted on the same axes is a
    # claim, and the claim is that an operator can pick a point on the curve
    # that dominates them.
    from atsc.agents.baselines import build_baseline
    # Not "actuated": SUMO's own vehicle-actuated program is supplied as an
    # additional file which this sweep does not load, and a reference point that
    # silently fell back to the fixed program would be a fabricated one.
    for ref in ("max_pressure_preempt", "fixed_time"):
        res = evaluate_controller(
            (lambda env, n=ref: build_baseline(n, env, cfg.demand)),
            cfg, seeds, ref, net_file=net_file,
            sensor=AcousticSensor([], cfg.acoustic, mode=cfg.sensing_mode,
                                  error_model=error_model, seed=args.seed_start),
            work_dir=RUNS_DIR / f"gammaref_{ref}",
            verbose=False)
        references[ref] = summarise(res["aggregate"])
        r = references[ref]
        print(f"  reference {ref:22s} delay={_f(r['delay_mean'])}s  "
              f"EV={_f(r['ev_delay_mean'])}s  "
              f"clean={_f(r['clean_pass'], 5, 3)}", flush=True)

    # gamma = 0 is the learned policy alone: no corridor can ever pay for
    # itself, so the sweep contains its own no-priority control point.
    for gamma in gammas:
        label = f"gamma_{gamma:g}"
        if gamma <= 0.0:
            factory = (lambda env, a=agent: LearnedPolicy(a, name="policy_only",
                                                          greedy=True))
        else:
            factory = (lambda env, a=agent, g=gamma:
                       CorridorPolicy(agent=a, name=f"arc_{g:g}", adaptive=False,
                                      scheduler=CorridorScheduler(ev_value_ratio=g)))
        res = evaluate_controller(
            factory, cfg, seeds, label, net_file=net_file,
            sensor=AcousticSensor([], cfg.acoustic, mode=cfg.sensing_mode,
                                  error_model=error_model, seed=args.seed_start),
            work_dir=RUNS_DIR / f"gamma_{gamma:g}",
        )
        agg = res["aggregate"]
        row = summarise(agg, {"gamma": gamma})
        rows.append(row)
        print(f"  gamma={gamma:6g}  delay={_f(row['delay_mean'])}s  "
              f"EV={_f(row['ev_delay_mean'])}s  "
              f"clean={_f(row['clean_pass'], 5, 3)}  "
              f"jam={agg.get('n_jammed', 0) or 0:.0f}/{len(seeds)}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"seeds": seeds, "horizon_s": args.horizon,
                   "ev_rate_per_hour": args.ev_rate, "regime": args.regime,
                   "checkpoint": args.checkpoint},
        "rows": rows,
        "references": references,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
