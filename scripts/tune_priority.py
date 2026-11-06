"""Sweep the priority overlay's engagement policy against the learned base.

The overlay's first configuration improved clean-pass (0.176 -> 0.274) but not
stopped delay, and stayed behind plain hard preemption (0.410 clean, 13.6 s).
The suspicion is over-restriction: the range, closing-speed, cooldown and
duty-cycle guards were each added to stop a specific failure, and together they
may simply be refusing to engage often enough.

This sweeps the engagement policy and includes, as a reference point, the
learned agent driven by :class:`HardPreemption` -- the rule already known to
work -- so the comparison separates "the overlay's guards are wrong" from "an
explicit overlay cannot help here".

    python scripts/tune_priority.py --seeds 6
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atsc.agents.baselines import HardPreemption, LearnedPolicy, PriorityAwarePolicy
from atsc.agents.dueling_ddqn import LagrangianDuelingDDQN
from atsc.config import (
    AgentConfig, DATA_DIR, DemandConfig, EnvConfig, MODELS_DIR, RESULTS_DIR, RUNS_DIR, ensure_dirs,
)
from atsc.control.priority import PriorityOverlay
from atsc.env import SumoSignalEnv
from atsc.evaluation.runner import run_episode
from atsc.netgen import build_network
from atsc.sensing.sensor import AcousticSensor, SensorErrorModel

# (label, overlay kwargs) -- each isolates one guard.
VARIANTS = [
    ("base(no overlay)", None),
    ("hardpreempt-rule", "hardpreempt"),
    ("loose", dict(commit_threshold=0.50, max_range_m=250.0, min_closing_ms=0.5,
                   cooldown_s=2.0, duty_cycle_cap=0.70, max_hold_s=45.0)),
    ("loose+closing", dict(commit_threshold=0.50, max_range_m=250.0, min_closing_ms=2.0,
                           cooldown_s=2.0, duty_cycle_cap=0.70, max_hold_s=45.0)),
    ("mid", dict(commit_threshold=0.55, max_range_m=200.0, min_closing_ms=1.0,
                 cooldown_s=6.0, duty_cycle_cap=0.55, max_hold_s=40.0)),
    ("strict", dict(commit_threshold=0.67, max_range_m=170.0, min_closing_ms=1.5,
                    cooldown_s=12.0, duty_cycle_cap=0.34, max_hold_s=40.0)),
]

KEYS = ["mean_time_loss_s", "ev_mean_waiting_time_s", "ev_mean_stops",
        "ev_zero_stop_rate", "mean_queue", "teleports", "max_movement_wait_s"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--seed-start", type=int, default=6200)
    p.add_argument("--horizon", type=int, default=2400)
    p.add_argument("--ev-rate", type=float, default=30.0)
    p.add_argument("--agent", type=str, default=str(MODELS_DIR / "siren_cmdp.h5"))
    p.add_argument("--out", type=str, default=str(RESULTS_DIR / "priority_tuning.json"))
    args = p.parse_args()

    ensure_dirs()
    cfg = EnvConfig()
    cfg.demand = DemandConfig(horizon_s=args.horizon, ev_rate_per_hour=args.ev_rate)
    cfg.collect_emissions = False
    net_file, _ = build_network(cfg.network, DATA_DIR / "networks")
    em_path = MODELS_DIR / "sensor_error_model.json"
    error_model = SensorErrorModel.load(em_path) if em_path.exists() else None

    probe = SumoSignalEnv(cfg, net_file=net_file,
                          work_dir=RUNS_DIR / "probe", label="probe_tune")
    probe.reset(seed=args.seed_start)
    state_dim, n_actions = probe.observation_dim, probe.n_actions
    probe.close()

    agent = LagrangianDuelingDDQN(state_dim, n_actions, AgentConfig())
    agent.load(Path(args.agent))
    seeds = [args.seed_start + i for i in range(args.seeds)]
    print(f"agent lambda={float(agent.lam.numpy()):.2f}   seeds={len(seeds)}   "
          f"horizon={args.horizon}s   ev_rate={args.ev_rate}/h\n", flush=True)

    def factory_for(spec):
        if spec is None:
            return lambda e: LearnedPolicy(agent, greedy=True)
        if spec == "hardpreempt":
            return lambda e: HardPreemption(base=LearnedPolicy(agent, greedy=True))
        return lambda e: PriorityAwarePolicy(
            agent=agent, overlay=PriorityOverlay(**spec), adaptive=False)

    results = {}
    for label, spec in VARIANTS:
        out = []
        for seed in seeds:
            env = SumoSignalEnv(
                cfg, net_file=net_file, work_dir=RUNS_DIR / "tune",
                label=f"tune_{label}_{seed}".replace("(", "").replace(")", "").replace(" ", ""),
                tripinfo_path=RUNS_DIR / "tune" / f"t_{label}_{seed}.xml".replace(" ", ""),
                sensor=AcousticSensor([], cfg.acoustic, mode=cfg.sensing_mode,
                                      error_model=error_model, seed=7),
            )
            out.append(run_episode(env, factory_for(spec), seed=seed))
        v = {k: float(np.nanmean([r.get(k, np.nan) for r in out])) for k in KEYS}
        ev = [r.get("ev_mean_waiting_time_s", np.nan) for r in out]
        v["ev_ci95"] = 1.96 * float(np.nanstd(ev, ddof=1)) / np.sqrt(len(out))
        results[label] = {"summary": v, "runs": out}
        print(f"  {label:<18} delay={v['mean_time_loss_s']:6.2f}  "
              f"EVstop={v['ev_mean_waiting_time_s']:6.2f}+-{v['ev_ci95']:.1f}  "
              f"stops={v['ev_mean_stops']:.2f}  clean={v['ev_zero_stop_rate']:.3f}  "
              f"telep={v['teleports']:.0f}", flush=True)

    print(f"\n{'variant':<18}{'delay':>9}{'EV stopped':>14}{'stops':>8}{'clean%':>9}{'telep':>7}")
    print("-" * 68)
    for label, r in results.items():
        v = r["summary"]
        print(f"{label:<18}{v['mean_time_loss_s']:9.2f}{v['ev_mean_waiting_time_s']:11.2f}"
              f"{v['ev_mean_stops']:11.2f}{v['ev_zero_stop_rate'] * 100:9.1f}"
              f"{v['teleports']:7.0f}")

    Path(args.out).write_text(json.dumps(
        {"config": vars(args), "results": results}, indent=2, default=float),
        encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
