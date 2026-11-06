"""Measure whether emergency-vehicle priority actually works.

This is the experiment that decides whether the priority feature earns its
place. It compares, on identical seeded demand:

  max_pressure       a strong efficiency baseline with no emergency awareness
  mp_preempt         the same baseline plus classic hard preemption
  learned_only       the trained policy, prioritisation left to the critic
  learned+priority   the trained policy plus the explicit belief-driven overlay

The headline question is whether ``learned+priority`` reduces emergency stopped
delay against ``learned_only`` by more than seed noise. Everything is paired by
seed and tested with a paired bootstrap, because the seed-to-seed spread here is
large enough that three-seed comparisons flip sign.

    python scripts/compare_priority.py --seeds 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atsc.agents.baselines import LearnedPolicy, PriorityAwarePolicy, build_baseline
from atsc.agents.dueling_ddqn import LagrangianDuelingDDQN
from atsc.config import (
    AgentConfig, DATA_DIR, DemandConfig, EnvConfig, MODELS_DIR, RESULTS_DIR, RUNS_DIR, ensure_dirs,
)
from atsc.env import SumoSignalEnv
from atsc.evaluation.metrics import paired_bootstrap_pvalue
from atsc.evaluation.runner import run_episode
from atsc.netgen import build_network
from atsc.sensing.sensor import AcousticSensor, SensorErrorModel

KEYS = ["mean_time_loss_s", "ev_mean_waiting_time_s", "ev_mean_stops",
        "ev_zero_stop_rate", "mean_queue", "teleports", "max_movement_wait_s"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--seed-start", type=int, default=5100)
    p.add_argument("--horizon", type=int, default=2400)
    p.add_argument("--ev-rate", type=float, default=30.0)
    p.add_argument("--regime", type=str, default="balanced")
    p.add_argument("--agent", type=str, default=str(MODELS_DIR / "siren_cmdp.h5"))
    p.add_argument("--out", type=str, default=str(RESULTS_DIR / "priority_comparison.json"))
    args = p.parse_args()

    ensure_dirs()
    cfg = EnvConfig()
    cfg.demand = DemandConfig(horizon_s=args.horizon, regime=args.regime,
                              ev_rate_per_hour=args.ev_rate)
    cfg.collect_emissions = False
    net_file, _ = build_network(cfg.network, DATA_DIR / "networks")
    em_path = MODELS_DIR / "sensor_error_model.json"
    error_model = SensorErrorModel.load(em_path) if em_path.exists() else None

    probe = SumoSignalEnv(cfg, net_file=net_file,
                          work_dir=RUNS_DIR / "probe", label="probe_pri")
    probe.reset(seed=args.seed_start)
    state_dim, n_actions = probe.observation_dim, probe.n_actions
    probe.close()

    agent = LagrangianDuelingDDQN(state_dim, n_actions, AgentConfig())
    agent.load(Path(args.agent))
    lam = float(agent.lam.numpy())
    print(f"agent {Path(args.agent).name}  lambda={lam:.2f}", flush=True)

    seeds = [args.seed_start + i for i in range(args.seeds)]
    controllers = {
        "max_pressure": lambda e: build_baseline("max_pressure", e, cfg.demand),
        "mp_preempt": lambda e: build_baseline("max_pressure_preempt", e, cfg.demand),
        "learned_only": lambda e: LearnedPolicy(agent, greedy=True),
        "learned+priority": lambda e: PriorityAwarePolicy(agent=agent),
    }

    runs: dict = {}
    for name, factory in controllers.items():
        out = []
        for seed in seeds:
            env = SumoSignalEnv(
                cfg, net_file=net_file, work_dir=RUNS_DIR / "priority",
                label=f"pri_{name}_{seed}",
                tripinfo_path=RUNS_DIR / "priority" / f"t_{name}_{seed}.xml",
                sensor=AcousticSensor([], cfg.acoustic, mode=cfg.sensing_mode,
                                      error_model=error_model, seed=7),
            )
            out.append(run_episode(env, factory, seed=seed))
        runs[name] = out
        v = {k: float(np.nanmean([r.get(k, np.nan) for r in out])) for k in KEYS}
        print(f"  {name:<18} delay={v['mean_time_loss_s']:6.2f}  "
              f"EVstop={v['ev_mean_waiting_time_s']:6.2f}  "
              f"clean={v['ev_zero_stop_rate']:.3f}  telep={v['teleports']:.0f}", flush=True)

    print(f"\n{'controller':<18}{'delay':>9}{'EV stopped':>16}{'EV stops':>10}"
          f"{'clean%':>9}{'queue':>8}{'telep':>7}   (n={len(seeds)})")
    print("-" * 80)
    summary = {}
    for name, rs in runs.items():
        v = {k: float(np.nanmean([r.get(k, np.nan) for r in rs])) for k in KEYS}
        ev = [r.get("ev_mean_waiting_time_s", np.nan) for r in rs]
        ci = 1.96 * float(np.nanstd(ev, ddof=1)) / np.sqrt(len(rs))
        v["ev_ci95"] = ci
        summary[name] = v
        print(f"{name:<18}{v['mean_time_loss_s']:9.2f}"
              f"{v['ev_mean_waiting_time_s']:11.2f}+-{ci:<3.1f}"
              f"{v['ev_mean_stops']:10.2f}{v['ev_zero_stop_rate'] * 100:9.1f}"
              f"{v['mean_queue']:8.2f}{v['teleports']:7.0f}")

    print("\npaired bootstrap vs learned_only (negative difference = better):")
    tests: dict = {}
    base = runs["learned_only"]
    for name in ("learned+priority", "mp_preempt", "max_pressure"):
        tests[name] = {}
        for key, label in (("ev_mean_waiting_time_s", "EV stopped delay"),
                           ("mean_time_loss_s", "background delay")):
            a = [r.get(key, np.nan) for r in runs[name]]
            b = [r.get(key, np.nan) for r in base]
            pair = [(x, y) for x, y in zip(a, b) if np.isfinite(x) and np.isfinite(y)]
            if len(pair) < 2:
                continue
            st = paired_bootstrap_pvalue([x for x, _ in pair], [y for _, y in pair])
            tests[name][key] = st
            print(f"  {name:<18} {label:<18} diff={st['diff']:+7.2f}s  p={st['p_value']:.4f}")

    out_path = Path(args.out)
    out_path.write_text(json.dumps({
        "config": {"seeds": seeds, "horizon": args.horizon,
                   "ev_rate": args.ev_rate, "regime": args.regime, "lambda": lam},
        "summary": summary,
        "runs": {k: v for k, v in runs.items()},
        "significance": tests,
    }, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
