"""Trace the efficiency / emergency-delay frontier, and show why the cheap way fails.

The frontier is built from agents **trained independently at different budgets**
``d0``. Each point is a separate training run that actually optimised for the
constraint it is plotted at.

The script also runs the cheap alternative --- take one trained agent and vary
``lambda`` at evaluation time, exploiting the fact that the greedy policy is
``argmax [Q_r - lambda Q_c]`` --- and reports it alongside as a **negative
control**. That shortcut is tempting and it does not work: during training the
multiplier settles in a narrow range, so the critics are only accurate for
policies near that value, and evaluating far from it is badly off-policy. In our
runs the swept curve is not even monotone --- large ``lambda`` degrades both
objectives at once rather than trading one for the other. Reporting this
matters, because the shortcut looks reasonable on paper and would silently
produce a fictitious frontier.

    python scripts/frontier.py --agents budget_tight,siren_cmdp,budget_loose,unconstrained
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atsc.agents.baselines import LearnedPolicy
from atsc.agents.dueling_ddqn import LagrangianDuelingDDQN
from atsc.config import (
    AgentConfig, DATA_DIR, DemandConfig, EnvConfig, MODELS_DIR, RESULTS_DIR, RUNS_DIR, ensure_dirs,
)
from atsc.env import SumoSignalEnv
from atsc.evaluation.runner import evaluate_controller
from atsc.netgen import build_network
from atsc.sensing.sensor import AcousticSensor, SensorErrorModel

METRIC_KEYS = {
    "delay_s": "mean_time_loss_s",
    "ev_wait_s": "ev_mean_waiting_time_s",
    "ev_stops": "ev_mean_stops",
    "ev_clean_pass": "ev_zero_stop_rate",
    "ev_time_loss_s": "ev_mean_time_loss_s",
    "mean_cost": "mean_cost",
    "queue": "mean_queue",
    "max_wait_s": "max_movement_wait_s",
}


def summarise(agg: dict) -> dict:
    out = {}
    for short, key in METRIC_KEYS.items():
        out[short] = agg.get(f"{key}_mean", float("nan"))
        out[f"{short}_ci95"] = agg.get(f"{key}_ci95", 0.0)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agents", type=str,
                   default="budget_tight,siren_cmdp,budget_loose,unconstrained")
    p.add_argument("--lambda-control", type=str, default="siren_cmdp",
                   help="agent used for the off-policy lambda-sweep negative control")
    p.add_argument("--lambdas", type=str, default="0,0.5,2,8,30")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed-start", type=int, default=70000)
    p.add_argument("--horizon", type=int, default=3600)
    p.add_argument("--regime", type=str, default="balanced")
    p.add_argument("--ev-rate", type=float, default=24.0)
    p.add_argument("--skip-lambda-control", action="store_true")
    args = p.parse_args()

    ensure_dirs()
    cfg = EnvConfig()
    cfg.demand = DemandConfig(horizon_s=args.horizon, regime=args.regime,
                              ev_rate_per_hour=args.ev_rate)
    cfg.collect_emissions = False
    seeds = [args.seed_start + i for i in range(args.seeds)]

    net_file, _ = build_network(cfg.network, DATA_DIR / "networks")
    em_path = MODELS_DIR / "sensor_error_model.json"
    error_model = SensorErrorModel.load(em_path) if em_path.exists() else None

    probe = SumoSignalEnv(cfg, net_file=net_file,
                          work_dir=RUNS_DIR / "probe", label="probe_frontier")
    probe.reset(seed=seeds[0])
    state_dim, n_actions = probe.observation_dim, probe.n_actions
    probe.close()

    def sensor_for(mode: str) -> AcousticSensor:
        return AcousticSensor([], cfg.acoustic, mode=mode,
                              error_model=error_model, seed=args.seed_start)

    def load(name: str):
        ckpt = MODELS_DIR / f"{name}.h5"
        if not ckpt.exists():
            return None, {}
        meta_path = ckpt.with_suffix(".agent.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        acfg = AgentConfig()
        acfg.constrained = bool(meta.get("constrained", True))
        acfg.cost_budget = float(meta.get("cost_budget", acfg.cost_budget))
        agent = LagrangianDuelingDDQN(state_dim, n_actions, acfg)
        agent.load(ckpt)
        return agent, meta

    # ---------------------------------------------------- frontier (the real one)
    print(f"frontier from independently trained agents "
          f"({len(seeds)} seeds, {args.horizon}s, {args.regime})\n")
    frontier = []
    for name in [n for n in args.agents.split(",") if n.strip()]:
        agent, meta = load(name)
        if agent is None:
            print(f"  skipping {name}: checkpoint not found")
            continue
        agent_cfg = copy.deepcopy(cfg)
        agent_cfg.sensing_mode = meta.get("sensing_mode", cfg.sensing_mode)
        res = evaluate_controller(
            lambda env, a=agent, n=name: LearnedPolicy(a, name=n, greedy=True),
            agent_cfg, seeds, f"frontier_{name}", net_file=net_file,
            sensor=sensor_for(agent_cfg.sensing_mode),
            work_dir=RUNS_DIR / f"frontier_{name}", verbose=False,
        )
        point = summarise(res["aggregate"])
        point.update({
            "name": name,
            "budget": None if not meta.get("constrained", True)
            else float(meta.get("cost_budget", float("nan"))),
            "trained_lambda": float(agent.lam.numpy()),
            "constrained": bool(meta.get("constrained", True)),
        })
        frontier.append(point)
        budget_str = "unconstrained" if point["budget"] is None else f"d0={point['budget']:.3f}"
        print(f"  {name:<16} {budget_str:<16} delay={point['delay_s']:6.2f}s  "
              f"EV wait={point['ev_wait_s']:6.2f}s  EV stops={point['ev_stops']:.2f}  "
              f"clean={point['ev_clean_pass']:.2f}  lambda={point['trained_lambda']:.2f}",
              flush=True)

    # ------------------------------------------- lambda sweep (negative control)
    control = []
    if not args.skip_lambda_control:
        agent, meta = load(args.lambda_control)
        if agent is not None:
            trained_lambda = float(agent.lam.numpy())
            print(f"\nnegative control: sweeping lambda on {args.lambda_control} "
                  f"(trained lambda={trained_lambda:.2f})")
            agent_cfg = copy.deepcopy(cfg)
            agent_cfg.sensing_mode = meta.get("sensing_mode", cfg.sensing_mode)
            for lam in [float(x) for x in args.lambdas.split(",") if x.strip()]:
                agent.lam.assign(lam)
                res = evaluate_controller(
                    lambda env, a=agent: LearnedPolicy(a, greedy=True),
                    agent_cfg, seeds[: max(len(seeds) // 2, 2)],
                    f"lamctl_{lam:g}", net_file=net_file,
                    sensor=sensor_for(agent_cfg.sensing_mode),
                    work_dir=RUNS_DIR / f"lamctl_{lam:g}", verbose=False,
                )
                point = summarise(res["aggregate"])
                point["lambda"] = lam
                control.append(point)
                print(f"  lambda={lam:6.2f}  delay={point['delay_s']:6.2f}s  "
                      f"EV wait={point['ev_wait_s']:6.2f}s  "
                      f"cost={point['mean_cost']:.4f}", flush=True)
            agent.lam.assign(trained_lambda)

            waits = [p["ev_wait_s"] for p in control if np.isfinite(p["ev_wait_s"])]
            monotone = all(b <= a + 1e-9 for a, b in zip(waits, waits[1:]))
            print(f"\n  swept curve monotone in EV wait: {monotone}"
                  f"{'' if monotone else '  <-- the shortcut does not trace a frontier'}")

    out = RESULTS_DIR / "frontier.json"
    out.write_text(json.dumps({
        "regime": args.regime,
        "seeds": seeds,
        "horizon_s": args.horizon,
        "ev_rate_per_hour": args.ev_rate,
        "frontier": frontier,
        "lambda_control": control,
        "lambda_control_agent": args.lambda_control,
    }, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
