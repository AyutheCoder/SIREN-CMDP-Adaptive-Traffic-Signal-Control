"""Train the SIREN-CMDP controller.

    python scripts/train_agent.py --episodes 150 --budget 0.06 --run-name siren_cmdp

Ablations are selected with the same script, so every variant shares one code
path and differences cannot come from implementation drift:

    --sensing oracle      perfect V2I detection (upper bound)
    --sensing none        no emergency awareness (efficiency-only agent)
    --unconstrained       scalarised reward, no dual variable
    --no-shield-bound     relax the starvation guarantee (shows what it costs)

Model selection is constraint-aware: the checkpoint kept is the best-performing
evaluation *among those that satisfy the cost budget*, falling back to the
lowest-cost evaluation if none do. Selecting on reward alone would happily
return a policy that ignores the constraint.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atsc.agents.baselines import LearnedPolicy
from atsc.agents.dueling_ddqn import LagrangianDuelingDDQN
from atsc.agents.replay import PrioritizedReplay, Transition
from atsc.config import (
    AgentConfig, DATA_DIR, DemandConfig, EnvConfig, MODELS_DIR, RESULTS_DIR, RUNS_DIR,
    TrainConfig, ensure_dirs, to_json,
)
from atsc.env import SumoSignalEnv
from atsc.evaluation.runner import run_episode
from atsc.netgen import build_network
from atsc.sensing.sensor import SensorErrorModel

MIXED_REGIMES = ("balanced", "asymmetric", "peak", "surge")
"""Demand patterns drawn from, one per episode, under ``--regime mixed``."""

LAMBDA_UPDATE_EVERY = 25
"""Dual-ascent period, in environment decisions. Slower than the critic update
(every ``train_every`` steps) to satisfy the two-timescale condition."""


def build_env_cfg(args) -> EnvConfig:
    cfg = EnvConfig()
    cfg.demand = DemandConfig(
        horizon_s=args.horizon,
        seed=args.seed,
        # "mixed" is resolved per episode in the training loop.
        regime="balanced" if args.regime == "mixed" else args.regime,
        base_flow_vph=args.flow,
        ev_rate_per_hour=args.ev_rate,
    )
    cfg.sensing_mode = args.sensing
    cfg.belief_mode = args.belief_mode
    cfg.seed = args.seed
    cfg.collect_emissions = False  # not needed while training; saves I/O
    if args.no_shield_bound:
        cfg.control.max_starvation_s = 600.0
    return cfg


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--horizon", type=int, default=1800)
    p.add_argument("--regime", type=str, default="balanced")
    p.add_argument("--flow", type=float, default=700.0)
    p.add_argument("--ev-rate", type=float, default=24.0,
                   help="emergency vehicles per hour; matched to the evaluation "
                        "rate so the cost critic trains on the deployment "
                        "distribution")
    p.add_argument("--sensing", type=str, default="surrogate",
                   choices=["surrogate", "oracle", "none", "end_to_end"])
    p.add_argument("--budget", type=float, default=0.06,
                   help="CMDP cost budget d0, in mean cost per decision epoch")
    p.add_argument("--unconstrained", action="store_true")
    p.add_argument("--no-shield-bound", action="store_true")
    p.add_argument("--belief-mode", type=str, default="joint",
                   choices=["joint", "independent"],
                   help="presence filter the policy's belief features come "
                        "from. A policy must be evaluated with the filter it "
                        "was trained on: the two produce differently scaled "
                        "features and swapping them measures a train/test "
                        "mismatch, not an ablation.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=15)
    p.add_argument("--eval-episodes", type=int, default=4,
                   help="held-out episodes per checkpoint evaluation. Two is too "
                        "few: emergency events are rare enough that the "
                        "per-episode cost estimate is noisy, and checkpoint "
                        "selection then turns on that noise.")
    p.add_argument("--run-name", type=str, default="siren_cmdp")
    p.add_argument("--error-model", type=str,
                   default=str(MODELS_DIR / "sensor_error_model.json"))
    args = p.parse_args()

    ensure_dirs()
    env_cfg = build_env_cfg(args)
    agent_cfg = AgentConfig(
        constrained=not args.unconstrained,
        cost_budget=args.budget,
        seed=args.seed,
    )
    train_cfg = TrainConfig(
        episodes=args.episodes,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        run_name=args.run_name,
    )

    run_dir = RESULTS_DIR / "training" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    to_json(env_cfg, run_dir / "env_config.json")
    to_json(agent_cfg, run_dir / "agent_config.json")

    net_file, _ = build_network(env_cfg.network, DATA_DIR / "networks")

    error_model = None
    em_path = Path(args.error_model)
    if em_path.exists():
        error_model = SensorErrorModel.load(em_path)
        print(f"loaded fitted sensor error model from {em_path}")
    else:
        print(f"WARNING: {em_path} not found; falling back to the hand-specified "
              "sensor error model. Run scripts/train_detector.py first for "
              "results that reflect the real detector.")

    env = SumoSignalEnv(env_cfg, net_file=net_file,
                        work_dir=RUNS_DIR / args.run_name,
                        label=f"train_{args.run_name}")
    env.reset(seed=args.seed)
    state_dim, n_actions = env.observation_dim, env.n_actions
    env.close()
    print(f"state_dim={state_dim}  n_actions={n_actions}  "
          f"constrained={agent_cfg.constrained}  budget={agent_cfg.cost_budget}  "
          f"sensing={env_cfg.sensing_mode}")

    agent = LagrangianDuelingDDQN(state_dim, n_actions, agent_cfg)
    buffer = PrioritizedReplay(
        capacity=agent_cfg.buffer_size,
        state_dim=state_dim,
        n_actions=n_actions,
        alpha=agent_cfg.per_alpha,
        beta_start=agent_cfg.per_beta_start,
        beta_final=agent_cfg.per_beta_final,
        beta_steps=agent_cfg.epsilon_decay_steps,
        eps=agent_cfg.per_eps,
        n_step=agent_cfg.n_step,
        gamma=agent_cfg.gamma,
        seed=args.seed,
    )

    if error_model is not None:
        from atsc.sensing.sensor import AcousticSensor
        shared_sensor = AcousticSensor(
            [], env_cfg.acoustic, mode=env_cfg.sensing_mode,
            error_model=error_model, seed=args.seed,
        )
    else:
        shared_sensor = None

    losses: list = []

    def on_transition(o, a, r, c, o2, done, next_mask):
        buffer.push(Transition(o, int(a), float(r), float(c), o2, bool(done), next_mask))
        agent.env_steps += 1
        agent.observe_cost(c)
        if agent.env_steps % agent_cfg.train_every == 0:
            loss = agent.learn(buffer)
            if loss is not None:
                losses.append(loss)
        if agent.env_steps % LAMBDA_UPDATE_EVERY == 0:
            agent.update_lambda()

    log_path = run_dir / "train_log.csv"
    fields = ["episode", "reward", "cost", "mean_cost", "mean_queue", "throughput",
              "ev_delay_per_vehicle_s", "max_movement_wait_s", "epsilon", "lambda",
              "cost_ema", "loss", "buffer", "wall_s"]
    log_file = log_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(log_file, fieldnames=fields)
    writer.writeheader()

    best_key = (float("inf"), float("inf"))
    best_score = float("inf")
    best_cost = float("inf")
    best_path = MODELS_DIR / f"{args.run_name}.h5"
    eval_history: list = []
    # Travels with the checkpoint so evaluation reproduces training conditions.
    checkpoint_meta = {
        "sensing_mode": env_cfg.sensing_mode,
        "belief_mode": env_cfg.belief_mode,
        "run_name": args.run_name,
        "train_episodes": train_cfg.episodes,
        "train_horizon_s": args.horizon,
        "train_regime": args.regime,
        "ev_rate_per_hour": args.ev_rate,
        "max_starvation_s": env_cfg.control.max_starvation_s,
    }

    print(f"\ntraining {train_cfg.episodes} episodes "
          f"({args.horizon}s each) -> {best_path}\n")
    t_start = time.time()

    for ep in range(train_cfg.episodes):
        t0 = time.time()
        if args.regime == "mixed":
            # One demand pattern per episode, drawn from the training seed, so
            # the policy sees balanced, asymmetric, peak and surge traffic in
            # roughly equal measure and the draw is reproducible.
            env_cfg.demand.regime = MIXED_REGIMES[
                np.random.default_rng(args.seed * 7919 + ep).integers(len(MIXED_REGIMES))]
        train_env = SumoSignalEnv(
            env_cfg, net_file=net_file,
            work_dir=RUNS_DIR / args.run_name,
            label=f"train_{args.run_name}",
            sensor=shared_sensor,
        )
        summary = run_episode(
            train_env,
            lambda e: LearnedPolicy(agent, greedy=False),
            seed=args.seed + 1000 + ep,
            on_transition=on_transition,
            parse_trips=False,
        )
        buffer.flush()

        row = {
            "episode": ep,
            "reward": round(summary["total_reward"], 3),
            "cost": round(summary["total_cost"], 4),
            "mean_cost": round(summary["mean_cost"], 5),
            "mean_queue": round(summary["mean_queue"], 3),
            "throughput": summary["throughput"],
            "ev_delay_per_vehicle_s": round(summary["ev_delay_per_vehicle_s"], 2),
            "max_movement_wait_s": round(summary["max_movement_wait_s"], 1),
            "epsilon": round(agent.epsilon, 4),
            "lambda": round(float(agent.lam.numpy()), 4),
            "cost_ema": round(agent.cost_ema, 5) if not np.isnan(agent.cost_ema) else "",
            "loss": round(float(np.mean(losses[-200:])), 5) if losses else "",
            "buffer": len(buffer),
            "wall_s": round(time.time() - t0, 1),
        }
        writer.writerow(row)
        log_file.flush()
        print(
            f"ep {ep:3d}  R={row['reward']:9.2f}  q={row['mean_queue']:6.2f}  "
            f"evdelay={row['ev_delay_per_vehicle_s']:6.2f}s  "
            f"lam={row['lambda']:6.3f}  cost_ema={row['cost_ema'] or 0:.4f}  "
            f"eps={row['epsilon']:.3f}  {row['wall_s']:.0f}s",
            flush=True,
        )

        if (ep + 1) % train_cfg.eval_every == 0 or ep == train_cfg.episodes - 1:
            ev = evaluate(agent, env_cfg, net_file, args, train_cfg.eval_episodes, shared_sensor)
            ev["episode"] = ep
            eval_history.append(ev)
            print(f"    eval: delay={ev['mean_time_loss_s']:.2f}s  "
                  f"ev_stopped={ev['ev_mean_waiting_time_s']:.2f}s  "
                  f"ev_stops={ev['ev_mean_stops']:.2f}  "
                  f"clean={ev['ev_zero_stop_rate']:.2f}  "
                  f"cost={ev['mean_cost']:.4f}  queue={ev['mean_queue']:.2f}",
                  flush=True)

            # Lexicographic selection on (constraint excess, delay).
            #
            # Among feasible evaluations the excess is zero for all of them, so
            # the rule reduces to "lowest delay". Among infeasible ones it
            # prefers the least violation first, which is what the constrained
            # problem asks for, but breaks ties on delay -- without that
            # tie-break, an infeasible run can keep a checkpoint that is
            # marginally cheaper on cost while being far worse for everyone
            # else, which is not a controller anyone would deploy.
            excess = 0.0 if not agent_cfg.constrained else max(
                ev["mean_cost"] - agent_cfg.cost_budget, 0.0)
            score = ev["mean_time_loss_s"]
            key = (round(excess, 6), score)
            if key < best_key:
                best_key = key
                best_score, best_cost = score, ev["mean_cost"]
                agent.save(best_path, extra=checkpoint_meta)
                print(f"    saved checkpoint (excess={excess:.4f}, delay={score:.2f}s)",
                      flush=True)

    log_file.close()
    if not best_path.exists():
        agent.save(best_path)
    (run_dir / "eval_history.json").write_text(
        json.dumps(eval_history, indent=2), encoding="utf-8")
    if best_key[0] <= 0.0:
        verdict = f"feasible; best delay {best_score:.2f}s at cost {best_cost:.4f}"
    else:
        verdict = (f"INFEASIBLE at d0={agent_cfg.cost_budget}: best achievable cost was "
                   f"{best_cost:.4f} (excess {best_key[0]:.4f}), delay {best_score:.2f}s")
    print(f"\ndone in {(time.time() - t_start) / 60:.1f} min; {verdict} -> {best_path}")
    return 0


def evaluate(agent, env_cfg, net_file, args, n_episodes: int, sensor) -> dict:
    """Greedy roll-outs on held-out seeds."""
    import copy
    cfg = copy.deepcopy(env_cfg)
    cfg.collect_emissions = True
    results = []
    for i in range(n_episodes):
        if args.regime == "mixed":
            # Evaluate on every pattern in turn, so checkpoint selection does
            # not favour whichever pattern the last training episode drew.
            cfg.demand.regime = MIXED_REGIMES[i % len(MIXED_REGIMES)]
        env = SumoSignalEnv(
            cfg, net_file=net_file,
            work_dir=RUNS_DIR / f"{args.run_name}_eval",
            label=f"eval_{args.run_name}_{i}",
            tripinfo_path=RUNS_DIR / f"{args.run_name}_eval" / f"tripinfo_{i}.xml",
            sensor=sensor,
        )
        results.append(run_episode(
            env, lambda e: LearnedPolicy(agent, greedy=True), seed=90000 + i
        ))
    # ev_mean_waiting_time_s (stopped delay) leads, not ev_mean_time_loss_s:
    # time loss is measured against the EV's own desired speed, which is higher
    # than surrounding traffic's, so it contains a component no controller can
    # remove and it obscures progress.
    keys = ["mean_time_loss_s", "ev_mean_waiting_time_s", "ev_mean_stops",
            "ev_zero_stop_rate", "mean_cost", "mean_queue", "throughput",
            "max_movement_wait_s", "ev_mean_time_loss_s", "p95_time_loss_s"]
    out = {}
    for k in keys:
        vals = [r.get(k, float("nan")) for r in results]
        vals = [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
