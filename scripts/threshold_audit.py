"""Record every preemption threshold the scheduler derives, and what drove it.

The central claim of the corridor layer is that the commitment threshold is not
a tuned constant but a quantity computed from the traffic state at each epoch:
``theta* = C / (gamma (1 + lambda) B)``, where ``C`` is the background delay a
preemption would cost and ``B`` the emergency delay it would avoid. A claim like
that is only worth making if the threshold demonstrably moves, and moves for the
stated reason.

This script logs one row per (epoch, approach-with-evidence): the derived
threshold, the calibrated posterior, the two sides of the trade in seconds, the
queue that made preemption expensive, and whether the corridor opened. The
resulting figure is the audit -- if the threshold were secretly a constant, it
would be a horizontal line.

    python scripts/threshold_audit.py --seeds 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from atsc.agents.baselines import CorridorPolicy
from atsc.agents.dueling_ddqn import LagrangianDuelingDDQN
from atsc.config import (
    AgentConfig, DATA_DIR, DemandConfig, EnvConfig, MODELS_DIR, RESULTS_DIR, RUNS_DIR, ensure_dirs)
from atsc.control.corridor import CorridorScheduler
from atsc.env import SumoSignalEnv
from atsc.netgen import build_network
from atsc.sensing.sensor import AcousticSensor, SensorErrorModel


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=str(MODELS_DIR / "siren_arc.h5"))
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--seed-start", type=int, default=770000,
                   help="disjoint from training, calibration and evaluation seeds")
    p.add_argument("--horizon", type=int, default=3600)
    p.add_argument("--ev-rate", type=float, default=40.0)
    p.add_argument("--value-ratio", type=float, default=50.0)
    p.add_argument("--out", default=str(RESULTS_DIR / "threshold_audit.json"))
    args = p.parse_args()

    ensure_dirs()
    cfg = EnvConfig()
    cfg.demand = DemandConfig(horizon_s=args.horizon, ev_rate_per_hour=args.ev_rate)
    cfg.gui = False
    cfg.collect_emissions = False

    net_file, _ = build_network(cfg.network, DATA_DIR / "networks")
    em = MODELS_DIR / "sensor_error_model.json"
    error_model = SensorErrorModel.load(em) if em.exists() else None

    rows = []
    agent = None
    for i in range(args.seeds):
        seed = args.seed_start + i
        sensor = AcousticSensor([], cfg.acoustic, mode=cfg.sensing_mode,
                                error_model=error_model, seed=seed)
        env = SumoSignalEnv(cfg, net_file=net_file,
                            work_dir=RUNS_DIR / "threshold_audit",
                            label=f"thr_{seed}", sensor=sensor)
        obs = env.reset(seed=seed)
        if agent is None:
            agent = LagrangianDuelingDDQN(env.observation_dim, env.n_actions,
                                          AgentConfig())
            agent.load(Path(args.checkpoint))
        ctrl = CorridorPolicy(
            agent=agent, name="arc",
            scheduler=CorridorScheduler(ev_value_ratio=args.value_ratio))
        ctrl.reset(env)

        done = False
        while not done:
            mask = env.action_mask()
            action = ctrl.act(env, obs, mask)
            decision = ctrl.last_decision
            if decision is not None:
                queues = env.phase_queues()
                total_q = float(queues.sum())
                for r in decision.requests or []:
                    # Cross-traffic queue is what makes a corridor expensive:
                    # the vehicles that would be denied service by taking this
                    # request's cheapest serving phase.
                    served_q = max(float(queues[ph]) for ph in r.phases)
                    rows.append({
                        "seed": seed,
                        "t": float(env.sim_time),
                        "approach": r.direction,
                        "p": float(r.presence),
                        "theta": (float(r.threshold)
                                  if np.isfinite(r.threshold) else None),
                        "benefit_s": float(r.benefit_s),
                        "cost_s": float(r.cost_s),
                        "queue_total": total_q,
                        "queue_cross": float(total_q - served_q),
                        "queue_own": float(r.queue),
                        "range_m": float(r.range_m),
                        "eta_s": float(r.eta_s) if np.isfinite(r.eta_s) else None,
                        # The just-in-time rule turns on these three: the red
                        # the vehicle is projected to meet when it arrives, the
                        # time the corridor needs, and how much longer the
                        # scheduler may wait before it must act.
                        "red_at_arrival_s": float(r.residual_red_s),
                        "open_time_s": float(r.open_time_s),
                        "slack_s": float(r.slack_s),
                        "due": bool(r.due),
                        "committed": bool(r.committed),
                        "engaged": bool(decision.engaged
                                        and decision.target is not None
                                        and decision.target.approach == r.approach),
                        "conflict": bool(decision.conflict),
                    })
            obs, _, _, done, _ = env.step(action, interrupt=ctrl.interrupt)
        env.close()
        print(f"  seed {seed}: {len(rows)} request-epochs so far", flush=True)

    finite = [r for r in rows if r["theta"] is not None]
    th = np.asarray([r["theta"] for r in finite])
    print("\n" + "=" * 66)
    print(f"  request-epochs logged        {len(rows)}")
    print(f"  with a finite threshold      {len(finite)}")
    if finite:
        print(f"  threshold  min / med / max   {th.min():.3f} / "
              f"{np.median(th):.3f} / {th.max():.3f}")
        print(f"  distinct values (3 dp)       {len(set(np.round(th, 3)))}")
        print(f"  committed                    {sum(r['committed'] for r in rows)}")
        print(f"  conflicts                    {sum(r['conflict'] for r in rows)}")
    print("=" * 66)

    out = Path(args.out)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
