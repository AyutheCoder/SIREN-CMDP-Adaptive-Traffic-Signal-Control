"""Record one emergency-vehicle approach for the qualitative belief figure.

Aggregate tables show that the controller works; they do not show *how*. This
records a single approach at decision-epoch resolution - true range, the
tracker's estimate, the presence posterior, and whether the signal was actually
serving that approach - so the mechanism can be read directly off a figure.

    python scripts/record_trace.py --agent models/siren_cmdp.h5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atsc.agents.baselines import CorridorPolicy, LearnedPolicy, MaxPressure
from atsc.agents.dueling_ddqn import LagrangianDuelingDDQN
from atsc.config import (
    AgentConfig, DATA_DIR, DemandConfig, EnvConfig, MODELS_DIR, RESULTS_DIR, RUNS_DIR, ensure_dirs,
)
from atsc.env import SumoSignalEnv
from atsc.netgen import build_network
from atsc.sensing.sensor import AcousticSensor, SensorErrorModel


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent", type=str, default=str(MODELS_DIR / "siren_arc.h5"))
    p.add_argument("--seed", type=int, default=31337)
    p.add_argument("--horizon", type=int, default=1800)
    p.add_argument("--ev-rate", type=float, default=25.0)
    p.add_argument("--max-traces", type=int, default=6)
    p.add_argument("--min-segment", type=int, default=8,
                   help="minimum decision epochs for an approach to be kept")
    p.add_argument("--no-corridor", action="store_true",
                   help="trace the learned policy alone, without the corridor layer")
    p.add_argument("--out", type=str, default=str(RESULTS_DIR / "belief_trace.json"))
    args = p.parse_args()

    ensure_dirs()
    cfg = EnvConfig()
    cfg.demand = DemandConfig(horizon_s=args.horizon, seed=args.seed,
                              ev_rate_per_hour=args.ev_rate)
    cfg.collect_emissions = False
    net_file, _ = build_network(cfg.network, DATA_DIR / "networks")

    em_path = MODELS_DIR / "sensor_error_model.json"
    sensor = AcousticSensor(
        [], cfg.acoustic, mode=cfg.sensing_mode,
        error_model=SensorErrorModel.load(em_path) if em_path.exists() else None,
        seed=args.seed,
    )
    env = SumoSignalEnv(cfg, net_file=net_file, work_dir=RUNS_DIR / "trace",
                        label="trace", sensor=sensor)
    obs = env.reset(seed=args.seed)

    ckpt = Path(args.agent)
    if ckpt.exists():
        agent = LagrangianDuelingDDQN(env.observation_dim, env.n_actions, AgentConfig())
        agent.load(ckpt)
        if args.no_corridor:
            controller = LearnedPolicy(agent, greedy=True)
            label = "learned policy"
        else:
            controller = CorridorPolicy(agent=agent, name="siren_arc")
            label = "SIREN-ARC"
        print(f"tracing with {ckpt} (lambda={float(agent.lam.numpy()):.3f})")
    else:
        controller = MaxPressure()
        label = "max-pressure"
        print(f"{ckpt} not found; tracing with the max-pressure baseline instead")
    controller.reset(env)

    min_segment = max(args.min_segment, 2)
    traces: dict = {}

    def sample(env_, req_by_approach, engaged_approach, info_queue):
        """Append one row per approach that currently has an inbound source."""
        scene = env_._acoustic_scene()
        beliefs = env_.belief.beliefs()
        serving = set(env_.phases[env_.current_phase].approaches)
        for approach, sources in scene.items():
            if not sources:
                continue
            nearest = min(sources, key=lambda s: s.distance_m)
            if nearest.off_axis_rad > math.radians(50) or nearest.radial_velocity_ms > 1.0:
                continue
            b = beliefs[approach]
            req = req_by_approach.get(approach)
            traces.setdefault(approach, []).append({
                "t": float(env_.sim_time),
                "approach": env_.approach_direction[approach],
                "true_range": float(nearest.distance_m),
                "tracked_range": float(b.range_m),
                "tracked_rate": float(b.range_rate_ms),
                "presence": float(b.presence),
                "eta": float(b.eta_s),
                "priority": float(b.expected_priority),
                "serving_ev": int(approach in serving),
                "queue": float(info_queue),
                # The corridor's own view: the calibrated probability it acted
                # on, the threshold it derived from the queue state this epoch,
                # and what each side of the trade was worth in seconds.
                "p": float(req.presence) if req is not None else None,
                "theta": (float(req.threshold)
                          if req is not None and math.isfinite(req.threshold) else None),
                "benefit_s": float(req.benefit_s) if req is not None else None,
                "slack_s": float(req.slack_s) if req is not None else None,
                "red_at_arrival_s": (float(req.residual_red_s)
                                     if req is not None else None),
                "cost_s": float(req.cost_s) if req is not None else None,
                "range_m": float(nearest.distance_m),
                "engaged": bool(engaged_approach == approach),
            })

    def snapshot(ctrl):
        """The corridor's per-approach requests and target, if it has any."""
        decision = getattr(ctrl, "last_decision", None)
        reqs, target = {}, None
        if decision is not None:
            for r in getattr(decision, "requests", []) or []:
                reqs[r.approach] = r
            if decision.engaged and decision.target is not None:
                target = decision.target.approach
        return reqs, target

    done = False
    while not done:
        mask = env.action_mask()
        action = controller.act(env, obs, mask)
        # An emergency vehicle crosses a 320 m approach in about 23 s, so
        # sampling once per 5 s decision epoch yields three or four points per
        # vehicle -- too few to show a mechanism. The corridor already
        # re-evaluates every simulated second, so the recorder rides along on
        # that same 1 Hz poll and gets the resolution for free.
        base_interrupt = getattr(controller, "interrupt", None)
        last_queue = [float(env._step_queue)]

        def recording_interrupt(env_, _c=controller, _q=last_queue):
            want = base_interrupt(env_) if base_interrupt is not None else None
            reqs, target = snapshot(_c)
            sample(env_, reqs, target, _q[0])
            return want

        obs, reward, cost, done, info = env.step(action,
                                                 interrupt=recording_interrupt)
        last_queue[0] = float(info["queue"])

        scene = env._acoustic_scene()
        beliefs = env.belief.beliefs()
        serving = set(env.phases[env.current_phase].approaches)


    env.close()

    # Keep the longest continuous approaches; those are the informative ones.
    segments = []
    for approach, rows in traces.items():
        rows.sort(key=lambda r: r["t"])
        current = [rows[0]]
        for prev, row in zip(rows, rows[1:]):
            if row["t"] - prev["t"] <= 6.0 and row["true_range"] < prev["true_range"] + 40:
                current.append(row)
            else:
                segments.append(current)
                current = [row]
        segments.append(current)

    # An approach is 320 m long and vehicles cross it at roughly 15 m/s, so a
    # single run lasts about 21 s -- only four or five decision epochs. Asking
    # for longer segments than that silently discards every approach, which is
    # what an earlier threshold of five did.
    segments = [s for s in segments if len(s) >= min_segment]
    segments.sort(key=len, reverse=True)
    segments = segments[: args.max_traces]

    out = Path(args.out)
    out.write_text(json.dumps({
        "controller": label,
        "seed": args.seed,
        "segments": segments,
    }, indent=2), encoding="utf-8")
    print(f"captured {len(segments)} approach segment(s) -> {out}")
    if segments:
        best = segments[0]
        print(f"  longest: approach {best[0]['approach']}, {len(best)} epochs, "
              f"range {best[0]['true_range']:.0f} m -> {best[-1]['true_range']:.0f} m")
        csv_path = out.with_suffix(".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(best[0].keys()))
            w.writeheader()
            w.writerows(best)
        print(f"  wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
