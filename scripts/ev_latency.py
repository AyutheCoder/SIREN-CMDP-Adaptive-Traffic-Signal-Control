"""The distribution of per-vehicle emergency delay, not its mean.

Why this script exists
----------------------
We evaluated S6 on mean emergency stopped delay and concluded from twelve seeds
that it did nothing (mean $15.49 \\to 15.06$ s, six seeds better of twelve,
paired sd $5.79$ s). That conclusion was drawn against the wrong statistic.

S6 is a **worst-case latency bound**. It does not promise to make the average
emergency vehicle faster -- most of them are already served promptly by the
corridor's economics, and for those the shield never fires at all. What it
promises is that no vehicle waits longer than the deadline once its request
stands. A guarantee on the maximum is invisible in the mean whenever the mean is
dominated by vehicles the guarantee never touches, and it is diluted further by
episode-level averaging, which buries one 60 s vehicle inside an episode of
twenty ordinary ones.

So this reports the per-vehicle distribution: median, p90, p95, maximum, and the
exceedance rate above a stated threshold. Vehicles are pooled across seeds
rather than averaged within them, because the tail is a property of the vehicle
population and episode means destroy it.

    python scripts/ev_latency.py --seeds 12
    python scripts/ev_latency.py --seeds 12 --ev-rate 45
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import traci

from atsc.agents.baselines import CorridorPolicy
from atsc.agents.dueling_ddqn import LagrangianDuelingDDQN
from atsc.config import (
    AgentConfig, DATA_DIR, DemandConfig, EnvConfig, MODELS_DIR, RESULTS_DIR,
    ensure_dirs)
from atsc.env import SumoSignalEnv
from atsc.netgen import build_network
from atsc.sensing.sensor import AcousticSensor, SensorErrorModel

SEED_BASE = 50000

ARMS = {
    "baseline (as shipped)": (False, False),
    "SIREN-ARC v2 (mast + S6)": (True, True),
}


def run_one(dual: bool, s6: bool, seed: int, horizon: int, ev_rate: float,
            checkpoint: Path):
    """One episode; returns the peak wait each emergency vehicle accrued."""
    cfg = EnvConfig()
    cfg.demand = DemandConfig(horizon_s=horizon, seed=seed, ev_rate_per_hour=ev_rate)
    cfg.sensing_mode, cfg.belief_mode, cfg.gui = "surrogate", "joint", False
    cfg.collect_emissions = False
    cfg.step_length_s = 1.0
    cfg.acoustic.dual_element_mast = bool(dual)
    if not s6:
        cfg.control.ev_service_deadline_s = float("inf")

    net_file, _ = build_network(cfg.network, DATA_DIR / "networks")
    em = MODELS_DIR / "sensor_error_model.json"
    sensor = AcousticSensor([], cfg.acoustic, mode=cfg.sensing_mode,
                            error_model=SensorErrorModel.load(em) if em.exists() else None,
                            seed=seed)
    env = SumoSignalEnv(cfg, net_file=net_file,
                        work_dir=DATA_DIR / "runs" / "evlatency",
                        label="evlatency", sensor=sensor)
    obs = env.reset(seed=seed)
    agent = LagrangianDuelingDDQN(env.observation_dim, env.n_actions, AgentConfig())
    agent.load(checkpoint)
    ctrl = CorridorPolicy(agent=agent, name="siren_arc")
    ctrl.reset(env)

    peak: dict = {}

    def hook(e):
        sc = e.scenario
        if not sc or not sc.emergency_vehicles:
            return
        try:
            live = set(e.conn.vehicle.getIDList()) & set(sc.emergency_vehicles)
            for vid in live:
                w = float(e.conn.vehicle.getAccumulatedWaitingTime(vid))
                if w > peak.get(vid, -1.0):
                    peak[vid] = w
        except traci.TraCIException:
            return

    env.step_hook = hook
    done = False
    while not done:
        mask = env.action_mask()
        action = ctrl.act(env, obs, mask)
        obs, _r, _c, done, _i = env.step(action, interrupt=ctrl.interrupt)
    summary = dict(env.episode_summary())
    env.close()
    return list(peak.values()), summary


def describe(waits, threshold: float) -> dict:
    a = np.asarray(waits, dtype=float)
    if a.size == 0:
        return {}
    return {
        "n_vehicles": int(a.size),
        "mean_s": float(a.mean()),
        "median_s": float(np.median(a)),
        "p90_s": float(np.percentile(a, 90)),
        "p95_s": float(np.percentile(a, 95)),
        "max_s": float(a.max()),
        "frac_over_threshold": float((a > threshold).mean()),
        "frac_clean_pass": float((a <= 0.1).mean()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=12)
    p.add_argument("--horizon", type=int, default=1800)
    p.add_argument("--ev-rate", type=float, default=30.0)
    p.add_argument("--threshold", type=float, default=30.0,
                   help="exceedance threshold, seconds of stopped delay")
    p.add_argument("--checkpoint", default=str(MODELS_DIR / "siren_arc.h5"))
    p.add_argument("--out", default=str(RESULTS_DIR / "ev_latency.json"))
    args = p.parse_args()

    ensure_dirs()
    seeds = [SEED_BASE + i for i in range(args.seeds)]
    pooled: dict = {k: [] for k in ARMS}
    per_seed: dict = {k: [] for k in ARMS}

    for seed in seeds:
        for label, (dual, s6) in ARMS.items():
            waits, summ = run_one(dual, s6, seed, args.horizon, args.ev_rate,
                                  Path(args.checkpoint))
            pooled[label].extend(waits)
            per_seed[label].append({"seed": seed, "waits": waits,
                                    "teleports": summ.get("teleports", 0)})
            print(f"  seed={seed} {label:26s} n={len(waits):3d} "
                  f"max={max(waits) if waits else 0:6.1f}s", flush=True)

    print("\n" + "=" * 96)
    print(f"per-vehicle emergency stopped delay, pooled over {args.seeds} seeds "
          f"at {args.ev_rate:g} EV/h")
    print("=" * 96)
    stats = {k: describe(pooled[k], args.threshold) for k in ARMS}
    keys = [("n_vehicles", "emergency vehicles", "{:.0f}"),
            ("mean_s", "mean (s)", "{:.2f}"),
            ("median_s", "median (s)", "{:.2f}"),
            ("p90_s", "p90 (s)", "{:.2f}"),
            ("p95_s", "p95 (s)", "{:.2f}"),
            ("max_s", "worst single vehicle (s)", "{:.2f}"),
            ("frac_over_threshold", f"P(wait > {args.threshold:.0f}s)", "{:.3f}"),
            ("frac_clean_pass", "clean pass (never stopped)", "{:.3f}")]
    header = f"{'statistic':<28}" + "".join(f"{k:>28}" for k in ARMS)
    print(header)
    print("-" * len(header))
    for key, label, fmt in keys:
        row = f"{label:<28}"
        for k in ARMS:
            row += f"{fmt.format(stats[k].get(key, float('nan'))):>28}"
        print(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seeds": seeds, "horizon_s": args.horizon,
                               "ev_rate_per_hour": args.ev_rate,
                               "threshold_s": args.threshold,
                               "stats": stats, "per_seed": per_seed}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
