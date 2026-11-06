"""How good is the emergency request, measured against simulator ground truth?

This is the diagnostic that found both defects this revision fixes, kept as a
script because the way it is scored is the point.

**Recall is reported by distance to the stop line, not in aggregate.** An
aggregate figure is the wrong summary for this system and it actively hid the
bug. A miss 300 m out costs nothing -- the just-in-time rule is deliberately not
acting yet -- while a miss at 20 m is the vehicle stopping at a red. Averaged
together, a collapse in the near bands is diluted by the far bands where a miss
is correct behaviour, and the mean barely moves. Split by band, the failure is
unmistakable: under the single-cardioid mast, recall falls in exactly the bands
where the mast is behind the vehicle.

**Ground truth comes from the simulator, not from the filter.** Every second,
for every emergency vehicle on an approach, the true edge and true distance to
the stop line are read from SUMO and compared against the request the controller
actually raised. Nothing here is scored against the belief, which is what makes
it able to say the belief is wrong.

    python scripts/request_quality.py --seeds 6
    python scripts/request_quality.py --seeds 6 --out results/request_quality.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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

BANDS = [(0.0, 10.0), (10.0, 25.0), (25.0, 50.0),
         (50.0, 100.0), (100.0, 200.0), (200.0, 400.0)]
LABELS = ["0-10 m", "10-25 m", "25-50 m", "50-100 m", "100-200 m", "200-400 m"]


def _band(d: float):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= d < hi:
            return i
    return None


def run_one(dual: bool, seed: int, horizon: int, ev_rate: float,
            checkpoint: Path):
    cfg = EnvConfig()
    cfg.demand = DemandConfig(horizon_s=horizon, seed=seed, ev_rate_per_hour=ev_rate)
    cfg.sensing_mode, cfg.belief_mode, cfg.gui = "surrogate", "joint", False
    cfg.collect_emissions = False
    cfg.step_length_s = 1.0
    cfg.acoustic.dual_element_mast = bool(dual)

    net_file, _ = build_network(cfg.network, DATA_DIR / "networks")
    em = MODELS_DIR / "sensor_error_model.json"
    sensor = AcousticSensor([], cfg.acoustic, mode=cfg.sensing_mode,
                            error_model=SensorErrorModel.load(em) if em.exists() else None,
                            seed=seed)
    env = SumoSignalEnv(cfg, net_file=net_file,
                        work_dir=DATA_DIR / "runs" / "reqquality",
                        label="reqquality", sensor=sensor)
    obs = env.reset(seed=seed)
    agent = LagrangianDuelingDDQN(env.observation_dim, env.n_actions, AgentConfig())
    agent.load(checkpoint)
    ctrl = CorridorPolicy(agent=agent, name="siren_arc")
    ctrl.reset(env)

    hit = Counter()
    tot = Counter()
    stats = {"ev_free_steps": 0, "ev_free_with_request": 0}

    def hook(e):
        sc = e.scenario
        try:
            live = (set(e.conn.vehicle.getIDList()) & set(sc.emergency_vehicles)
                    if sc and sc.emergency_vehicles else set())
        except traci.TraCIException:
            return
        req = set(e.ev_request_phases)
        on_approach = False
        for vid in live:
            try:
                edge = e.conn.vehicle.getRoadID(vid)
                if edge not in e.approaches:
                    continue
                lane = e.conn.vehicle.getLaneID(vid)
                d = (e.conn.lane.getLength(lane)
                     - e.conn.vehicle.getLanePosition(vid))
            except traci.TraCIException:
                continue
            on_approach = True
            b = _band(float(d))
            if b is None:
                continue
            tot[b] += 1
            if req & set(e.phases_serving_approach(edge)):
                hit[b] += 1
        # A spurious request is one raised with no emergency vehicle anywhere on
        # any approach -- the only unambiguous false positive available, since a
        # request naming the wrong approach while a vehicle is present is a
        # localisation error rather than a detection one.
        if not on_approach:
            stats["ev_free_steps"] += 1
            if req:
                stats["ev_free_with_request"] += 1

    env.step_hook = hook
    done = False
    while not done:
        mask = env.action_mask()
        action = ctrl.act(env, obs, mask)
        obs, _r, _c, done, _i = env.step(action, interrupt=ctrl.interrupt)
    summary = dict(env.episode_summary())
    env.close()
    return hit, tot, stats, summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--horizon", type=int, default=1800)
    p.add_argument("--ev-rate", type=float, default=30.0)
    p.add_argument("--checkpoint", default=str(MODELS_DIR / "siren_arc.h5"))
    p.add_argument("--out", default=str(RESULTS_DIR / "request_quality.json"))
    args = p.parse_args()

    ensure_dirs()
    seeds = [SEED_BASE + i for i in range(args.seeds)]
    acc = {}
    for arm, dual in (("single", False), ("dual", True)):
        hit, tot = Counter(), Counter()
        free = free_req = 0
        delays = []
        for seed in seeds:
            h, t, st, summ = run_one(dual, seed, args.horizon, args.ev_rate,
                                     Path(args.checkpoint))
            hit.update(h)
            tot.update(t)
            free += st["ev_free_steps"]
            free_req += st["ev_free_with_request"]
            delays.append(summ.get("ev_delay_per_vehicle_s", float("nan")))
            print(f"  {arm:7s} seed={seed} "
                  f"ev_delay={delays[-1]:6.2f}s", flush=True)
        acc[arm] = {"hit": dict(hit), "tot": dict(tot),
                    "ev_free_steps": free, "ev_free_with_request": free_req,
                    "ev_delay_per_vehicle_s": delays}

    bands = []
    for i, label in enumerate(LABELS):
        row = {"label": label}
        for arm in ("single", "dual"):
            t = acc[arm]["tot"].get(i, 0)
            row[f"recall_{arm}"] = (acc[arm]["hit"].get(i, 0) / t) if t else float("nan")
            row[f"n_{arm}"] = t
        bands.append(row)

    false_rate = {arm: (acc[arm]["ev_free_with_request"]
                        / max(acc[arm]["ev_free_steps"], 1))
                  for arm in ("single", "dual")}

    print("\n" + "=" * 74)
    print(f"{'band':>12} {'single recall':>16} {'dual recall':>16}")
    print("-" * 74)
    for row in bands:
        print(f"{row['label']:>12} "
              f"{100 * row['recall_single']:15.1f}% "
              f"{100 * row['recall_dual']:15.1f}%")
    print("-" * 74)
    for arm in ("single", "dual"):
        print(f"  {arm:7s} spurious-request rate "
              f"{100 * false_rate[arm]:5.1f}%   "
              f"mean EV delay "
              f"{np.nanmean(acc[arm]['ev_delay_per_vehicle_s']):6.2f}s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seeds": seeds, "horizon_s": args.horizon,
                               "ev_rate_per_hour": args.ev_rate,
                               "bands": bands, "false_rate": false_rate,
                               "raw": acc}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
