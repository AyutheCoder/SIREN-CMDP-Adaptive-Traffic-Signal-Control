"""How tight should the S6 emergency-service deadline be?

The deadline is the operator's service-level promise, but it is also, and less
obviously, a *trigger sensitivity*. S6 begins forcing at
``tau_ev = deadline - (g_ev + poll + yellow + all_red)``, which on the shipped
timings is ``16 - 14 = 2`` s: a request that stands for two seconds already
compels the shield. At that setting S6 is not a backstop behind the risk
calculus, it is a hair-trigger preemptor in front of it, and it acts on every
request the detector raises -- including the wrong ones.

That matters because the detector's precision is finite. Measured on this
network, the single-cardioid mast raises a request with no emergency vehicle
anywhere on 11.7% of steps, and S6 at a 2 s trigger converts each of those into
a forced green on the wrong approach: emergency delay rose from 15.5 s to
21.0 s, background queue from 10.7 to 15.1 vehicles, and one seed of twelve
gridlocked. The two-element mast halves the spurious rate and the damage
disappears -- but the sensitivity of the result to a single timing constant is
itself worth measuring rather than assuming.

Loosening the deadline weakens the guarantee and gives the corridor's economics
more room to act first. This sweeps that trade on shared seeds.

    python scripts/s6_sweep.py --seeds 12
    python scripts/s6_sweep.py --seeds 12 --single-element   # the fragile case
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from atsc.agents.baselines import CorridorPolicy
from atsc.agents.dueling_ddqn import LagrangianDuelingDDQN
from atsc.config import (
    AgentConfig, DATA_DIR, DemandConfig, EnvConfig, MODELS_DIR, RESULTS_DIR,
    ensure_dirs)
from atsc.env import SumoSignalEnv
from atsc.netgen import build_network
from atsc.sensing.sensor import AcousticSensor, SensorErrorModel

SEED_BASE = 50000
DEADLINES = [16.0, 25.0, 40.0, 60.0, float("inf")]
"""``inf`` is S6 disabled, kept in the sweep as its own column so the guarantee
is priced against having none rather than against the tightest setting."""


def run_one(deadline: float, dual: bool, seed: int, horizon: int,
            ev_rate: float, checkpoint: Path) -> dict:
    cfg = EnvConfig()
    cfg.demand = DemandConfig(horizon_s=horizon, seed=seed, ev_rate_per_hour=ev_rate)
    cfg.sensing_mode, cfg.belief_mode, cfg.gui = "surrogate", "joint", False
    cfg.collect_emissions = False
    cfg.step_length_s = 1.0
    cfg.acoustic.dual_element_mast = bool(dual)
    cfg.control.ev_service_deadline_s = float(deadline)

    net_file, _ = build_network(cfg.network, DATA_DIR / "networks")
    em = MODELS_DIR / "sensor_error_model.json"
    sensor = AcousticSensor([], cfg.acoustic, mode=cfg.sensing_mode,
                            error_model=SensorErrorModel.load(em) if em.exists() else None,
                            seed=seed)
    env = SumoSignalEnv(cfg, net_file=net_file, work_dir=DATA_DIR / "runs" / "s6sweep",
                        label="s6sweep", sensor=sensor)
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
            for vid in set(e.conn.vehicle.getIDList()) & set(sc.emergency_vehicles):
                w = float(e.conn.vehicle.getAccumulatedWaitingTime(vid))
                if w > peak.get(vid, -1.0):
                    peak[vid] = w
        except Exception:
            return

    env.step_hook = hook
    done = False
    while not done:
        mask = env.action_mask()
        obs, _r, _c, done, _i = env.step(ctrl.act(env, obs, mask),
                                         interrupt=ctrl.interrupt)
    out = dict(env.episode_summary())
    out["ev_waits"] = list(peak.values())
    out["s6_forced"] = float(env.shield.n_ev_forced)
    env.close()
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=12)
    p.add_argument("--horizon", type=int, default=1800)
    p.add_argument("--ev-rate", type=float, default=30.0)
    p.add_argument("--single-element", action="store_true",
                   help="run the sweep on the deaf single-cardioid mast, where "
                        "S6's sensitivity to its own trigger is largest")
    p.add_argument("--checkpoint", default=str(MODELS_DIR / "siren_arc.h5"))
    p.add_argument("--out", default=str(RESULTS_DIR / "s6_sweep.json"))
    args = p.parse_args()

    ensure_dirs()
    dual = not args.single_element
    seeds = [SEED_BASE + i for i in range(args.seeds)]
    rows = {}
    t0 = time.time()

    for d in DEADLINES:
        runs = [run_one(d, dual, s, args.horizon, args.ev_rate,
                        Path(args.checkpoint)) for s in seeds]
        rows[str(d)] = runs
        waits = np.array([w for r in runs for w in r["ev_waits"]], dtype=float)
        print(f"  deadline={d:>5}s  ev_delay="
              f"{np.nanmean([r.get('ev_delay_per_vehicle_s', np.nan) for r in runs]):6.2f}s "
              f"queue={np.nanmean([r.get('mean_queue', np.nan) for r in runs]):6.2f} "
              f"p95={np.percentile(waits, 95) if waits.size else float('nan'):6.1f}s "
              f"max={waits.max() if waits.size else float('nan'):6.1f}s "
              f"teleports={sum(r.get('teleports', 0) for r in runs):.0f}", flush=True)

    print(f"\n{'=' * 92}")
    print(f"{'S6 deadline':>14} {'EV delay (s)':>14} {'queue (veh)':>13} "
          f"{'p95 (s)':>9} {'max (s)':>9} {'P(>30s)':>9} {'jam':>5}")
    print("-" * 92)
    for d in DEADLINES:
        runs = rows[str(d)]
        waits = np.array([w for r in runs for w in r["ev_waits"]], dtype=float)
        label = "off" if d == float("inf") else f"{d:.0f} s"
        print(f"{label:>14} "
              f"{np.nanmean([r.get('ev_delay_per_vehicle_s', np.nan) for r in runs]):14.2f} "
              f"{np.nanmean([r.get('mean_queue', np.nan) for r in runs]):13.2f} "
              f"{np.percentile(waits, 95) if waits.size else float('nan'):9.1f} "
              f"{waits.max() if waits.size else float('nan'):9.1f} "
              f"{(waits > 30).mean() if waits.size else float('nan'):9.3f} "
              f"{sum(r.get('teleports', 0) for r in runs):5.0f}")
    print(f"\n{time.time() - t0:.0f}s wall   mast="
          f"{'two-element' if dual else 'single cardioid'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seeds": seeds, "dual_element": dual,
                               "ev_rate_per_hour": args.ev_rate,
                               "horizon_s": args.horizon, "rows": rows}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
