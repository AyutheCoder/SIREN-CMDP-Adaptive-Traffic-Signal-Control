"""Record one emergency vehicle's approach under several controllers.

Aggregate tables say a controller is better. They do not show *what it did*, and
for a preemption scheme that is the whole question: did the light change before
the vehicle arrived, or after it had already stopped?

This records the classic time-space diagram. The demand realisation is fixed by
the seed, so the same emergency vehicle -- same id, same entry time, same route
-- enters under every controller, and the only thing that differs is the signal.
For each run we log, once per simulated second:

* the vehicle's signed distance to the stop line, positive upstream;
* its speed;
* whether the signal is showing green to its approach.

The result is a figure in which a controller that anticipates shows an unbroken
line into the junction against a green band, and one that reacts shows a line
that flattens at the stop line and a green band that arrives late. Nothing is
averaged, nothing is aggregated, and the reader can check the claim by eye.

    python scripts/record_trajectory.py
    python scripts/record_trajectory.py --seed 4041 --controllers siren_arc,fixed_time
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from atsc.agents.baselines import CorridorPolicy, LearnedPolicy, build_baseline
from atsc.agents.dueling_ddqn import LagrangianDuelingDDQN
from atsc.config import (
    AgentConfig, DATA_DIR, DemandConfig, EnvConfig, MODELS_DIR, RESULTS_DIR, RUNS_DIR, ensure_dirs)
from atsc.control.corridor import CorridorScheduler
from atsc.env import SumoSignalEnv
from atsc.netgen import build_network, write_actuated_program
from atsc.sensing.sensor import AcousticSensor, SensorErrorModel

TRACE_SEED_BASE = 4040


def make_controller(name: str, env, cfg, agent):
    """Trained run names go through the corridor; anything else is a baseline."""
    if name in ("siren_arc", "arc_base_indep", "siren_cmdp", "no_sensing"):
        return CorridorPolicy(agent=agent, name=name, adaptive=False,
                              scheduler=CorridorScheduler())
    if name == "base_only":
        return LearnedPolicy(agent, name=name, greedy=True)
    return build_baseline(name, env, cfg.demand)


def record_run(cfg, net_file, error_model, seed, name, agent, additional):
    sensor = AcousticSensor([], cfg.acoustic, mode=cfg.sensing_mode,
                            error_model=error_model, seed=seed)
    env = SumoSignalEnv(cfg, net_file=net_file,
                        work_dir=RUNS_DIR / "trajectory",
                        label=f"traj_{name}_{seed}", sensor=sensor,
                        additional_files=additional)
    obs = env.reset(seed=seed)
    controller = make_controller(name, env, cfg, agent)
    controller.reset(env)

    tracks: dict = {}

    def sample(e):
        conn = e.conn
        evs = getattr(e.scenario, "emergency_vehicles", None) or {}
        if not evs:
            return
        for vid in set(conn.vehicle.getIDList()) & set(evs):
            try:
                road = conn.vehicle.getRoadID(vid)
                px, py = conn.vehicle.getPosition(vid)
                speed = float(conn.vehicle.getSpeed(vid))
            except Exception:
                continue
            entry = evs[vid].entry_edge
            if entry not in e.stop_line:
                continue
            sx, sy = e.stop_line[entry]
            d = float(np.hypot(px - sx, py - sy))
            # Signed: still on the entry edge means upstream, anything else
            # means the vehicle has crossed and the sign flips.
            signed = d if road == entry else -d
            rec = tracks.setdefault(vid, {"t": [], "d": [], "v": [], "green": [],
                                          "approach": entry,
                                          "veh_class": evs[vid].veh_class})
            rec["t"].append(float(e.sim_time))
            rec["d"].append(signed)
            rec["v"].append(speed)
            rec["green"].append(bool(entry in e._serving_approaches))

    env.step_hook = sample
    done = False
    while not done:
        mask = env.action_mask()
        action = controller.act(env, obs, mask)
        obs, _, _, done, _ = env.step(action, interrupt=controller.interrupt)
    summary = env.episode_summary()
    env.close()
    return tracks, summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=TRACE_SEED_BASE)
    p.add_argument("--horizon", type=int, default=1200)
    p.add_argument("--ev-rate", type=float, default=30.0)
    p.add_argument("--controllers", type=str,
                   default="siren_arc,base_only,max_pressure_preempt,fixed_time")
    p.add_argument("--checkpoint", type=str, default=str(MODELS_DIR / "siren_arc.h5"))
    p.add_argument("--out", type=str, default=str(RESULTS_DIR / "trajectories.json"))
    args = p.parse_args()

    ensure_dirs()
    cfg = EnvConfig()
    cfg.demand = DemandConfig(horizon_s=args.horizon, seed=args.seed,
                              ev_rate_per_hour=args.ev_rate)
    cfg.collect_emissions = False
    # One sample per simulated second is enough to draw a trajectory and keeps
    # the recording from changing the thing it records.
    cfg.step_length_s = 1.0

    net_file, _ = build_network(cfg.network, DATA_DIR / "networks")
    actuated = [write_actuated_program(
        net_file, DATA_DIR / "networks" / "actuated.add.xml",
        min_dur=cfg.control.min_green_s, max_dur=cfg.control.max_green_s)]
    em_path = MODELS_DIR / "sensor_error_model.json"
    error_model = SensorErrorModel.load(em_path) if em_path.exists() else None

    agent = None
    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        probe = SumoSignalEnv(cfg, net_file=net_file,
                              work_dir=RUNS_DIR / "trajectory", label="probe")
        probe.reset(seed=args.seed)
        agent = LagrangianDuelingDDQN(probe.observation_dim, probe.n_actions,
                                      AgentConfig())
        probe.close()
        agent.load(ckpt)
    else:
        print(f"  no checkpoint at {ckpt}; learned entries will be skipped")

    out: dict = {"seed": args.seed, "horizon_s": args.horizon, "runs": {}}
    for name in [c.strip() for c in args.controllers.split(",") if c.strip()]:
        if name in ("siren_arc", "base_only", "arc_base_indep", "siren_cmdp") \
                and agent is None:
            continue
        adds = actuated if name == "actuated" else None
        tracks, summary = record_run(cfg, net_file, error_model, args.seed,
                                     name, agent, adds)
        out["runs"][name] = {
            "tracks": tracks,
            "ev_delay_per_vehicle_s": summary.get("ev_delay_per_vehicle_s"),
            "mean_queue": summary.get("mean_queue"),
        }
        stopped = sum(1 for r in tracks.values() if min(r["v"]) < 0.3)
        print(f"  {name:24s} {len(tracks)} emergency vehicles, "
              f"{stopped} of them came to a stop, "
              f"mean stopped delay {summary.get('ev_delay_per_vehicle_s', float('nan')):.1f}s",
              flush=True)

    # Vehicles common to every run: only those support a like-for-like picture.
    common = None
    for run in out["runs"].values():
        ids = set(run["tracks"])
        common = ids if common is None else (common & ids)
    out["common_vehicles"] = sorted(common or [])
    Path(args.out).write_text(json.dumps(out, indent=2, default=float),
                              encoding="utf-8")
    print(f"\n{len(out['common_vehicles'])} emergency vehicles appear in every run")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
