"""Where does an emergency vehicle's stopped time go?

    python scripts/ev_wait_breakdown.py --seeds 12

S4 bounds the time from a standing emergency *request* to a green for the
requested approach (16 s, or 88 s when S3 must first serve starving phases).
The headline metric, stopped delay, counts every second an emergency vehicle
stands still -- including seconds before any request exists and seconds spent
behind a queue that is already moving off a green. This script runs SIREN-ARC
on the evaluation seeds and labels each stopped second by what the vehicle's
own signal was showing at that moment:

* ``green``            -- its light is green; it is waiting for the queue ahead;
* ``red_with_request`` -- red, and a request for its approach is standing, so
                          the S4 deadline is running;
* ``red_no_request``   -- red, and no request has been raised for its approach
                          (not yet detected, or detected but not yet requested).

It also records, per vehicle, the longest unbroken run of ``red_with_request``
seconds, which is the quantity Theorem 2 bounds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atsc.agents.baselines import CorridorPolicy
from atsc.config import DATA_DIR, DemandConfig, EnvConfig, MODELS_DIR, RESULTS_DIR, RUNS_DIR
from atsc.control.corridor import CorridorScheduler
from atsc.env import SumoSignalEnv
from atsc.evaluation.runner import evaluate_controller
from atsc.netgen import build_network
from atsc.sensing.sensor import AcousticSensor, SensorErrorModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import load_agent  # noqa: E402

LOG: dict = {}


def _patched_collect(orig):
    def collect(self):
        orig(self)
        conn = self.conn
        t = conn.simulation.getTime()
        for vid in conn.vehicle.getIDList():
            if not vid.startswith("ev_"):
                continue
            rec = LOG.setdefault((self.label, vid), {
                "green": 0, "red_with_request": 0, "red_no_request": 0,
                "other": 0, "run": 0, "max_run": 0})
            if conn.vehicle.getSpeed(vid) >= 0.1:
                rec["run"] = 0
                continue
            nxt = conn.vehicle.getNextTLS(vid)
            if not nxt:
                rec["other"] += 1
                rec["run"] = 0
                continue
            state = nxt[0][3]
            if state in "Gg":
                rec["green"] += 1
                rec["run"] = 0
                continue
            edge = conn.vehicle.getRoadID(vid)
            tls = nxt[0][0]
            full = conn.trafficlight.getRedYellowGreenState(tls)
            links = conn.trafficlight.getControlledLinks(tls)
            mine = [i for i, l in enumerate(links)
                    if l and l[0][0].rsplit("_", 1)[0] == edge]
            if "y" in full.lower():
                rec["changeover"] = rec.get("changeover", 0) + 1
                rec["run"] = 0
                continue
            if any(full[i] in "Gg" for i in mine):
                # Another movement on the vehicle's own road is green: the
                # request counts as served, but this vehicle's turn is not.
                rec["other_movement_green"] = rec.get("other_movement_green", 0) + 1
                rec["run"] = 0
                continue
            requested = any(edge in self.phases[p].approaches
                            for p in self.ev_request_phases
                            if 0 <= p < len(self.phases))
            if requested:
                rec["red_with_request"] += 1
                rec["run"] += 1
                rec["max_run"] = max(rec["max_run"], rec["run"])
            else:
                rec["red_no_request"] += 1
                rec["run"] = 0
    return collect


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=12)
    p.add_argument("--seed-start", type=int, default=50000)
    p.add_argument("--out", type=str,
                   default=str(RESULTS_DIR / "camera_ready" / "ev_wait_breakdown.json"))
    args = p.parse_args()

    cfg = EnvConfig()
    cfg.demand = DemandConfig(horizon_s=3600, regime="balanced",
                              base_flow_vph=700.0, ev_rate_per_hour=30.0)
    cfg.sensing_mode = "surrogate"
    cfg.belief_mode = "joint"
    seeds = [args.seed_start + i for i in range(args.seeds)]
    net_file, _ = build_network(cfg.network, DATA_DIR / "networks")
    error_model = SensorErrorModel.load(MODELS_DIR / "sensor_error_model.json")

    probe = SumoSignalEnv(cfg, net_file=net_file, work_dir=RUNS_DIR / "probe_bd",
                          label="probe_bd")
    probe.reset(seed=seeds[0])
    state_dim, n_actions = probe.observation_dim, probe.n_actions
    probe.close()

    agent, _ = load_agent(MODELS_DIR / "siren_arc.h5", state_dim, n_actions,
                          run_name="siren_arc")
    SumoSignalEnv._collect_step = _patched_collect(SumoSignalEnv._collect_step)
    factory = (lambda env: CorridorPolicy(
        agent=agent, name="siren_arc",
        calibration_path=MODELS_DIR / "presence_calibration.json",
        scheduler=CorridorScheduler(ev_value_ratio=50.0)))
    res = evaluate_controller(
        factory, cfg, seeds, "siren_arc_bd", net_file=net_file,
        sensor=AcousticSensor([], cfg.acoustic, mode="surrogate",
                              error_model=error_model, seed=args.seed_start),
        work_dir=RUNS_DIR / "balanced_siren_arc_bd")

    per_ev = [dict(v, label=k[0], vid=k[1]) for k, v in LOG.items()]
    cats = ("green", "other_movement_green", "changeover", "red_with_request",
            "red_no_request", "other")
    for r in per_ev:
        for c in cats:
            r.setdefault(c, 0)
    total = {c: int(sum(r[c] for r in per_ev)) for c in cats}
    stopped = sum(total.values())
    long_ = [r for r in per_ev if sum(r[c] for c in cats) > 16]
    total_long = {c: int(sum(r[c] for r in long_)) for c in cats}
    out = {
        "seeds": seeds,
        "n_ev": len(per_ev),
        "stopped_seconds": stopped,
        "share_all": {c: total[c] / max(stopped, 1) for c in cats},
        "n_ev_stopped_over_16s": len(long_),
        "share_over_16s": {c: total_long[c] / max(sum(total_long.values()), 1)
                           for c in cats},
        "max_red_with_request_run_s": int(max((r["max_run"] for r in per_ev), default=0)),
        "red_with_request_run_p95_s": float(np.percentile(
            [r["max_run"] for r in per_ev], 95)) if per_ev else 0.0,
        "median_ev_stopped_s_check": float(np.median(
            [r["ev_mean_waiting_time_s"] for r in res["runs"]])),
        "per_ev": per_ev,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "per_ev"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
