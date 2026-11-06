"""Stress-test the safety shield in the full SUMO simulation.

    python scripts/shield_stress.py --seeds 6

Theorems 1 and 2 hold for *any* controller and *any* detector output. This
checks that claim in the running system rather than in isolation: a hostile
controller drives the real intersection, the shield is the only thing between
it and the signal, and fake emergency requests stand in for a fooled or
malfunctioning detector.

Scenarios
---------
``starve``        the controller always proposes the phase served most
                  recently, i.e. it tries to starve the longest-waiting one;
                  no emergency requests at all.
``spoof_single``  as ``starve``, plus one fake request at a time on a random
                  road, moved every 40 s, which the controller refuses to serve.
``spoof_flood``   as ``starve``, plus fake requests on random, often
                  conflicting, sets of phases redrawn every second.
``random_flood``  a uniformly random controller under the same flood.

Reported per scenario: the longest time any phase went without green (S3,
bound 180 s) and the longest a request stood unserved (S4: 16 s when S3 does
not intervene, 88 s in the worst case when it does).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atsc.config import DATA_DIR, DemandConfig, EnvConfig, MODELS_DIR, RESULTS_DIR, RUNS_DIR
from atsc.env import SumoSignalEnv
from atsc.evaluation.runner import evaluate_controller
from atsc.netgen import build_network
from atsc.sensing.sensor import AcousticSensor, SensorErrorModel

PEAK = {"max_phase_unserved_s": 0.0, "max_request_age_s": 0.0}


def _patched_collect(orig):
    def collect(self):
        orig(self)
        PEAK["max_phase_unserved_s"] = max(PEAK["max_phase_unserved_s"],
                                           float(np.max(self.time_since_served)))
        ages = self.ev_request_age_s()
        if ages.size:
            PEAK["max_request_age_s"] = max(PEAK["max_request_age_s"], float(ages.max()))
    return collect


class Adversary:
    """A controller that works against the guarantees, as hard as it can."""

    def __init__(self, scenario: str, seed: int):
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        self.name = scenario
        self._target = None
        self._next_move = 0.0

    def reset(self, env) -> None:
        self._target = None
        self._next_move = 0.0
        env.clear_ev_request()

    def _requests(self, env):
        n = env.n_actions
        t = float(env.sim_time)
        if self.scenario == "spoof_single":
            if self._target is None or t >= self._next_move:
                road = env.phases[int(self.rng.integers(n))].approaches[0]
                self._target = [p.action_index for p in env.phases
                                if road in p.approaches]
                self._next_move = t + 40.0
            env.set_ev_request(self._target)
        elif self.scenario in ("spoof_flood", "random_flood"):
            k = int(self.rng.integers(1, n + 1))
            env.set_ev_request(list(self.rng.choice(n, size=k, replace=False)))
        else:
            env.set_ev_request(None)

    def act(self, env, obs, mask) -> int:
        self._requests(env)
        if self.scenario == "random_flood":
            return int(self.rng.integers(env.n_actions))
        # Propose the phase served most recently, avoiding requested phases:
        # the proposal most likely to starve someone and to refuse a request.
        waits = np.asarray(env.time_since_served, dtype=float).copy()
        for p in env.ev_request_phases:
            waits[p] = np.inf
        return int(np.argmin(waits))

    def interrupt(self, env):
        self._requests(env)
        return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--seed-start", type=int, default=50000)
    p.add_argument("--scenarios", type=str,
                   default="starve,spoof_single,spoof_flood,random_flood")
    p.add_argument("--out", type=str,
                   default=str(RESULTS_DIR / "camera_ready" / "shield_stress.json"))
    args = p.parse_args()

    cfg = EnvConfig()
    cfg.demand = DemandConfig(horizon_s=3600, regime="balanced",
                              base_flow_vph=700.0, ev_rate_per_hour=30.0)
    cfg.sensing_mode = "surrogate"
    net_file, _ = build_network(cfg.network, DATA_DIR / "networks")
    error_model = SensorErrorModel.load(MODELS_DIR / "sensor_error_model.json")
    SumoSignalEnv._collect_step = _patched_collect(SumoSignalEnv._collect_step)

    seeds = [args.seed_start + i for i in range(args.seeds)]
    out = {"seeds": seeds, "bounds": {"S3_s": cfg.control.max_starvation_s,
                                      "S4_s": cfg.control.ev_service_deadline_s},
           "scenarios": {}}
    for scenario in args.scenarios.split(","):
        rows = []
        for seed in seeds:
            PEAK["max_phase_unserved_s"] = PEAK["max_request_age_s"] = 0.0
            res = evaluate_controller(
                lambda env, s=scenario, sd=seed: Adversary(s, sd), cfg, [seed],
                f"stress_{scenario}", net_file=net_file,
                sensor=AcousticSensor([], cfg.acoustic, mode="surrogate",
                                      error_model=error_model, seed=seed),
                work_dir=RUNS_DIR / f"stress_{scenario}", verbose=False)
            run = res["runs"][0]
            rows.append({"seed": seed,
                         "max_phase_unserved_s": PEAK["max_phase_unserved_s"],
                         "max_request_age_s": PEAK["max_request_age_s"],
                         "max_movement_wait_s": run.get("max_movement_wait_s"),
                         "jammed": bool(run.get("jammed", False))})
        out["scenarios"][scenario] = {
            "runs": rows,
            "worst_phase_unserved_s": max(r["max_phase_unserved_s"] for r in rows),
            "worst_request_age_s": max(r["max_request_age_s"] for r in rows),
            "worst_movement_wait_s": max(r["max_movement_wait_s"] or 0 for r in rows),
        }
        print(scenario, {k: v for k, v in out["scenarios"][scenario].items()
                         if k != "runs"}, flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
