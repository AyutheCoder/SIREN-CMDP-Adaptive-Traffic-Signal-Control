"""How the controller degrades when the microphones do.

This is the study a transponder-based system cannot run. An infrared or V2I
preemption link either works or it does not; there is no partial failure mode
between them, and no story about what the intersection does in the gap. A
passive acoustic front end degrades continuously --- with the ambient noise
floor, and with the loss of individual masts --- and the interesting question is
whether the *controller* degrades continuously too, or falls off a cliff.

Three axes:

* **Ambient noise.** The urban floor is raised in 5 dB steps. Every mast loses
  signal-to-noise together, so this is the common-mode failure: detections get
  later and less certain everywhere at once.
* **Mast loss.** One and then two microphones are switched off entirely. This is
  the differential failure, and the one the joint filter should survive best,
  because the remaining masts still constrain where the source is --- an
  independent per-approach filter simply goes blind on the dead arm. A failed
  mast is *silent*, not noisy, and the filter is told which masts are out, which
  is the correct marginalisation over a missing observation.
* **Nothing at all.** Sensing disabled, as the floor.

For each condition we report both halves: what the sensing did (precision and
warning distance) and what the intersection did (emergency stopped delay,
background delay, gridlock). Reporting only the second would hide whether a
degradation was absorbed by the decision rule or never reached it.

    python scripts/robustness.py --seeds 6
    python scripts/robustness.py --conditions nominal,noise+10,mic1
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from atsc.agents.baselines import CorridorPolicy, LearnedPolicy
from atsc.agents.dueling_ddqn import LagrangianDuelingDDQN
from atsc.config import (
    AgentConfig, DATA_DIR, DemandConfig, EnvConfig, MODELS_DIR, RESULTS_DIR, RUNS_DIR, ensure_dirs)
from atsc.control.corridor import CorridorScheduler
from atsc.env import SumoSignalEnv
from atsc.evaluation.runner import evaluate_controller
from atsc.netgen import build_network
from atsc.sensing.sensor import AcousticSensor, SensorErrorModel

ROBUST_SEED_BASE = 610000
"""Disjoint from training, evaluation, calibration and the sensing benchmark."""

CONDITIONS = {
    "nominal":  {"noise_db": 0.0, "dead": 0, "sensing": "surrogate"},
    "noise+5":  {"noise_db": 5.0, "dead": 0, "sensing": "surrogate"},
    "noise+10": {"noise_db": 10.0, "dead": 0, "sensing": "surrogate"},
    "noise+15": {"noise_db": 15.0, "dead": 0, "sensing": "surrogate"},
    "mic1":     {"noise_db": 0.0, "dead": 1, "sensing": "surrogate"},
    "mic2":     {"noise_db": 0.0, "dead": 2, "sensing": "surrogate"},
    "deaf":     {"noise_db": 0.0, "dead": 0, "sensing": "none"},
}


class _FailingSensor(AcousticSensor):
    """An acoustic sensor with some masts switched off.

    The set of dead approaches is resolved lazily, on the first call, because
    the approach labels are network-dependent and only exist once SUMO has been
    started and the signal program discovered.
    """

    def __init__(self, *a, n_dead: int = 0, **kw):
        super().__init__(*a, **kw)
        self.n_dead = int(n_dead)
        self._resolved = False

    def reset_stats(self):
        # Called by the environment immediately after it has told the sensor
        # what the approaches are and before it builds the belief filter, which
        # is exactly when the dead set has to be known: the filter needs it to
        # drop those masts from its likelihood.
        super().reset_stats()
        if self.approaches:
            # Deterministic choice: the first n approaches in the canonical
            # N, E, S, W order, so the condition is identical across seeds and
            # a difference between seeds is demand, not which mast died.
            self.dead_approaches = set(self.approaches[:self.n_dead])


def build(cfg, spec, error_model, seed):
    cfg = copy.deepcopy(cfg)
    lo, hi = cfg.acoustic.ambient_noise_db_range
    cfg.acoustic.ambient_noise_db_range = (lo + spec["noise_db"],
                                           hi + spec["noise_db"])
    cfg.sensing_mode = spec["sensing"]
    sensor = _FailingSensor([], cfg.acoustic, mode=cfg.sensing_mode,
                            error_model=error_model, seed=seed,
                            n_dead=spec["dead"])
    return cfg, sensor


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--horizon", type=int, default=3600)
    p.add_argument("--ev-rate", type=float, default=30.0)
    p.add_argument("--checkpoint", type=str, default=str(MODELS_DIR / "siren_arc.h5"))
    p.add_argument("--conditions", type=str, default=",".join(CONDITIONS))
    p.add_argument("--out", type=str, default=str(RESULTS_DIR / "robustness.json"))
    args = p.parse_args()

    ensure_dirs()
    base = EnvConfig()
    base.demand = DemandConfig(horizon_s=args.horizon, ev_rate_per_hour=args.ev_rate)
    base.collect_emissions = False
    seeds = [ROBUST_SEED_BASE + i for i in range(args.seeds)]

    net_file, _ = build_network(base.network, DATA_DIR / "networks")
    em_path = MODELS_DIR / "sensor_error_model.json"
    error_model = SensorErrorModel.load(em_path) if em_path.exists() else None

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt}; train the base policy first")
    probe = SumoSignalEnv(base, net_file=net_file,
                          work_dir=RUNS_DIR / "robust", label="probe")
    probe.reset(seed=seeds[0])
    state_dim, n_actions = probe.observation_dim, probe.n_actions
    probe.close()
    agent = LagrangianDuelingDDQN(state_dim, n_actions, AgentConfig())
    agent.load(ckpt)

    results = {}
    print(f"\n{len(seeds)} seeds x {args.horizon}s, {args.ev_rate:.0f} EV/h, "
          f"policy {ckpt.name}\n")
    for name in [c.strip() for c in args.conditions.split(",") if c.strip()]:
        spec = CONDITIONS.get(name)
        if spec is None:
            print(f"  unknown condition {name!r}; skipping")
            continue
        cfg, sensor = build(base, spec, error_model, seeds[0])
        if spec["sensing"] == "none":
            factory = (lambda env, a=agent: LearnedPolicy(a, name="deaf", greedy=True))
        else:
            factory = (lambda env, a=agent, n=name:
                       CorridorPolicy(agent=a, name=n, adaptive=False,
                                      scheduler=CorridorScheduler()))
        print(f"[{name}] ambient +{spec['noise_db']:.0f} dB, "
              f"{spec['dead']} mast(s) dead, sensing={spec['sensing']}")
        results[name] = evaluate_controller(
            factory, cfg, seeds, name, net_file=net_file, sensor=sensor,
            work_dir=RUNS_DIR / f"robust_{name}")
        results[name]["condition"] = spec

    print("\n" + "=" * 92)
    print(f"{'condition':<12}{'EV stopped (s)':>18}{'clean pass':>14}"
          f"{'delay (s)':>14}{'jam':>8}{'corridors/h':>14}")
    print("-" * 92)
    for name, res in results.items():
        agg = res["aggregate"]
        eng = agg.get("corridor_priority_engagements_mean", float("nan"))
        print(f"{name:<12}"
              f"{agg.get('ev_mean_waiting_time_s_mean', float('nan')):>12.2f}"
              f" +-{agg.get('ev_mean_waiting_time_s_ci95', 0.0):<4.2f}"
              f"{agg.get('ev_zero_stop_rate_mean', float('nan')):>14.3f}"
              f"{agg.get('mean_time_loss_s_mean', float('nan')):>14.2f}"
              f"{agg.get('n_jammed', 0):>5.0f}/{agg.get('n_runs', 0):<2.0f}"
              f"{eng:>14.1f}")
    print("=" * 92)

    out = Path(args.out)
    out.write_text(json.dumps({
        "config": {"seeds": seeds, "horizon_s": args.horizon,
                   "ev_rate_per_hour": args.ev_rate, "checkpoint": str(ckpt)},
        "conditions": CONDITIONS,
        "results": results,
    }, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
