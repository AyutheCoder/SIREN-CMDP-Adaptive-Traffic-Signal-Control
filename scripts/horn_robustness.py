"""Does the controller survive traffic that honks?

The question, and why it had no answer before
---------------------------------------------
A road horn is the one urban sound genuinely confusable with a siren. It is
harmonic, its tones sit on the low end of the siren band, and it is sounded from
the queue -- metres from the mast -- while the siren that matters is still
hundreds of metres away. Evaluated on this geometry, a horn at 30 m reaches the
microphone at 23 dB SNR against an approaching ambulance's 15.5 dB: **the
confuser is the louder signal**, so no level threshold separates them.

Until this script existed there were no horns anywhere in the model. The
detector's only negative class was broadband traffic noise, which is an easy
negative -- it is rejected almost regardless of level -- so the network was
never asked to learn the distinction that matters, and "our detector rejects
horns" was not a claim anyone could have checked.

That gap is not cosmetic given the rest of this system. The shield's S6 rule
acts on a request *decisively*: a spurious request is not a missed opportunity
but a forced green on the wrong approach. We have already measured what happens
when S6 consumes an imprecise detector -- emergency delay up 36%, background
queue up 41%, and the only gridlocked seed in the study. A confuser that fires
the detector once every fifteen seconds is precisely that failure mode, arriving
through the front door.

What is compared
----------------
Three arms on shared seeds, differing only in what the microphones hear and what
the detector knows:

``silent``      no horns at all -- the model this project used previously, and
                the optimistic reference.
``horn-blind``  horns sound, but the detector's error model has no horn class,
                so a horn is scored on the *siren* detection curve. This is the
                honest reading of a detector that has never seen the confuser:
                it cannot reject what it has no concept of.
``horn-aware``  horns sound, and the detector was trained with them as an
                explicit class, so the surrogate scores them on the measured
                horn-confusion curve.

    python scripts/horn_robustness.py --seeds 8
    python scripts/horn_robustness.py --seeds 8 --horn-rate 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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

def _load_error_model(horn_aware: bool) -> SensorErrorModel:
    """The same fitted error model, with or without its horn curve.

    Both arms are built from the **current** model and differ in exactly one
    field. Loading the archived pre-horn file for the blind arm would have been
    the obvious thing to do and is wrong: that file was fitted on a different
    network, with a different class head and a different readout, so the arms
    would differ in their siren curves, their noise floor and their range error
    as well as in horn-awareness. Measured that way the "blind" arm came out
    *better* on the first seed, which is not a finding about horns at all.

    Stripping the horn curve makes `SensorErrorModel.horn_logit` fall back to
    the siren curve -- exactly the behaviour of a detector that has no concept
    of the confuser, and nothing else changes.
    """
    path = MODELS_DIR / "sensor_error_model.json"
    em = SensorErrorModel.load(path) if path.exists() else SensorErrorModel.default()
    if not horn_aware:
        em.horn_logit_mean, em.horn_logit_std = [], []
    return em


def run_arm(horns: bool, horn_aware: bool, seed: int, horizon: int,
            ev_rate: float, horn_rate: float, checkpoint: Path) -> dict:
    cfg = EnvConfig()
    cfg.demand = DemandConfig(horizon_s=horizon, seed=seed, ev_rate_per_hour=ev_rate)
    cfg.sensing_mode, cfg.belief_mode, cfg.gui = "surrogate", "joint", False
    cfg.collect_emissions = False
    cfg.step_length_s = 1.0
    cfg.acoustic.horn_rate_per_veh_hour = float(horn_rate) if horns else 0.0

    net_file, _ = build_network(cfg.network, DATA_DIR / "networks")
    sensor = AcousticSensor([], cfg.acoustic, mode=cfg.sensing_mode,
                            error_model=_load_error_model(horn_aware), seed=seed)
    env = SumoSignalEnv(cfg, net_file=net_file, work_dir=DATA_DIR / "runs" / "horn",
                        label="horn", sensor=sensor)
    obs = env.reset(seed=seed)
    agent = LagrangianDuelingDDQN(env.observation_dim, env.n_actions, AgentConfig())
    agent.load(checkpoint)
    ctrl = CorridorPolicy(agent=agent, name="siren_arc")
    ctrl.reset(env)

    stat = {"ev_free_steps": 0, "ev_free_with_request": 0, "horn_seconds": 0}
    peak: dict = {}

    def hook(e):
        sc = e.scenario
        stat["horn_seconds"] += len(getattr(e, "_horn_until", {}) or {})
        on_approach = False
        try:
            live = set(e.conn.vehicle.getIDList()) & set(sc.emergency_vehicles or ())
            for vid in live:
                if e.conn.vehicle.getRoadID(vid) in e.approaches:
                    on_approach = True
                w = float(e.conn.vehicle.getAccumulatedWaitingTime(vid))
                if w > peak.get(vid, -1.0):
                    peak[vid] = w
        except traci.TraCIException:
            return
        if not on_approach:
            stat["ev_free_steps"] += 1
            if e.ev_request_phases:
                stat["ev_free_with_request"] += 1

    env.step_hook = hook
    done = False
    while not done:
        obs, _r, _c, done, _i = env.step(ctrl.act(env, obs, env.action_mask()),
                                         interrupt=ctrl.interrupt)
    out = dict(env.episode_summary())
    out["ev_waits"] = list(peak.values())
    out["spurious_rate"] = (stat["ev_free_with_request"]
                            / max(stat["ev_free_steps"], 1))
    out["horn_seconds"] = stat["horn_seconds"]
    out["s6_forced"] = float(env.shield.n_ev_forced)
    env.close()
    return out


ARMS = {
    "silent traffic":        (False, True),
    "horns, detector blind": (True, False),
    "horns, detector aware": (True, True),
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--horizon", type=int, default=1800)
    p.add_argument("--ev-rate", type=float, default=30.0)
    p.add_argument("--horn-rate", type=float, default=15.0)
    p.add_argument("--checkpoint", default=str(MODELS_DIR / "siren_arc.h5"))
    p.add_argument("--out", default=str(RESULTS_DIR / "horn_robustness.json"))
    args = p.parse_args()

    ensure_dirs()
    seeds = [SEED_BASE + i for i in range(args.seeds)]
    results = {k: [] for k in ARMS}
    t0 = time.time()

    for seed in seeds:
        for label, (horns, aware) in ARMS.items():
            r = run_arm(horns, aware, seed, args.horizon, args.ev_rate,
                        args.horn_rate, Path(args.checkpoint))
            results[label].append(r)
            print(f"  seed={seed} {label:24s} ev_delay="
                  f"{r.get('ev_delay_per_vehicle_s', float('nan')):6.2f}s "
                  f"spurious={100 * r['spurious_rate']:5.1f}% "
                  f"queue={r.get('mean_queue', float('nan')):6.2f} "
                  f"horn_s={r['horn_seconds']:4d}", flush=True)

    print("\n" + "=" * 100)
    print(f"{args.seeds} shared seeds, {args.horizon}s, {args.ev_rate:g} EV/h, "
          f"{args.horn_rate:g} horn presses per queued vehicle-hour "
          f"({time.time() - t0:.0f}s wall)")
    print("=" * 100)
    hdr = f"{'metric':<34}" + "".join(f"{k:>24}" for k in ARMS)
    print(hdr)
    print("-" * len(hdr))
    rows = [
        ("spurious request rate (%)", lambda rs: 100 * np.mean([r["spurious_rate"] for r in rs])),
        ("EV stopped delay (s/veh)", lambda rs: np.nanmean([r.get("ev_delay_per_vehicle_s", np.nan) for r in rs])),
        ("worst EV wait (s)", lambda rs: max((max(r["ev_waits"]) if r["ev_waits"] else 0.0) for r in rs)),
        ("background queue (veh)", lambda rs: np.nanmean([r.get("mean_queue", np.nan) for r in rs])),
        ("S6 forcings", lambda rs: np.mean([r["s6_forced"] for r in rs])),
        ("teleports (gridlock)", lambda rs: sum(r.get("teleports", 0) for r in rs)),
    ]
    for label, fn in rows:
        print(f"{label:<34}" + "".join(f"{fn(results[k]):24.2f}" for k in ARMS))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seeds": seeds, "horn_rate": args.horn_rate,
                               "ev_rate_per_hour": args.ev_rate,
                               "horizon_s": args.horizon, "arms": results}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
