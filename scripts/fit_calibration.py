"""Fit the presence-posterior calibration map, on seeds held out from results.

The corridor scheduler compares ``p * gamma * B`` against ``C``. That is only
arithmetic if ``p`` is a probability, and the recursive presence filter does not
produce one -- it ranks well (AUC ~0.95) and is over-confident by roughly four
times in the band that matters. This script measures the miscalibration against
ground truth and fits the monotone correction.

Seeds here are deliberately disjoint from the evaluation seeds (which start at
50000) and from the training seeds, so the calibration map is never fitted on
anything a reported number is computed from.

    python scripts/fit_calibration.py --episodes 8
    python scripts/fit_calibration.py --sensing oracle --out models/cal_oracle.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from atsc.agents.baselines import LearnedPolicy, build_baseline
from atsc.agents.dueling_ddqn import LagrangianDuelingDDQN
from atsc.config import (
    AgentConfig, DATA_DIR, DemandConfig, EnvConfig, MODELS_DIR, RESULTS_DIR, RUNS_DIR, ensure_dirs)
from atsc.env import SumoSignalEnv
from atsc.netgen import build_network
from atsc.sensing.calibration import PresenceCalibrator
from atsc.sensing.sensor import AcousticSensor, SensorErrorModel

CALIBRATION_SEED_BASE = 900000
"""Disjoint from training (0..) and evaluation (50000..) by construction."""

DRIVER_FOR_BELIEF = {
    "joint": MODELS_DIR / "siren_arc.h5",
    "independent": MODELS_DIR / "arc_base_indep.h5",
}
"""Which policy drives the signal while calibration data is collected.

The map is a correction to one filter's score distribution *under the phase
behaviour it will be used with*, and both halves of that matter. It used to be
collected under a single fixed checkpoint for both filters, which broke the
second half twice over: that checkpoint was trained on the independent filter,
so the joint map was fitted while the signal was driven by a policy reading
features it had never been trained on, and it predates the observation the
policies now receive, so it is not the controller either map will be deployed
beneath.

The project's own rule elsewhere is that a policy is evaluated with the filter
it was trained on. This is the same rule applied one stage earlier: each map is
fitted under the policy that will consume it. Pass --checkpoint to override,
and if the file is absent the run falls back to --baseline, which keeps the
stage runnable before any policy has been trained.
"""


def inbound_truth(env, radius: float) -> dict:
    """Ground truth per approach: an emergency vehicle inbound and in range.

    'Inbound' means still on the approach edge and still closing on the stop
    line. This is the event the controller can actually act on -- a vehicle that
    has crossed is no longer something a green can help -- so it is the event
    the probability should be calibrated against.
    """
    out = {a: False for a in env.approaches}
    evs = getattr(env.scenario, "emergency_vehicles", None) or {}
    if not evs:
        return out
    conn = env.conn
    for vid in set(conn.vehicle.getIDList()) & set(evs):
        try:
            road = conn.vehicle.getRoadID(vid)
            if road not in out:
                continue
            px, py = conn.vehicle.getPosition(vid)
            speed = conn.vehicle.getSpeed(vid)
        except Exception:
            continue
        sx, sy = env.stop_line[road]
        d = float(np.hypot(px - sx, py - sy))
        if d <= radius and speed >= 0.0:
            out[road] = True
    return out


def collect(cfg, net_file, error_model, seeds, controller_name, checkpoint):
    scores, truth = [], []
    agent = None
    for seed in seeds:
        sensor = AcousticSensor([], cfg.acoustic, mode=cfg.sensing_mode,
                                error_model=error_model, seed=seed)
        env = SumoSignalEnv(cfg, net_file=net_file,
                            work_dir=RUNS_DIR / "calibration",
                            label=f"cal_{seed}", sensor=sensor)
        obs = env.reset(seed=seed)
        if checkpoint is not None:
            if agent is None:
                agent = LagrangianDuelingDDQN(env.observation_dim, env.n_actions,
                                              AgentConfig())
                agent.load(checkpoint)
            controller = LearnedPolicy(agent, name="policy", greedy=True)
        else:
            controller = build_baseline(controller_name, env, cfg.demand)
        controller.reset(env)

        radius = float(cfg.network.arm_length_m) + 5.0
        done = False
        while not done:
            mask = env.action_mask()
            action = controller.act(env, obs, mask)
            t = inbound_truth(env, radius)
            beliefs = env.belief.beliefs()
            for a in env.approaches:
                scores.append(float(beliefs[a].presence))
                truth.append(bool(t[a]))
            obs, _, _, done, _ = env.step(action)
        env.close()
        print(f"  seed {seed}: {len(scores)} approach-epochs so far", flush=True)
    return np.asarray(scores), np.asarray(truth)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--horizon", type=int, default=3600)
    p.add_argument("--ev-rate", type=float, default=40.0)
    p.add_argument("--regime", default="balanced")
    p.add_argument("--sensing", default="surrogate",
                   choices=["surrogate", "oracle", "none", "end_to_end"])
    p.add_argument("--checkpoint", default=None,
                   help="policy driving the signal while data is collected. "
                        "Defaults to the policy this map will actually be used "
                        "under, which is chosen by --belief-mode; see "
                        "DRIVER_FOR_BELIEF.")
    p.add_argument("--baseline", default="max_pressure",
                   help="used instead of --checkpoint if the checkpoint is absent")
    p.add_argument("--bins", type=int, default=20)
    p.add_argument("--belief-mode", default="joint",
                   choices=["joint", "independent"],
                   help="which presence filter to calibrate; the map is only "
                        "valid for the filter it was fitted on")
    p.add_argument("--diagnostics", default=None,
                   help="where to write the calibration diagnostics JSON")
    p.add_argument("--out", default=str(MODELS_DIR / "presence_calibration.json"))
    args = p.parse_args()

    ensure_dirs()
    cfg = EnvConfig()
    cfg.demand = DemandConfig(horizon_s=args.horizon, regime=args.regime,
                              ev_rate_per_hour=args.ev_rate)
    cfg.sensing_mode = args.sensing
    cfg.belief_mode = args.belief_mode
    cfg.gui = False
    cfg.collect_emissions = False

    net_file, _ = build_network(cfg.network, DATA_DIR / "networks")
    em_path = MODELS_DIR / "sensor_error_model.json"
    error_model = SensorErrorModel.load(em_path) if em_path.exists() else None

    ckpt = Path(args.checkpoint) if args.checkpoint else DRIVER_FOR_BELIEF[args.belief_mode]
    checkpoint = ckpt if ckpt.exists() else None
    if checkpoint is None:
        print(f"  no checkpoint at {ckpt}; collecting under '{args.baseline}'")
    else:
        print(f"  collecting under {ckpt.name}, the policy this map is for")

    seeds = [CALIBRATION_SEED_BASE + i for i in range(args.episodes)]
    print(f"fitting presence calibration on {len(seeds)} held-out episodes "
          f"(seeds {seeds[0]}..{seeds[-1]}, sensing={args.sensing})")
    scores, truth = collect(cfg, net_file, error_model, seeds,
                            args.baseline, checkpoint)

    # An isotonic map lands on the diagonal on its own training data by
    # construction, so quoting in-sample calibration error would say nothing.
    # Half the episodes fit the map and the other half score it; the shipped
    # map is then refitted on everything, which is standard and is what the
    # controller loads.
    n_fit = max(1, args.episodes // 2)
    per_episode = len(scores) // max(args.episodes, 1)
    split = n_fit * per_episode
    fit_s, fit_y = scores[:split], truth[:split]
    ho_s, ho_y = scores[split:], truth[split:]
    if ho_s.size < 100:
        fit_s, fit_y, ho_s, ho_y = scores, truth, scores, truth
        print("  too few episodes to hold out; quoted errors are in-sample")

    probe = PresenceCalibrator.fit(fit_s, fit_y, n_bins=args.bins)
    ho_cal = probe.transform_many(ho_s)
    ece_before = PresenceCalibrator.expected_calibration_error(ho_s, ho_y)
    ece_after = PresenceCalibrator.expected_calibration_error(ho_cal, ho_y)
    brier_before = PresenceCalibrator.brier(ho_s, ho_y)
    brier_after = PresenceCalibrator.brier(ho_cal, ho_y)
    print(f"  held-out episodes for scoring: {args.episodes - n_fit} "
          f"({ho_s.size} approach-epochs)")

    cal = PresenceCalibrator.fit(scores, truth, n_bins=args.bins)
    calibrated = cal.transform_many(scores)

    # AUC is invariant under any monotone map, so it must not move; reporting it
    # is the check that calibration changed the scale and not the ranking.
    def auc(s, y):
        y = np.asarray(y, dtype=bool)
        order = np.argsort(s)
        ranks = np.empty(len(s), float)
        ranks[order] = np.arange(1, len(s) + 1)
        n1, n0 = y.sum(), (~y).sum()
        if n1 == 0 or n0 == 0:
            return float("nan")
        return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    print("\n" + "=" * 70)
    print(f"  samples                 {len(scores)}")
    print(f"  base rate (occupancy)   {truth.mean():.4f}")
    print(f"  AUC before / after      {auc(scores, truth):.4f} / "
          f"{auc(calibrated, truth):.4f}   (must be equal: monotone map)")
    print(f"  ECE  before / after     {ece_before:.4f} / {ece_after:.4f}"
          f"   (held out)")
    print(f"  Brier before / after    {brier_before:.4f} / {brier_after:.4f}"
          f"   (held out)")
    print("\n  reliability (raw -> calibrated, with support):")
    for raw, cp, n in zip(cal.raw_means, cal.probs, cal.counts):
        if n <= 0:
            continue
        print(f"    {raw:6.3f} -> {cp:6.3f}   n={int(n):6d}")
    print("=" * 70)

    cal.save(Path(args.out))
    print(f"\nwrote {args.out}")

    diag = Path(args.diagnostics) if args.diagnostics else         RESULTS_DIR / "calibration_diagnostics.json"
    diag.write_text(json.dumps({
        "sensing": args.sensing,
        "belief_mode": args.belief_mode,
        "n_samples": int(len(scores)),
        "occupancy": float(truth.mean()),
        "auc": auc(scores, truth),
        "ece_before": ece_before, "ece_after": ece_after,
        "brier_before": brier_before, "brier_after": brier_after,
        "scored_out_of_sample": bool(ho_s.size != scores.size),
        "n_scored": int(ho_s.size),
        "raw_means": cal.raw_means.tolist(),
        "calibrated": cal.probs.tolist(),
        "counts": cal.counts.tolist(),
    }, indent=2), encoding="utf-8")
    print(f"wrote {diag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
