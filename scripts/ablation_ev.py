"""Ablation of the two changes that fixed emergency service, over shared seeds.

What is being separated
-----------------------
Two independent defects were found by instrumenting the controller against
ground truth rather than against its own summary statistics, and each has its
own fix. This script measures them apart and together, so the paper can say
which one did the work instead of reporting their sum.

**The mast is deaf where it matters.** With the mast ``mic_offset_m`` upstream
of its stop line and read as a single cardioid looking *away* from the
junction, every vehicle between the mast and the line is in its rear null. At
the 70 m offset this project shipped, an ambulance 1 m from the stop line
delivered 60.8 dB against 63.5 dB from one 300 m away and an ambient floor of
55-80 dB: quietest exactly where the vehicle is queued. ``--dual-element``
reads the same two capsules as a steerable endfire pair instead, which recovers
the level over that stretch and adds the front-minus-back ratio as a
side-of-mast observable.

This arm is retained as the ablation rather than as the fix. Reading the louder
of two opposed lobes also flattens the pattern to nearly omnidirectional, and
the shipped mast now sits at the signal pole as a single cardioid instead --
where an inbound vehicle is on-axis all the way in and the rear null does
useful work rejecting one that has crossed. See ``AcousticConfig.mic_offset_m``
for the separation measurements that decided it. Run this with
``--mic-offset 70`` to reproduce the geometry the arm was written for.

**Emergency service is an option, not a guarantee.** Every layer above the
shield may decline: the posterior may fall short of its derived threshold, the
Bayes rule may be outbid by the cost of the cross-street queue, the token bucket
may be empty. Nothing anywhere promised that a vehicle which *is* there gets a
green. ``--s6`` adds the shield rule that does, bounding the latency from a
standing request to service, underneath everything else and above nothing that
matters -- S1, S2, S3 all still outrank it.

    python scripts/ablation_ev.py --seeds 12
    python scripts/ablation_ev.py --seeds 12 --out results/ablation_ev.json
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
from atsc.evaluation.metrics import paired_bootstrap_pvalue
from atsc.netgen import build_network
from atsc.sensing.sensor import AcousticSensor, SensorErrorModel

SEED_BASE = 50000
"""The evaluation seed block, disjoint from training (0..) and calibration."""

ARMS = {
    # label                     dual-element mast   S6 deadline guarantee
    "baseline (as shipped)":    (False, False),
    "+ two-element mast":       (True,  False),
    "+ S6 deadline only":       (False, True),
    "+ both (SIREN-ARC v2)":    (True,  True),
}

METRICS = [
    ("ev_delay_per_vehicle_s", "EV stopped delay / veh (s)", "min"),
    ("ev_stops", "EV full stops", "min"),
    ("mean_queue", "background mean queue (veh)", "min"),
    ("mean_wait_s", "background mean wait (s)", "min"),
    ("throughput", "throughput (veh)", "max"),
    ("max_movement_wait_s", "worst movement wait (s)", "min"),
    ("teleports", "teleports (gridlock)", "min"),
]


def run_arm(dual: bool, s6: bool, seed: int, horizon: int, ev_rate: float,
            checkpoint: Path, mic_offset: float | None = None) -> dict:
    cfg = EnvConfig()
    cfg.demand = DemandConfig(horizon_s=horizon, seed=seed, ev_rate_per_hour=ev_rate)
    cfg.sensing_mode, cfg.belief_mode, cfg.gui = "surrogate", "joint", False
    cfg.collect_emissions = False
    cfg.step_length_s = 1.0
    cfg.acoustic.dual_element_mast = bool(dual)
    if mic_offset is not None:
        cfg.acoustic.mic_offset_m = float(mic_offset)
    if not s6:
        # Disabling S6 has to mean "the shield never forces emergency service",
        # and the honest way to express that is a deadline no request can reach
        # rather than a flag the rule can be written around.
        cfg.control.ev_service_deadline_s = float("inf")

    net_file, _ = build_network(cfg.network, DATA_DIR / "networks")
    em = MODELS_DIR / "sensor_error_model.json"
    sensor = AcousticSensor([], cfg.acoustic, mode=cfg.sensing_mode,
                            error_model=SensorErrorModel.load(em) if em.exists() else None,
                            seed=seed)
    env = SumoSignalEnv(cfg, net_file=net_file,
                        work_dir=DATA_DIR / "runs" / "ablation",
                        label="ablation", sensor=sensor)
    obs = env.reset(seed=seed)
    agent = LagrangianDuelingDDQN(env.observation_dim, env.n_actions, AgentConfig())
    agent.load(checkpoint)
    ctrl = CorridorPolicy(agent=agent, name="siren_arc")
    ctrl.reset(env)

    done = False
    while not done:
        mask = env.action_mask()
        action = ctrl.act(env, obs, mask)
        obs, _r, _c, done, _i = env.step(action, interrupt=ctrl.interrupt)
    out = dict(env.episode_summary())
    out["s6_forced"] = float(env.shield.n_ev_forced)
    env.close()
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=12)
    p.add_argument("--horizon", type=int, default=1800)
    p.add_argument("--ev-rate", type=float, default=30.0)
    p.add_argument("--checkpoint", default=str(MODELS_DIR / "siren_arc.h5"))
    p.add_argument("--mic-offset", type=float, default=None, metavar="M",
                   help="mast offset upstream of the stop line. Pass 70 to "
                        "reproduce the geometry the two-element arm was "
                        "written for; the default is whatever the config ships.")
    p.add_argument("--out", default=str(RESULTS_DIR / "ablation_ev.json"))
    args = p.parse_args()

    ensure_dirs()
    seeds = [SEED_BASE + i for i in range(args.seeds)]
    results = {k: [] for k in ARMS}
    t0 = time.time()

    for si, seed in enumerate(seeds):
        for label, (dual, s6) in ARMS.items():
            r = run_arm(dual, s6, seed, args.horizon, args.ev_rate,
                        Path(args.checkpoint), mic_offset=args.mic_offset)
            results[label].append(r)
            print(f"[{si + 1}/{len(seeds)}] seed={seed} {label:26s} "
                  f"ev_delay={r.get('ev_delay_per_vehicle_s', float('nan')):6.2f}s "
                  f"queue={r.get('mean_queue', float('nan')):6.2f} "
                  f"teleports={r.get('teleports', 0):.0f}", flush=True)

    print("\n" + "=" * 104)
    print(f"{args.seeds} shared seeds, {args.horizon}s each, "
          f"{args.ev_rate:g} EV/h   ({time.time() - t0:.0f}s wall)")
    print("=" * 104)
    header = f"{'metric':<30}" + "".join(f"{k:>18}" for k in ARMS)
    print(header)
    print("-" * len(header))
    for key, label, _ in METRICS:
        row = f"{label:<30}"
        for k in ARMS:
            vals = [r.get(key, float("nan")) for r in results[k]]
            row += f"{np.nanmean(vals):18.2f}"
        print(row)

    # Paired test against the shipped baseline, on the metric that motivated
    # the work. Paired because every arm saw the identical seed block.
    base = np.array([r.get("ev_delay_per_vehicle_s", np.nan)
                     for r in results["baseline (as shipped)"]])
    print("-" * len(header))
    print("paired change in EV stopped delay vs baseline (negative = better):")
    for k in ARMS:
        if k == "baseline (as shipped)":
            continue
        arm = np.array([r.get("ev_delay_per_vehicle_s", np.nan) for r in results[k]])
        d = arm - base
        stat = paired_bootstrap_pvalue(list(arm), list(base))
        pval = float(stat.get("p_value", stat.get("p", float("nan"))))
        print(f"   {k:26s} {np.nanmean(d):+7.2f}s "
              f"({100 * np.nanmean(d) / np.nanmean(base):+5.1f}%)   p={pval:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"seeds": seeds, "horizon_s": args.horizon, "ev_rate_per_hour": args.ev_rate,
         "arms": {k: results[k] for k in ARMS}}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
