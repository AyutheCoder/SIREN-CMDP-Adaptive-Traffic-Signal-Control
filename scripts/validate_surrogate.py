"""Check that the fast surrogate sensor is faithful to the real detector.

Training the RL agent against real audio is impractical - rendering four
microphone frames and running the CNN at every decision epoch is roughly 30x
slower than sampling from a fitted error model. The whole argument for the
surrogate therefore rests on it being *statistically indistinguishable* from
the real thing, and that has to be demonstrated rather than asserted.

Two levels of check:

**Sensor level.** Draw matched scenes across the operating envelope and compare
detection rate by SNR, class posterior, and range error between the CNN and the
surrogate.

**System level.** Run complete control episodes both ways on identical seeds and
compare the metrics that the paper actually reports. If the closed-loop delay
and emergency-vehicle delay agree within noise, the substitution is sound.

    python scripts/validate_surrogate.py --scenes 3000 --episodes 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atsc.agents.baselines import HardPreemption, MaxPressure
from atsc.config import (
    AcousticConfig, DATA_DIR, DemandConfig, EnvConfig, MODELS_DIR, RESULTS_DIR, RUNS_DIR, ensure_dirs,
)
from atsc.env import SumoSignalEnv
from atsc.evaluation.metrics import paired_bootstrap_pvalue
from atsc.evaluation.runner import run_episode
from atsc.netgen import build_network
from atsc.sensing.detector import SirenDetector
from atsc.sensing.sensor import AcousticSensor, SensorErrorModel, SourceState

SNR_EDGES = [-20.0, -10.0, -5.0, 0.0, 5.0, 15.0]


def sensor_level(detector, error_model, acoustic: AcousticConfig, n_scenes: int, seed: int):
    """Compare per-frame behaviour of the CNN and the surrogate on matched scenes."""
    rng = np.random.default_rng(seed)
    e2e = AcousticSensor(["N"], acoustic, mode="end_to_end",
                         detector=detector, mel=detector.mel, seed=seed)
    sur = AcousticSensor(["N"], acoustic, mode="surrogate",
                         error_model=error_model, seed=seed + 1)

    rows = []
    t0 = time.time()
    for i in range(n_scenes):
        if rng.random() < 0.35:
            scene = {"N": []}
            src = None
        else:
            veh_class = str(rng.choice(["ambulance", "fire", "police"]))
            distance = float(rng.uniform(5.0, acoustic.detection_radius_m))
            v_r = -float(rng.uniform(3.0, acoustic.max_radial_speed_ms))
            theta = float(rng.uniform(0.0, np.deg2rad(20.0))
                          if rng.random() < 0.5
                          else rng.uniform(np.deg2rad(55.0), np.pi))
            src = SourceState(veh_class, distance, v_r, theta,
                              acoustic.directivity_front_back_db)
            scene = {"N": [src]}

        a = e2e.sense(scene)["N"]
        b = sur.sense(scene)["N"]
        snr = (src.received_spl_db() - 67.5) if src is not None else -np.inf
        rows.append({
            "has_siren": src is not None,
            "snr": snr,
            "true_range": src.distance_m if src is not None else np.nan,
            "e2e_p": a.p_siren,
            "sur_p": b.p_siren,
            "e2e_r": a.distance_m if a.distance_m is not None else np.nan,
            "sur_r": b.distance_m if b.distance_m is not None else np.nan,
        })
    elapsed = time.time() - t0

    has = np.array([r["has_siren"] for r in rows])
    snr = np.array([r["snr"] for r in rows])
    e2e_p = np.array([r["e2e_p"] for r in rows])
    sur_p = np.array([r["sur_p"] for r in rows])

    out = {
        "n_scenes": n_scenes,
        "wall_time_s": elapsed,
        "false_positive_rate_e2e": float(np.mean(e2e_p[~has] >= 0.5)) if (~has).any() else float("nan"),
        "false_positive_rate_surrogate": float(np.mean(sur_p[~has] >= 0.5)) if (~has).any() else float("nan"),
        "detection_rate_e2e": float(np.mean(e2e_p[has] >= 0.5)) if has.any() else float("nan"),
        "detection_rate_surrogate": float(np.mean(sur_p[has] >= 0.5)) if has.any() else float("nan"),
        "mean_p_e2e": float(np.mean(e2e_p)),
        "mean_p_surrogate": float(np.mean(sur_p)),
    }

    bins = np.digitize(snr, SNR_EDGES)
    by_snr = []
    for b in range(len(SNR_EDGES) + 1):
        sel = has & (bins == b)
        if sel.sum() < 5:
            continue
        lo = "-inf" if b == 0 else f"{SNR_EDGES[b - 1]:g}"
        hi = "inf" if b == len(SNR_EDGES) else f"{SNR_EDGES[b]:g}"
        by_snr.append({
            "snr_bin": f"[{lo}, {hi})",
            "n": int(sel.sum()),
            "detect_e2e": float(np.mean(e2e_p[sel] >= 0.5)),
            "detect_surrogate": float(np.mean(sur_p[sel] >= 0.5)),
        })
    out["by_snr"] = by_snr
    out["max_detection_rate_gap"] = (
        max(abs(r["detect_e2e"] - r["detect_surrogate"]) for r in by_snr) if by_snr else float("nan")
    )

    true_r = np.array([r["true_range"] for r in rows])
    for tag in ("e2e", "sur"):
        est = np.array([r[f"{tag}_r"] for r in rows])
        ok = has & np.isfinite(est) & np.isfinite(true_r)
        out[f"range_mape_{tag}"] = (
            float(np.mean(np.abs(est[ok] - true_r[ok]) / np.maximum(true_r[ok], 1.0)))
            if ok.any() else float("nan")
        )
    return out


def system_level(acoustic, error_model, detector, n_episodes: int, horizon: int, seed: int):
    """Run whole control episodes under each sensing mode on matched seeds.

    The controller must be one that *consumes* the belief, or the comparison is
    vacuous. An earlier version of this script used max-pressure, which ignores
    emergency-vehicle sensing entirely; unsurprisingly the two sensing modes
    produced bit-identical trajectories and a p-value of exactly 1.0, which
    looks like a perfect validation and is in fact no validation at all. Hard
    preemption reads the belief directly, so any divergence between the
    surrogate and the real detector shows up in the control outcome.
    """
    net_file, _ = build_network(EnvConfig().network, DATA_DIR / "networks")
    out = {}
    for mode in ("surrogate", "end_to_end"):
        cfg = EnvConfig()
        cfg.demand = DemandConfig(horizon_s=horizon, ev_rate_per_hour=60.0)
        cfg.sensing_mode = mode
        cfg.collect_emissions = False
        runs = []
        t0 = time.time()
        for i in range(n_episodes):
            sensor = AcousticSensor(
                [], acoustic, mode=mode,
                error_model=error_model,
                detector=detector if mode == "end_to_end" else None,
                mel=detector.mel if mode == "end_to_end" else None,
                seed=seed + i,
            )
            env = SumoSignalEnv(
                cfg, net_file=net_file,
                work_dir=RUNS_DIR / f"validate_{mode}",
                label=f"val_{mode}_{i}",
                tripinfo_path=RUNS_DIR / f"validate_{mode}" / f"trip_{i}.xml",
                sensor=sensor,
            )
            runs.append(run_episode(
                env, lambda e: HardPreemption(base=MaxPressure()), seed=seed + i))
            print(f"  {mode:<11} ep {i}  delay={runs[-1]['mean_time_loss_s']:.2f}s  "
                  f"ev={runs[-1].get('ev_mean_time_loss_s', float('nan')):.2f}s", flush=True)
        out[mode] = {
            "runs": runs,
            "wall_time_s": time.time() - t0,
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes", type=int, default=3000)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--horizon", type=int, default=1200)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--detector", type=str, default=str(MODELS_DIR / "siren_detector.h5"))
    p.add_argument("--error-model", type=str,
                   default=str(MODELS_DIR / "sensor_error_model.json"))
    args = p.parse_args()

    ensure_dirs()
    acoustic = AcousticConfig()
    det_path, em_path = Path(args.detector), Path(args.error_model)
    if not det_path.exists() or not em_path.exists():
        raise SystemExit("run scripts/train_detector.py first")

    detector = SirenDetector.from_saved(det_path, acoustic)
    error_model = SensorErrorModel.load(em_path)

    print(f"sensor-level comparison over {args.scenes} matched scenes ...")
    sensor = sensor_level(detector, error_model, acoustic, args.scenes, args.seed)
    print(f"  detection rate   CNN={sensor['detection_rate_e2e']:.3f}  "
          f"surrogate={sensor['detection_rate_surrogate']:.3f}")
    print(f"  false positives  CNN={sensor['false_positive_rate_e2e']:.3f}  "
          f"surrogate={sensor['false_positive_rate_surrogate']:.3f}")
    print(f"  range MAPE       CNN={sensor['range_mape_e2e']:.3f}  "
          f"surrogate={sensor['range_mape_sur']:.3f}")
    print("  detection rate by SNR:")
    for row in sensor["by_snr"]:
        print(f"    {row['snr_bin']:>14s}  n={row['n']:5d}  "
              f"CNN={row['detect_e2e']:.3f}  surrogate={row['detect_surrogate']:.3f}")
    print(f"  largest gap: {sensor['max_detection_rate_gap']:.3f}")

    print(f"\nsystem-level comparison over {args.episodes} matched episodes ...")
    system = system_level(acoustic, error_model, detector,
                          args.episodes, args.horizon, args.seed)

    comparison = {}
    for key in ("mean_time_loss_s", "ev_mean_waiting_time_s", "ev_mean_stops",
                "mean_queue"):
        a = [r.get(key, float("nan")) for r in system["surrogate"]["runs"]]
        b = [r.get(key, float("nan")) for r in system["end_to_end"]["runs"]]
        pair = [(x, y) for x, y in zip(a, b) if np.isfinite(x) and np.isfinite(y)]
        if len(pair) < 2:
            continue
        stat = paired_bootstrap_pvalue([x for x, _ in pair], [y for _, y in pair])
        comparison[key] = stat
        print(f"  {key:<24} surrogate-vs-CNN diff={stat['diff']:+7.3f}  p={stat['p_value']:.3f}")

    speedup = system["end_to_end"]["wall_time_s"] / max(system["surrogate"]["wall_time_s"], 1e-9)
    print(f"\n  wall time: surrogate {system['surrogate']['wall_time_s']:.1f}s  "
          f"vs end-to-end {system['end_to_end']['wall_time_s']:.1f}s  ({speedup:.1f}x)")

    out_path = RESULTS_DIR / "surrogate_validation.json"
    out_path.write_text(json.dumps({
        "sensor_level": sensor,
        "system_level": {k: {"runs": v["runs"], "wall_time_s": v["wall_time_s"]}
                         for k, v in system.items()},
        "comparison": comparison,
        "speedup": speedup,
    }, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
