"""Recompute the headline numbers of the paper from results/.

Every figure quoted in the manuscript should be reproducible from the stored
evaluation output. This script recomputes them and prints them next to what the
manuscript claims, so a mismatch is visible rather than inferred.

    python scripts/verify_paper_numbers.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SEED_KEY = "seed"
N_BOOT = 20000
RNG = np.random.default_rng(12345)


def load(regime: str) -> dict:
    p = RESULTS / f"evaluation_{regime}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def per_seed(res: dict, controller: str, metric: str):
    """Metric value per seed, keyed by seed so pairing is by seed not by order."""
    runs = res.get("results", {}).get(controller, {}).get("runs", [])
    out = {}
    for r in runs:
        s = r.get(SEED_KEY)
        v = r.get(metric)
        if s is not None and v is not None and np.isfinite(v):
            out[s] = float(v)
    return out


def paired_bootstrap(a: dict, b: dict):
    """Paired bootstrap of mean(a) - mean(b) over the seeds both share."""
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 3:
        return None
    da = np.array([a[s] for s in seeds])
    db = np.array([b[s] for s in seeds])
    d = da - db
    obs = float(d.mean())
    idx = RNG.integers(0, len(d), size=(N_BOOT, len(d)))
    boot = d[idx].mean(axis=1)
    # two-sided p for H0: mean difference = 0, centred on the observed mean
    centred = boot - obs
    p = float((np.abs(centred) >= abs(obs)).mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(n=len(seeds), diff=obs, p=max(p, 1.0 / N_BOOT),
                ci=(float(lo), float(hi)),
                mean_a=float(da.mean()), mean_b=float(db.mean()),
                med_a=float(np.median(da)), med_b=float(np.median(db)))


def show(title, r, claim=""):
    if r is None:
        print(f"  {title:44s} INSUFFICIENT PAIRED SEEDS")
        return
    print(f"  {title:44s} n={r['n']:2d}  "
          f"{r['mean_b']:8.3f} -> {r['mean_a']:8.3f}  "
          f"diff {r['diff']:+8.3f}  p={r['p']:.4f}   {claim}")


def main() -> int:
    bal = load("balanced")
    if not bal:
        print("no balanced evaluation found")
        return 1

    PROPOSED = "siren_arc"
    NOSENSE = "base_no_sensing"
    PREEMPT = "max_pressure_preempt"

    print("\n=== Table V: proposed vs learned baseline WITHOUT acoustic sensing ===")
    for metric, label in (("ev_mean_waiting_time_s", "EV stopped delay (s)"),
                          ("ev_zero_stop_rate", "clean-pass fraction"),
                          ("mean_time_loss_s", "background delay (s)")):
        a = per_seed(bal, PROPOSED, metric)
        b = per_seed(bal, NOSENSE, metric)
        show(label, paired_bootstrap(a, b))

    print("\n=== proposed vs max-pressure + hard preemption ===")
    for metric, label in (("ev_mean_waiting_time_s", "EV stopped delay (s)"),
                          ("mean_time_loss_s", "background delay (s)"),
                          ("ev_zero_stop_rate", "clean-pass fraction")):
        a = per_seed(bal, PROPOSED, metric)
        b = per_seed(bal, PREEMPT, metric)
        show(label, paired_bootstrap(a, b))

    print("\n=== proposed vs oracle sensing ceiling ===")
    a = per_seed(bal, PROPOSED, "ev_mean_waiting_time_s")
    b = per_seed(bal, "arc_oracle", "ev_mean_waiting_time_s")
    show("EV stopped delay (s)", paired_bootstrap(a, b))

    print("\n=== Table IV: means / medians actually stored ===")
    want = [("fixed_time", "Fixed-time (Webster)"),
            ("actuated", "Actuated (SUMO)"),
            ("greedy_queue", "Greedy queue"),
            ("max_pressure", "Max-pressure"),
            (PREEMPT, "Max-pressure + preemption"),
            (NOSENSE, "Learned policy, no sensing"),
            ("base_arc", "Learned policy alone"),
            ("base_cmdp", "Constrained policy (CMDP only)"),
            (PROPOSED, "SIREN-CMDP (proposed)"),
            ("arc_oracle", "Ceiling: proposed + oracle"),
            ("max_pressure_preempt_oracle", "Ceiling: preemption + oracle")]
    print(f"  {'controller':34s} {'bg med':>8s} {'bg mean':>8s} {'queue':>7s} "
          f"{'EV stop':>8s} {'clean':>6s} {'maxwait':>8s} {'jam':>4s}")
    for key, label in want:
        agg = bal.get("results", {}).get(key, {}).get("aggregate", {})
        if not agg:
            print(f"  {label:34s} MISSING")
            continue
        runs = bal["results"][key]["runs"]
        bgs = [r["mean_time_loss_s"] for r in runs if r.get("mean_time_loss_s") is not None]
        jam = sum(1 for r in runs if r.get("teleports", 0) and r["teleports"] > 0)
        print(f"  {label:34s} {np.median(bgs):8.2f} {np.mean(bgs):8.2f} "
              f"{agg.get('mean_queue_mean', float('nan')):7.2f} "
              f"{agg.get('ev_mean_waiting_time_s_mean', float('nan')):8.2f} "
              f"{agg.get('ev_zero_stop_rate_mean', float('nan')):6.3f} "
              f"{agg.get('max_movement_wait_s_max', float('nan')):8.0f} "
              f"{jam:2d}/{len(runs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
