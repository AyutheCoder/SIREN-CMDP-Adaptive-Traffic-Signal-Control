"""Print every number the prose quotes, from the files that produced them.

The paper's tables and figures are generated, but its sentences are written by
hand, and a hand-written sentence is exactly where a stale number survives a
re-run. This prints the quotable quantities in one place so the narrative can be
checked against them in one pass, and re-checked after any re-run.

    python scripts/report_numbers.py
    python scripts/report_numbers.py --section control
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atsc.config import RESULTS_DIR


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def g(agg: dict, key: str, suffix: str = "_mean"):
    v = agg.get(f"{key}{suffix}")
    return float("nan") if v is None else float(v)


def line(label: str, value, unit: str = "", places: int = 2) -> None:
    if isinstance(value, float):
        txt = "--" if not math.isfinite(value) else f"{value:.{places}f}"
    else:
        txt = str(value)
    print(f"  {label:<46s} {txt:>10s} {unit}")


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def report_sensing(results: Path) -> None:
    bench = load(results / "sensing_benchmark.json")
    if not bench:
        print("  (no sensing_benchmark.json)")
        return
    modes = bench["modes"]
    ep = max(float(bench.get("config", {}).get("episodes", 1) or 1), 1.0)
    for name in ("independent", "joint"):
        d = modes.get(name)
        if not d:
            continue
        print(f"\n  [{name}]")
        line("precision @0.5", d["prf@0.5"]["precision"], "", 3)
        line("recall @0.5", d["prf@0.5"]["recall"], "", 3)
        line("F1 @0.5", d["prf@0.5"]["f1"], "", 3)
        line("false detections", d["prf@0.5"]["false_alarms_per_hour"] / ep, "per hour", 0)
        line("arms above 0.5 when one occupied", d.get("arms_above@0.5"), "", 2)
        line("localisation top-1", d.get("localisation_top1"), "", 3)
        line("AUC", d.get("auc"), "", 4)
        line("ECE", d.get("ece"), "", 4)
        line("Brier", d.get("brier"), "", 4)
        line("median range error", d.get("range_median_ae_m"), "m", 1)
        line("arrival-time MAE", d.get("eta_mae_s"), "s", 1)
        line("median warning distance", d.get("warning_range_median_m@0.5"), "m", 0)
        line("arrivals detected", d.get("arrival_detection_rate@0.5"), "", 3)
        line("class accuracy", d.get("class_accuracy"), "", 3)
        line("env step cost", d.get("median_epoch_us", float("nan")) / 1000.0, "ms", 1)
        fp = d.get("false_positive_kinds", {})
        for k, v in sorted(fp.items()):
            line(f"  false positives: {k}", v, "", 3)
    diag = load(results / "calibration_diagnostics.json")
    if diag:
        print("\n  [calibration, joint filter]")
        for k in ("auc", "ece_before", "ece_after", "brier_before", "brier_after",
                  "occupancy", "n_samples"):
            line(k, diag.get(k), "", 4)
        cal = diag.get("calibrated") or []
        line("max attainable calibrated probability", max(cal) if cal else float("nan"),
             "", 3)


def report_control(results: Path) -> None:
    data = load(results / "evaluation_balanced.json")
    if not data:
        print("  (no evaluation_balanced.json)")
        return
    res = data["results"]
    hdr = (f"  {'controller':<30}{'delay':>8}{'med':>8}{'jam':>7}"
           f"{'EVstop':>9}{'stops':>7}{'clean':>7}{'maxwait':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, r in res.items():
        a = r["aggregate"]
        print(f"  {name:<30}{g(a,'mean_time_loss_s'):8.2f}"
              f"{g(a,'mean_time_loss_s','_median'):8.2f}"
              f"{a.get('n_jammed',0):4.0f}/{a.get('n_runs',0):<2.0f}"
              f"{g(a,'ev_mean_waiting_time_s'):9.2f}"
              f"{g(a,'ev_mean_stops'):7.2f}"
              f"{g(a,'ev_zero_stop_rate'):7.3f}"
              f"{g(a,'max_movement_wait_s','_max'):9.0f}")

    print("\n  corridor behaviour")
    for name, r in res.items():
        a = r["aggregate"]
        eng = a.get("corridor_priority_engagements_mean")
        if eng is None or not math.isfinite(float(eng)):
            continue
        print(f"  {name:<30} engagements={eng:5.1f}/h  "
              f"duty={g(a,'corridor_priority_duty_cycle'):.3f}  "
              f"lead={g(a,'corridor_mean_lead_time_s'):5.1f}s  "
              f"open={g(a,'corridor_mean_open_time_s'):5.1f}s  "
              f"slack={g(a,'corridor_mean_slack_left_s'):5.1f}s  "
              f"conflict={g(a,'corridor_conflict_rate'):.3f}  "
              f"falsepre={g(a,'corridor_expected_false_preemptions'):.2f}")

    print("\n  emergency delay by class (mean stopped delay, s)")
    for name, r in res.items():
        a = r["aggregate"]
        vals = [g(a, f"ev_{c}_mean_waiting_time_s") for c in ("ambulance", "fire", "police")]
        if not any(math.isfinite(v) for v in vals):
            continue
        print(f"  {name:<30} ambulance={vals[0]:6.2f}  fire={vals[1]:6.2f}  "
              f"police={vals[2]:6.2f}  "
              f"(pooled n={a.get('ev_n_vehicles', 0):.0f}, "
              f"median={a.get('ev_waiting_pooled_median_s', float('nan')):.1f}, "
              f"p90={a.get('ev_waiting_pooled_p90_s', float('nan')):.1f})")

    sig = data.get("significance") or {}
    if sig:
        print(f"\n  paired bootstrap against {data['config'].get('reference', 'max_pressure')}")
        for name, tests in sig.items():
            for key, stat in tests.items():
                print(f"  {name:<26}{key:<26} diff={stat['diff']:+8.2f}  "
                      f"p={stat['p_value']:.4f}")


def report_frontier(results: Path) -> None:
    sweep = load(results / "value_sweep.json")
    if not sweep:
        print("  (no value_sweep.json)")
        return
    rows = sweep if isinstance(sweep, list) else sweep.get("rows", [])
    refs = {} if isinstance(sweep, list) else sweep.get("references", {})
    print(f"  {'gamma':>8}{'delay':>9}{'EVstop':>9}{'clean':>8}{'jam':>7}{'preempt':>9}")
    for r in rows:
        print(f"  {r['gamma']:8g}{r['delay_mean']:9.2f}{r['ev_delay_mean']:9.2f}"
              f"{(r.get('clean_pass') or float('nan')):8.3f}"
              f"{(r.get('n_jammed') or 0):4.0f}   "
              f"{(r.get('preemptions') or float('nan')):9.1f}")
    for name, r in refs.items():
        print(f"  {name:>8}{r['delay_mean']:9.2f}{r['ev_delay_mean']:9.2f}"
              f"{(r.get('clean_pass') or float('nan')):8.3f}"
              f"{(r.get('n_jammed') or 0):4.0f}")


def report_robustness(results: Path) -> None:
    data = load(results / "robustness.json")
    if not data:
        print("  (no robustness.json)")
        return
    print(f"  {'condition':<12}{'delay':>9}{'EVstop':>9}{'clean':>8}{'jam':>7}{'corridors':>11}")
    for name, r in data["results"].items():
        a = r["aggregate"]
        print(f"  {name:<12}{g(a,'mean_time_loss_s'):9.2f}"
              f"{g(a,'ev_mean_waiting_time_s'):9.2f}"
              f"{g(a,'ev_zero_stop_rate'):8.3f}"
              f"{a.get('n_jammed',0):4.0f}/{a.get('n_runs',0):<2.0f}"
              f"{g(a,'corridor_priority_engagements'):11.1f}")


def report_surge(results: Path) -> None:
    data = load(results / "evaluation_surge.json")
    if not data:
        print("  (no evaluation_surge.json)")
        return
    for name, r in data["results"].items():
        a = r["aggregate"]
        print(f"  {name:<30}{g(a,'mean_time_loss_s'):9.2f}"
              f"{g(a,'mean_time_loss_s','_median'):9.2f}"
              f"{a.get('n_jammed',0):4.0f}/{a.get('n_runs',0):<2.0f}"
              f"{g(a,'ev_mean_waiting_time_s'):9.2f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default=str(RESULTS_DIR))
    p.add_argument("--section", default="all",
                   choices=["all", "sensing", "control", "frontier",
                            "robustness", "surge"])
    args = p.parse_args()
    results = Path(args.results)

    want = args.section
    for name, fn in (("sensing", report_sensing), ("control", report_control),
                     ("frontier", report_frontier), ("robustness", report_robustness),
                     ("surge", report_surge)):
        if want in ("all", name):
            section(name.upper())
            fn(results)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
