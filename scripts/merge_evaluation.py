"""Fold an extra controller into an existing evaluation, and re-test.

A full evaluation over twelve seeds and fifteen controllers takes hours, so
adding one more controller by re-running the whole thing is wasteful. It is also
only sound under one condition: the addendum must have been produced on the
*same seeds*, at the same horizon and demand, by the same code. This checks that
condition rather than assuming it, refuses the merge if it does not hold, and
then recomputes the paired bootstrap over the union -- because a significance
test that still refers to the old controller set is worse than none.

    python scripts/merge_evaluation.py --into results/evaluation_balanced.json \
        --add results/_addendum_cmdp.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from atsc.evaluation.metrics import paired_bootstrap_pvalue

COMPARABLE = ("regime", "flow", "ev_rate", "horizon", "seeds", "sensing")


def compatible(a: dict, b: dict) -> list:
    """Reasons the two runs are not comparable; empty means they are."""
    problems = []
    for key in COMPARABLE:
        if a.get(key) != b.get(key):
            problems.append(f"{key}: {a.get(key)!r} vs {b.get(key)!r}")
    return problems


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--into", required=True)
    p.add_argument("--add", action="append", required=True)
    p.add_argument("--reference", default="base_arc",
                   help="controller the recomputed paired tests compare against")
    p.add_argument("--force", action="store_true",
                   help="merge even if the run configurations differ (they must "
                        "not, and this exists only for deliberate exceptions)")
    args = p.parse_args()

    base_path = Path(args.into)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    added = []

    for extra in args.add:
        path = Path(extra)
        if not path.exists():
            print(f"  skipping {path}: not found")
            continue
        other = json.loads(path.read_text(encoding="utf-8"))
        problems = compatible(base.get("config", {}), other.get("config", {}))
        if problems and not args.force:
            print(f"refusing to merge {path.name}: configurations differ")
            for msg in problems:
                print(f"  - {msg}")
            return 1
        for name, res in other.get("results", {}).items():
            if name in base["results"]:
                print(f"  replacing existing entry {name!r}")
            base["results"][name] = res
            added.append(name)

    if not added:
        print("nothing to merge")
        return 0

    # Recompute the paired tests over the union. Pairing is by position in the
    # seed list, which is why the configurations had to match.
    ref = args.reference
    tests = {}
    if ref in base["results"]:
        for name, res in base["results"].items():
            if name == ref:
                continue
            tests[name] = {}
            for key in ("mean_time_loss_s", "ev_mean_waiting_time_s",
                        "ev_zero_stop_rate"):
                x = [r.get(key, float("nan")) for r in res["runs"]]
                y = [r.get(key, float("nan")) for r in base["results"][ref]["runs"]]
                pair = [(u, v) for u, v in zip(x, y)
                        if np.isfinite(u) and np.isfinite(v)]
                if len(pair) < 3:
                    continue
                tests[name][key] = paired_bootstrap_pvalue(
                    [u for u, _ in pair], [v for _, v in pair])
        base["significance"] = tests
        base.setdefault("config", {})["reference"] = ref
    else:
        print(f"  warning: reference {ref!r} not present; significance not recomputed")

    base_path.write_text(json.dumps(base, indent=2, default=float), encoding="utf-8")
    print(f"merged {len(set(added))} controller(s) into {base_path.name}: "
          f"{', '.join(sorted(set(added)))}")
    print(f"significance recomputed against {ref}")

    # Keep the CSV beside the JSON in step, since make_tables reads the JSON but
    # a human reading the directory will open the CSV.
    csv_path = base_path.with_suffix(".csv")
    if csv_path.exists():
        keys = sorted({k for res in base["results"].values()
                       for k in res["aggregate"] if k.endswith("_mean")})
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["controller", "delay_median", "n_jammed"] + keys)
            for name, res in base["results"].items():
                agg = res["aggregate"]
                w.writerow([name, agg.get("mean_time_loss_s_median", ""),
                            agg.get("n_jammed", "")]
                           + [agg.get(k, "") for k in keys])
        print(f"rewrote {csv_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
