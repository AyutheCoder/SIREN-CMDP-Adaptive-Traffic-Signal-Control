"""Reproduce the whole study end to end.

Stage order matters and is not arbitrary:

1. **detector** -- the CNN and the error model fitted to its held-out
   predictions. Everything downstream consumes that error model, including the
   joint belief filter's likelihood, so nothing else can run first.
2. **cost check** -- verify the constraint cost can rank fixed controllers of
   known quality *before* spending training runs on it. A cost that cannot tell
   max-pressure from hard preemption will not teach a policy either, and
   discovering that after six trainings is expensive.
3. **train** -- the base policies. The sensing ablation is trained, not swapped
   at test time: a policy's belief features come from a particular filter, and
   running it against the other one measures a train/test mismatch.
4. **calibrate** -- the isotonic map from filter output to probability, on seeds
   disjoint from everything else. It must follow training because it is fitted
   under the phase behaviour it will be used with.
5. **sensing benchmark** -- score the filters against ground truth.
6. **surrogate validation, evaluation, sweeps, robustness, traces, tables,
   figures.**

Each stage is skipped if its output already exists, so an interrupted run
resumes rather than restarting.

    python scripts/run_all.py                 # full study (several hours)
    python scripts/run_all.py --quick         # reduced settings, ~1 hour
    python scripts/run_all.py --only evaluate # a single stage
    python scripts/run_all.py --force         # ignore existing outputs
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from atsc.config import FIGURES_DIR, MODELS_DIR, RESULTS_DIR, ensure_dirs

PY = sys.executable
REGIMES = ("balanced", "asymmetric", "peak", "surge")

HEADLINE_BUDGET = 0.12
"""Cost budget for the constrained run, on the shaped (anticipatory) cost scale.

It has to *bind*, or the constrained and unconstrained agents are the same agent
and the comparison means nothing. The efficiency-only agent converges at a
shaped cost of 0.13--0.22 unaided, so 0.12 sits just below what it reaches for
free. The run then terminates infeasible, which we report rather than relaxing
the budget until it is satisfied: a budget below what the intersection can
deliver is not a solver failure, and moving it until the solver succeeds is how
a constrained method comes to mean nothing.

How far infeasible is itself a measurement, and it moved. Before the corridor's
economics were fixed the best achievable cost at this budget was 0.207, an
excess of 0.087; it is now 0.169, an excess of 0.049. Nothing about the
constraint, the budget or the dual ascent changed -- what changed is that the
intersection can now actually deliver more emergency priority at the same
demand, so the same requirement is closer to being met. The residual excess is
the real one: this junction runs near capacity and 0.12 is below what it can
give.

Run scripts/check_cost_signal.py to re-derive the scale if the cost or the demand
changes."""

BASE_RUNS = [
    # (run name, extra arguments, what it is for)
    ("siren_arc", ["--unconstrained", "--belief-mode", "joint"],
     "SIREN-ARC's base: efficiency only, joint belief"),
    ("siren_cmdp", ["--budget", str(HEADLINE_BUDGET), "--belief-mode", "joint"],
     "SIREN-CMDP's base: efficiency subject to an emergency-delay budget"),
    ("arc_base_indep", ["--unconstrained", "--belief-mode", "independent"],
     "sensing ablation, trained end to end on the independent filter"),
    ("no_sensing", ["--unconstrained", "--sensing", "none"],
     "no emergency-vehicle awareness at all"),
]

CORRIDOR_ENTRIES = [
    # (evaluation name, checkpoint stem, flags appended to the path)
    ("siren_arc", "siren_arc", ""),
    ("cmdp_decoupled", "siren_cmdp", ""),
    # The overlay's exchange rate coupled to the policy's dual variable. Kept
    # because it is the ablation that shows why they are decoupled by default.
    ("siren_cmdp", "siren_cmdp", ":adapt"),
    ("arc_myopic", "siren_arc", ":myopic"),
    ("arc_nocal", "siren_arc", ":nocal"),
    ("arc_greedy", "siren_arc", ":greedy"),
    ("arc_on_indep", "siren_arc", ":indep"),
    ("arc_oracle", "siren_arc", ":oracle"),
    ("arc_indep", "arc_base_indep", ""),
]


def run(cmd: list, label: str) -> None:
    print(f"\n{'=' * 78}\n>> {label}\n   {' '.join(str(c) for c in cmd)}\n{'=' * 78}",
          flush=True)
    t0 = time.time()
    result = subprocess.run([str(c) for c in cmd], cwd=REPO)
    if result.returncode != 0:
        raise SystemExit(f"stage failed: {label} (exit {result.returncode})")
    print(f"-- {label} finished in {(time.time() - t0) / 60:.1f} min", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true",
                   help="reduced episode counts and seeds for a fast smoke run")
    p.add_argument("--force", action="store_true",
                   help="rerun stages even if outputs exist")
    p.add_argument("--only", type=str, default=None,
                   choices=["detector", "train", "calibrate", "sensing",
                            "surrogate", "evaluate", "sweep", "robustness",
                            "trace", "tables", "figures"])
    p.add_argument("--budget", type=float, default=HEADLINE_BUDGET)
    args = p.parse_args()

    ensure_dirs()
    quick = args.quick

    detector_epochs = 6 if quick else 22
    detector_clips = 4000 if quick else 20000
    episodes = 30 if quick else 200
    horizon = 900 if quick else 1800
    eval_seeds = 3 if quick else 12
    eval_horizon = 1800 if quick else 3600
    cal_episodes = 3 if quick else 8
    sense_episodes = 2 if quick else 6
    sweep_seeds = 2 if quick else 5
    regimes = ("balanced",) if quick else REGIMES

    def want(stage: str) -> bool:
        return args.only is None or args.only == stage

    def missing(path: Path) -> bool:
        return args.force or not Path(path).exists()

    t_start = time.time()

    # ------------------------------------------------------------ detector
    if want("detector") and missing(MODELS_DIR / "siren_detector.h5"):
        run([PY, "scripts/train_detector.py",
             "--epochs", detector_epochs,
             "--train-clips", detector_clips,
             "--val-clips", max(detector_clips // 7, 500),
             "--test-clips", max(detector_clips // 4, 1000)],
            "train acoustic detector")
    elif want("detector"):
        print("skip: detector already trained")

    # ------------------------------------------------- constraint-cost sanity
    if want("train") and missing(RESULTS_DIR / "cost_signal_check.json"):
        run([PY, "scripts/check_cost_signal.py", "--episodes", 2, "--horizon", 1200],
            "check the constraint cost is attributable to actions")

    # --------------------------------------------------------------- agents
    if want("train"):
        plan = BASE_RUNS if not quick else BASE_RUNS[:2]
        for name, extra, purpose in plan:
            ckpt = MODELS_DIR / f"{name}.h5"
            if not missing(ckpt):
                print(f"skip: {name} already trained ({purpose})")
                continue
            run([PY, "scripts/train_agent.py",
                 "--episodes", episodes, "--horizon", horizon,
                 "--ev-rate", 30, "--run-name", name] + extra,
                f"train {name}: {purpose}")

    # ----------------------------------------------------------- calibration
    # One map per filter. A map fitted on one filter's scores is meaningless
    # applied to the other's, and applying it anyway is how a decision rule ends
    # up multiplying a cost by something that is not a probability.
    if want("calibrate"):
        for mode, out, diag in (
            ("joint", MODELS_DIR / "presence_calibration.json",
             RESULTS_DIR / "calibration_diagnostics.json"),
            ("independent", MODELS_DIR / "presence_calibration_independent.json",
             RESULTS_DIR / "calibration_diagnostics_independent.json"),
        ):
            if not missing(out):
                print(f"skip: {mode} calibration already fitted")
                continue
            run([PY, "scripts/fit_calibration.py",
                 "--episodes", cal_episodes, "--belief-mode", mode,
                 "--out", out, "--diagnostics", diag],
                f"fit the {mode} presence calibration")

    # ------------------------------------------------------ sensing benchmark
    if want("sensing") and missing(RESULTS_DIR / "sensing_benchmark.json"):
        run([PY, "scripts/sensing_benchmark.py", "--episodes", sense_episodes],
            "score both presence filters against ground truth")
    elif want("sensing"):
        print("skip: sensing benchmark already run")

    # ---------------------------------------------------------- validation
    if want("surrogate") and missing(RESULTS_DIR / "surrogate_validation.json"):
        run([PY, "scripts/validate_surrogate.py",
             "--scenes", 800 if quick else 3000,
             "--episodes", 2 if quick else 3,
             "--horizon", 900 if quick else 1200],
            "validate the surrogate sensor against the real detector")
    elif want("surrogate"):
        print("skip: surrogate validation already done")

    # ---------------------------------------------------------- evaluation
    if want("evaluate"):
        entry_args = []
        for name, stem, flags in CORRIDOR_ENTRIES:
            ckpt = MODELS_DIR / f"{stem}.h5"
            if ckpt.exists():
                entry_args += ["--corridor", f"{name}={ckpt}{flags}"]
        # The same policies with no priority layer, so the corridor's own
        # contribution is measurable rather than inferred.
        for name, label in (("siren_arc", "base_arc"), ("siren_cmdp", "base_cmdp"),
                            ("arc_base_indep", "base_indep"),
                            ("no_sensing", "base_no_sensing")):
            ckpt = MODELS_DIR / f"{name}.h5"
            if ckpt.exists():
                entry_args += ["--agent", f"{label}={ckpt}"]
        for regime in regimes:
            out = RESULTS_DIR / f"evaluation_{regime}.json"
            if not missing(out):
                print(f"skip: evaluation for {regime} already done")
                continue
            run([PY, "scripts/evaluate.py",
                 "--seeds", eval_seeds, "--horizon", eval_horizon,
                 "--ev-rate", 30, "--regime", regime] + entry_args,
                f"evaluate all controllers ({regime} demand)")

    # --------------------------------------------------------------- sweeps
    if want("sweep"):
        if missing(RESULTS_DIR / "frontier.json"):
            available = [n for n in ("siren_cmdp", "siren_arc")
                         if (MODELS_DIR / f"{n}.h5").exists()]
            run([PY, "scripts/frontier.py", "--agents", ",".join(available),
                 "--seeds", sweep_seeds, "--horizon", eval_horizon],
                "trace the trade-off frontier from per-budget agents")
        else:
            print("skip: frontier already computed")
        if missing(RESULTS_DIR / "value_sweep.json"):
            run([PY, "scripts/value_sweep.py", "--seeds", sweep_seeds],
                "sweep the emergency/background exchange rate")
        else:
            print("skip: value sweep already computed")
        if missing(RESULTS_DIR / "threshold_audit.json"):
            run([PY, "scripts/threshold_audit.py"],
                "audit that the commitment threshold is derived, not constant")
        else:
            print("skip: threshold audit already computed")

    # ---------------------------------------------------- sensor degradation
    if want("robustness") and missing(RESULTS_DIR / "robustness.json"):
        run([PY, "scripts/robustness.py", "--seeds", sweep_seeds],
            "measure degradation under noise and mast failure")
    elif want("robustness"):
        print("skip: robustness study already run")

    # --------------------------------------------------------------- trace
    if want("trace") and missing(RESULTS_DIR / "trajectories.json"):
        run([PY, "scripts/record_trajectory.py"],
            "record one emergency vehicle under several controllers")
    if want("trace") and missing(RESULTS_DIR / "belief_trace.json"):
        run([PY, "scripts/record_trace.py", "--horizon", 1800],
            "record a qualitative emergency-approach trace")
    elif want("trace"):
        print("skip: trace already recorded")

    # ------------------------------------------------------ tables, figures
    if want("tables"):
        run([PY, "scripts/make_tables.py"], "generate tables")
        # The README quotes the same numbers as the paper. It used to be
        # updated by hand, which is the one place a stale table could survive
        # a re-run; it is now spliced from the same generated file.
        run([PY, "scripts/update_readme.py"], "splice the tables into README.md")
    if want("figures"):
        run([PY, "scripts/make_figures.py", "--budget", args.budget],
            "generate figures")

    print(f"\nall stages complete in {(time.time() - t_start) / 60:.1f} min")
    print(f"  results -> {RESULTS_DIR}")
    print(f"  figures -> {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
