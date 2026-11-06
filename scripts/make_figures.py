"""Regenerate every paper figure from the result files.

Figures are never hand-drawn or hand-edited: each one is a pure function of a
JSON/CSV artefact produced by the experiment scripts, so a figure cannot drift
away from the numbers it is supposed to show. Missing inputs are skipped with a
message naming the script that produces them, rather than faked or silently
omitted.

    python scripts/make_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atsc.config import FIGURES_DIR, RESULTS_DIR, ensure_dirs
from atsc.evaluation import plots

made: list = []
skipped: list = []


def record(name: str, path) -> None:
    (made if path else skipped).append(name)
    print(f"  wrote {path}" if path else f"  skipped {name} (no usable input)")


def _rows(data, key: str = "rows"):
    """Some result files are a bare list of rows and some wrap them in a dict.

    Both shapes are written by scripts in this repo and both are legitimate;
    the loader should not care which, and it certainly should not crash on the
    one it did not expect.
    """
    if not data:
        return None
    if isinstance(data, list):
        return data
    return data.get(key) or None


def missing(name: str, how: str) -> None:
    skipped.append(name)
    print(f"  skipped {name}: {how}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=str, default=str(RESULTS_DIR))
    p.add_argument("--figures", type=str, default=str(FIGURES_DIR))
    p.add_argument("--run-name", type=str, default="siren_arc")
    p.add_argument("--budget", type=float, default=0.35,
                   help="cost budget of the headline run, drawn on the lambda figure")
    p.add_argument("--highlight", type=str, default="siren_arc",
                   help="controller emphasised in the comparison figures")
    args = p.parse_args()

    ensure_dirs()
    results = Path(args.results)
    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)
    plots.apply_style()

    # ---------------------------------------------------------------- detector
    print("detector figures:")
    det = plots.load_json(results / "detector_metrics.json")
    if det:
        record("detector_snr", plots.plot_detector_by_snr(
            det, figures / "fig_detector_snr.pdf"))
        record("detector_range", plots.plot_detector_range(
            det, figures / "fig_detector_range.pdf"))
        record("confusion", plots.plot_confusion(det, figures / "fig_confusion.pdf"))
    else:
        missing("detector figures", "run scripts/train_detector.py")

    # ------------------------------------------------------- mast geometry
    # No result file: this one is the acoustic model evaluated along the arm,
    # so it is a pure function of the code and cannot drift from it.
    print("mast geometry:")
    record("mast_response", plots.plot_mast_response(
        figures / "fig_mast_response.pdf"))

    # ----------------------------------------------------------------- sensing
    print("sensing figures:")

    ablation = plots.load_json(results / "ablation_ev.json")
    if ablation:
        record("ev_ablation", plots.plot_ev_ablation(
            ablation, figures / "fig_ev_ablation.pdf"))
    else:
        missing("emergency-service ablation", "run scripts/ablation_ev.py")

    # Horn discrimination: read straight from the fitted error model, so the
    # figure cannot disagree with the model the surrogate is sampling from.
    try:
        from atsc.config import MODELS_DIR
        from atsc.sensing.sensor import SensorErrorModel
        em_path = MODELS_DIR / "sensor_error_model.json"
        if em_path.exists():
            blind_path = MODELS_DIR / "sensor_error_model.preHorn.json"
            record("horn_discrimination", plots.plot_horn_discrimination(
                SensorErrorModel.load(em_path),
                figures / "fig_horn_discrimination.pdf",
                blind=(SensorErrorModel.load(blind_path)
                       if blind_path.exists() else None)))
        else:
            missing("horn discrimination", "run scripts/train_detector.py")
    except Exception as exc:                                   # pragma: no cover
        missing("horn discrimination", f"{exc}")

    reqq = plots.load_json(results / "request_quality.json")
    if reqq:
        record("request_quality", plots.plot_request_quality(
            reqq, figures / "fig_request_quality.pdf"))
    else:
        missing("request quality", "run scripts/request_quality.py")
    bench = plots.load_json(results / "sensing_benchmark.json")
    if bench:
        record("sensing_ablation", plots.plot_sensing_ablation(
            bench, figures / "fig_sensing_ablation.pdf"))
        record("false_positives", plots.plot_false_positive_decomposition(
            bench, figures / "fig_false_positives.pdf"))
    else:
        missing("sensing ablation", "run scripts/sensing_benchmark.py")

    diag = plots.load_json(results / "calibration_diagnostics.json")
    if diag:
        record("calibration", plots.plot_calibration(
            diag, figures / "fig_calibration.pdf"))
        record("reliability", plots.plot_reliability(
            diag, figures / "fig_reliability.pdf"))
    else:
        missing("calibration", "run scripts/fit_calibration.py")

    surrogate = plots.load_json(results / "surrogate_validation.json")
    if surrogate:
        record("surrogate", plots.plot_surrogate_validation(
            surrogate, figures / "fig_surrogate.pdf"))
    else:
        missing("surrogate validation", "run scripts/validate_surrogate.py")

    trace = plots.load_json(results / "belief_trace.json")
    if trace and trace.get("segments"):
        # Not segments[0]: the recorder saves every approach it followed, and
        # most of them are approaches where nothing was ever committed. A
        # figure captioned "one emergency vehicle, from first evidence to
        # clearing the stop line" has to be drawn from a segment that contains
        # a commitment, or it shows the mechanism not firing.
        def _informative(segments):
            return max(segments, key=lambda seg: (
                sum(r.get("theta") is not None for r in seg),
                sum(bool(r.get("engaged")) for r in seg),
                len(seg)))
        seg = _informative(trace["segments"])
        record("belief_trace", plots.plot_belief_trace(
            seg, figures / "fig_belief_trace.pdf"))
        record("corridor_event", plots.plot_corridor_event(
            seg, figures / "fig_corridor_event.pdf"))
    else:
        missing("belief trace", "run scripts/record_trace.py")

    # ---------------------------------------------------------------- training
    print("training figures:")
    log = plots.load_csv(results / "training" / args.run_name / "train_log.csv")
    if log:
        record("training", plots.plot_training(log, figures / "fig_training.pdf"))
        record("lambda", plots.plot_lambda_and_cost(
            log, args.budget, figures / "fig_lambda.pdf"))
    else:
        missing("training figures", f"run scripts/train_agent.py --run-name {args.run_name}")

    # -------------------------------------------------------------- comparison
    print("comparison figures:")
    by_regime = {}
    for regime in ("balanced", "asymmetric", "peak", "surge"):
        data = plots.load_json(results / f"evaluation_{regime}.json")
        if data:
            by_regime[regime] = data["results"]

    balanced = by_regime.get("balanced")
    if balanced:
        record("controllers", plots.plot_controller_comparison(
            balanced, figures / "fig_controllers.pdf", highlight=args.highlight))
        record("ev_delay_distribution", plots.plot_ev_delay_distribution(
            balanced, figures / "fig_ev_delay_cdf.pdf",
            show=[args.highlight, "max_pressure_preempt", "actuated", "fixed_time"]))
        record("ev_class_delay", plots.plot_ev_class_delay(
            balanced, figures / "fig_ev_class.pdf",
            show=[args.highlight, "max_pressure_preempt", "fixed_time"]))
        record("corridor_timing", plots.plot_corridor_timing(
            balanced, figures / "fig_corridor_timing.pdf",
            show=[args.highlight, "arc_myopic"]))
        record("ablation_ladder", plots.plot_ablation_ladder(
            balanced, figures / "fig_ablation.pdf", full=args.highlight))
    else:
        missing("controller comparison", "run scripts/evaluate.py")

    if len(by_regime) > 1:
        record("regimes", plots.plot_regime_robustness(
            by_regime, figures / "fig_regimes.pdf", highlight=args.highlight))
    else:
        missing("regime robustness", "run scripts/evaluate.py for another regime")

    robust = plots.load_json(results / "robustness.json")
    if robust:
        record("robustness", plots.plot_robustness(
            robust, figures / "fig_robustness.pdf"))
    else:
        missing("robustness", "run scripts/robustness.py")

    traj = plots.load_json(results / "trajectories.json")
    if traj:
        record("ev_trajectory", plots.plot_ev_trajectory(
            traj, figures / "fig_trajectory.pdf",
            order=[args.highlight, "base_only", "max_pressure_preempt",
                   "fixed_time"]))
    else:
        missing("trajectory", "run scripts/record_trajectory.py")

    frontier = plots.load_json(results / "frontier.json")
    if frontier and balanced:
        baselines = {k: v for k, v in balanced.items()
                     if k in ("fixed_time", "actuated", "max_pressure",
                              "max_pressure_preempt", "greedy_queue")}
        record("frontier", plots.plot_frontier(
            frontier["frontier"], baselines, frontier.get("lambda_control"),
            figures / "fig_frontier.pdf"))
    else:
        missing("frontier", "run scripts/frontier.py")

    sweep_raw = plots.load_json(results / "value_sweep.json")
    sweep = _rows(sweep_raw)
    if sweep:
        refs = sweep_raw.get("references") if isinstance(sweep_raw, dict) else None
        record("value_frontier", plots.plot_value_frontier(
            sweep, figures / "fig_value_frontier.pdf", references=refs))
    else:
        missing("value sweep", "run scripts/value_sweep.py")

    audit = _rows(plots.load_json(results / "threshold_audit.json"))
    if audit:
        record("threshold_audit", plots.plot_threshold_audit(
            audit, figures / "fig_threshold_audit.pdf"))
        record("conflicts", plots.plot_conflict_summary(
            audit, figures / "fig_conflict.pdf"))
    else:
        missing("threshold audit", "run scripts/threshold_audit.py")

    lit = _rows(plots.load_json(results / "literature.json"), key="entries")
    if lit:
        ours = None
        if balanced and args.highlight in balanced and "fixed_time" in balanced:
            a = balanced[args.highlight]["aggregate"]
            b = balanced["fixed_time"]["aggregate"]
            key = "ev_mean_waiting_time_s_mean"
            if a.get(key) and b.get(key):
                ours = {"label": "SIREN-ARC (this work, acoustic only)",
                        "value": 100.0 * (b[key] - a[key]) / b[key]}
        record("literature", plots.plot_literature_context(
            plots.literature_rows(lit, ours), figures / "fig_literature.pdf"))
    else:
        missing("literature context", "results/literature.json is missing")

    print(f"\n{len(made)} figure(s) written to {figures}")
    if skipped:
        print(f"{len(skipped)} skipped: {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
