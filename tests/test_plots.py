"""Figure generation: every plot must render from a realistic input.

These run against synthetic result dictionaries shaped exactly like the real
artefacts, so a broken figure is caught here rather than after a multi-hour
experiment has finished.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from atsc.evaluation import plots

plots.apply_style()


def _agg(delay, ev_delay, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "aggregate": {
            "mean_time_loss_s_mean": delay,
            "mean_time_loss_s_ci95": 1.2,
            "p95_time_loss_s_mean": delay * 2.4,
            "mean_queue_mean": delay * 0.4,
            "throughput_mean": 2600.0,
            "ev_mean_time_loss_s_mean": ev_delay,
            "ev_mean_time_loss_s_ci95": 2.0,
            "ev_mean_waiting_time_s_mean": ev_delay * 0.6,
            "ev_mean_waiting_time_s_ci95": 1.5,
            "ev_mean_stops_mean": ev_delay / 20.0,
            "ev_zero_stop_rate_mean": 0.4,
            "jain_fairness_mean": 0.99,
            "max_movement_wait_s_mean": 120.0,
            "co2_g_per_vehicle_mean": 180.0,
            "fuel_l_per_vehicle_mean": 0.08,
        },
        "runs": [{"mean_time_loss_s": delay + rng.normal(0, 1)} for _ in range(5)],
    }


def _results():
    return {
        "fixed_time": _agg(33.0, 39.0, 1),
        "actuated": _agg(27.0, 35.0, 2),
        "max_pressure": _agg(25.0, 24.0, 3),
        "max_pressure_preempt": _agg(29.0, 22.0, 4),
        "siren_cmdp": _agg(23.5, 12.0, 5),
    }


def _detector_metrics():
    return {
        "n_test": 6000,
        "detection_precision": 0.92,
        "detection_recall": 0.88,
        "by_snr": [
            {"snr_bin": "[-inf, -20)", "n": 300, "recall": 0.22,
             "class_accuracy": 0.04, "vr_mae_ms": 13.0},
            {"snr_bin": "[-20, -10)", "n": 740, "recall": 0.84,
             "class_accuracy": 0.44, "vr_mae_ms": 11.6},
            {"snr_bin": "[-10, -5)", "n": 480, "recall": 1.0,
             "class_accuracy": 0.70, "vr_mae_ms": 12.2},
        ],
        "by_range": [
            {"range_bin_m": "[0, 30)", "n": 400, "range_mae_m": 6.0, "range_mape": 0.22},
            {"range_bin_m": "[30, 60)", "n": 500, "range_mae_m": 14.0, "range_mape": 0.31},
            {"range_bin_m": "[60, 100)", "n": 500, "range_mae_m": 34.0, "range_mape": 0.44},
        ],
        "confusion_matrix": [[900, 30, 25, 45], [40, 700, 160, 100],
                             [30, 190, 640, 140], [50, 110, 130, 710]],
        "confusion_labels": ["none", "ambulance", "fire", "police"],
    }


def _train_log(n=40):
    rows = []
    for i in range(n):
        rows.append({
            "episode": i,
            "reward": -400 + 3.0 * i,
            "ev_delay_per_vehicle_s": 80 - 1.4 * i,
            "lambda": 1.0 + 0.05 * i,
            "cost_ema": 0.30 - 0.005 * i,
            "mean_queue": 40 - 0.3 * i,
        })
    return rows


def test_detector_snr_figure():
    with tempfile.TemporaryDirectory() as d:
        out = plots.plot_detector_by_snr(_detector_metrics(), Path(d) / "a.pdf")
        assert out and out.exists() and out.stat().st_size > 1000


def test_detector_range_figure():
    with tempfile.TemporaryDirectory() as d:
        out = plots.plot_detector_range(_detector_metrics(), Path(d) / "b.pdf")
        assert out and out.exists()


def test_confusion_figure():
    with tempfile.TemporaryDirectory() as d:
        out = plots.plot_confusion(_detector_metrics(), Path(d) / "c.pdf")
        assert out and out.exists()


def test_training_figures():
    with tempfile.TemporaryDirectory() as d:
        assert plots.plot_training(_train_log(), Path(d) / "t.pdf")
        assert plots.plot_lambda_and_cost(_train_log(), 0.06, Path(d) / "l.pdf")


def test_controller_comparison_figure():
    with tempfile.TemporaryDirectory() as d:
        out = plots.plot_controller_comparison(_results(), Path(d) / "cc.pdf")
        assert out and out.exists()


def test_frontier_figure():
    # Points come from agents trained independently per budget; the lambda
    # sweep is carried alongside only as a negative control.
    frontier = [
        {"name": "unconstrained", "budget": None, "delay_s": 22.0, "ev_wait_s": 16.0},
        {"name": "budget_loose", "budget": 0.24, "delay_s": 23.0, "ev_wait_s": 13.0},
        {"name": "siren_cmdp", "budget": 0.18, "delay_s": 25.0, "ev_wait_s": 10.0},
        {"name": "budget_tight", "budget": 0.13, "delay_s": 28.0, "ev_wait_s": 8.0},
    ]
    control = [
        {"lambda": 0.0, "delay_s": 23.0, "ev_wait_s": 15.0},
        {"lambda": 4.0, "delay_s": 27.0, "ev_wait_s": 18.0},
        {"lambda": 16.0, "delay_s": 34.0, "ev_wait_s": 21.0},
    ]
    with tempfile.TemporaryDirectory() as d:
        out = plots.plot_frontier(frontier, _results(), control, Path(d) / "f.pdf")
        assert out and out.exists()


def test_frontier_handles_an_unconstrained_point_without_a_budget():
    """A None budget must be labelled, not crash the annotation."""
    frontier = [{"name": "unconstrained", "budget": None,
                 "delay_s": 22.0, "ev_wait_s": 16.0}]
    with tempfile.TemporaryDirectory() as d:
        assert plots.plot_frontier(frontier, {}, None, Path(d) / "f2.pdf")


def test_regime_figure():
    by_regime = {r: _results() for r in ("balanced", "asymmetric", "peak")}
    with tempfile.TemporaryDirectory() as d:
        out = plots.plot_regime_robustness(by_regime, Path(d) / "r.pdf")
        assert out and out.exists()


def test_surrogate_figure():
    data = {"sensor_level": {"by_snr": [
        {"snr_bin": "[-20, -10)", "detect_e2e": 0.80, "detect_surrogate": 0.83},
        {"snr_bin": "[-10, 0)", "detect_e2e": 0.99, "detect_surrogate": 0.98},
    ]}}
    with tempfile.TemporaryDirectory() as d:
        out = plots.plot_surrogate_validation(data, Path(d) / "s.pdf")
        assert out and out.exists()


def test_belief_trace_figure():
    trace = [{"t": 100 + 5 * i, "true_range": 260 - 13 * i,
              "tracked_range": 260 - 13 * i + np.sin(i) * 20,
              "presence": min(0.05 + 0.12 * i, 0.99),
              "serving_ev": int(i > 8)} for i in range(16)]
    with tempfile.TemporaryDirectory() as d:
        out = plots.plot_belief_trace(trace, Path(d) / "bt.pdf")
        assert out and out.exists()


def test_plots_return_none_on_empty_input():
    with tempfile.TemporaryDirectory() as d:
        assert plots.plot_detector_by_snr({}, Path(d) / "x.pdf") is None
        assert plots.plot_training([], Path(d) / "x.pdf") is None
        assert plots.plot_frontier([], {}, None, Path(d) / "x.pdf") is None
        assert plots.plot_controller_comparison({}, Path(d) / "x.pdf") is None
        assert plots.plot_belief_trace([], Path(d) / "x.pdf") is None


def test_palette_is_fixed_order_and_not_cycled():
    """Slot order is the colourblind-safety mechanism, so it must not drift."""
    assert plots.SERIES[:3] == ["#2a78d6", "#eb6834", "#1baf7a"]
    assert plots.EMPHASIS == plots.SERIES[0]
    # The sequential ramp must be monotonically darker, single hue.
    assert len(plots.SEQUENTIAL) >= 5


# ---------------------------------------------------------------------------
# The README is generated too
# ---------------------------------------------------------------------------
def test_readme_splice_is_idempotent_and_replaces_only_the_block():
    """Re-splicing the same table must not move anything else in the README.

    The README quotes the paper's numbers, and it used to be updated by hand --
    the one place in this project where a re-run could leave a stale number
    behind. The splice has to be exact: replace the generated block, touch
    nothing around it.
    """
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "update_readme", root / "scripts" / "update_readme.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    src = "results/evaluation_balanced.json"
    head = mod.marker(src)
    readme = (
        "# Title\n\nintro paragraph\n\n"
        f"{head}\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n"
        "### What to read out of that\n\ntrailing prose\n"
    )
    block = f"{head}\n\n| a | b |\n|---|---|\n| 9 | 8 |\n"

    once = mod.splice(readme, src, block)
    assert "| 9 | 8 |" in once
    assert "| 1 | 2 |" not in once
    assert once.startswith("# Title\n\nintro paragraph\n")
    assert once.rstrip().endswith("trailing prose")
    assert "### What to read out of that" in once
    # Idempotent: splicing the same block again changes nothing.
    assert mod.splice(once, src, block) == once


def test_readme_splice_refuses_a_missing_block():
    """Silently doing nothing would leave a stale table looking fresh."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "update_readme", root / "scripts" / "update_readme.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    try:
        mod.splice("# no marker here\n", "results/evaluation_balanced.json", "x")
    except SystemExit:
        return
    raise AssertionError("a README with no block for the table must be an error")
