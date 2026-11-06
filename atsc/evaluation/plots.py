"""Publication figures, generated from the result files rather than by hand.

Design rules applied throughout (they are not stylistic preferences - each one
prevents a specific way of misreading a chart):

* **No dual-axis plots.** Two quantities on two y-scales invent a correlation
  that is not in the data. Where two measures must appear together they are
  drawn as separate stacked panels sharing an x-axis.
* **Colour carries identity, never magnitude.** In a comparison across
  controllers there is one series, so every bar takes the same hue and the
  method under test is picked out by emphasis, not by its own colour.
* **Categorical hues are assigned in a fixed, validated order** - blue, orange,
  aqua - and never cycled. That three-colour set clears colourblind separation
  on all pairs, which matters because these figures will be printed and
  photocopied.
* **Recessive chrome.** Solid hairline gridlines on the value axis only; no
  dashed grids (dashes read as "threshold"); no chart junk.
* Values are also tabulated in the paper, which satisfies the relief rule for
  the two hues that sit below 3:1 contrast on white.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# --- validated categorical slots (fixed order, never cycled) ---
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
EMPHASIS = "#2a78d6"
RECESSIVE = "#b9b8b1"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
ALERT = "#b4442e"
"""Failure, kept clear of EMPHASIS so one hue does not mean two things:
blue marks the method under test, this marks a run that gridlocked."""
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Sequential ramp (single hue, light -> dark) for magnitude only.
SEQUENTIAL = ["#e8f1fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
    })


def _finish(ax, ylabel: str = "", xlabel: str = "", title: str = "", grid_axis: str = "y"):
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, color=INK, loc="left", pad=8)
    ax.grid(axis=grid_axis)
    ax.grid(axis="x" if grid_axis == "y" else "y", visible=False)
    return ax


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Detector figures
# ---------------------------------------------------------------------------
def plot_detector_by_snr(metrics: Dict, out: Path) -> Optional[Path]:
    rows = metrics.get("by_snr") or []
    if not rows:
        return None
    labels = [r["snr_bin"] for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    ax.plot(x, [r["recall"] for r in rows], marker="o", color=SERIES[0], label="detection recall")
    ax.plot(x, [r["class_accuracy"] for r in rows], marker="s", color=SERIES[1],
            label="vehicle-class accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    _finish(ax, "rate", "signal-to-noise ratio (dB)",
            "Detector performance degrades gracefully with SNR")
    ax.legend(loc="lower right")
    return _save(fig, out)


def plot_detector_range(metrics: Dict, out: Path) -> Optional[Path]:
    """Range error by true range, in metres and as a fraction.

    Both panels are shown because they say different things and only one of
    them is operationally meaningful. Absolute error is what the tracker and
    the controller consume - an error of 40 m is about three seconds of travel
    - and it grows with range as the upper harmonics sink below the noise
    floor. Relative error is worst very close in, where the model has the least
    training mass and where presence is in any case already near-certain, so
    the estimate barely matters. Reporting only the fraction would imply the
    detector is at its worst exactly where it is in fact most useful.
    """
    rows = metrics.get("by_range") or []
    if not rows:
        return None
    labels = [r["range_bin_m"] for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(5.4, 4.6), sharex=True)

    bars = axes[0].bar(x, [r["range_mae_m"] for r in rows], width=0.62,
                       color=SERIES[0], edgecolor="none")
    for rect, r in zip(bars, rows):
        axes[0].annotate(f"{r['range_mae_m']:.0f}",
                         (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                         ha="center", va="bottom", fontsize=7.5, color=INK_SECONDARY)
    _finish(axes[0], "absolute error (m)", "",
            "Absolute range error grows with range; relative error does not")

    bars = axes[1].bar(x, [r["range_mape"] for r in rows], width=0.62,
                       color=SERIES[1], edgecolor="none")
    for rect, r in zip(bars, rows):
        axes[1].annotate(f"{r['range_mape']:.2f}",
                         (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                         ha="center", va="bottom", fontsize=7.5, color=INK_SECONDARY)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    _finish(axes[1], "error / true range", "true range (m)")
    fig.align_ylabels(axes)
    return _save(fig, out)


def plot_confusion(metrics: Dict, out: Path) -> Optional[Path]:
    cm = metrics.get("confusion_matrix")
    labels = metrics.get("confusion_labels")
    if not cm or not labels:
        return None
    m = np.asarray(cm, dtype=float)
    norm = m / np.maximum(m.sum(axis=1, keepdims=True), 1)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if norm[i, j] > 0.55 else INK)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Class confusion (row-normalised)", color=INK, loc="left", pad=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).outline.set_visible(False)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def plot_training(log_rows: List[Dict], out: Path) -> Optional[Path]:
    if not log_rows:
        return None

    def col(key):
        vals = []
        for r in log_rows:
            try:
                vals.append(float(r[key]))
            except (KeyError, TypeError, ValueError):
                vals.append(np.nan)
        return np.asarray(vals)

    ep = col("episode")
    # Stacked panels sharing x rather than a dual axis: reward and the dual
    # variable have unrelated units and putting them on one plot with two
    # scales would imply a relationship the data does not contain.
    fig, axes = plt.subplots(3, 1, figsize=(5.4, 6.0), sharex=True)

    axes[0].plot(ep, col("reward"), color=SERIES[0])
    _finish(axes[0], "episode return", "", "Learning progress")

    axes[1].plot(ep, col("ev_delay_per_vehicle_s"), color=SERIES[1])
    _finish(axes[1], "EV delay (s/veh)")

    axes[2].plot(ep, col("lambda"), color=SERIES[2])
    _finish(axes[2], "dual variable $\\lambda$", "episode")

    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.align_ylabels(axes)
    return _save(fig, out)


def plot_lambda_and_cost(log_rows: List[Dict], budget: float, out: Path) -> Optional[Path]:
    """The dual variable reacting to constraint violation - the CMDP mechanism."""
    if not log_rows:
        return None

    def col(key):
        vals = []
        for r in log_rows:
            try:
                vals.append(float(r[key]))
            except (KeyError, TypeError, ValueError):
                vals.append(np.nan)
        return np.asarray(vals)

    ep, lam, cost = col("episode"), col("lambda"), col("cost_ema")
    fig, axes = plt.subplots(2, 1, figsize=(5.4, 4.2), sharex=True)

    axes[0].plot(ep, cost, color=SERIES[1], label="running cost $\\hat{J}_c$")
    axes[0].axhline(budget, color=RECESSIVE, linewidth=1.2)
    # Label the budget line at the left edge and below it. At the right edge and
    # above it the running-cost trace passes straight through the text, and the
    # number a reader most needs from this panel is the one that gets lost.
    axes[0].annotate(f"budget $d_0$ = {budget:g}", (ep[0] if len(ep) else 0, budget),
                     xytext=(4, -4), textcoords="offset points",
                     ha="left", va="top", fontsize=7.5, color=INK_SECONDARY,
                     bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                               edgecolor="none", alpha=0.85))
    _finish(axes[0], "cost per decision", "",
            "The dual variable rises exactly when the constraint is violated")
    axes[0].legend(loc="upper right")

    axes[1].plot(ep, lam, color=SERIES[2])
    _finish(axes[1], "$\\lambda$", "episode")
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.align_ylabels(axes)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# Controller comparison
# ---------------------------------------------------------------------------
PRETTY = {
    "fixed_time": "Fixed-time (Webster)",
    "actuated": "Actuated (SUMO)",
    "greedy_queue": "Greedy queue",
    "max_pressure": "Max-pressure",
    "max_pressure_preempt": "Max-pressure + preemption",
    "max_pressure_preempt_oracle": "Ceiling: preemption + perfect detection",
    "fixed_time_preempt": "Fixed-time + preemption",
    # An ablation, not a configuration of the method -- see the note in
    # scripts/make_tables.py. It must not carry the method's name.
    "siren_cmdp": r"ablation: $\lambda$ coupled into $\gamma$",
    "cmdp_decoupled": "SIREN-CMDP (ours, constrained base)",
    "siren_priority": "Learned + priority overlay",
    "siren_priority_oracle": "Ceiling: our overlay + perfect detection",
    "learned_base": "Learned policy alone",
    "unconstrained": "Unconstrained RL",
    "no_sensing": "RL, no EV sensing",
    "oracle": "SIREN-CMDP (oracle sensing)",
    "siren_arc": "SIREN-ARC (ours)",
    "arc_indep": "ours, independent filters",
    "arc_myopic": "ours, act-on-sight",
    "arc_nocal": "ours, uncalibrated posterior",
    "arc_oracle": "Ceiling: ours + perfect detection",
    "arc_base_indep": "ours, independent filters (retrained)",
    "siren_cmdp_arc": "SIREN-CMDP (ours, constrained base)",
    "base_cmdp": "Constrained policy alone",
    "arc_on_indep": "ours, filter swapped at test time",
    "base_only": "Learned policy alone",
    "base_siren_arc": "Learned policy alone",
    "base_siren_cmdp": "Constrained policy alone",
    "base_no_sensing": "Policy with no EV sensing",
    # These three appear in every current evaluation and had no entry,
    # so the figures drew their raw keys next to properly named rows.
    "base_arc": "Learned policy alone",
    "base_indep": "Learned alone, independent filters",
    "arc_greedy": "ours, greedy conflict horizon",
}

FILTER_PRETTY = {
    "joint": "Joint exclusive filter (ours)",
    "independent": "Independent per-approach filters",
}


def plot_controller_comparison(
    results: Dict[str, Dict], out: Path, highlight: str = "siren_priority"
) -> Optional[Path]:
    """Background delay and emergency stopped delay, two panels, one hue each.

    Controllers are nominal categories, not a series, so they all take the same
    colour; the method under test is distinguished by emphasis. Colouring each
    bar differently would burn the colour channel on information the bar length
    already carries.

    Bars are **medians** over seeds with whiskers at the interquartile range.
    Means were shown here originally and they misled: on a heavy-tailed
    distribution one gridlocked seed in twelve moved SIREN-CMDP from 24.2 to
    57.9 s, so the chart implied a catastrophe where the typical run was
    ordinary. Medians alone would hide that tail, so any controller with a
    gridlocked seed is marked with its count in the emphasis colour -- the
    central tendency and the tail risk are then both legible from one figure.

    The emergency panel reports **stopped delay**, not time loss. SUMO's
    ``timeLoss`` is measured against each vehicle's own desired speed, and
    emergency vehicles are configured to want to travel faster than the traffic
    around them, so they accrue time loss merely by being stuck behind
    law-abiding cars -- a component no signal controller can remove. Stopped
    delay is the part that signal control actually governs.
    """
    names = [n for n in results if results[n].get("aggregate")]
    if not names:
        return None

    def get(name, key, suffix="_median"):
        return results[name]["aggregate"].get(f"{key}{suffix}", np.nan)

    def per_seed(name, key):
        """The raw per-seed values, for a dispersion measure that matches."""
        xs = [r.get(key) for r in results[name].get("runs", [])]
        xs = np.array([x for x in xs if x is not None], dtype=float)
        return xs[np.isfinite(xs)]

    def jammed(name):
        return sum(1 for r in results[name].get("runs", [])
                   if (r.get("teleports") or 0) > 0)

    order = sorted(names, key=lambda n: get(n, "mean_time_loss_s"))
    labels = [PRETTY.get(n, n) for n in order]
    y = np.arange(len(order))

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 0.45 * len(order) + 2.2), sharey=True)
    for ax, key, title, xlabel in (
        (axes[0], "mean_time_loss_s", "Background traffic delay",
         "median time loss (s)"),
        (axes[1], "ev_mean_waiting_time_s", "Emergency-vehicle stopped delay",
         "median stopped delay (s)"),
    ):
        vals = np.array([get(n, key) for n in order], dtype=float)
        # Whiskers are the interquartile range, not a confidence interval on a
        # mean: the bar is a median, and the two must describe the same thing.
        lo, hi = [], []
        for n, v in zip(order, vals):
            xs = per_seed(n, key)
            if xs.size and np.isfinite(v):
                lo.append(max(v - np.percentile(xs, 25), 0.0))
                hi.append(max(np.percentile(xs, 75) - v, 0.0))
            else:
                lo.append(0.0)
                hi.append(0.0)
        errs = np.vstack([lo, hi])
        colors = [EMPHASIS if n == highlight else RECESSIVE for n in order]
        ax.barh(y, vals, xerr=errs, height=0.6, color=colors, edgecolor="none",
                error_kw={"ecolor": MUTED, "elinewidth": 0.9, "capsize": 2})
        for yi, v, n in zip(y, vals, order):
            if not np.isfinite(v):
                continue
            # A median hides the tail, so the tail is annotated rather than
            # left for the reader to discover from a table.
            j = jammed(n)
            tag = f"{v:.1f}" + (f"  ({j} jam)" if j else "")
            ax.annotate(tag, (v + (errs[1][yi] if errs.shape[1] > yi else 0.0), yi),
                        xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=7.5,
                        color=ALERT if j else INK_SECONDARY)
        ax.set_title(title, color=INK, loc="left", pad=8)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.22)

    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    return _save(fig, out)


def plot_frontier(
    frontier: List[Dict],
    baselines: Dict[str, Dict],
    lambda_control: Optional[List[Dict]] = None,
    out: Path = Path("frontier.pdf"),
) -> Optional[Path]:
    """The achievable trade-off, plus the shortcut that fails to reproduce it.

    Lower-left is better. Two things are being shown at once:

    * the frontier itself, from agents trained independently at different cost
      budgets, each of which actually optimised for the point it is plotted at;
    * the tempting shortcut -- take one agent and vary ``lambda`` at evaluation
      -- which is off-policy and, in our runs, is not even monotone. It is drawn
      in a recessive colour and explicitly labelled, because it is a negative
      result rather than a competing method.
    """
    if not frontier:
        return None
    fig, ax = plt.subplots(figsize=(6.0, 4.6))

    pts = sorted((p for p in frontier if np.isfinite(p.get("delay_s", np.nan))),
                 key=lambda p: p["delay_s"])
    fx = [p["delay_s"] for p in pts]
    fy = [p["ev_wait_s"] for p in pts]

    baseline_pts = []
    for name, res in baselines.items():
        agg = res.get("aggregate", {})
        x = agg.get("mean_time_loss_s_mean")
        y = agg.get("ev_mean_waiting_time_s_mean")
        if x is None or y is None or not (np.isfinite(x) and np.isfinite(y)):
            continue
        baseline_pts.append((x, y, PRETTY.get(name, name)))

    # A controller that gridlocked sits two orders of magnitude out and would
    # compress every meaningful difference into a corner. Frame the axes on the
    # controllers that stayed stable and mark the rest at the edge, rather than
    # dropping them (which would hide a real failure) or letting them set the
    # scale (which would hide everything else).
    scale_pool = fx + [x for x, _, _ in baseline_pts]
    ref = float(np.median(scale_pool)) if scale_pool else 1.0
    x_cut = max(ref * 2.2, max(fx) * 1.15 if fx else ref)
    y_pool = fy + [y for _, y, _ in baseline_pts]
    y_ref = float(np.median(y_pool)) if y_pool else 1.0
    y_cut = max(y_ref * 2.2, max(fy) * 1.15 if fy else y_ref)

    ax.plot(fx, fy, marker="o", color=SERIES[0], zorder=4,
            label="SIREN-CMDP, trained per budget $d_0$")
    for i, p in enumerate(pts):
        tag = "unconstrained" if p.get("budget") is None else f"$d_0$={p['budget']:g}"
        ax.annotate(tag, (p["delay_s"], p["ev_wait_s"]),
                    xytext=(6, 7 if i % 2 == 0 else -12),
                    textcoords="offset points", fontsize=7, color=INK_SECONDARY)

    inside = [(x, y, l) for x, y, l in baseline_pts if x <= x_cut and y <= y_cut]
    outside = [(x, y, l) for x, y, l in baseline_pts if x > x_cut or y > y_cut]
    if inside:
        ax.scatter([x for x, _, _ in inside], [y for _, y, _ in inside],
                   marker="s", s=44, color=SERIES[1], zorder=5,
                   edgecolors="#fcfcfb", linewidths=1.4, label="baseline controllers")
        for i, (x, y, lab) in enumerate(sorted(inside)):
            ax.annotate(lab, (x, y), xytext=(7, -11 if i % 2 == 0 else 8),
                        textcoords="offset points", fontsize=7, color=INK_SECONDARY)

    if lambda_control:
        cx = [min(p["delay_s"], x_cut) for p in lambda_control]
        cy = [min(p["ev_wait_s"], y_cut) for p in lambda_control]
        ax.plot(cx, cy, marker="^", linestyle=":", color=RECESSIVE, zorder=3,
                label="negative control: sweeping $\\lambda$ post hoc")

    ax.set_xlim(right=x_cut)
    ax.set_ylim(top=y_cut)
    for x, y, lab in outside:
        ax.annotate(f"{lab} → ({x:.0f}, {y:.0f})",
                    (min(x, x_cut), min(y, y_cut)),
                    xytext=(-6, -6), textcoords="offset points", ha="right",
                    fontsize=7, color=MUTED)

    _finish(ax, "emergency-vehicle stopped delay (s)",
            "background traffic delay (s)", "Lower-left is better", grid_axis="both")
    ax.grid(axis="both")
    ax.legend(loc="upper left", fontsize=7.5)
    return _save(fig, out)


def plot_regime_robustness(by_regime: Dict[str, Dict], out: Path,
                           highlight: str = "siren_priority") -> Optional[Path]:
    """Does the advantage survive demand patterns the agent never trained on?"""
    regimes = [r for r in by_regime if by_regime[r]]
    if not regimes:
        return None
    controllers = sorted({c for r in regimes for c in by_regime[r]})
    controllers = [c for c in controllers if c in ("max_pressure", "actuated",
                                                   "max_pressure_preempt",
                                                   "learned_base", highlight)]
    if not controllers:
        return None

    x = np.arange(len(regimes))
    width = 0.8 / max(len(controllers), 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for ax, key, title in (
        (axes[0], "mean_time_loss_s", "Background delay"),
        (axes[1], "ev_mean_waiting_time_s", "Emergency-vehicle stopped delay"),
    ):
        for i, c in enumerate(controllers):
            vals = [by_regime[r].get(c, {}).get("aggregate", {}).get(f"{key}_mean", np.nan)
                    for r in regimes]
            color = SERIES[0] if c == highlight else SERIES[(i % 3) + 1]
            ax.bar(x + i * width - 0.4 + width / 2, vals, width=width * 0.88,
                   color=color, edgecolor="none",
                   label=PRETTY.get(c, c) if ax is axes[0] else None)
        ax.set_xticks(x, regimes)
        ax.set_title(title, color=INK, loc="left", pad=8)
        ax.set_ylabel("mean time loss (s)")
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
    # The legend goes under the panels rather than inside the left one. Placed
    # inside, the longest entry ("Max-pressure + preemption") ran past the axes
    # and was clipped, and the box sat on top of the bars it was labelling.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(labels), 4), fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.26)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# Sensing
# ---------------------------------------------------------------------------
def plot_surrogate_validation(data: Dict, out: Path) -> Optional[Path]:
    rows = data.get("sensor_level", {}).get("by_snr") or []
    if not rows:
        return None
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    ax.plot(x, [r["detect_e2e"] for r in rows], marker="o", color=SERIES[0],
            label="end-to-end CNN on synthesised audio")
    ax.plot(x, [r["detect_surrogate"] for r in rows], marker="s", color=SERIES[1],
            label="fitted surrogate sensor")
    ax.set_xticks(x, [r["snr_bin"] for r in rows], rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    _finish(ax, "detection rate", "signal-to-noise ratio (dB)",
            "The surrogate reproduces the detector it replaces")
    ax.legend(loc="lower right")
    return _save(fig, out)


def plot_belief_trace(trace: List[Dict], out: Path) -> Optional[Path]:
    """One emergency-vehicle approach, from first detection to clearance."""
    if not trace:
        return None
    t = np.array([r["t"] for r in trace], dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(5.6, 5.4), sharex=True)

    axes[0].plot(t, [r["true_range"] for r in trace], color=RECESSIVE, linewidth=2.4,
                 label="true range")
    axes[0].plot(t, [r["tracked_range"] for r in trace], color=SERIES[0],
                 label="tracked range")
    _finish(axes[0], "range (m)", "", "Acoustic belief during one emergency approach")
    axes[0].legend(loc="upper right")

    axes[1].plot(t, [r["presence"] for r in trace], color=SERIES[1])
    axes[1].axhline(0.6, color=RECESSIVE, linewidth=1.0)
    axes[1].annotate("commit threshold", (t[-1], 0.6), xytext=(-4, 4),
                     textcoords="offset points", ha="right", va="bottom",
                     fontsize=7.5, color=INK_SECONDARY)
    axes[1].set_ylim(0, 1.05)
    _finish(axes[1], "P(EV present)")

    axes[2].step(t, [r["serving_ev"] for r in trace], where="post", color=SERIES[2])
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].set_yticks([0, 1], ["other", "serving\nEV approach"])
    _finish(axes[2], "", "simulation time (s)")
    fig.align_ylabels(axes)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# Calibration, corridor economics, and the operating-point frontier
# ---------------------------------------------------------------------------
def plot_calibration(diag: Dict, out: Path) -> Optional[Path]:
    """Reliability diagram: what the filter claims against what happens.

    The 45-degree line is where a calibrated probability sits. The raw filter
    curve falling far below it is the whole argument for this step -- a Bayes
    decision rule cannot be run on a score that says 0.97 when the event
    happens 16 % of the time.
    """
    raw = np.asarray(diag.get("raw_means", []), dtype=float)
    cal = np.asarray(diag.get("calibrated", []), dtype=float)
    counts = np.asarray(diag.get("counts", []), dtype=float)
    if raw.size == 0 or cal.size == 0:
        return None
    keep = counts > 0
    raw, cal, counts = raw[keep], cal[keep], counts[keep]

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.plot([0, 1], [0, 1], color=RECESSIVE, linewidth=1.0, zorder=1)
    ax.annotate("perfectly calibrated", xy=(0.62, 0.62), xytext=(0.30, 0.80),
                color=MUTED, fontsize=7.5,
                arrowprops=dict(arrowstyle="-", color=RECESSIVE, linewidth=0.8))

    # Marker area carries bin support; area, not radius, so it reads linearly.
    sizes = 18.0 + 90.0 * counts / max(counts.max(), 1.0)
    # Only the measured curve is drawn. Plotting the fitted map as well would
    # be tautological -- an isotonic fit lies on the diagonal by construction --
    # so the correction is quantified by the held-out error in the caption
    # instead, which is the number that means something.
    ax.plot(raw, cal, color=SERIES[0], linewidth=1.8, zorder=3)
    ax.scatter(raw, cal, s=sizes, color=SERIES[0], zorder=4,
               edgecolor="white", linewidth=0.8)

    # Direct-label the reading that carries the argument, chosen as the highest
    # bin whose raw score is confident but whose outcome is not.
    conf = np.flatnonzero(raw > 0.9)
    if conf.size:
        i = int(conf[0])
        ax.annotate(f"filter says {raw[i]:.2f},\nhappens {cal[i]:.0%} of the time",
                    xy=(raw[i], cal[i]), xytext=(-14, 34),
                    textcoords="offset points", fontsize=8, color=INK_SECONDARY,
                    ha="right",
                    arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.8))

    ece_b = diag.get("ece_before")
    ece_a = diag.get("ece_after")
    if ece_b is not None and ece_a is not None:
        tag = "held out" if diag.get("scored_out_of_sample") else "in sample"
        ax.text(0.03, 0.955,
                f"expected calibration error {ece_b:.3f} $\\rightarrow$ {ece_a:.3f}"
                f"\nafter isotonic correction ({tag})",
                transform=ax.transAxes, fontsize=8, color=INK_SECONDARY, va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _finish(ax, ylabel="observed frequency of an inbound EV",
            xlabel="filter presence output",
            title="The presence posterior ranks well and is not a probability",
            grid_axis="both")
    return _save(fig, out)


def plot_value_frontier(rows: List[Dict], out: Path,
                        references: Optional[Dict[str, Dict]] = None) -> Optional[Path]:
    """Operating points traced by the emergency/background exchange rate.

    One connected scatter, because the two axes are two costs and the curve is
    the set of choices available -- not a time series and not a ranking. Gamma
    is direct-labelled at the ends and at the shipped operating point rather
    than at every marker.
    """
    rows = [r for r in rows if r.get("delay_median") is not None
            and r.get("ev_delay_median") is not None]
    if len(rows) < 2:
        return None
    rows = sorted(rows, key=lambda r: float(r["gamma"]))
    g = np.asarray([float(r["gamma"]) for r in rows])
    # Medians, not means. At the seed count this sweep runs, a single seed that
    # tips into spillback moves a mean by tens of seconds and would draw a
    # frontier whose shape is that one seed. The medians are well behaved and
    # the jam counts are reported separately, which is the honest split.
    x = np.asarray([float(r["delay_median"]) for r in rows])
    y = np.asarray([float(r["ev_delay_median"]) for r in rows])
    xe = np.zeros_like(x)
    ye = np.zeros_like(y)

    fig, ax = plt.subplots(figsize=(5.8, 3.8))

    # The achievable frontier: points nothing else beats on both axes. Drawn
    # instead of a line through the points in gamma order, which would assert a
    # monotonicity the seed count cannot support.
    order = np.argsort(x)
    front, best_y = [], np.inf
    for i in order:
        if y[i] < best_y - 1e-9:
            front.append(i)
            best_y = y[i]
    front = np.asarray(front, dtype=int)
    if front.size >= 2:
        ax.step(x[front], y[front], where="post", color=SERIES[0],
                linewidth=1.6, zorder=2, label="achievable frontier")

    dominated = np.setdiff1d(np.arange(len(rows)), front)
    ax.scatter(x[dominated], y[dominated], s=34, color=RECESSIVE, zorder=3,
               edgecolor="white", linewidth=0.8)
    ax.scatter(x[front], y[front], s=48, color=SERIES[0], zorder=4,
               edgecolor="white", linewidth=0.9)

    # Label the no-priority control point, the shipped operating point, and
    # every point on the frontier -- those are the ones an operator can pick.
    to_label = set(int(i) for i in front)
    for i, r in enumerate(rows):
        if float(r["gamma"]) <= 0 or abs(float(r["gamma"]) - 50.0) < 1e-9:
            to_label.add(i)
    for i in sorted(to_label):
        label = "no priority" if g[i] <= 0 else f"$\\gamma$={g[i]:g}"
        ax.annotate(label, (x[i], y[i]), textcoords="offset points",
                    xytext=(6, 5), fontsize=7.5, color=INK_SECONDARY)

    # Fixed alternatives, on the same seeds and the same axes. Without them the
    # curve shows only that the dial does something; with them it shows whether
    # any point on it dominates what an operator could deploy instead.
    for i, (name, r) in enumerate(sorted((references or {}).items())):
        rx, ry = r.get("delay_median"), r.get("ev_delay_median")
        if rx is None or ry is None:
            continue
        ax.scatter([rx], [ry], s=52, marker="D", zorder=4,
                   color=SERIES[(i + 1) % len(SERIES)],
                   edgecolor="white", linewidth=0.9)
        ax.annotate(PRETTY.get(name, name), (rx, ry), textcoords="offset points",
                    xytext=(-6, -12), ha="right", fontsize=7.5,
                    color=INK_SECONDARY)

    _finish(ax, ylabel="emergency stopped delay (s)",
            xlabel="background delay (s/veh)",
            title="Operating points from one policy (medians)",
            grid_axis="both")
    ax.margins(0.18)
    return _save(fig, out)


def plot_corridor_event(trace: List[Dict], out: Path) -> Optional[Path]:
    """One emergency vehicle, from first evidence to clearing the stop line.

    Two stacked panels on a shared time axis rather than one panel with two
    scales. The upper panel is the decision: the calibrated posterior against
    the threshold the risk rule derives from the current queue state, which is
    the mechanism this paper is about -- the threshold moves, and the corridor
    opens where the curves cross.
    """
    if not trace:
        return None
    def col(key, default=np.nan):
        # Epochs before the scheduler had a request for this approach record
        # nulls rather than zeros, and a zero would read as "certainly absent".
        out = []
        for i, r in enumerate(trace):
            v = r.get(key, None)
            out.append(float(v) if v is not None else (float(i) if key == "t" else default))
        return np.asarray(out, dtype=float)

    t = col("t")
    p = col("p")
    th = col("theta")
    rng = col("range_m")
    engaged = np.asarray([bool(r.get("engaged", False)) for r in trace])
    if not np.isfinite(p).any():
        return None
    th = np.where(np.isfinite(th), np.clip(th, 0.0, 1.0), np.nan)

    fig, axes = plt.subplots(2, 1, figsize=(5.2, 4.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.25, 1.0]})
    ax0, ax1 = axes

    # Shade the epochs where a corridor was actually open.
    for i in np.flatnonzero(engaged):
        lo = t[i] - (t[1] - t[0]) / 2 if len(t) > 1 else t[i] - 2.5
        hi = t[i] + (t[1] - t[0]) / 2 if len(t) > 1 else t[i] + 2.5
        for ax in axes:
            ax.axvspan(lo, hi, color=SERIES[2], alpha=0.16, linewidth=0, zorder=0)

    ax0.plot(t, p, color=SERIES[0], linewidth=1.8, label="calibrated presence")
    ax0.plot(t, th, color=SERIES[1], linewidth=1.8, label="derived threshold $\\theta^*$")
    ax0.set_ylim(0, 1)
    ax0.legend(loc="upper left")
    _finish(ax0, ylabel="probability", title="A corridor opens when the sums turn",
            grid_axis="y")

    ax1.plot(t, rng, color=SERIES[3], linewidth=1.8, label="tracked range")
    ax1.axhline(0.0, color=RECESSIVE, linewidth=1.0)
    _finish(ax1, ylabel="range to stop line (m)", xlabel="simulation time (s)",
            grid_axis="y")
    ax1.legend(loc="upper right")
    fig.align_ylabels(axes)
    return _save(fig, out)


def plot_threshold_audit(rows: List[Dict], out: Path) -> Optional[Path]:
    """The derived preemption threshold against the traffic that sets it.

    This is the audit of the paper's central claim. If the commitment threshold
    were a tuned constant -- as it is in every preemption scheme we could find,
    including our own earlier version -- this plot would be a horizontal line.
    It is not: the threshold is computed from the delay a corridor would cost
    and the delay it would save, so it climbs as the conflicting queue grows and
    the same acoustic evidence stops being worth acting on.

    Colour separates the two outcomes rather than encoding a magnitude, and the
    ceiling the calibrated posterior can reach is drawn because it is what makes
    the upper region unreachable: above it, no evidence this sensor can produce
    is sufficient, which is the correct behaviour and not a failure to fire.
    """
    rows = [r for r in rows if r.get("theta") is not None]
    if len(rows) < 10:
        return None
    q = np.asarray([float(r["queue_cross"]) for r in rows])
    th = np.asarray([float(r["theta"]) for r in rows])
    committed = np.asarray([bool(r.get("committed")) for r in rows])
    p_max = max((float(r["p"]) for r in rows), default=0.0)

    fig, ax = plt.subplots(figsize=(5.0, 3.6))

    if p_max > 0:
        ax.axhspan(p_max, 1.0, color=RECESSIVE, alpha=0.16, linewidth=0, zorder=0)
        ax.axhline(p_max, color=AXIS, linewidth=0.9, zorder=1)
        ax.text(0.985, min(p_max + 0.03, 0.96),
                "above the sensor's best attainable probability:\nno evidence suffices",
                transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=7.5, color=MUTED)

    ax.scatter(q[~committed], th[~committed], s=13, color=RECESSIVE, zorder=2,
               linewidth=0, label="declined")
    ax.scatter(q[committed], th[committed], s=15, color=SERIES[0], zorder=3,
               linewidth=0, label="corridor opened")

    # A median trend, binned on the x variable, so the shape is legible through
    # the scatter without implying a fitted model.
    edges = np.linspace(q.min(), np.percentile(q, 98), 9)
    cx, cy = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (q >= lo) & (q < hi)
        if m.sum() >= 8:
            cx.append(0.5 * (lo + hi))
            cy.append(float(np.median(th[m])))
    if len(cx) >= 3:
        ax.plot(cx, cy, color=SERIES[1], linewidth=2.0, zorder=4,
                label="median threshold")

    ax.set_ylim(0, 1)
    ax.legend(loc="center right", ncol=1)
    _finish(ax, ylabel="derived threshold $\\theta^*$",
            xlabel="conflicting queue at the moment of decision (veh)",
            title="The commitment threshold is computed, not chosen",
            grid_axis="both")
    return _save(fig, out)


def plot_conflict_summary(rows: List[Dict], out: Path) -> Optional[Path]:
    """How often several emergency vehicles compete for the same intersection.

    Reviews of preemption name simultaneous conflicting requests as an open
    problem, and deployed systems serve one vehicle and drop the rest. This is
    how common the case actually is at the emergency rate used here -- frequent
    enough that a scheme with no answer for it is answering the wrong question.

    What the scheduler *does* about it is not shown here, because a per-epoch
    count cannot show it: in any one epoch only one phase can be green, so a
    "served this epoch" bar would read as a failure whatever the scheduler did.
    The outcome is measured instead by ablation against a greedy scheduler that
    serves the strongest request and re-decides, in the main results table.
    """
    if not rows:
        return None
    total = len(rows)
    conflict = sum(bool(r.get("conflict")) for r in rows)
    if not conflict:
        return None
    conf_rows = [r for r in rows if r.get("conflict")]
    committed_conf = sum(bool(r.get("committed")) for r in conf_rows)

    labels = ["request-epochs with\na rival request",
              "of those, rivals that\ncleared their own threshold"]
    vals = [100.0 * conflict / total,
            100.0 * committed_conf / max(conflict, 1)]
    counts = [f"{conflict}/{total}", f"{committed_conf}/{conflict}"]

    fig, ax = plt.subplots(figsize=(5.2, 2.3))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, height=0.45, color=EMPHASIS, zorder=2)
    for yy, v, c in zip(y, vals, counts):
        ax.annotate(f"{v:.0f}%  ({c})", (v, yy), textcoords="offset points",
                    xytext=(6, -3), fontsize=8, color=INK_SECONDARY)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 100)
    _finish(ax, xlabel="percent", title="Concurrent emergency requests are common",
            grid_axis="x")
    return _save(fig, out)


def literature_rows(entries: List[Dict], ours: Optional[Dict] = None) -> List[Dict]:
    """Turn ``results/literature.json`` into rows the dot plot can draw.

    Each published entry contributes the authors' own headline reduction, on
    their own network and simulator. Ours is computed here from the evaluation
    run against the same reference controller the field usually reports against
    (fixed-time), so at least our own number is not a different quantity from
    the one it sits beside. The figure's caption -- and its title -- carry the
    warning that the column is context, not a ranking.
    """
    rows = []
    for e in entries:
        v = e.get("reported_ev_improvement_pct")
        if v is None:
            continue
        rows.append({"label": e.get("label", e.get("key", "?")),
                     "value": float(v), "ours": False})
    if ours:
        rows.append({"label": ours["label"], "value": float(ours["value"]),
                     "ours": True})
    return rows


def plot_literature_context(rows: List[Dict], out: Path) -> Optional[Path]:
    """Where reported improvements sit, ours among them.

    A dot plot, not bars: these numbers come from different networks, demands
    and simulators, so the visual must not invite a magnitude comparison of
    filled areas. The measured-here entries are emphasised and everything else
    is drawn recessively, because that is exactly the epistemic difference.
    """
    rows = [r for r in rows if r.get("value") is not None]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: float(r["value"]))
    labels = [r["label"] for r in rows]
    vals = np.asarray([float(r["value"]) for r in rows])
    ours = np.asarray([bool(r.get("ours", False)) for r in rows])
    ypos = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(5.6, 0.34 * len(rows) + 1.4))
    ax.hlines(ypos, 0, vals, color=RECESSIVE, linewidth=1.0, zorder=1)
    ax.scatter(vals[~ours], ypos[~ours], s=46, color=RECESSIVE, zorder=2,
               edgecolor="white", linewidth=0.8, label="reported elsewhere")
    ax.scatter(vals[ours], ypos[ours], s=58, color=EMPHASIS, zorder=3,
               edgecolor="white", linewidth=0.8, label="measured here")
    for y, v in zip(ypos, vals):
        ax.annotate(f"{v:.0f}%", (v, y), textcoords="offset points",
                    xytext=(7, -3), fontsize=8, color=INK_SECONDARY)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, max(vals.max() * 1.22, 5))
    ax.legend(loc="lower right")
    _finish(ax, xlabel="reported reduction in emergency-vehicle delay or travel time (%)",
            title="Reported gains are not measured on a common benchmark",
            grid_axis="x")
    return _save(fig, out)


def load_json(path: Path) -> Optional[Dict]:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> List[Dict]:
    import csv
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Sensing: what modelling cross-talk jointly is worth
# ---------------------------------------------------------------------------
def plot_sensing_ablation(bench: Dict, out: Path) -> Optional[Path]:
    """Joint versus independent presence filtering, on the metrics that matter.

    Four panels rather than one composite score, because the two filters differ
    in *opposite* directions on two of them and a single number would hide that:
    the independent filter has the higher recall (it fires on everything) and
    much the worse precision and calibration (it fires on everything). The
    controller consumes a probability, so precision and calibration are the
    quantities its decisions are made of; recall is reported so the trade is
    visible rather than buried.
    """
    modes = bench.get("modes") or {}
    order = [m for m in ("independent", "joint") if m in modes]
    if len(order) < 2:
        return None
    apply_style()

    panels = [
        ("precision", "Precision at p >= 0.5", lambda d: d["prf@0.5"]["precision"], None),
        ("recall", "Recall at p >= 0.5", lambda d: d["prf@0.5"]["recall"], None),
        ("crosstalk", "Approaches above 0.5 when\nexactly one is occupied",
         lambda d: d.get("arms_above@0.5", np.nan), 1.0),
        ("ece", "Expected calibration error", lambda d: d.get("ece", np.nan), None),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(9.4, 2.9))
    x = np.arange(len(order))
    for ax, (_key, title, fn, reference) in zip(axes, panels):
        vals = [float(fn(modes[m])) for m in order]
        colors = [RECESSIVE if m == "independent" else EMPHASIS for m in order]
        ax.bar(x, vals, width=0.58, color=colors, edgecolor="none")
        for xi, v in zip(x, vals):
            ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, 3),
                        textcoords="offset points", ha="center",
                        fontsize=7.5, color=INK_SECONDARY)
        if reference is not None:
            ax.axhline(reference, color=INK_SECONDARY, lw=0.9, ls=(0, (4, 3)))
            ax.annotate("ideal", (len(order) - 0.5, reference), xytext=(0, 4),
                        textcoords="offset points", ha="right", fontsize=7,
                        color=INK_SECONDARY)
        ax.set_xticks(x, ["independent", "joint"], fontsize=8)
        ax.set_title(title, color=INK, loc="left", pad=8, fontsize=9)
        ax.margins(y=0.24)
        _finish(ax)
    return _save(fig, out)


def plot_false_positive_decomposition(bench: Dict, out: Path) -> Optional[Path]:
    """Where the remaining false detections come from, per filter.

    A false-alarm *rate* says nothing about what to do next. Split by cause it
    does: cross-talk (a vehicle inbound on a different arm) is what the joint
    filter is for, a stale tail is a state-space problem, and a hallucination is
    the detector's own noise floor, which no filter over its outputs can remove.
    """
    modes = bench.get("modes") or {}
    order = [m for m in ("independent", "joint") if m in modes]
    if not order:
        return None
    episodes = max(float(bench.get("config", {}).get("episodes", 1) or 1), 1.0)
    kinds = ["mislocalised", "stale_tail", "hallucination"]
    labels = {"mislocalised": "cross-talk (vehicle on another arm)",
              "stale_tail": "stale tail (vehicle already crossed)",
              "hallucination": "detector noise (nothing there)"}
    apply_style()

    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    y = np.arange(len(order))
    left = np.zeros(len(order))
    for i, kind in enumerate(kinds):
        vals = np.array([
            modes[m].get("false_positive_kinds", {}).get(kind, 0.0)
            * modes[m].get("false_positive_total", 0.0) / episodes
            for m in order], dtype=float)
        ax.barh(y, vals, left=left, height=0.55, color=SEQUENTIAL[6 - 2 * i],
                edgecolor="none", label=labels[kind])
        left += vals
    for yi, tot in zip(y, left):
        ax.annotate(f"{tot:.0f}/h", (tot, yi), xytext=(5, 0),
                    textcoords="offset points", va="center",
                    fontsize=7.5, color=INK_SECONDARY)
    ax.set_yticks(y, [FILTER_PRETTY.get(m, m) for m in order])
    ax.invert_yaxis()
    ax.legend(loc="lower right", ncols=1)
    ax.margins(x=0.22)
    _finish(ax, xlabel="false detections per hour (all four approaches)",
            grid_axis="x")
    return _save(fig, out)


def plot_reliability(diag: Dict, out: Path) -> Optional[Path]:
    """Reliability diagram: what a reported presence of p actually means."""
    raw = diag.get("raw_means")
    cal = diag.get("calibrated")
    counts = diag.get("counts")
    if not raw or not cal:
        return None
    apply_style()
    raw = np.asarray(raw, float)
    cal = np.asarray(cal, float)
    keep = np.asarray(counts, float) > 0 if counts else np.ones_like(raw, bool)

    fig, ax = plt.subplots(figsize=(4.3, 3.4))
    ax.plot([0, 1], [0, 1], color=INK_SECONDARY, lw=0.9, ls=(0, (4, 3)),
            label="perfectly calibrated")
    ax.plot(raw[keep], cal[keep], "o-", color=EMPHASIS,
            label="filter output vs. ground truth")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ece = diag.get("ece_before")
    if ece is not None:
        ax.annotate(f"ECE {float(ece):.3f}", (0.04, 0.90), fontsize=8,
                    color=INK_SECONDARY)
    ax.legend(loc="lower right")
    _finish(ax, ylabel="observed frequency of an inbound vehicle",
            xlabel="reported presence", grid_axis="both")
    return _save(fig, out)


# ---------------------------------------------------------------------------
# Control: what the corridor does, and what it is worth
# ---------------------------------------------------------------------------
def plot_ev_delay_distribution(results: Dict[str, Dict], out: Path,
                               show: Optional[List[str]] = None) -> Optional[Path]:
    """Survival curves of per-vehicle emergency stopped delay.

    Means hide the thing that matters here. A controller that halves the mean by
    helping vehicles that were barely delayed anyway is not the same as one that
    removes the twenty-second waits, and only the distribution separates them.
    Read the curve as the fraction of emergency vehicles that waited longer than
    the x value.
    """
    names = [n for n in (show or results) if n in results
             and results[n].get("aggregate", {}).get("ev_waiting_times_pooled_s")]
    if not names:
        return None
    apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    for i, name in enumerate(names):
        v = np.sort(np.asarray(
            results[name]["aggregate"]["ev_waiting_times_pooled_s"], float))
        surv = 1.0 - np.arange(len(v)) / len(v)
        colour = EMPHASIS if i == 0 else SERIES[min(i, len(SERIES) - 1)]
        ax.step(v, surv, where="post", color=colour,
                lw=2.0 if i == 0 else 1.4,
                label=f"{PRETTY.get(name, name)}  (n={len(v)})")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    _finish(ax, ylabel="fraction of emergency vehicles delayed longer",
            xlabel="stopped delay (s)", grid_axis="both")
    return _save(fig, out)


def plot_ev_class_delay(results: Dict[str, Dict], out: Path,
                        show: Optional[List[str]] = None) -> Optional[Path]:
    """Stopped delay by emergency-vehicle class.

    The scheduler weights an ambulance above a fire appliance above a police
    car, using the acoustic class posterior. If that weighting does anything,
    the realised delays should come out in the same order -- and if they do not,
    the triage term is decoration. This is the figure that says which.
    """
    classes = ("ambulance", "fire", "police")
    names = [n for n in (show or results) if n in results and any(
        np.isfinite(results[n]["aggregate"].get(
            f"ev_{c}_mean_waiting_time_s_mean", np.nan)) for c in classes)]
    if not names:
        return None
    apply_style()
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    x = np.arange(len(classes))
    width = 0.8 / max(len(names), 1)
    for i, name in enumerate(names):
        agg = results[name]["aggregate"]
        vals = [agg.get(f"ev_{c}_mean_waiting_time_s_mean", np.nan) for c in classes]
        errs = [agg.get(f"ev_{c}_mean_waiting_time_s_ci95", 0.0) for c in classes]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.9, yerr=errs,
               color=EMPHASIS if i == 0 else SERIES[min(i, len(SERIES) - 1)],
               edgecolor="none", label=PRETTY.get(name, name),
               error_kw={"ecolor": MUTED, "elinewidth": 0.8, "capsize": 2})
    ax.set_xticks(x, ["Ambulance\ntriage 1.0", "Fire appliance\n0.9",
                      "Police\n0.7"])
    ax.legend(loc="upper left")
    _finish(ax, ylabel="mean stopped delay (s)")
    return _save(fig, out)


def plot_corridor_timing(results: Dict[str, Dict], out: Path,
                         show: Optional[List[str]] = None) -> Optional[Path]:
    """When the corridor opens, against when it has to.

    The *open time* is changeover plus queue discharge: the deadline the
    scheduler works back from. The *lead time* is the arrival time still
    remaining when it actually acted. The just-in-time rule is exactly the claim
    that the second should sit just above the first; acting on sight makes the
    gap between them large, and every second of that gap is bought with
    cross-traffic delay that buys the emergency vehicle nothing.
    """
    names = [n for n in (show or results) if n in results
             and np.isfinite(results[n]["aggregate"].get(
                 "corridor_mean_lead_time_s_mean", np.nan))]
    if not names:
        return None
    apply_style()
    fig, ax = plt.subplots(figsize=(6.6, 0.5 * len(names) + 1.9))
    y = np.arange(len(names))
    lead = np.array([results[n]["aggregate"]["corridor_mean_lead_time_s_mean"]
                     for n in names], float)
    open_t = np.array([results[n]["aggregate"].get(
        "corridor_mean_open_time_s_mean", np.nan) for n in names], float)
    ax.barh(y, lead, height=0.5, color=RECESSIVE, edgecolor="none",
            label="lead time when the corridor was opened")
    ax.barh(y, open_t, height=0.5, color=EMPHASIS, edgecolor="none",
            label="time it needs: changeover + queue discharge")
    for yi, lv, ov in zip(y, lead, open_t):
        if np.isfinite(lv) and np.isfinite(ov):
            ax.annotate(f"{lv - ov:+.1f}s slack", (lv, yi), xytext=(5, 0),
                        textcoords="offset points", va="center",
                        fontsize=7.5, color=INK_SECONDARY)
    ax.set_yticks(y, [PRETTY.get(n, n) for n in names])
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    ax.margins(x=0.26)
    _finish(ax, xlabel="seconds before the vehicle reaches the stop line",
            grid_axis="x")
    return _save(fig, out)


def plot_ablation_ladder(results: Dict[str, Dict], out: Path,
                         full: str = "siren_arc",
                         steps: Optional[List[str]] = None) -> Optional[Path]:
    """What each component is worth, one removal at a time.

    Two panels sharing a y-axis, because removing a component can help one
    objective while hurting the other and a single composite score would let a
    component look useful for the wrong reason.
    """
    steps = [s for s in (steps or ["arc_myopic", "arc_indep", "arc_nocal"])
             if s in results]
    if full not in results or not steps:
        return None
    apply_style()
    order = [full] + steps
    labels = [PRETTY.get(n, n) for n in order]
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 0.5 * len(order) + 2.0), sharey=True)
    for ax, key, title, xlabel in (
        (axes[0], "ev_mean_waiting_time_s", "Emergency stopped delay",
         "mean stopped delay (s)"),
        (axes[1], "mean_time_loss_s", "Background traffic delay",
         "mean time loss (s)"),
    ):
        vals = np.array([results[n]["aggregate"].get(f"{key}_mean", np.nan)
                         for n in order], float)
        errs = np.array([results[n]["aggregate"].get(f"{key}_ci95", 0.0)
                         for n in order], float)
        colors = [EMPHASIS if n == full else RECESSIVE for n in order]
        ax.barh(y, vals, xerr=errs, height=0.55, color=colors, edgecolor="none",
                error_kw={"ecolor": MUTED, "elinewidth": 0.9, "capsize": 2})
        base = vals[0]
        for yi, v in zip(y, vals):
            if not np.isfinite(v):
                continue
            tag = f"{v:.1f}" if yi == 0 else f"{v:.1f}  ({v - base:+.1f})"
            ax.annotate(tag, (v, yi), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=7.5, color=INK_SECONDARY)
        ax.set_title(title, color=INK, loc="left", pad=8)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.28)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    return _save(fig, out)


# ---------------------------------------------------------------------------
# Time-space: what the controller actually did to one vehicle
# ---------------------------------------------------------------------------
def plot_ev_trajectory(data: Dict, out: Path, vehicle: Optional[str] = None,
                       order: Optional[List[str]] = None) -> Optional[Path]:
    """One emergency vehicle's approach, under each controller, on one axis.

    The demand realisation is fixed by the seed, so this is the same vehicle
    entering at the same second on the same route in every run; only the signal
    differs. A controller that anticipates draws an unbroken line into the
    junction. One that reacts draws a line that flattens against the stop line
    and then leaves --- the flat segment is the delay, in seconds, readable off
    the axis.

    The green bands beneath the lines are when each controller was actually
    showing green to that approach, which is what separates *deciding* early
    from *arriving* early.
    """
    runs = data.get("runs") or {}
    common = data.get("common_vehicles") or []
    if not runs or not common:
        return None

    names = [n for n in (order or runs) if n in runs]
    if not names:
        return None

    # Pick the vehicle with the most to show: the largest spread in stopped
    # delay across controllers. A vehicle that sails through under every
    # controller is a true observation and a useless picture.
    def spread(vid):
        stops = []
        for n in names:
            rec = runs[n]["tracks"].get(vid)
            if not rec:
                return -1.0
            v = np.asarray(rec["v"], float)
            d = np.asarray(rec["d"], float)
            up = d >= 0
            stops.append(float((v[up] < 0.3).sum()) if up.any() else 0.0)
        return float(max(stops) - min(stops))

    vid = vehicle or max(common, key=spread)
    if vid not in runs[names[0]]["tracks"]:
        return None

    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    band_h = 0.0
    t0 = min(float(runs[n]["tracks"][vid]["t"][0]) for n in names)
    ymax = 0.0

    for i, name in enumerate(names):
        rec = runs[name]["tracks"][vid]
        t = np.asarray(rec["t"], float) - t0
        d = np.asarray(rec["d"], float)
        v = np.asarray(rec["v"], float)
        colour = EMPHASIS if i == 0 else SERIES[min(i, len(SERIES) - 1)]
        stopped = int((v[d >= 0] < 0.3).sum()) if (d >= 0).any() else 0
        ax.plot(t, d, color=colour, lw=2.0 if i == 0 else 1.4,
                label=f"{PRETTY.get(name, name)}  ({stopped}s stopped)")
        ymax = max(ymax, float(np.nanmax(d)))

    # Green bands, one lane per controller, below the axis origin.
    lane = max(ymax * 0.055, 6.0)
    for i, name in enumerate(names):
        rec = runs[name]["tracks"][vid]
        t = np.asarray(rec["t"], float) - t0
        g = np.asarray(rec["green"], bool)
        colour = EMPHASIS if i == 0 else SERIES[min(i, len(SERIES) - 1)]
        y = -(i + 1) * lane
        band_h = y - lane * 0.7
        ax.fill_between(t, y - lane * 0.35, y + lane * 0.35, where=g,
                        step="post", color=colour, alpha=0.75, linewidth=0)
        ax.fill_between(t, y - lane * 0.35, y + lane * 0.35, where=~g,
                        step="post", color=GRID, linewidth=0)
        ax.annotate(PRETTY.get(name, name), (t[-1], y), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=6.5,
                    color=INK_SECONDARY)

    ax.axhline(0.0, color=INK_SECONDARY, lw=0.9)
    ax.annotate("stop line", (0.0, 0.0), xytext=(2, 4),
                textcoords="offset points", fontsize=7.5, color=INK_SECONDARY)
    ax.set_ylim(band_h, ymax * 1.06)
    ax.legend(loc="upper right")
    _finish(ax, ylabel="distance to the stop line (m)",
            xlabel="seconds since the vehicle entered the network",
            title=f"One {runs[names[0]]['tracks'][vid].get('veh_class', 'emergency')} "
                  f"vehicle, identical demand; bands show green to its approach",
            grid_axis="y")
    return _save(fig, out)


ROBUST_PRETTY = {
    "nominal": "nominal",
    "noise+5": "+5 dB ambient",
    "noise+10": "+10 dB ambient",
    "noise+15": "+15 dB ambient",
    "mic1": "1 mast dead",
    "mic2": "2 masts dead",
    "deaf": "no sensing at all",
}


def plot_robustness(data: Dict, out: Path) -> Optional[Path]:
    """Degradation under a rising noise floor and under mast loss.

    Two panels sharing a category axis. The emergency panel is the one under
    test; the background panel is there because a controller can hold emergency
    delay flat by preempting more aggressively as it gets less certain, and that
    trade has to be visible rather than inferred. The dashed line is the
    no-sensing floor: how the same policy behaves with the microphones removed
    entirely, which is what any degradation is heading towards.
    """
    results = data.get("results") or {}
    order = [k for k in ("nominal", "noise+5", "noise+10", "noise+15",
                         "mic1", "mic2") if k in results]
    if len(order) < 2:
        return None
    apply_style()
    floor = results.get("deaf", {}).get("aggregate", {})

    fig, axes = plt.subplots(2, 1, figsize=(6.2, 4.6), sharex=True)
    x = np.arange(len(order))
    for ax, key, ylabel in (
        (axes[0], "ev_mean_waiting_time_s", "EV stopped delay (s)"),
        (axes[1], "mean_time_loss_s", "background delay (s)"),
    ):
        vals = np.array([results[k]["aggregate"].get(f"{key}_mean", np.nan)
                         for k in order], float)
        errs = np.array([results[k]["aggregate"].get(f"{key}_ci95", 0.0)
                         for k in order], float)
        colors = [EMPHASIS if k == "nominal" else RECESSIVE for k in order]
        ax.bar(x, vals, width=0.6, yerr=errs, color=colors, edgecolor="none",
               error_kw={"ecolor": MUTED, "elinewidth": 0.9, "capsize": 2})
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                ax.annotate(f"{v:.1f}", (xi, v), xytext=(0, 3),
                            textcoords="offset points", ha="center",
                            fontsize=7.5, color=INK_SECONDARY)
        fl = floor.get(f"{key}_mean")
        if fl is not None and np.isfinite(fl):
            ax.axhline(fl, color=INK_SECONDARY, lw=0.9, ls=(0, (4, 3)))
            ax.annotate("no sensing", (len(order) - 0.4, fl), xytext=(0, 3),
                        textcoords="offset points", ha="right", fontsize=7,
                        color=INK_SECONDARY)
        ax.margins(y=0.22)
        _finish(ax, ylabel=ylabel)
    axes[1].set_xticks(x, [ROBUST_PRETTY.get(k, k) for k in order], fontsize=8)
    axes[0].set_title("Degradation is continuous, not a cliff",
                      color=INK, loc="left", pad=8)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# The two defects found by instrumenting against ground truth, and their fixes
# ---------------------------------------------------------------------------
def plot_mast_response(out: Path, cfg=None) -> Optional[Path]:
    """What each mast siting can and cannot tell apart, along the approach.

    The explanatory figure for the sensing revision, and it needs no result
    file -- it is the acoustic model evaluated over the junction geometry, so
    it cannot drift from the code it describes.

    Two panels sharing an x-axis, and they answer two different questions.
    *Audibility* is whether the mast hears the vehicle at all, which is what
    the steered two-element pair was introduced to fix. *Separation* is whether
    the correct mast hears it more loudly than the wrong ones do, which is the
    question the controller actually asks -- and it is the one the steered pair
    made worse, because reading the louder of two opposed cardioids flattens
    the pattern that separates the arms.

    The x-axis runs right to left, from far upstream to the stop line, so
    reading order matches the direction of travel: the vehicle enters at the
    left of the page and arrives at the right.
    """
    import math
    from ..config import AcousticConfig
    from ..sensing.sensor import (SourceState, cardioid_gain_db,
                                  endfire_pair_gain_db)
    from ..sensing.siren_synth import received_spl_db

    cfg = cfg or AcousticConfig()
    fb = float(cfg.directivity_front_back_db)
    amb_lo, amb_hi = cfg.ambient_noise_db_range
    noise = float(getattr(cfg, "level_noise_db", 2.5))

    # A square junction with the stop lines 10 m out, which is the geometry
    # netgen produces. Only the ratios matter, so nothing here is fitted.
    arms = ["N", "E", "S", "W"]
    axis = {"N": (0.0, 1.0), "E": (1.0, 0.0), "S": (0.0, -1.0), "W": (-1.0, 0.0)}
    stop = {a: (axis[a][0] * 10.0, axis[a][1] * 10.0) for a in arms}

    def level_at(offset, dual, src_arm, s_m, mast):
        ax, ay = axis[src_arm]
        px, py = stop[src_arm][0] + ax * s_m, stop[src_arm][1] + ay * s_m
        bx = stop[mast][0] + axis[mast][0] * offset
        by = stop[mast][1] + axis[mast][1] * offset
        dx, dy = px - bx, py - by
        dist = max(math.hypot(dx, dy), 1.0)
        ux, uy = axis[mast]
        theta = math.acos(max(-1.0, min(1.0, (dx * ux + dy * uy) / dist)))
        if dual:
            f, b = endfire_pair_gain_db(theta, fb)
            gain = max(f, b)
        else:
            gain = cardioid_gain_db(theta, fb)
        return received_spl_db("ambulance", dist, gain)

    sitings = [
        (70.0, True, "70 m upstream, steered pair", SERIES[1]),
        (float(cfg.mic_offset_m), bool(cfg.dual_element_mast),
         f"{cfg.mic_offset_m:.0f} m (signal pole), cardioid", SERIES[0]),
    ]
    d = np.linspace(320.0, 1.0, 320)

    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.6), sharex=True)
    ax = axes[0]
    ax.axhspan(amb_lo, amb_hi, color=RECESSIVE, alpha=0.30, lw=0,
               label="ambient noise floor (55-80 dB)")
    for offset, dual, label, colour in sitings:
        own = [level_at(offset, dual, "W", float(s_m), "W") for s_m in d]
        ax.plot(d, own, color=colour, lw=2.0, label=label)
    _finish(ax, ylabel="level at own mast (dB)",
            title="Audibility: both sitings hear the vehicle all the way in")
    ax.legend(loc="upper left")

    ax2 = axes[1]
    ax2.axhspan(0.0, 2.0 * noise, color=RECESSIVE, alpha=0.30, lw=0,
                label=f"within 2 sd of the level noise ({noise:g} dB)")
    for offset, dual, label, colour in sitings:
        sep = []
        for s_m in d:
            heard = {m: level_at(offset, dual, "W", float(s_m), m) for m in arms}
            sep.append(heard["W"] - max(v for k, v in heard.items() if k != "W"))
        ax2.plot(d, sep, color=colour, lw=2.0, label=label)
        # The number the figure exists for is the one at the stop line.
        ax2.annotate(f"{sep[-1]:.1f} dB", (d[-1], sep[-1]), xytext=(8, 0),
                     textcoords="offset points", fontsize=8, color=colour)
    _finish(ax2, ylabel="margin over wrong masts (dB)",
            xlabel="distance from the stop line (m); the vehicle travels left to right",
            title="Separation: only one of them can say which arm, at the line")
    ax2.legend(loc="upper left")
    ax2.invert_xaxis()
    return _save(fig, out)


_SHORT_ARM = {
    "baseline (as shipped)": "baseline",
    "+ two-element mast": "+ mast",
    "+ S6 deadline only": "+ S6",
    "+ both (SIREN-ARC v2)": "+ both",
}
"""Tick labels for :func:`plot_ev_ablation`. The full names are spelled out in
the caption; repeating them under three panels only produces collisions."""


def plot_ev_ablation(data: Dict, out: Path) -> Optional[Path]:
    """The two fixes, separately and together, on shared seeds.

    Three panels, one measure each, sharing the arm order. Emergency delay and
    background queue are deliberately not overlaid on twin axes: the question a
    reader has is whether the emergency gain was bought with background delay,
    and that comparison should be made by the reader between two panels rather
    than asserted by the chart.

    Per-seed points are drawn over the bars because the seed distribution is
    heavy-tailed. A mean alone would hide whether an arm fixed the tail or
    merely moved the centre.
    """
    arms = data.get("arms") or {}
    if not arms:
        return None
    labels = list(arms)
    x = np.arange(len(labels))

    panels = [
        ("ev_delay_per_vehicle_s", "Emergency stopped delay", "s per vehicle"),
        ("mean_queue", "Background traffic", "mean halting vehicles"),
        ("ev_stops", "Emergency full stops", "stops per episode"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.2))
    rng = np.random.default_rng(0)
    for ax, (key, title, ylabel) in zip(axes, panels):
        series = [np.array([r.get(key, np.nan) for r in arms[k]], dtype=float)
                  for k in labels]
        means = np.array([np.nanmean(s) for s in series])
        # One hue: the arms are nominal categories, not a series, so bar length
        # already carries the magnitude. The arm under test takes the emphasis.
        colours = [EMPHASIS if i == len(labels) - 1 else RECESSIVE
                   for i in range(len(labels))]
        ax.bar(x, means, width=0.62, color=colours, edgecolor="none", zorder=2)
        for xi, s in zip(x, series):
            jitter = rng.uniform(-0.13, 0.13, size=s.shape)
            ax.scatter(xi + jitter, s, s=9, color=INK_SECONDARY, alpha=0.45,
                       linewidths=0, zorder=3)
        for xi, m in zip(x, means):
            ax.annotate(f"{m:.2f}", (xi, m), xytext=(0, 5),
                        textcoords="offset points", ha="center",
                        fontsize=7.5, color=INK_SECONDARY, zorder=4)
        _finish(ax, ylabel=ylabel, title=title)
        # Short tick labels, spelled out once in the caption rather than four
        # times across three panels: at this width the full arm names collide
        # with their neighbours whether they are rotated or wrapped.
        ax.set_xticks(x, [_SHORT_ARM.get(v, v) for v in labels], fontsize=8)
        ax.margins(y=0.20)
    return _save(fig, out)


def plot_request_quality(data: Dict, out: Path) -> Optional[Path]:
    """The precision/recall trade the second element actually makes.

    Aggregate recall is the wrong summary for this system -- a miss 300 m out
    costs nothing, because the corridor is deliberately not acting yet, while a
    miss at 20 m is the vehicle stopping at a red -- so recall is reported by
    distance band rather than pooled.

    The measured trade is not the one we expected. Making the mast able to hear
    behind itself does not simply raise recall: the second lobe and the
    front-back ratio give the joint filter enough geometry to *reject*
    hypotheses it previously accepted, so the filter becomes more conservative
    and raises fewer requests overall. Recall falls in most bands while the
    spurious-request rate roughly halves.

    That is the favourable direction here, and the reason is S6. A guarantee
    acts on a request decisively, so a wrong request is not merely wasted -- it
    forces the wrong green. Precision, not recall, is what the guarantee
    consumes.
    """
    bands = data.get("bands") or []
    if not bands:
        return None
    labels = [b["label"] for b in bands]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    ax = axes[0]
    for arm, colour, marker, name in (
        ("dual", SERIES[0], "o", "two-element mast"),
        ("single", SERIES[1], "s", "single cardioid"),
    ):
        vals = [100.0 * b.get(f"recall_{arm}", np.nan) for b in bands]
        if not any(np.isfinite(v) for v in vals):
            continue
        ax.plot(x, vals, color=colour, marker=marker, label=name)
        ax.annotate(f"{vals[-1]:.0f}%", (x[-1], vals[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center",
                    fontsize=7.5, color=colour)
    _finish(ax, ylabel="request recall (%)",
            xlabel="emergency vehicle distance to the stop line",
            title="Fewer requests raised, band by band")
    ax.set_xticks(x, labels, fontsize=7.5, rotation=30, ha="right")
    ax.set_ylim(0, 108)
    ax.margins(x=0.10)
    ax.legend(loc="lower left")

    ax2 = axes[1]
    fp = data.get("false_rate") or {}
    keys = [k for k in ("single", "dual") if k in fp]
    xv = np.arange(len(keys))
    vals = [100.0 * fp[k] for k in keys]
    colours = [RECESSIVE if k == "single" else EMPHASIS for k in keys]
    ax2.bar(xv, vals, width=0.5, color=colours, edgecolor="none")
    for xi, v in zip(xv, vals):
        ax2.annotate(f"{v:.1f}%", (xi, v), xytext=(0, 5),
                     textcoords="offset points", ha="center",
                     fontsize=7.5, color=INK_SECONDARY)
    _finish(ax2, ylabel="spurious requests (% of EV-free steps)",
            title="...and half as many of them are wrong")
    ax2.set_xticks(xv, [{"single": "single cardioid",
                         "dual": "two-element mast"}[k] for k in keys],
                   fontsize=8)
    ax2.margins(y=0.24)
    return _save(fig, out)


def plot_horn_discrimination(model, out: Path, blind=None) -> Optional[Path]:
    """Can the detector tell a siren from a car horn?

    Two panels, because two different questions are being asked and conflating
    them is what hid the bug.

    *Left* is the detector's own behaviour: the probability it reports a siren,
    against SNR, for a true siren, for a road horn, and for bare traffic noise.
    A usable detector separates the first curve from the other two. The horn
    curve is the one that matters, because bare noise was never a hard negative
    -- it is broadband, and a harmonic detector rejects it almost regardless of
    level.

    *Right* is the readout. The same network, the same weights, scored two ways:
    ``1 - P(none)``, which counts a confidently-identified horn as a siren, and
    the sum over the siren classes, which does not. The gap between them is not
    a modelling improvement -- it is information the network had already
    extracted and the readout was throwing away.
    """
    def sig(x):
        return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))

    edges = list(model.snr_edges)
    n = len(model.siren_logit_mean)
    if n == 0:
        return None
    # Bin centres, with the open ends given the width of their neighbour so the
    # x-axis stays linear in dB rather than compressing the tails.
    widths = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)] or [5.0]
    centres = ([edges[0] - widths[0] / 2.0]
               + [(edges[i] + edges[i + 1]) / 2.0 for i in range(len(edges) - 1)]
               + [edges[-1] + widths[-1] / 2.0])
    centres = centres[:n]

    siren = sig(model.siren_logit_mean[:n])
    horn = sig(model.horn_logit_mean[:n]) if model.horn_logit_mean else None
    noise = np.full(n, float(sig(model.noise_logit_mean)))

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))

    ax = axes[0]
    ax.plot(centres, siren, color=SERIES[0], marker="o", label="true siren")
    if horn is not None:
        ax.plot(centres, horn, color=SERIES[1], marker="s", label="road horn")
    ax.plot(centres, noise, color=RECESSIVE, linestyle="-", marker=None,
            label="broadband traffic noise")
    # Direct labels in the band the guarantee actually acts in: a horn is
    # sounded from the queue, metres from the mast, so it arrives loud.
    if horn is not None and n >= 5:
        ax.annotate(f"{horn[-1]:.3f}", (centres[-1], horn[-1]), xytext=(6, -2),
                    textcoords="offset points", fontsize=8, color=SERIES[1])
    ax.annotate(f"{siren[-1]:.3f}", (centres[-1], siren[-1]), xytext=(6, -2),
                textcoords="offset points", fontsize=8, color=SERIES[0])
    _finish(ax, ylabel="P(detector reports a siren)",
            xlabel="signal-to-noise ratio at the mast (dB)",
            title="The confuser is rejected where it is loud")
    ax.set_ylim(-0.04, 1.08)
    ax.legend(loc="center left")

    ax2 = axes[1]
    if blind is not None and blind.horn_logit_mean:
        before = sig(blind.horn_logit_mean[:n])
    elif blind is not None:
        # A model fitted before horns existed has no horn curve at all, and
        # falls back to the siren curve -- which is the honest reading of a
        # detector that has never seen the confuser.
        before = sig(blind.siren_logit_mean[:n])
    else:
        before = siren
    after = horn if horn is not None else np.zeros(n)
    ax2.plot(centres, before, color=SERIES[1], marker="s",
             label="readout: 1 - P(none)")
    ax2.plot(centres, after, color=SERIES[0], marker="o",
             label="readout: sum over siren classes")
    ax2.fill_between(centres, after, before, color=SERIES[1], alpha=0.12, lw=0)
    # The shaded gap is the message, so the caption sits inside it and the
    # legend moves out of its way rather than the other way round.
    ax2.annotate("same network, same weights;\nthe difference is the readout",
                 (centres[-2], 0.58), fontsize=8, color=INK_SECONDARY,
                 ha="right")
    _finish(ax2, ylabel="P(a road horn is called a siren)",
            xlabel="signal-to-noise ratio at the mast (dB)",
            title="What the readout was discarding")
    ax2.set_ylim(-0.04, 1.12)
    ax2.legend(loc="lower left", fontsize=7.5)
    return _save(fig, out)
