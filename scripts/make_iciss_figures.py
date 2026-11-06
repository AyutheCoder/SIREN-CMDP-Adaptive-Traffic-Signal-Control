"""Every figure in the ICISS paper, built from the files in ``results/``.

Nothing here is typed by hand: each figure is a pure function of a result file,
so re-running an experiment and re-running this script keeps the paper honest.
Output goes to ``paper_iciss/figures`` as PDF (for LaTeX) and PNG (for review).

    python scripts/make_iciss_figures.py            # all of them
    python scripts/make_iciss_figures.py detector   # just one

Every figure is drawn at the LNNS text width (122 mm) and included at natural
size, so the point sizes below are the point sizes on the printed page. Drawing
wide and scaling down is what makes axis labels unreadable in a proceedings PDF.

Colour: a validated categorical palette (worst all-pairs CVD dE 9.2 on the three
slots used for scatter identity). Magnitude-only charts use a single hue with
the proposed method highlighted; signed quantities use a two-hue diverging pair
about a neutral midpoint. Identity is never carried by colour alone -- every
series is either direct-labelled or given a distinct marker as well.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT = REPO / "paper_iciss" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# LNNS (llncs.cls) is one column, 122mm = 4.75in of text width. Every figure
# is drawn at that width and included at natural size, so the point sizes
# below are the point sizes on the printed page -- drawing wide and letting
# LaTeX scale down is what makes axis labels unreadable in proceedings.
TW = 4.75   # LNNS text width, inches

# ----------------------------------------------------------------- palette
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
VIOLET = "#4a3aa7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8a85"
GRID = "#dedede"
NEUTRAL = "#c9c9c4"          # unhighlighted bars
HILITE = BLUE                # the proposed method
SEQ = LinearSegmentedColormap.from_list("seq", ["#f2f7fd", "#2a78d6", "#123a68"])

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7.0,
    "axes.labelsize": 7.0,
    "axes.titlesize": 7.2,
    "xtick.labelsize": 6.2,
    "ytick.labelsize": 6.2,
    "legend.fontsize": 6.2,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "text.color": INK,
    "axes.labelcolor": INK,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "figure.dpi": 240,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf")


def tidy(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


def panel(ax, text):
    ax.set_title(text, loc="left", color=INK2, fontsize=8.3, pad=3)


# ------------------------------------------------------------------- data
EVAL = json.loads((RESULTS / "evaluation_balanced.json").read_text())
RES = EVAL["results"]

LABELS = {
    "fixed_time": "Fixed-time (Webster)",
    "actuated": "Actuated (SUMO)",
    "greedy_queue": "Greedy queue",
    "max_pressure": "Max-pressure",
    "max_pressure_preempt": "MaxP + preemption",
    "base_no_sensing": "DQN, no sensing",
    "base_arc": "Learned policy alone",
    "base_cmdp": "Constrained policy alone",
    "siren_arc": "SIREN-ARC (ours)",
    "siren_cmdp": "SIREN-CMDP (ours)",
    "arc_oracle": "Ours + oracle sensing",
    "arc_indep": "Ours, independent filters",
    "arc_myopic": "Ours, act on sight",
    "arc_nocal": "Ours, no calibration",
    "arc_greedy": "Ours, greedy conflicts",
}


def runs(c):
    return sorted(RES[c]["runs"], key=lambda r: r["seed"])


def series(c, k):
    return np.array([r[k] for r in runs(c)], dtype=float)


def med(c, k):
    return float(np.median(series(c, k)))


def jam(c):
    return int(RES[c]["aggregate"]["n_jammed"])


def boot_median_ci(v, n_boot=20000, seed=0):
    """Percentile bootstrap interval for the median.

    The seed distribution is heavy-tailed: a run that tips into spillback
    reaches several hundred seconds against a ~24 s median for the same
    controller. A normal-theory interval around the *mean* would therefore put
    an error bar of 60 s on a bar of height 17. The bar shows the median, so
    the interval has to be the median's.
    """
    v = np.asarray(v, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.median(v[rng.integers(0, v.size, size=(n_boot, v.size))], axis=1)
    return tuple(np.percentile(draws, [2.5, 97.5]))


# ============================================================ 1. architecture
def fig_architecture():
    """The pipeline, drawn as the closed loop it actually is."""
    fig, ax = plt.subplots(figsize=(TW, 1.95))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def lane(y, h, label, color, fc):
        ax.add_patch(FancyBboxPatch((13.5, y), 84.5, h,
                                    boxstyle="round,pad=0.4,rounding_size=1.0",
                                    fc=fc, ec=color, lw=0.9, zorder=1))
        ax.text(12.4, y + h / 2, label, ha="right", va="center", fontsize=7.4,
                weight="bold", color=color, rotation=90)

    def cell(x, w, y, h, title, body, color, tsize=5.7, bsize=4.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.3,rounding_size=0.8",
                                    fc="white", ec=color, lw=0.7, zorder=2))
        ax.text(x + w / 2, y + h - 1.5, title, ha="center", va="top",
                fontsize=tsize, weight="bold", color=INK, zorder=3)
        ax.text(x + w / 2, y + h - 4.1, body, ha="center", va="top",
                fontsize=bsize, color=INK2, zorder=3, linespacing=1.35)

    def down(x, y1, y2, label):
        ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>",
                                     mutation_scale=7, lw=0.9, color=INK2,
                                     zorder=4))
        # The gap between lanes is shorter than the label, so the label is set
        # on an opaque patch rather than allowed to sit on the lane borders.
        ax.text(x + 1.4, (y1 + y2) / 2, label, ha="left", va="center",
                fontsize=5.2, color=MUTED, style="italic", zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8))

    # ---- lane 1: perception, runs at 1 Hz ----------------------------------
    lane(33.6, 11.4, "Perception", ORANGE, "#fdf6f2")
    cell(15.0, 19.0, 34.6, 9.4, "Four roadside masts",
         "one cardioid per arm,\npower-preserving log-mel", ORANGE)
    cell(35.0, 19.0, 34.6, 9.4, "Siren CNN",
         "class, range, range-rate\n(siren, horn, none)", ORANGE)
    cell(55.0, 25.0, 34.6, 9.4, "Joint belief filter",
         "exact inference over 2731\nhypotheses: which arm, and where",
         ORANGE)
    cell(81.0, 16.5, 34.6, 9.4, "Calibration",
         "isotonic map,\nscore to probability", ORANGE)
    for x in (33.4, 53.4, 79.4):
        ax.add_patch(FancyArrowPatch((x, 39.3), (x + 1.4, 39.3),
                                     arrowstyle="-|>", mutation_scale=6,
                                     lw=0.8, color=MUTED, zorder=3))

    # ---- lane 2: decision, 5 s epoch with a 1 Hz re-check ------------------
    lane(19.6, 11.4, "Decision", AQUA, "#f2fbf8")
    cell(15.0, 24.5, 20.6, 9.4, "Lagrangian dueling DDQN",
         "acts greedily in $Q_r - \\lambda Q_c$;\n$\\lambda$ solves the CMDP dual", AQUA)
    cell(40.0, 33.0, 20.6, 9.4, "Risk-calibrated corridor",
         "open iff $p\\,\\gamma B > C$, i.e. $p > \\theta^{\\star}\\!=C/(\\gamma B)$;\n"
         "$\\theta^{\\star}$ re-derived each epoch, opened just in time", AQUA)
    cell(75.0, 21.5, 20.6, 9.4, "Conflict scheduler",
         "exact minimum-cost\nordering over requests", AQUA)

    # ---- lane 3: safety ----------------------------------------------------
    # Two body lines at this size need about 4.5 units; the box is sized for
    # that rather than for the text that used to be here.
    lane(6.6, 11.0, "Safety", VIOLET, "#f6f4fc")
    cell(15.0, 81.5, 7.4, 9.0,
         "Safety shield: projects the proposed phase onto the legal action set",
         "S1 minimum green and a full yellow/all-red changeover  $\\cdot$  "
         "S2 maximum green\nS3 no movement waits beyond $W_{\\max}\\!=\\!180$ s"
         "  $\\cdot$  S6 an emergency request is served within 16 s",
         VIOLET, tsize=6.2, bsize=4.7)

    # ---- lane 4: environment -----------------------------------------------
    # Labelled inline, not with a rotated lane title: at this font size a
    # vertical "Environment" runs up into the "Safety" label beside it.
    ax.add_patch(FancyBboxPatch((13.5, 0.4), 84.5, 3.6,
                                boxstyle="round,pad=0.4,rounding_size=1.0",
                                fc="#fafafa", ec=MUTED, lw=0.9, ls=(0, (3, 2)),
                                zorder=1))
    ax.text(55.75, 2.2, "Environment — SUMO: four-arm junction, 700 veh/h "
            "per approach, 30 emergency vehicles/h",
            ha="center", va="center", fontsize=5.6, color=INK)

    down(55.75, 33.2, 31.0, "calibrated posterior $p$, range, arrival time")
    down(55.75, 19.2, 17.8, "proposed phase")
    down(55.75, 6.4, 4.6, "signal command")

    # the environment closes the loop
    for seg in (((13.3, 2.2), (4.2, 2.2)), ((4.2, 2.2), (4.2, 39.3))):
        ax.add_patch(FancyArrowPatch(*seg, arrowstyle="-", lw=0.9,
                                     color=MUTED, zorder=1))
    for y in (39.3, 25.3):
        ax.add_patch(FancyArrowPatch((4.2, y), (13.2, y), arrowstyle="-|>",
                                     mutation_scale=7, lw=0.9, color=MUTED,
                                     zorder=1))
    ax.text(2.6, 21.0, "acoustic scene, queues, phase state", ha="center",
            va="center", fontsize=5.8, color=MUTED, style="italic", rotation=90)
    save(fig, "fig_architecture")


# ============================================================ 2. SUMO frames
def fig_sumo():
    demo = REPO / "figures" / "demo"
    # One row rather than 2x2: the frames are nearly square, so a 2x2 block
    # leaves half the text width empty and costs twice the height for the same
    # four pictures.
    # Panel labels stay bare; what each one shows is in the caption, because at
    # a quarter of the text width a descriptive title collides with its
    # neighbours.
    panels = [
        ("seq_00_t83s.png", "(a) $t=83$ s"),
        ("seq_01_t92s.png", "(b) $t=92$ s"),
        ("seq_02_t99s.png", "(c) $t=99$ s"),
        ("corridor_S_312s.png", "(d) $t=312$ s"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(TW, 1.12))
    for ax, (fname, title) in zip(axes, panels):
        p = demo / fname
        if not p.exists():
            ax.axis("off")
            continue
        img = mpimg.imread(p)
        h, w = img.shape[:2]
        ax.imshow(img[int(h * 0.04):, int(w * 0.29):int(w * 0.71)])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(MUTED); s.set_linewidth(0.6)
        ax.set_title(title, fontsize=7.3, pad=2.0, color=INK)
    fig.subplots_adjust(wspace=0.05)
    save(fig, "fig_sumo")


# ============================================ 3. controllers + trade-off plane
def fig_controllers():
    order = ["actuated", "fixed_time", "greedy_queue", "max_pressure",
             "base_arc", "base_cmdp", "max_pressure_preempt",
             "siren_cmdp", "siren_arc"]
    fig = plt.figure(figsize=(TW * 0.92, 2.58))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.02], height_ratios=[2.0, 1.0],
                          hspace=0.85, wspace=0.46)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    ax_cdf = fig.add_subplot(gs[1, :])

    # --- (a) the two objectives, side by side, as horizontal bars
    ax = axes[0]
    y = np.arange(len(order))
    ev = np.array([med(c, "ev_mean_waiting_time_s") for c in order])
    bg = np.array([med(c, "mean_time_loss_s") for c in order])
    evl = np.array([[v - boot_median_ci(series(c, "ev_mean_waiting_time_s"))[0],
                     boot_median_ci(series(c, "ev_mean_waiting_time_s"))[1] - v]
                    for c, v in zip(order, ev)]).T
    bgl = np.array([[v - boot_median_ci(series(c, "mean_time_loss_s"))[0],
                     boot_median_ci(series(c, "mean_time_loss_s"))[1] - v]
                    for c, v in zip(order, bg)]).T
    ax.barh(y - 0.20, ev, 0.38, color=ORANGE, zorder=3, xerr=evl,
            error_kw=dict(ecolor=INK2, lw=0.5, capsize=1.1),
            label="emergency stopped delay")
    ax.barh(y + 0.20, bg, 0.38, color=BLUE, zorder=3, xerr=bgl,
            error_kw=dict(ecolor=INK2, lw=0.5, capsize=1.1),
            label="background delay")
    # No per-bar value labels: eighteen of them at this size collide with the
    # bars and with each other, and every one of these numbers is already in
    # Table 2. The chart is here to show the ordering, not to be read off.
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[c] for c in order], fontsize=7.1)
    for lbl, c in zip(ax.get_yticklabels(), order):
        if c.startswith("siren"):
            lbl.set_color(HILITE); lbl.set_fontweight("bold")
    ax.invert_yaxis()
    # Every bar runs the width of the axes, so an inside legend sits on data
    # whatever corner it is put in; it goes above the axes instead.
    ax.set_ylim(len(order) - 0.4, -0.6)
    ax.set_xlim(0, 30)
    ax.set_xlabel("median delay over 12 seeds (s)")
    panel(ax, "(a) Both objectives at once")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.10),
              ncol=2, fontsize=6.6, handlelength=1.0, handletextpad=0.3,
              borderpad=0.0, columnspacing=1.0)
    tidy(ax, grid="x")

    # --- (b) the plane the two objectives live in
    ax = axes[1]
    groups = {
        "fixed/actuated": (["fixed_time", "actuated"], BLUE, "o"),
        "adaptive, no priority": (["max_pressure", "base_arc"], ORANGE, "s"),
        "preemption": (["max_pressure_preempt"], AQUA, "^"),
        "proposed": (["siren_arc", "siren_cmdp"], INK, "D"),
    }
    # No per-point text labels. Seven points across a 55mm panel cannot carry
    # legible annotations without colliding, and identity is already given three
    # ways: the legend groups them by colour and marker, panel (a) names every
    # controller, and Table 2 has the numbers. The panel's job is the shape of
    # the trade-off, which the grouping shows directly.
    for name, (cs, col, mk) in groups.items():
        xs = [med(c, "mean_time_loss_s") for c in cs]
        ys = [med(c, "ev_mean_waiting_time_s") for c in cs]
        ax.scatter(xs, ys, s=34, marker=mk, facecolor=col, edgecolor="white",
                   linewidth=0.8, zorder=4, label=name)
    # the ceiling, marked as such rather than treated as a competitor
    xo = med("arc_oracle", "mean_time_loss_s")
    yo = med("arc_oracle", "ev_mean_waiting_time_s")
    ax.scatter([xo], [yo], s=36, marker="D", facecolor="white", edgecolor=INK,
               linewidth=0.9, zorder=4, label="oracle ceiling")

    pts = sorted((med(c, "mean_time_loss_s"), med(c, "ev_mean_waiting_time_s"))
                 for cs, _, _ in groups.values() for c in cs)
    front, best = [], np.inf
    for x, y_ in pts:
        if y_ < best:
            front.append((x, y_)); best = y_
    fx, fy = zip(*front)
    ax.step(fx, fy, where="post", color=MUTED, lw=0.8, ls=(0, (4, 2)), zorder=2,
            label="Pareto front")
    # Headroom above the highest point (actuated, ~19.5 s) for the legend, so
    # it does not land on the "Fixed-time" and "Actuated" annotations.
    # Headroom above the highest point (~19.5 s) for the two-row legend, so it
    # never lands on the point labels below it.
    ax.set_xlim(21.6, 29.0); ax.set_ylim(3.0, 30.0)
    ax.set_xlabel("background delay, median (s)")
    ax.set_ylabel("EV stopped delay (s)")
    panel(ax, "(b) The trade-off plane")
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=6.3,
              handletextpad=0.2, columnspacing=0.6, borderpad=0.0,
              labelspacing=0.22)
    tidy(ax, grid="both")

    # --- (c) the whole per-vehicle distribution, not just its median.
    # A median hides the tail, and for an emergency service the tail is the
    # operationally relevant statistic.
    ax = ax_cdf
    styles = {"siren_arc": (HILITE, "-", 1.6),
              "max_pressure_preempt": (AQUA, "-", 1.1),
              "base_arc": (ORANGE, "--", 1.1),
              "actuated": (VIOLET, "-.", 1.0),
              "fixed_time": (MUTED, ":", 1.0)}
    for c in ["fixed_time", "actuated", "base_arc", "max_pressure_preempt",
              "siren_arc"]:
        w = np.sort([x for r in runs(c) for x in r.get("ev_waiting_times_s", [])])
        col, ls, lw = styles[c]
        ax.plot(w, np.arange(1, w.size + 1) / w.size, ls, color=col, lw=lw,
                zorder=3, label=LABELS[c])
    ax.set_xlim(0, 70)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("stopped delay per emergency vehicle (s)")
    ax.set_ylabel("fraction of EVs")
    # This panel is wide and short, so a five-entry legend is taller than any
    # gap between the curves. Title on the left, legend in one row on the
    # right, both above the axes and clear of the data.
    panel(ax, "(c) Per-vehicle distribution (CDF)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.70),
              ncol=3, fontsize=6.4, handlelength=1.6, columnspacing=1.2,
              handletextpad=0.35, borderpad=0.0, labelspacing=0.3)
    tidy(ax, grid="both")
    save(fig, "fig_controllers")


# ============================================== 4. multi-metric comparison
def fig_heatmap():
    order = ["fixed_time", "actuated", "greedy_queue", "max_pressure",
             "max_pressure_preempt", "base_arc", "base_cmdp", "siren_cmdp",
             "siren_arc"]
    metrics = [
        ("mean_waiting_time_s", "Waiting\ntime (s)", -1),
        ("mean_time_loss_s", "Delay\n(s)", -1),
        ("p95_time_loss_s", "Delay\np95 (s)", -1),
        ("mean_queue", "Queue\n(veh)", -1),
        ("mean_stops", "Stops /\nveh", -1),
        ("co2_g_per_vehicle", "CO$_2$\n(g/veh)", -1),
        ("ev_mean_waiting_time_s", "EV stop.\ndelay (s)", -1),
        ("ev_p95_waiting_time_s", "EV delay\np95 (s)", -1),
        ("ev_max_waiting_time_s", "EV worst\nwait (s)", -1),
        ("ev_mean_stops", "EV stops\n/ veh", -1),
        ("ev_zero_stop_rate", "EV clean\npass", +1),
    ]
    M = np.zeros((len(order), len(metrics)))
    raw = np.zeros_like(M)
    for j, (k, _, sgn) in enumerate(metrics):
        col = np.array([med(c, k) for c in order])
        raw[:, j] = col
        lo, hi = col.min(), col.max()
        z = (col - lo) / (hi - lo) if hi > lo else np.zeros_like(col)
        M[:, j] = z if sgn < 0 else 1.0 - z      # 0 = best, 1 = worst

    fig, ax = plt.subplots(figsize=(TW, 1.90))
    im = ax.imshow(M, cmap=SEQ, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[1] for m in metrics], fontsize=6.4, linespacing=1.2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABELS[c] for c in order], fontsize=7.0)
    for i in range(len(order)):
        for j in range(len(metrics)):
            v = raw[i, j]
            txt = f"{v:.2f}" if v < 2 else (f"{v:.1f}" if v < 100 else f"{v:.0f}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=5.7,
                    color="white" if M[i, j] > 0.55 else INK)
    ax.set_xticks(np.arange(-.5, len(metrics), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(order), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.0)
    ax.tick_params(which="minor", length=0)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="both", length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)
    for lbl in ax.get_yticklabels():
        if "ours" in lbl.get_text():
            lbl.set_color(HILITE); lbl.set_fontweight("bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.012, ticks=[0.04, 0.96])
    cb.ax.set_yticklabels(["best", "worst"], fontsize=6.6)
    cb.outline.set_visible(False)
    ax.axvline(5.5, color=INK, lw=1.0)
    # The group headers sit above the two-line column labels. Placing them at a
    # fixed data-y collided with those labels once the figure got shorter, so
    # they are anchored in axes fraction: x in data, y above the axes.
    from matplotlib.transforms import blended_transform_factory
    tf = blended_transform_factory(ax.transData, ax.transAxes)
    for xc, s in ((2.5, "Background traffic"), (8.5, "Emergency vehicles")):
        ax.text(xc, 1.30, s, ha="center", va="bottom", fontsize=7.6,
                color=INK, weight="bold", transform=tf)
    save(fig, "fig_heatmap")


# ================================ 5. perception and the commitment rule
def fig_perception():
    """Detector, filter and the Bayes rule, in one 2x2 block.

    These four panels were three separate figures. At the LNNS page budget a
    caption costs about as much vertical space as a short panel, so merging
    figures that share a subject buys a third of a page for nothing.
    """
    D = json.loads((RESULTS / "detector_metrics.json").read_text())
    cm = np.array(D["confusion_matrix"], dtype=float)
    labels = [l.capitalize() for l in D["confusion_labels"]]
    row = cm / cm.sum(axis=1, keepdims=True)

    A = json.loads((RESULTS / "threshold_audit.json").read_text())
    # theta is None on epochs screened out before pricing, and unbounded when
    # the projected benefit is zero. Both are off the plot.
    A = [r for r in A if isinstance(r["theta"], (int, float))
         and np.isfinite(r["theta"]) and r["theta"] <= 1.0]
    q = np.array([r["queue_cross"] for r in A])
    th = np.array([r["theta"] for r in A])
    pp = np.array([r["p"] for r in A])
    cmt = np.array([bool(r["committed"]) for r in A])

    fig, axes = plt.subplots(2, 2, figsize=(TW, 2.46))

    # (a) confusion -------------------------------------------------------
    ax = axes[0, 0]
    im = ax.imshow(row, cmap=SEQ, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6.4, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6.4)
    # Five columns across a quarter of the text width leaves about 11pt per
    # cell: "0.02" at 4.9pt runs into its neighbour, so drop the leading zero
    # and print an exact zero as a dash.
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = row[i, j]
            s = "--" if v == 0 else f"{v:.2f}".lstrip("0")
            ax.text(j, i, s, ha="center", va="center",
                    fontsize=5.2, color="white" if v > 0.55 else INK)
    ax.set_xlabel("predicted", fontsize=7.1)
    ax.set_ylabel("true", fontsize=7.1)
    panel(ax, "(a) Detector confusion")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

    # (b) performance against loudness ------------------------------------
    ax = axes[0, 1]
    bins = D["by_snr"]
    x = np.arange(len(bins))
    ax.plot(x, [b["recall"] for b in bins], "-o", color=BLUE, lw=1.3, ms=3.2,
            zorder=3, label="siren detected")
    ax.plot(x, [b["class_accuracy"] for b in bins], "--s", color=ORANGE, lw=1.3,
            ms=3.2, zorder=3, label="vehicle class correct")
    ticks = ["$<\\!-20$", "$-20..\\!-10$", "$-10..\\!-5$", "$-5..0$",
             "$0..5$", "$5..15$", "$>\\!15$"]
    ax.set_xticks(x)
    ax.set_xticklabels(ticks[:len(bins)], fontsize=5.9, rotation=35, ha="right")
    ax.set_xlabel("SNR at the mast (dB)", fontsize=7.1)
    ax.set_ylabel("rate", fontsize=7.1)
    ax.set_ylim(0, 1.06)
    panel(ax, "(b) Against loudness")
    ax.legend(frameon=False, loc="lower right", fontsize=6.4)
    tidy(ax)

    # (c) the threshold follows the state ---------------------------------
    ax = axes[1, 0]
    ax.scatter(q[~cmt], th[~cmt], s=4, color=NEUTRAL, alpha=0.6, lw=0, zorder=2,
               label="declined")
    ax.scatter(q[cmt], th[cmt], s=5, color=BLUE, alpha=0.85, lw=0, zorder=3,
               label="corridor opened")
    ax.axhline(0.6, color=ORANGE, lw=1.0, ls=(0, (4, 2)), zorder=4)
    # Left of the line, not right: the legend sits in the upper-right corner.
    ax.text(q.max() * 0.02, 0.64, "fixed threshold 0.6", ha="left",
            fontsize=6.1, color=ORANGE, zorder=5)
    ax.set_xlabel("vehicles queued on conflicting approaches", fontsize=7.1)
    ax.set_ylabel(r"$\theta^{\star}=C/(\gamma B)$", fontsize=7.1)
    ax.set_ylim(-0.03, 1.02)
    panel(ax, "(c) Threshold vs. queue")
    # No legend box: the points fill this panel and every placement lands on
    # either the scatter or the threshold line. The two colours are named in
    # the caption instead, and they differ in lightness as well as hue.
    tidy(ax, grid="both")

    # (d) the Bayes rule as executed --------------------------------------
    ax = axes[1, 1]
    ax.scatter(th[~cmt], pp[~cmt], s=4, color=NEUTRAL, alpha=0.6, lw=0, zorder=2)
    ax.scatter(th[cmt], pp[cmt], s=5, color=BLUE, alpha=0.85, lw=0, zorder=3)
    lim = np.array([0.0, 1.0])
    ax.plot(lim, lim, color=INK, lw=0.8, zorder=4)
    ax.fill_between(lim, lim, 1.02, color=BLUE, alpha=0.07, zorder=1)
    ax.text(0.56, 0.50, r"$p=\theta^{\star}$", fontsize=6.6, color=INK,
            rotation=36)
    ax.text(0.08, 0.90, "act", fontsize=7.1, color=BLUE, weight="bold")
    ax.text(0.70, 0.05, "decline", fontsize=7.1, color=MUTED, weight="bold")
    ax.set_xlabel(r"derived threshold $\theta^{\star}$", fontsize=7.1)
    ax.set_ylabel("calibrated posterior $p$", fontsize=7.1)
    ax.set_xlim(-0.03, 1.02)
    ax.set_ylim(-0.03, 1.02)
    panel(ax, "(d) The rule as executed")
    tidy(ax, grid="both")

    # The top row carries rotated tick labels and the bottom row carries panel
    # titles, so the two rows need more clearance than the columns do.
    fig.subplots_adjust(wspace=0.55, hspace=1.35)
    save(fig, "fig_perception")


# ============================ 6. ablation, degradation and the operating point
def fig_ablation():
    fig, axes = plt.subplots(1, 3, figsize=(TW, 1.30),
                             gridspec_kw=dict(width_ratios=[1.3, 0.95, 1.05]))

    # --- (a) what each component is worth
    ax = axes[0]
    base_ev = med("siren_arc", "ev_mean_waiting_time_s")
    base_bg = med("siren_arc", "mean_time_loss_s")
    rows = [("arc_myopic", "act on sight"),
            ("arc_nocal", "no calibration"),
            ("arc_indep", "independent filters"),
            ("arc_greedy", "greedy conflicts"),
            ("base_arc", "no priority layer"),
            ("base_no_sensing", "no acoustic sensing")]
    y = np.arange(len(rows))
    dev = [med(c, "ev_mean_waiting_time_s") - base_ev for c, _ in rows]
    dbg = [med(c, "mean_time_loss_s") - base_bg for c, _ in rows]
    ax.barh(y - 0.19, dev, 0.36, color=ORANGE, zorder=3, label="emergency delay")
    ax.barh(y + 0.19, dbg, 0.36, color=BLUE, zorder=3, label="background delay")
    for yi, (a, b) in enumerate(zip(dev, dbg)):
        ax.text(a + (0.12 if a >= 0 else -0.12), yi - 0.19, f"{a:+.1f}",
                va="center", ha="left" if a >= 0 else "right", fontsize=6.1,
                color=INK2)
        ax.text(b + (0.12 if b >= 0 else -0.12), yi + 0.19, f"{b:+.1f}",
                va="center", ha="left" if b >= 0 else "right", fontsize=6.1,
                color=INK2)
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(y); ax.set_yticklabels([r[1] for r in rows], fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlabel("change in median delay (s)", fontsize=7.3)
    ax.set_xlim(-1.8, max(dev + dbg) * 1.45)
    ax.set_ylim(len(rows) - 0.4, -1.35)
    panel(ax, "(a) Removing one component")
    ax.legend(frameon=False, loc="upper right", fontsize=6.5, handlelength=1.0,
              handletextpad=0.3, borderpad=0.0, labelspacing=0.2)
    tidy(ax, grid="x")

    # --- (b) degradation as the sensor fails
    ax = axes[1]
    R = json.loads((RESULTS / "robustness.json").read_text())["results"]
    names = {"nominal": "4 masts", "noise+5": "floor $+5$ dB",
             "noise+10": "floor $+10$ dB", "noise+15": "floor $+15$ dB",
             "mic1": "1 mast dead", "mic2": "2 masts dead",
             "deaf": "no sensing"}
    keys = [k for k in names if k in R]
    if not keys:
        raise KeyError(f"robustness.json has {sorted(R)}, expected {sorted(names)}")
    ev = [R[k]["aggregate"]["ev_delay_per_vehicle_s_mean"] for k in keys]
    err = [R[k]["aggregate"].get("ev_delay_per_vehicle_s_ci95", 0.0) for k in keys]
    cols = [HILITE if k == "nominal" else (ORANGE if k == "deaf" else NEUTRAL)
            for k in keys]
    x = np.arange(len(keys))
    ax.bar(x, ev, 0.72, color=cols, zorder=3, yerr=err,
           error_kw=dict(ecolor=INK2, lw=0.5, capsize=1.1))
    ax.set_xticks(x)
    ax.set_xticklabels([names[k] for k in keys], fontsize=6.3, rotation=55,
                       ha="right")
    ax.set_ylabel("EV stopped delay (s)", fontsize=7.3)
    panel(ax, "(b) Losing the sensor")
    tidy(ax)

    # --- (c) the operating point, swept by the exchange rate gamma.
    # Drawn against gamma rather than as a plane: both series are in seconds,
    # so one axis is honest, and a connected scatter in the (delay, EV delay)
    # plane would invite the reader to see a frontier the data cannot support
    # -- past about gamma = 10 the movement is inside seed noise.
    ax = axes[2]
    V = json.loads((RESULTS / "value_sweep.json").read_text())
    rows_g = sorted(V["rows"], key=lambda r: r["gamma"])
    x = np.arange(len(rows_g))
    ax.plot(x, [r["ev_delay_median"] for r in rows_g], "-o", color=ORANGE,
            lw=1.2, ms=3.0, zorder=3, label="emergency")
    ax.plot(x, [r["delay_median"] for r in rows_g], "--s", color=BLUE, lw=1.2,
            ms=3.0, zorder=3, label="background")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['gamma']:g}" for r in rows_g], fontsize=6.1)
    ax.set_xlabel(r"exchange rate $\gamma$", fontsize=7.3)
    ax.set_ylabel("median delay (s)", fontsize=7.3)
    ax.set_ylim(6, 30)
    ax.annotate("never\npreempts", (0, 15.35), (0.35, 21.5), fontsize=6.1,
                color=MUTED, linespacing=1.2,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.6))
    panel(ax, r"(c) Sweeping $\gamma$")
    ax.legend(frameon=False, loc="lower right", fontsize=6.3, handlelength=1.2,
              handletextpad=0.3, borderpad=0.0, labelspacing=0.2)
    tidy(ax)
    fig.subplots_adjust(wspace=0.50)
    save(fig, "fig_ablation")


FIGS = {
    "architecture": fig_architecture,
    "sumo": fig_sumo,
    "controllers": fig_controllers,
    "heatmap": fig_heatmap,
    "perception": fig_perception,
    "ablation": fig_ablation,
}

if __name__ == "__main__":
    only = sys.argv[1:] or None
    for name, fn in FIGS.items():
        if only and name not in only:
            continue
        print(f"[{name}]")
        fn()
    print(f"figures -> {OUT}")
