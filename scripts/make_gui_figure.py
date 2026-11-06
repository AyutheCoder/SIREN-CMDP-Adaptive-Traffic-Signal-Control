"""Compose the SUMO-GUI figure from frames captured by ``scripts/demo.py``.

The two panels are captured on the same seed and the same demand, and differ
only in whether the priority layer is running:

    python scripts/demo.py --priority none --screenshot-at 65 \
        --screenshot-tag nopriority --seed 2029 --view-radius 150 ...
    python scripts/demo.py --screenshot-ev --follow-ev --seed 2029 \
        --view-radius 150 ...

Each frame is cropped to the junction before it is placed, because the SUMO
view is framed by a boundary in metres and leaves a wide margin of background
that carries no information and, at figure scale, shrinks the part that does.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "figures" / "demo"
OUT = ROOT / "figures" / "fig_sumo_gui.pdf"

# Fraction of the frame kept around the centre. The SUMO view is much wider
# than it is tall, so the horizontal crop is the tighter of the two.
CROP_X = (0.19, 0.88)
CROP_Y = (0.13, 0.93)


def load(name: str) -> Image.Image:
    path = DEMO / name
    if not path.exists():
        sys.exit(f"missing frame: {path}\nRun scripts/demo.py first (see module docstring).")
    img = Image.open(path).convert("RGB")
    w, h = img.size
    return img.crop((int(CROP_X[0] * w), int(CROP_Y[0] * h),
                     int(CROP_X[1] * w), int(CROP_Y[1] * h)))


def main() -> int:
    top = load("nopriority_t0065s.png")
    bottom = load("corridor_E_65s.png")

    # Generated at the width it is placed at (0.62\textwidth of an IEEE page),
    # so the panel labels render at their stated point size instead of being
    # scaled down by LaTeX.
    fig, axes = plt.subplots(2, 1, figsize=(4.4, 2.9))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, hspace=0.14)

    panels = [
        (axes[0], top, "(a)  priority layer disabled"),
        (axes[1], bottom, "(b)  SIREN-ARC, corridor committed"),
    ]

    for ax, img, title in panels:
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("0.35")
            spine.set_linewidth(0.8)
        ax.set_title(title, fontsize=8.5, fontweight="bold", loc="left", pad=2.0)

    fig.savefig(OUT, bbox_inches="tight", dpi=300)
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
