"""Distinct on-map badges for each emergency-vehicle class.

SUMO already draws the three ``guiShape`` values used here -- ``emergency``,
``firebrigade`` and ``police`` -- as different vehicle bodies, but at the zoom
level from which a whole junction is legible those bodies are about eight
pixels long and all three read as "a small box". Whether the vehicle the
controller just gave a corridor to was an ambulance or a police car is exactly
the thing a viewer wants to know, because the triage weight it carries is what
the scheduler used to rank it.

So each vehicle also gets a badge rendered above it as a SUMO point of
interest, whose image is generated here rather than shipped as a binary: a
white cross on red for the ambulance, a flame on amber for the fire appliance,
and a chequered band with a beacon for police. The badge follows the vehicle
every simulation step and is removed when it leaves the network.

The images are pure matplotlib and written once to ``figures/icons``; nothing
downloads and nothing needs to be committed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, PathPatch, Polygon, Rectangle

CLASS_COLOUR: Dict[str, tuple] = {
    "ambulance": (235, 30, 40, 255),
    "fire": (255, 120, 0, 255),
    "police": (30, 90, 235, 255),
}
"""Primary livery colour per class, as SUMO RGBA."""

CLASS_FLASH: Dict[str, tuple] = {
    "ambulance": (255, 255, 255, 255),
    "fire": (255, 230, 60, 255),
    "police": (235, 30, 40, 255),
}
"""The colour each class alternates *to*, once per simulated second, so the
vehicle reads as running lights rather than as a painted box. Police alternate
blue and red, which is what a beacon bar actually does; the ambulance flashes
white and the fire appliance amber."""

CLASS_LABEL: Dict[str, str] = {
    "ambulance": "Ambulance",
    "fire": "Fire appliance",
    "police": "Police",
}


def _canvas(size_px: int = 96):
    fig = plt.figure(figsize=(size_px / 96, size_px / 96), dpi=96)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    return fig, ax


def _disc(ax, rgb, edge="white"):
    ax.add_patch(Circle((0.5, 0.5), 0.47, facecolor=rgb, edgecolor=edge, lw=3.0))


def _ambulance(ax) -> None:
    _disc(ax, "#EB1E28")
    ax.add_patch(Rectangle((0.42, 0.18), 0.16, 0.64, facecolor="white", lw=0))
    ax.add_patch(Rectangle((0.18, 0.42), 0.64, 0.16, facecolor="white", lw=0))


def _fire(ax) -> None:
    _disc(ax, "#FF7800")
    # A flame drawn with cubics: a straight-edged polygon at this size reads as
    # an arrowhead, which is exactly the wrong thing for a traffic display.
    C4 = MplPath.CURVE4
    verts = [(0.50, 0.88),
             (0.30, 0.66), (0.44, 0.60), (0.40, 0.46),   # left shoulder in
             (0.36, 0.30), (0.30, 0.24), (0.32, 0.16),   # left lobe down
             (0.40, 0.10), (0.60, 0.10), (0.68, 0.16),   # round the base
             (0.72, 0.24), (0.64, 0.34), (0.62, 0.48),   # right lobe up
             (0.60, 0.64), (0.66, 0.68), (0.50, 0.88)]   # back to the tip
    codes = [MplPath.MOVETO] + [C4] * 15
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor="white", lw=0))
    # An inner tongue, so the flame still reads as fire when it is 10 px wide.
    inner = [(0.50, 0.55),
             (0.42, 0.44), (0.44, 0.36), (0.50, 0.26),
             (0.56, 0.36), (0.58, 0.44), (0.50, 0.55)]
    ax.add_patch(PathPatch(MplPath(inner, [MplPath.MOVETO] + [C4] * 6),
                           facecolor="#FF7800", lw=0))


def _police(ax) -> None:
    _disc(ax, "#1E5AEB")
    # Beacon bar: red one side, white the other, which is what the light bar
    # on the roof actually looks like and is legible at eight pixels.
    ax.add_patch(Rectangle((0.20, 0.60), 0.30, 0.18, facecolor="#EB1E28", lw=0))
    ax.add_patch(Rectangle((0.50, 0.60), 0.30, 0.18, facecolor="#FFFFFF", lw=0))
    # Chequered band beneath it: the internationally recognisable half of the
    # livery. Two rows of four, kept well inside the disc.
    for row in range(2):
        for col in range(4):
            dark = (row + col) % 2 == 0
            ax.add_patch(Rectangle((0.22 + col * 0.14, 0.24 + row * 0.14),
                                   0.14, 0.14,
                                   facecolor="#111111" if dark else "white", lw=0))


_DRAW = {"ambulance": _ambulance, "fire": _fire, "police": _police}


def write_icons(out_dir: Path, size_px: int = 96) -> Dict[str, Path]:
    """Render one badge per class and return ``{class: png path}``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for name, draw in _DRAW.items():
        path = out_dir / f"ev_{name}.png"
        if not path.exists():
            fig, ax = _canvas(size_px)
            draw(ax)
            fig.savefig(path, transparent=True, dpi=96)
            plt.close(fig)
        paths[name] = path
    return paths


def ev_class_of(veh_id: str, scenario=None) -> str:
    """Which class a vehicle belongs to, from the scenario or from its id."""
    if scenario is not None:
        ev = getattr(scenario, "emergency_vehicles", {}).get(veh_id)
        if ev is not None:
            return ev.veh_class
    for name in _DRAW:
        if name in veh_id:
            return name
    return "ambulance"


if __name__ == "__main__":  # pragma: no cover
    from ..config import FIGURES_DIR
    for k, v in write_icons(FIGURES_DIR / "icons").items():
        print(f"{k:12s} -> {v}")
