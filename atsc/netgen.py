"""Programmatic SUMO network generation.

Builds a parameterised signalised network (a single 4-arm intersection, or an
R x C grid of them) from plain node/edge/connection XML and lets ``netconvert``
compile it. Generating the network rather than shipping a fixed ``.net.xml``
means the geometry that produced every reported number is reproducible from the
config alone, and the same code yields the corridor networks used in the
multi-intersection experiment.

Lane assignment per approach (SUMO numbers lanes right-to-left, lane 0 = kerb):

    lane 0  -> through + right turn
    lane 1  -> through
    lane 2  -> protected left turn

With ``lanes_per_approach`` other than 3 the assignment degrades gracefully:
the leftmost lane always takes the left turn and lane 0 always takes the right.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

from sumolib import checkBinary

from .config import NetworkConfig

# Direction unit vectors in (x, y).
DIRECTIONS: Dict[str, Tuple[int, int]] = {
    "N": (0, 1),
    "E": (1, 0),
    "S": (0, -1),
    "W": (-1, 0),
}
_VEC_TO_DIR = {v: k for k, v in DIRECTIONS.items()}


def opposite(direction: str) -> str:
    dx, dy = DIRECTIONS[direction]
    return _VEC_TO_DIR[(-dx, -dy)]


def rotate_ccw(direction: str) -> str:
    dx, dy = DIRECTIONS[direction]
    return _VEC_TO_DIR[(-dy, dx)]


def rotate_cw(direction: str) -> str:
    dx, dy = DIRECTIONS[direction]
    return _VEC_TO_DIR[(dy, -dx)]


def turn_targets(entry_direction: str) -> Dict[str, str]:
    """Exit directions for a vehicle that *entered* from ``entry_direction``.

    A vehicle entering from the north is heading south, so its through movement
    leaves to the south, its left to the east and its right to the west.
    """
    heading = opposite(entry_direction)
    return {
        "through": heading,
        "left": rotate_ccw(heading),
        "right": rotate_cw(heading),
    }


class NetworkBuilder:
    """Writes plain XML and invokes ``netconvert``."""

    def __init__(self, cfg: NetworkConfig, out_dir: Path):
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.spacing = cfg.arm_length_m

        self.junctions: List[str] = []
        self.junction_pos: Dict[str, Tuple[float, float]] = {}
        self.border_pos: Dict[str, Tuple[float, float]] = {}
        # junction -> entry direction -> upstream node id
        self.neighbour: Dict[str, Dict[str, str]] = {}
        self._layout()

    # ------------------------------------------------------------------ layout
    def _layout(self) -> None:
        rows, cols = self.cfg.rows, self.cfg.cols
        for r in range(rows):
            for c in range(cols):
                jid = f"J{r}_{c}"
                self.junctions.append(jid)
                self.junction_pos[jid] = (c * self.spacing, (rows - 1 - r) * self.spacing)

        grid_index = {f"J{r}_{c}": (r, c) for r in range(rows) for c in range(cols)}
        step = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}

        for jid in self.junctions:
            r, c = grid_index[jid]
            self.neighbour[jid] = {}
            jx, jy = self.junction_pos[jid]
            for d, (dr, dc) in step.items():
                nr, nc = r + dr, c + dc
                nid = f"J{nr}_{nc}"
                if 0 <= nr < rows and 0 <= nc < cols:
                    self.neighbour[jid][d] = nid
                else:
                    bid = f"B{r}_{c}_{d}"
                    dx, dy = DIRECTIONS[d]
                    self.border_pos[bid] = (jx + dx * self.spacing, jy + dy * self.spacing)
                    self.neighbour[jid][d] = bid

    def edge_id(self, frm: str, to: str) -> str:
        return f"{frm}to{to}"

    def incoming_edges(self, jid: str) -> Dict[str, str]:
        """Entry direction -> incoming edge id for a junction."""
        return {d: self.edge_id(up, jid) for d, up in self.neighbour[jid].items()}

    def outgoing_edges(self, jid: str) -> Dict[str, str]:
        """Exit direction -> outgoing edge id for a junction."""
        return {d: self.edge_id(jid, down) for d, down in self.neighbour[jid].items()}

    # -------------------------------------------------------------- xml writers
    def _write_nodes(self, path: Path) -> None:
        lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<nodes>"]
        for jid in self.junctions:
            x, y = self.junction_pos[jid]
            lines.append(
                f'    <node id="{jid}" x="{x:.2f}" y="{y:.2f}" '
                f'type="{self.cfg.junction_type}" tlType="static"/>'
            )
        for bid, (x, y) in sorted(self.border_pos.items()):
            lines.append(f'    <node id="{bid}" x="{x:.2f}" y="{y:.2f}" type="priority"/>')
        lines.append("</nodes>")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_edges(self, path: Path) -> None:
        n = self.cfg.lanes_per_approach
        v = self.cfg.speed_limit_ms
        lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<edges>"]
        seen = set()
        for jid in self.junctions:
            for d, other in self.neighbour[jid].items():
                for frm, to in ((other, jid), (jid, other)):
                    eid = self.edge_id(frm, to)
                    if eid in seen:
                        continue
                    seen.add(eid)
                    lines.append(
                        f'    <edge id="{eid}" from="{frm}" to="{to}" '
                        f'numLanes="{n}" speed="{v:.2f}" priority="3"/>'
                    )
        lines.append("</edges>")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_connections(self, path: Path) -> None:
        n = self.cfg.lanes_per_approach
        left_lane = n - 1
        lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<connections>"]
        for jid in self.junctions:
            inc = self.incoming_edges(jid)
            out = self.outgoing_edges(jid)
            for entry_dir, in_edge in inc.items():
                targets = turn_targets(entry_dir)
                through_edge = out[targets["through"]]
                left_edge = out[targets["left"]]
                right_edge = out[targets["right"]]

                if self.cfg.left_turn_lane and n >= 2:
                    lines.append(
                        f'    <connection from="{in_edge}" to="{left_edge}" '
                        f'fromLane="{left_lane}" toLane="{min(left_lane, n - 1)}"/>'
                    )
                    through_lanes = list(range(0, left_lane))
                else:
                    through_lanes = list(range(0, n))
                    lines.append(
                        f'    <connection from="{in_edge}" to="{left_edge}" '
                        f'fromLane="{left_lane}" toLane="{n - 1}"/>'
                    )

                for lane in through_lanes:
                    lines.append(
                        f'    <connection from="{in_edge}" to="{through_edge}" '
                        f'fromLane="{lane}" toLane="{lane}"/>'
                    )
                if self.cfg.right_turn_on_lane0:
                    lines.append(
                        f'    <connection from="{in_edge}" to="{right_edge}" '
                        f'fromLane="0" toLane="0"/>'
                    )
        lines.append("</connections>")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------- build
    def build(self, force: bool = False) -> Path:
        net_path = self.out_dir / f"{self.cfg.name}.net.xml"
        if net_path.exists() and not force:
            return net_path

        nod = self.out_dir / f"{self.cfg.name}.nod.xml"
        edg = self.out_dir / f"{self.cfg.name}.edg.xml"
        con = self.out_dir / f"{self.cfg.name}.con.xml"
        self._write_nodes(nod)
        self._write_edges(edg)
        self._write_connections(con)

        netconvert = checkBinary("netconvert")
        cmd = [
            netconvert,
            "--node-files", str(nod),
            "--edge-files", str(edg),
            "--connection-files", str(con),
            "--output-file", str(net_path),
            "--tls.default-type", "static",
            "--tls.left-green.time", "6",
            "--tls.yellow.time", "3",
            "--tls.allred.time", "2",
            "--tls.green.time", "31",
            "--no-turnarounds", "true",
            "--junctions.corner-detail", "5",
            "--rectangular-lane-cut", "true",
            "--no-warnings", "true",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not net_path.exists():
            raise RuntimeError(
                "netconvert failed:\n"
                f"  cmd: {' '.join(cmd)}\n"
                f"  stdout: {result.stdout}\n"
                f"  stderr: {result.stderr}"
            )
        return net_path


def build_network(cfg: NetworkConfig, out_dir: Path, force: bool = False) -> Tuple[Path, NetworkBuilder]:
    builder = NetworkBuilder(cfg, out_dir)
    return builder.build(force=force), builder


def write_actuated_program(
    net_file: Path,
    out_path: Path,
    program_id: str = "actuated",
    min_dur: float = 8.0,
    max_dur: float = 55.0,
    max_gap: float = 3.0,
) -> Path:
    """Emit a SUMO ``actuated`` variant of the network's own signal program.

    Using SUMO's native gap-out logic rather than a hand-rolled imitation makes
    the vehicle-actuated baseline a genuinely competitive comparison instead of
    a straw man: it is the same controller a practitioner would deploy, with the
    same green bounds this work's shield enforces.
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(str(net_file)).getroot()
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<additional>"]
    for tl in root.findall("tlLogic"):
        tid = tl.get("id")
        lines.append(
            f'    <tlLogic id="{tid}" type="actuated" programID="{program_id}" offset="0">'
        )
        lines.append(f'        <param key="max-gap" value="{max_gap}"/>')
        lines.append('        <param key="detector-gap" value="2.0"/>')
        lines.append('        <param key="show-detectors" value="false"/>')
        for ph in tl.findall("phase"):
            state = ph.get("state", "")
            duration = float(ph.get("duration", "5"))
            is_green = any(c in "Gg" for c in state) and "y" not in state.lower()
            if is_green:
                lines.append(
                    f'        <phase duration="{max_dur:g}" minDur="{min_dur:g}" '
                    f'maxDur="{max_dur:g}" state="{state}"/>'
                )
            else:
                lines.append(f'        <phase duration="{duration:g}" state="{state}"/>')
        lines.append("    </tlLogic>")
    lines.append("</additional>")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
