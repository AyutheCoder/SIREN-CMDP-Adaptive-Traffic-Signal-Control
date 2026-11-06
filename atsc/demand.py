"""Stochastic traffic demand generation, including emergency-vehicle injection.

Routes are computed directly from the generated topology (BFS over the node
graph), so the same generator serves both the single intersection and the grid
networks without an external router.

Design note on realism vs. attribution
-------------------------------------
SUMO ships a ``bluelight`` device that makes ordinary traffic pull aside for an
emergency vehicle. It is deliberately **disabled by default** here: with it on,
part of the measured emergency-vehicle delay reduction would come from driver
yielding rather than from signal control, confounding the quantity under study.
Set ``bluelight=True`` to enable it for a realism ablation.
"""

from __future__ import annotations

import json
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .config import DemandConfig
from .netgen import NetworkBuilder, turn_targets

VEHICLE_TYPES = """    <vType id="car" vClass="passenger" accel="2.6" decel="4.5" sigma="{sigma}"
           length="5.0" minGap="2.5" maxSpeed="{vmax:.2f}" tau="1.0"
           speedFactor="normc(1.00,0.10,0.85,1.15)" carFollowModel="Krauss"
           emissionClass="HBEFA3/PC_G_EU4" color="0.85,0.85,0.9"/>
    <vType id="ambulance" vClass="emergency" accel="2.9" decel="5.5" sigma="0.1"
           length="6.5" minGap="2.5" maxSpeed="{ev_vmax:.2f}" tau="0.8"
           speedFactor="normc(1.25,0.05,1.15,1.35)" guiShape="emergency"
           emissionClass="HBEFA3/LDV_G_EU4" color="1,1,1"{bluelight}
    <vType id="fire" vClass="emergency" accel="2.2" decel="5.0" sigma="0.1"
           length="9.0" minGap="3.0" maxSpeed="{ev_vmax:.2f}" tau="0.9"
           speedFactor="normc(1.15,0.05,1.05,1.25)" guiShape="firebrigade"
           emissionClass="HBEFA3/HDV_D_EU4" color="1,0.1,0.1"{bluelight}
    <vType id="police" vClass="emergency" accel="3.2" decel="6.0" sigma="0.1"
           length="5.2" minGap="2.5" maxSpeed="{ev_vmax:.2f}" tau="0.7"
           speedFactor="normc(1.30,0.05,1.20,1.40)" guiShape="police"
           emissionClass="HBEFA3/PC_G_EU4" color="0.1,0.2,1"{bluelight}
"""

_PLAIN_END = "/>"
"""Close an emergency ``vType`` that carries no child element."""

_BLUELIGHT = """>
        <param key="has.bluelight.device" value="true"/>
    </vType>"""
"""Close it around a ``bluelight`` device instead.

The closing delimiter lives here, in both branches, rather than in the template.
While the template ended ``{bluelight}/>`` the device branch emitted
``</vType/>``, which SUMO rejects outright with "unterminated end tag" -- so
``bluelight=True`` had never once produced a runnable scenario.
"""


@dataclass
class EmergencyVehicle:
    """Ground-truth record for one injected emergency vehicle."""

    veh_id: str
    veh_class: str
    priority: float
    depart_s: float
    route_edges: Tuple[str, ...]
    entry_edge: str
    junctions: Tuple[str, ...]

    def to_dict(self) -> Dict:
        return {
            "veh_id": self.veh_id,
            "veh_class": self.veh_class,
            "priority": self.priority,
            "depart_s": self.depart_s,
            "route_edges": list(self.route_edges),
            "entry_edge": self.entry_edge,
            "junctions": list(self.junctions),
        }


@dataclass
class Scenario:
    """A generated demand instance plus the metadata needed to score it."""

    route_file: Path
    emergency_vehicles: Dict[str, EmergencyVehicle]
    n_background: int
    config: DemandConfig
    meta_file: Optional[Path] = None
    """Sidecar JSON recording the emergency-vehicle ground truth for this run."""

    @property
    def n_emergency(self) -> int:
        return len(self.emergency_vehicles)


class DemandGenerator:
    def __init__(self, builder: NetworkBuilder, cfg: DemandConfig):
        self.builder = builder
        self.cfg = cfg
        self._adj = self._build_adjacency()
        self._entry_borders = self._collect_entry_borders()
        self._route_cache: Dict[Tuple[str, str], Optional[Tuple[str, ...]]] = {}

    # ------------------------------------------------------------------ routing
    def _build_adjacency(self) -> Dict[str, List[str]]:
        adj: Dict[str, List[str]] = {}
        for jid, neigh in self.builder.neighbour.items():
            adj.setdefault(jid, [])
            for other in neigh.values():
                adj[jid].append(other)
                adj.setdefault(other, [])
                if jid not in adj[other]:
                    adj[other].append(jid)
        return adj

    def _collect_entry_borders(self) -> List[str]:
        return sorted(self.builder.border_pos.keys())

    def _node_path(self, src: str, dst: str) -> Optional[List[str]]:
        if src == dst:
            return None
        queue = deque([[src]])
        seen = {src}
        while queue:
            path = queue.popleft()
            node = path[-1]
            for nxt in self._adj.get(node, []):
                if nxt in seen:
                    continue
                # Border nodes are terminals: never route *through* one.
                if nxt.startswith("B") and nxt != dst:
                    continue
                if nxt == dst:
                    return path + [nxt]
                seen.add(nxt)
                queue.append(path + [nxt])
        return None

    def route_edges(self, src: str, dst: str) -> Optional[Tuple[str, ...]]:
        key = (src, dst)
        if key in self._route_cache:
            return self._route_cache[key]
        path = self._node_path(src, dst)
        edges: Optional[Tuple[str, ...]] = None
        if path is not None and len(path) >= 2:
            edges = tuple(
                self.builder.edge_id(path[i], path[i + 1]) for i in range(len(path) - 1)
            )
        self._route_cache[key] = edges
        return edges

    def junctions_on_route(self, edges: Sequence[str]) -> Tuple[str, ...]:
        out: List[str] = []
        for eid in edges:
            for jid in self.builder.junctions:
                if eid.endswith(f"to{jid}") and jid not in out:
                    out.append(jid)
        return tuple(out)

    # ------------------------------------------------------------- demand shape
    def _approach_weight(self, border: str, t: float) -> float:
        """Per-approach arrival intensity multiplier at simulation time ``t``."""
        direction = border.rsplit("_", 1)[-1]
        horizon = max(self.cfg.horizon_s, 1)
        frac = t / horizon
        regime = self.cfg.regime

        if regime == "balanced":
            return 1.0
        if regime == "asymmetric":
            return 1.6 if direction in ("E", "W") else 0.7
        if regime == "peak":
            # Smooth build-up to a mid-horizon peak and back down.
            return 0.55 + 1.05 * math.sin(math.pi * frac) ** 2
        if regime == "surge":
            # Sudden sustained burst on the N-S axis in the middle third.
            in_surge = 0.35 <= frac <= 0.65
            if direction in ("N", "S"):
                return 2.4 if in_surge else 0.8
            return 0.9
        raise ValueError(f"unknown demand regime: {regime!r}")

    def _sample_exit(self, entry_border: str, rng: random.Random) -> Optional[str]:
        """Pick a destination border consistent with the configured turn ratios."""
        entry_dir = entry_border.rsplit("_", 1)[-1]
        targets = turn_targets(entry_dir)
        p_left, p_through, p_right = self.cfg.turn_ratios
        u = rng.random()
        if u < p_left:
            movement = "left"
        elif u < p_left + p_through:
            movement = "through"
        else:
            movement = "right"
        exit_dir = targets[movement]

        candidates = [b for b in self._entry_borders if b.rsplit("_", 1)[-1] == exit_dir]
        candidates = [b for b in candidates if b != entry_border]
        if not candidates:
            return None
        # Prefer the border reachable by the straightest path (matters on grids).
        rng.shuffle(candidates)
        for cand in candidates:
            if self.route_edges(entry_border, cand):
                return cand
        return None

    # -------------------------------------------------------- emergency timing
    def _ev_departure_times(self, rng: random.Random) -> List[float]:
        """When each emergency vehicle enters the network.

        Two placements from one rate; see ``DemandConfig.ev_schedule``. Both
        consume the generator's randomness *after* the background traffic has
        been drawn, so switching between them leaves ordinary demand for a
        given seed byte-identical and only the emergency arrivals move.
        """
        cfg = self.cfg
        horizon = int(cfg.horizon_s)
        rate_per_s = cfg.ev_rate_per_hour / 3600.0
        if cfg.ev_schedule == "poisson":
            return [float(t) for t in range(horizon) if rng.random() < rate_per_s]
        if cfg.ev_schedule != "spread":
            raise ValueError(f"unknown ev_schedule: {cfg.ev_schedule!r}")

        n = int(round(rate_per_s * horizon))
        if n <= 0 or horizon <= 0:
            return []
        # Interior placement: the gap is measured between arrivals, and half of
        # one sits at each end. Dividing the horizon by ``n - 1`` instead would
        # put the first vehicle at t=0, before the network has any traffic on
        # it to preempt, and the last one at the horizon, where it is cut off
        # mid-approach.
        gap = horizon / float(n)
        jitter = gap * max(cfg.ev_spread_jitter, 0.0)
        times: List[float] = []
        for i in range(n):
            t = (i + 0.5) * gap + rng.uniform(-jitter, jitter)
            times.append(float(min(max(t, 1.0), horizon - 1.0)))
        times.sort()
        return times

    # ------------------------------------------------------------------ writing
    def generate(self, path: Path, seed: Optional[int] = None, bluelight: bool = False) -> Scenario:
        cfg = self.cfg
        rng = random.Random(cfg.seed if seed is None else seed)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        entries: List[Tuple[float, str, str, Tuple[str, ...]]] = []  # depart, vtype, id, edges
        emergency: Dict[str, EmergencyVehicle] = {}

        per_approach_rate = cfg.base_flow_vph / 3600.0
        veh_nr = 0

        for t in range(cfg.horizon_s):
            for border in self._entry_borders:
                rate = per_approach_rate * self._approach_weight(border, t)
                if rng.random() >= rate:
                    continue
                exit_border = self._sample_exit(border, rng)
                if exit_border is None:
                    continue
                edges = self.route_edges(border, exit_border)
                if not edges:
                    continue
                entries.append((float(t), "car", f"car_{veh_nr}", edges))
                veh_nr += 1

        # Emergency vehicles: a homogeneous Poisson process over the horizon,
        # or the evenly-spread placement the demonstration uses.
        ev_nr = 0
        classes = list(cfg.ev_class_weights.keys())
        weights = [cfg.ev_class_weights[c] for c in classes]
        for t in self._ev_departure_times(rng):
            border = rng.choice(self._entry_borders)
            exit_border = self._sample_exit(border, rng)
            if exit_border is None:
                continue
            edges = self.route_edges(border, exit_border)
            if not edges:
                continue
            veh_class = rng.choices(classes, weights=weights, k=1)[0]
            vid = f"ev_{veh_class}_{ev_nr}"
            ev_nr += 1
            entries.append((float(t), veh_class, vid, edges))
            emergency[vid] = EmergencyVehicle(
                veh_id=vid,
                veh_class=veh_class,
                priority=cfg.ev_priority.get(veh_class, 1.0),
                depart_s=float(t),
                route_edges=edges,
                entry_edge=edges[0],
                junctions=self.junctions_on_route(edges),
            )

        entries.sort(key=lambda item: (item[0], item[2]))

        vtypes = VEHICLE_TYPES.format(
            sigma=cfg.car_following_sigma,
            vmax=13.89,
            ev_vmax=19.5,
            bluelight=_BLUELIGHT if bluelight else _PLAIN_END,
        )

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
            ' xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">',
            vtypes.rstrip(),
            "",
        ]
        for depart, vtype, vid, edges in entries:
            lines.append(
                f'    <vehicle id="{vid}" type="{vtype}" depart="{depart:.2f}" '
                f'departSpeed="{cfg.depart_speed}" departLane="best">'
            )
            lines.append(f'        <route edges="{" ".join(edges)}"/>')
            lines.append("    </vehicle>")
        lines.append("</routes>")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "n_background": len(entries) - len(emergency),
                    "n_emergency": len(emergency),
                    "regime": cfg.regime,
                    "seed": cfg.seed if seed is None else seed,
                    "emergency_vehicles": {k: v.to_dict() for k, v in emergency.items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return Scenario(
            route_file=path,
            emergency_vehicles=emergency,
            n_background=len(entries) - len(emergency),
            config=cfg,
            meta_file=meta_path,
        )


def write_sumocfg(
    path: Path,
    net_file: Path,
    route_file: Path,
    step_length: float = 1.0,
    teleport_s: int = 300,
    emissions: bool = False,
    additional_files: Optional[Sequence[Path]] = None,
) -> Path:
    """Write a minimal ``.sumocfg`` pointing at the generated network and routes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = ""
    if emissions:
        extra = '        <device.emissions.probability value="1.0"/>\n'
    add_line = ""
    if additional_files:
        joined = ",".join(Path(p).as_posix() for p in additional_files)
        add_line = f'        <additional-files value="{joined}"/>\n'
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
               xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="{net_file.as_posix()}"/>
        <route-files value="{route_file.as_posix()}"/>
{add_line}    </input>
    <time>
        <step-length value="{step_length}"/>
    </time>
    <processing>
        <time-to-teleport value="{teleport_s}"/>
        <collision.action value="warn"/>
        <lateral-resolution value="0.8"/>
{extra}    </processing>
    <report>
        <xml-validation value="never"/>
        <duration-log.disable value="true"/>
        <no-step-log value="true"/>
        <no-warnings value="true"/>
    </report>
</configuration>
""",
        encoding="utf-8",
    )
    return path
