"""Clearing the road ahead of an emergency vehicle, by distance.

What this is, and what it is not
--------------------------------
Signal preemption opens the *junction*. It does nothing about the cars standing
between the emergency vehicle and the stop line, and on the demonstration seed
the corridor was green on 97% of the epochs it was engaged for while the
emergency vehicles still lost twelve seconds each -- a green they cannot reach
is not a green. So this module is the other half: the actuation on ordinary
traffic that a real deployment gets from drivers hearing the siren, and that
SUMO's own ``bluelight`` device models by making everyone dive for the kerb.

It is written separately from that device for one reason: ``bluelight``
deadlocks this intersection outright on some seeds (measured, 1 run in 6 went
from 24 s of delay to 213 s with a 95-vehicle queue), because it makes traffic
yield *regardless of whether there is anywhere to yield to*. This does
something narrower and reversible.

What it is measured to do
-------------------------
**It does not reduce emergency delay on this network, and the honest number is
here rather than in a footnote.** Over twelve seeds at 24 emergency vehicles an
hour, mean stopped delay per emergency vehicle was 15.5 s without it and 15.6 s
with it -- a paired difference of +0.1 s with a standard error of 2.3, which is
no effect at all rather than a small one. Background delay was likewise
unchanged (mean queue 10.9 against 10.7 veh), no run gridlocked, and the
shield's starvation bound held everywhere, so it is not *harmful*; it simply is
not where the delay is.

The reason is geometry, and it is worth stating because it bounds where this
rule would help. Each approach here is three lanes carrying 700 veh/h, so the
queue standing in any one lane in front of an arriving emergency vehicle is
typically two or three cars, and a lane change in the sublane model takes about
as long as those cars take to discharge once the corridor's green arrives. The
mechanism is real and visible; the margin it operates on is not. On a
single-lane approach, or at a saturation where the queue reaches back past the
detection range, the arithmetic would look different -- but that is a claim
about a network this study does not run, so it is left as a hypothesis.

What it *is* for, then, is the demonstration: it makes the "clear a path"
half of emergency priority visible on the map, next to the signal half, and it
is the ablation (``--no-clear-lane``) that shows how much of the behaviour is
which. It is off in every evaluation run and no reported number depends on it.

The rule
--------
While an emergency vehicle is within :attr:`trigger_m` of a stop line on a
controlled approach, every ordinary vehicle **in its lane, ahead of it, within
:attr:`ahead_m`** is asked to move to another lane of the same approach. Three
constraints keep it from doing harm:

* **Only sideways, never slower.** Nothing is stopped, held or slowed. The only
  command issued is a lane-change request, which SUMO executes if and only if it
  is safe; a vehicle with nowhere to go simply stays where it is.
* **Only to a lane its own route can use.** The target lane is checked against
  the vehicle's next edge through the junction's connection table, so a
  right-turning car is never pushed into a left-only lane, where it would stop
  dead and block the approach it was moved out of. This is the check
  ``bluelight`` does not make and is why that device jams.
* **Only for as long as the vehicle is coming.** The request carries a short
  duration and is refreshed per step while the emergency vehicle is inside the
  trigger range; the lane-change model is restored the moment it is not, so no
  vehicle is left driving under modified rules after the event.

The emergency vehicle itself is left in its own lane by default; see
:attr:`CorridorClearance.move_ev` for the measurement behind that.

Why a distance and not the belief
---------------------------------
The acoustic belief decides the *signal*, and it is deliberately the only thing
that does -- that is the claim the study makes. This layer is downstream of a
vehicle that has already been identified, and a deployment identifies it the
same way it dispatches it: the responding vehicle's own position, over V2X or
the CAD link. Driving it from the belief instead would mean recolouring traffic
on approaches where the filter is merely uncertain, which is both worse
behaviour and a different claim. It is off unless asked for, and it is not
enabled in any evaluation run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

DEFAULT_LC_MODE = 1621
"""SUMO's default lane-change mode, restored when a vehicle is released."""

CLEARING_LC_MODE = 1285
"""Lane-change mode while a vehicle is being moved out of the way.

Strategic and cooperative changes stay on -- a vehicle still has to be able to
reach the lane its route needs -- while "keep right" and "speed gain" are
switched off, so the two models that would immediately undo the request are the
two that are disabled. Collision avoidance is untouched.
"""


@dataclass
class CorridorClearance:
    """Move ordinary traffic out of an approaching emergency vehicle's lane."""

    env: object
    enabled: bool = True

    trigger_m: float = 220.0
    """Distance from the stop line at which clearing starts.

    Sized from the time it takes rather than from the road: at the 19.5 m/s an
    emergency vehicle runs at, this is about eleven seconds, which is a lane
    change (four to six seconds in the sublane model, more from a standstill)
    plus enough margin for the vehicle in front of *that* one to move first.
    Starting earlier does not clear more, because a queue only discharges from
    the front; it just holds more of the approach under modified rules."""

    ahead_m: float = 180.0
    """How far ahead of the emergency vehicle to clear.

    The whole distance to the stop line, in practice, which is the point: the
    lane has to be clear where the vehicle is *going*, not where it is."""

    move_ev: bool = False
    """May the emergency vehicle itself be steered to a freer lane?

    Off, and the measurement is the reason. It looks like the obvious extra
    win -- an emergency driver reads the queues and picks the short lane, so
    why not do it here -- and over twelve seeds it was the *worst* of the
    variants tried, +1.8 s of emergency delay per vehicle against leaving the
    vehicle where it is (paired, s.e. 2.8, so not a significant loss either;
    the point is that there is no gain to bank). The mechanism is visible in
    the picture: a lane change in the sublane model is not free, the vehicle
    slides sideways at a bounded lateral speed while its longitudinal progress
    is governed by whatever is now in front of it, and on a three-lane approach
    where each lane holds two or three cars the swap costs about what it saves.

    Kept as an option because it is the honest ablation of the clearing rule
    below, and because on an approach with longer queues than this one the
    arithmetic could plausibly go the other way."""

    request_s: float = 8.0
    """Duration of each lane-change request, refreshed every step while the
    emergency vehicle is still coming. Long enough to survive the seconds a
    change takes from a standstill, short enough that the road returns to
    normal on its own if this object stops running."""

    # ---------------------------------------------------------------- runtime
    _lane_exits: Dict[str, Set[str]] = field(default_factory=dict, init=False)
    """Lane id -> the set of edges reachable from it. Static; read once."""

    _lanes_on_edge: Dict[str, int] = field(default_factory=dict, init=False)
    _held: Dict[str, float] = field(default_factory=dict, init=False)
    """Vehicle id -> sim time its lane-change mode must be restored at."""

    n_moved: int = field(default=0, init=False)
    n_events: int = field(default=0, init=False)
    _active: Set[str] = field(default_factory=set, init=False)
    """Emergency vehicles currently being cleared for, for the event count."""

    # ------------------------------------------------------------------ utils
    def _safe(self, fn, *a, **k):
        try:
            return fn(*a, **k)
        except Exception:
            return None

    @property
    def _conn(self):
        return getattr(self.env, "conn", None)

    def _evs(self) -> Dict[str, object]:
        scenario = getattr(self.env, "scenario", None)
        return getattr(scenario, "emergency_vehicles", None) or {}

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        """Read the junction's connection table. Cheap, and done once per run."""
        self._lane_exits = {}
        self._lanes_on_edge = {}
        self._held = {}
        self._active = set()
        self.n_moved = 0
        self.n_events = 0
        if not self.enabled or self._conn is None:
            return
        conn = self._conn
        approaches = {self.env.lane_to_approach[ln]
                      for ln in getattr(self.env, "controlled_lanes", [])}
        for edge in approaches:
            n = self._safe(conn.edge.getLaneNumber, edge)
            if not n:
                continue
            self._lanes_on_edge[edge] = int(n)
            for idx in range(int(n)):
                lane = f"{edge}_{idx}"
                exits: Set[str] = set()
                for link in self._safe(conn.lane.getLinks, lane) or ():
                    # (approachedLane, hasPrio, isOpen, hasFoe, ...) -- only the
                    # first field is needed and its position is stable across
                    # the TraCI versions this runs on.
                    target = link[0] if link else None
                    if not target:
                        continue
                    # An internal lane inside the junction names the edge it
                    # leads to via its own successor; the via lane's edge id
                    # starts with ':' and is not a route edge.
                    eid = self._safe(conn.lane.getEdgeID, target)
                    if eid and not eid.startswith(":"):
                        exits.add(eid)
                self._lane_exits[lane] = exits

    # ------------------------------------------------------------- per second
    def update(self) -> None:
        """Clear ahead of every emergency vehicle that is close enough.

        Called once per simulation step from the demo's frame hook, so the
        picture and the actuation move together.
        """
        if not self.enabled or self._conn is None:
            return
        conn = self._conn
        evs = self._evs()
        if not evs:
            return
        now = float(getattr(self.env, "sim_time", 0.0))
        present = set(self._safe(conn.vehicle.getIDList) or ())
        live = present & set(evs)

        wanted: Set[str] = set()
        active_now: Set[str] = set()
        for vid in sorted(live):
            edge = self._safe(conn.vehicle.getRoadID, vid)
            if edge not in self._lanes_on_edge:
                continue  # inside the junction, or downstream of it
            lane = self._safe(conn.vehicle.getLaneID, vid)
            pos = self._safe(conn.vehicle.getLanePosition, vid)
            if lane is None or pos is None:
                continue
            length = self._safe(conn.lane.getLength, lane)
            if length is None or (length - pos) > self.trigger_m:
                continue
            active_now.add(vid)
            wanted |= self._clear_ahead(vid, edge, lane, float(pos), present)

        self.n_events += len(active_now - self._active)
        self._active = active_now

        # Release anything whose request has run out. The lane-change request
        # itself expires on its own; this is only the mode, and it has to be
        # put back or the vehicle drives the rest of its trip with keep-right
        # disabled.
        for v in [v for v, t in self._held.items() if v not in wanted and now >= t]:
            self._safe(conn.vehicle.setLaneChangeMode, v, DEFAULT_LC_MODE)
            self._held.pop(v, None)

    def _clear_ahead(self, ev: str, edge: str, ev_lane: str, ev_pos: float,
                     present: Set[str]) -> Set[str]:
        """Empty the lane the emergency vehicle is going to use.

        Which lane that is, is a question only when :attr:`move_ev` is set: an
        approach here carries a left-only lane, a shared through-and-right lane
        and a through lane, and which of them is the fast one is a property of
        the queue rather than of the vehicle. By default the answer is "the one
        it is in" -- see :attr:`move_ev` for the measurement.

        When the vehicle *is* being moved, the clearing applies to the lane it
        is in **and** the lane it is moving to. Both, because a lane change
        takes seconds it may not get: clearing only the destination leaves it
        stuck behind the queue it was trying to leave, and clearing only the
        origin sends it into a fresh one.
        """
        conn = self._conn
        now = float(getattr(self.env, "sim_time", 0.0))
        evs = set(self._evs())
        try:
            ev_idx = int(ev_lane.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return set()
        n_lanes = self._lanes_on_edge.get(edge, 0)
        if n_lanes < 2:
            return set()

        # One read of the approach, reused for every decision below. Asking
        # per-vehicle instead costs three TraCI round-trips per car per step,
        # which at the 0.15 s step the GUI runs is what turns a smooth demo
        # into a slideshow.
        ahead: Dict[int, List[Tuple[float, str]]] = {i: [] for i in range(n_lanes)}
        for v in self._safe(conn.edge.getLastStepVehicleIDs, edge) or ():
            if v == ev or v not in present:
                continue
            pos = self._safe(conn.vehicle.getLanePosition, v)
            idx = self._safe(conn.vehicle.getLaneIndex, v)
            if pos is None or idx is None or not (0 <= int(idx) < n_lanes):
                continue
            gap = float(pos) - ev_pos
            if 0.0 < gap <= self.ahead_m:
                ahead[int(idx)].append((gap, v))

        usable = self._route_lanes(ev, edge, n_lanes) if self.move_ev else []
        target_idx = ev_idx
        if usable:
            # Fewest vehicles between here and the stop line wins; ties go to
            # the lane it is already in, so it does not weave for nothing.
            pick = min(usable, key=lambda i: (len(ahead[i]), i != ev_idx,
                                              abs(i - ev_idx)))
            if pick != ev_idx and len(ahead[pick]) + 1 < len(ahead[ev_idx]):
                target_idx = pick
                self._safe(conn.vehicle.changeLane, ev, int(target_idx),
                           self.request_s)

        keep_clear = {ev_idx, target_idx}
        moved: Set[str] = set()
        for idx in keep_clear:
            for _gap, v in ahead[idx]:
                if v in evs:
                    continue
                dest = self._target_lane(v, edge, keep_clear, n_lanes, ahead)
                if dest is None:
                    continue
                if v not in self._held:
                    self._safe(conn.vehicle.setLaneChangeMode, v, CLEARING_LC_MODE)
                    self.n_moved += 1
                self._held[v] = now + self.request_s
                self._safe(conn.vehicle.changeLane, v, int(dest), self.request_s)
                moved.add(v)
        return moved

    def _route_lanes(self, veh: str, edge: str, n_lanes: int) -> List[int]:
        """Lane indices on ``edge`` from which ``veh`` can still make its turn."""
        nxt = self._next_edge(veh)
        if nxt is None:
            return []
        out = []
        for idx in range(n_lanes):
            exits = self._lane_exits.get(f"{edge}_{idx}")
            if exits and nxt in exits:
                out.append(idx)
        return out

    def _target_lane(self, veh: str, edge: str, keep_clear: Set[int],
                     n_lanes: int,
                     ahead: Dict[int, List[Tuple[float, str]]]) -> Optional[int]:
        """The emptiest lane of this approach that still serves ``veh``'s route.

        Route-compatibility is the constraint that matters. Moving a vehicle
        into a lane its next edge cannot be reached from does not clear the
        approach, it blocks it: the vehicle arrives at the stop line unable to
        take its turn, stops, and the queue it was moved out of now has a
        second one beside it. On this network that is not a corner case -- the
        left lane takes only left turns and only lane 0 may turn right -- so a
        rule that just shoved everything sideways would jam two movements in
        three.
        """
        best: Optional[Tuple[Tuple[int, int], int]] = None
        for idx in self._route_lanes(veh, edge, n_lanes):
            if idx in keep_clear:
                continue
            key = (len(ahead.get(idx, ())), abs(idx - min(keep_clear)))
            if best is None or key < best[0]:
                best = (key, idx)
        return None if best is None else best[1]

    def _next_edge(self, veh: str) -> Optional[str]:
        route = self._safe(self._conn.vehicle.getRoute, veh)
        i = self._safe(self._conn.vehicle.getRouteIndex, veh)
        if not route or i is None:
            return None
        i = int(i) + 1
        return route[i] if 0 <= i < len(route) else None

    # ------------------------------------------------------------------ close
    def close(self) -> None:
        """Hand every vehicle back its own lane-change model."""
        conn = self._conn
        if conn is None:
            return
        for v in list(self._held):
            self._safe(conn.vehicle.setLaneChangeMode, v, DEFAULT_LC_MODE)
        self._held.clear()

    def stats(self) -> Dict[str, float]:
        return {
            "clearance_events": float(self.n_events),
            "vehicles_moved_aside": float(self.n_moved),
        }
