"""Drawing the controller's reasoning onto the SUMO map.

Watching cars queue and discharge tells a viewer nothing about *why*. Three
things are therefore drawn on top of the simulation, and each corresponds to a
quantity the controller actually computed:

**Which vehicle is it, and how much is it worth.** Emergency vehicles carry a
class badge (:mod:`atsc.evaluation.icons`) and flash between their two livery
colours once per simulated second, so an ambulance, a fire appliance and a
police car are distinguishable at junction zoom -- and so is the fact that they
are running lights at all. The triage weight the scheduler ranks them by is a
property of the class, so the class has to be visible.

**Where the corridor is.** When the scheduler commits to an approach, the lanes
of that approach are overlaid with a translucent band and the ordinary vehicles
on it are recoloured. The band appears at the moment of commitment, which is
before the signal changes, so the *decision* is visible separately from its
effect -- that is the whole mechanism under study.

The running state of the controller -- phase, queues, per-approach belief, and
for each commitment the projected red at arrival, the delay saved, the slack
remaining and the cost -- goes to the terminal commentary rather than onto the
map, because SUMO's view has no text layer and a legible caption cannot be
faked out of polygons.

None of this affects the simulation. Every call is wrapped so that a TraCI
version without one of these calls degrades to not drawing it rather than to
killing the run, and the whole class is inert when the GUI is off.

Camera
------
The frame is set by :meth:`GuiPainter._frame_box` and nothing else touches it.
Two things had to be learned the hard way, both of the form *the TraCI getters
and the rendered picture disagree*:

* ``gui.trackVehicle`` zooms the rendered view hard in while reporting nothing
  --- ``getZoom`` and ``getBoundary`` both keep describing the wide frame. It is
  therefore not used; the camera is panned by setting the boundary directly,
  which makes a constant scale arithmetic rather than a hope.
* ``gui.setBoundary`` *fits* its box into the window rather than filling it, so
  a square box shows far more than it asks for on a wide window --- 520 m
  square became 1241 m across at 2.39:1. The box is shaped to the window's
  measured aspect instead, so the longest visible axis is always
  ``2 * view_radius_m`` on any screen.

Smoothness
----------
The other half of "make it watchable" is the step length. SUMO redraws once per
simulation step, so at the 1 s step the controller is trained on, a vehicle at
50 km/h jumps fourteen metres between frames and then freezes -- which looks
like a stutter, not like traffic. The demo therefore runs the *simulation* at a
fraction of a second per step while leaving the 5 s decision epoch untouched;
the controller is unchanged and the picture is continuous. See
``--step`` in ``scripts/demo.py``.
"""

from __future__ import annotations

import math

import numpy as np
from typing import Dict, Optional, Sequence

from .icons import CLASS_COLOUR, CLASS_FLASH, ev_class_of, write_icons

CORRIDOR_BAND = (40, 220, 90, 90)
"""Translucent green band over the approach the scheduler has committed to."""

HEARD_BAND = (255, 170, 30, 70)
"""Translucent amber band over an approach the belief says is carrying a siren
but that has not been committed to.

This is the other half of the picture and it was missing. A green band appears
only once the scheduler has decided to act, so everything the system does
*before* that -- hearing a vehicle, tracking it, pricing the corridor, waiting
for the last moment at which opening it still works -- was invisible, and a
viewer watching an emergency vehicle approach a red had no way to tell "the
controller has not heard it" from "the controller has heard it and is timing
the opening". Those are the two hypotheses the whole demonstration is about,
and amber-then-green distinguishes them."""

CORRIDOR_VEHICLE = (40, 220, 90, 255)
NORMAL_VEHICLE = (235, 235, 240, 255)
HALO_ALPHA = 140


class GuiPainter:
    """Draws the controller's internal state onto the SUMO map."""

    view_radius_m = 150.0
    """Half of the longest distance the view spans, metres, centred on the
    junction.

    ``gui.setBoundary`` *fits* the box it is given into the window rather than
    filling it, so the window's aspect ratio decides how much of the other axis
    comes along. A square 520 m box in a 1.66:1 window is fitted by its height
    and shows **865 m** across -- the whole network, plus green margin, with the
    vehicles too small to read. In a 2.39:1 window the same call shows 1240 m.
    The same number therefore framed the junction on one machine and the entire
    map on another, which is exactly the complaint this attribute's previous
    docstring claimed a boundary in metres had solved.

    So the box is no longer square. :meth:`_frame_box` measures the window once,
    and shapes every request to that aspect so the *longer* visible axis is
    always ``2 * view_radius_m`` and the shorter follows. The span is then the
    same on any window, which is what was wanted all along.

    150 m puts 300 m across the long axis: the junction and a hundred and fifty
    metres of every arm, which covers any queue this intersection forms while
    leaving a car about thirty pixels wide on a 1900 px window. Nothing in this
    class changes it during a run --- but **the viewer may**, with the mouse, and
    :meth:`refresh_follow` stands down permanently as soon as they do."""

    badge_size_m = 6.0
    badge_offset_m = 7.5
    """The badge floats above the vehicle rather than on it, so at close zoom
    the vehicle body and its livery are still visible underneath.

    Sized against the road, not against the vehicle: a lane is 3.2 m and a car
    is about 4.3 m, so 6 m reads clearly at junction zoom while still leaving
    the queue behind the vehicle visible. The 11 m this started at drew a badge
    wider than the three-lane approach it sat on, which hid the very traffic
    the corridor is being judged against."""

    halo_radius_m = 10.0
    """Throw of the beacon. Kept just larger than the badge: at the 18 m it
    started at the halo covered most of a queue."""

    follow_max_offset_m = 55.0
    """How far the frame's centre may leave the junction, metres.

    The junction is the subject; following only *biases* the view toward the arm
    where something is happening, and this is deliberately a small fraction of
    the frame. At the 110 m it started at, following a vehicle on the west
    approach put the junction hard against the right-hand edge with two thirds
    of the window showing empty road --- which is not a demonstration of an
    intersection. Clamped again to a third of the view radius below, so whatever
    frame the viewer picks, the junction stays near the middle of it."""

    follow_ease = 0.08
    """Fraction of the remaining distance the camera closes each step.

    At the demo's ten frames per second this settles a full-offset move in
    about three seconds, which reads as a glide. Set it to 1.0 for the old
    behaviour of cutting straight to the target."""

    follow_range_m = 200.0
    """Distance from the stop line inside which ``--follow-ev`` will acquire a
    vehicle, metres.

    A bound on how far the camera may jump when it takes hold of one, not a
    judgement about which vehicles are interesting. Acquiring a vehicle three
    hundred metres up an arm moves the centre further than the view is deep, so
    the picture cuts rather than pans; inside this range the vehicle is already
    on screen before the camera moves to it, and stays on screen while it does.

    Clamped to :attr:`view_radius_m`, so a vehicle is only ever picked up once
    it is inside the frame already on screen --- the camera biases toward
    something the viewer can already see rather than toward an empty arm.

    It is deliberately larger than the corridor's own commit range: the point
    of watching is to see the corridor open *ahead* of the vehicle, which
    happens before it is committed to."""

    def __init__(self, env, enabled: bool = True, icons_dir=None,
                 flash_period_s: float = 0.5, view_radius_m: Optional[float] = None):
        self.env = env
        self.enabled = bool(enabled and env.cfg.gui)
        self.flash_period_s = float(flash_period_s)
        if view_radius_m is not None:
            self.view_radius_m = float(view_radius_m)
        self._painted: set = set()
        self._badges: Dict[str, str] = {}
        self._halo_until: Dict[str, float] = {}
        self._bands: set = set()
        self._band_colour: Dict[str, tuple] = {}
        self._tracking: Optional[str] = None
        self._follow_approach: Optional[str] = None
        self._camera: Optional[tuple] = None
        self._aspect: Optional[float] = None
        self._last_fitted: Optional[tuple] = None
        self._user_drove: bool = False
        self._icons: Dict[str, str] = {}
        if self.enabled and icons_dir is not None:
            try:
                self._icons = {k: str(v) for k, v in write_icons(icons_dir).items()}
            except Exception:
                self._icons = {}

    # ------------------------------------------------------------------ utils
    def _safe(self, fn, *a, **k):
        if not self.enabled:
            return None
        try:
            return fn(*a, **k)
        except Exception:
            return None

    def _evs(self) -> Dict[str, object]:
        scenario = getattr(self.env, "scenario", None)
        return getattr(scenario, "emergency_vehicles", None) or {}

    # ------------------------------------------------------------------ setup
    def _junction_centre(self):
        """Where the intersection is, in network coordinates.

        Averaged over the stop lines rather than read from a junction id, so it
        is correct for whatever geometry ``netgen`` produced and needs no extra
        TraCI call.
        """
        stops = list(getattr(self.env, "stop_line", {}).values())
        if not stops:
            return None
        return (sum(x for x, _ in stops) / len(stops),
                sum(y for _, y in stops) / len(stops))

    def _window_aspect(self) -> Optional[float]:
        """Width/height of the view, measured once from SUMO itself.

        There is no TraCI call for the window's shape, so it is inferred: ask
        for a square boundary, read back what was actually fitted, and the
        ratio of the two is the aspect. ``None`` when the view cannot be
        measured -- no GUI, or a TraCI without ``getBoundary`` -- and every
        caller then falls back to a square box, which is the old behaviour.
        """
        if self._aspect is not None:
            return self._aspect
        centre = self._junction_centre()
        if centre is None:
            return None
        cx, cy = centre
        r = self.view_radius_m
        # Guarded as a whole rather than per call: _safe protects the *call*,
        # and a TraCI without getBoundary fails at the attribute lookup before
        # it is reached.
        try:
            self.env.conn.gui.setBoundary("View #0", cx - r, cy - r, cx + r, cy + r)
            (x0, y0), (x1, y1) = self.env.conn.gui.getBoundary("View #0")
            w, h = float(x1) - float(x0), float(y1) - float(y0)
            if w > 0.0 and h > 0.0:
                self._aspect = w / h
        except Exception:
            self._aspect = None
        return self._aspect

    def _frame_box(self, cx: float, cy: float):
        """The boundary to request for a view centred on ``(cx, cy)``.

        Sized so the longer visible axis spans ``2 * view_radius_m`` whatever
        the window's shape, and so every call returns a box of identical size
        -- which is what makes the scale constant by construction rather than
        by hoping SUMO leaves it alone.
        """
        r = self.view_radius_m
        aspect = self._window_aspect()
        if not aspect or not math.isfinite(aspect) or aspect <= 0.0:
            return cx - r, cy - r, cx + r, cy + r
        if aspect >= 1.0:                      # wide window: width is the long axis
            hw, hh = r, r / aspect
        else:                                  # tall window
            hw, hh = r * aspect, r
        return cx - hw, cy - hh, cx + hw, cy + hh

    def frame_junction(self) -> None:
        """Point the camera at the intersection and leave it there."""
        if not self.enabled:
            return
        centre = self._junction_centre()
        if centre is None:
            return
        cx, cy = centre
        self._camera = (cx, cy)
        self._set_frame(cx, cy)

    def setup(self) -> None:
        if not self.enabled:
            return
        self._safe(self.env.conn.gui.setSchema, "View #0", "real world")
        self.frame_junction()

    # ------------------------------------------------------------- per second
    def paint_emergency(self) -> None:
        """Flash each emergency vehicle in its livery and keep its badge on it.

        Called every simulation step. The flash phase is derived from simulated
        time rather than from a frame counter so the blink rate is the same
        whatever playback speed the viewer picked.
        """
        if not self.enabled:
            return
        conn = self.env.conn
        evs = self._evs()
        if not evs:
            return
        present = set(self._safe(conn.vehicle.getIDList) or ())
        live = present & set(evs)
        t = float(getattr(self.env, "sim_time", 0.0))
        on = int(t / max(self.flash_period_s, 1e-3)) % 2 == 0

        for vid in live:
            cls = ev_class_of(vid, getattr(self.env, "scenario", None))
            colour = CLASS_COLOUR.get(cls) if on else CLASS_FLASH.get(cls)
            self._safe(conn.vehicle.setColor, vid, colour or CLASS_COLOUR["ambulance"])
            pos = self._safe(conn.vehicle.getPosition, vid)
            if pos is None:
                continue
            badge = self._badges.get(vid)
            if badge is None:
                badge = f"badge_{vid}"
                img = self._icons.get(cls, "")
                self._safe(
                    conn.poi.add, badge, pos[0], pos[1] + self.badge_offset_m,
                    CLASS_COLOUR.get(cls, (255, 255, 255, 255)),
                    poiType=f"ev_{cls}", layer=12, imgFile=img,
                    width=self.badge_size_m, height=self.badge_size_m)
                self._badges[vid] = badge
            else:
                self._safe(conn.poi.setPosition, badge,
                           pos[0], pos[1] + self.badge_offset_m)
            # A halo, re-armed as it expires, gives the beacon a visible throw
            # of light rather than only recolouring the body.
            if t >= self._halo_until.get(vid, -1.0):
                base = CLASS_COLOUR.get(cls, (255, 60, 60, 255))
                self._safe(conn.vehicle.highlight, vid,
                           (base[0], base[1], base[2], HALO_ALPHA),
                           self.halo_radius_m, 255, 1.2, 0)
                self._halo_until[vid] = t + 1.0

        for vid in list(self._badges):
            if vid not in live:
                self._safe(conn.poi.remove, self._badges.pop(vid))
                self._halo_until.pop(vid, None)

    # ---------------------------------------------------------- per decision
    def paint_corridor(self, approach: Optional[str],
                       heard: Sequence[str] = ()) -> None:
        """Band the committed approach and recolour the traffic it releases.

        ``heard`` names approaches the belief is currently tracking a siren on
        without the scheduler having committed to them; they get the amber band
        so the *hearing* is visible separately from the *acting*.

        Lane *recolouring* is not used: SUMO's lane colour belongs to the active
        visualisation scheme, so setting it fights the scheme rather than adding
        to it. A translucent polygon laid over the lane shapes composites on top
        of any scheme and reads unambiguously as an overlay.
        """
        if not self.enabled:
            return
        conn = self.env.conn
        want_bands = set()
        banded = {a: CORRIDOR_BAND for a in ((approach,) if approach else ())}
        for a in heard:
            banded.setdefault(a, HEARD_BAND)
        # What SUMO actually holds, asked once per epoch. This is the test of
        # whether a band exists: polygon.add returns nothing on success, so its
        # return value cannot be it, and the local _bands set cannot be it
        # either -- an id goes in there whether or not the add worked.
        existing = set(self._safe(conn.polygon.getIDList) or ())
        for band_approach, colour in banded.items():
            for ln in getattr(self.env, "controlled_lanes", []):
                if self.env.lane_to_approach.get(ln) != band_approach:
                    continue
                pid = f"corridor_{ln}"
                want_bands.add(pid)
                # An amber band that becomes green keeps its id and changes
                # colour, so the id alone is not enough to decide whether the
                # polygon SUMO holds is the one we want. Recorded per id and
                # re-added when it differs; without this an approach the
                # scheduler commits to goes on being drawn in the colour it was
                # first heard in, which says the opposite of what happened.
                if pid in existing and self._band_colour.get(pid) != colour:
                    self._safe(conn.polygon.remove, pid)
                    existing.discard(pid)
                shape = self._safe(conn.lane.getShape, ln)
                if not shape or len(shape) < 2:
                    continue
                width = self._safe(conn.lane.getWidth, ln) or 3.2
                band = _thicken(shape, width * 0.9)
                # Add only what is missing. Re-adding every epoch also works
                # -- the duplicate raises and _safe swallows it -- but the
                # exception is raised by the *server*, which logs "Could not
                # add polygon" to the GUI message window first, so a corridor
                # held for n epochs printed 3n errors at the viewer for
                # something that was working correctly. Checking the id list
                # keeps the retry this replaced was there for: a band that
                # genuinely failed to appear is absent from `existing` and is
                # attempted again on the next epoch.
                if pid not in existing:
                    self._safe(conn.polygon.add, pid, band, colour,
                               True, "corridor", 8)
                    self._band_colour[pid] = colour
                # Recorded whether or not the add landed, so that a band which
                # did appear is always removed when the corridor moves on.
                # Otherwise every approach ever served stays painted green.
                self._bands.add(pid)
        for pid in self._bands - want_bands:
            # Same reasoning as the add: removing an id SUMO does not hold logs
            # a server-side error at the viewer. An id can be in _bands without
            # being in the simulation if its add failed.
            if pid in existing:
                self._safe(conn.polygon.remove, pid)
            self._band_colour.pop(pid, None)
        self._bands &= want_bands

        evs = set(self._evs())
        want = set()
        if approach is not None:
            for vid in self._safe(conn.vehicle.getIDList) or ():
                if vid in evs:
                    continue
                if self._safe(conn.vehicle.getRoadID, vid) == approach:
                    want.add(vid)
        for vid in self._painted - want:
            self._safe(conn.vehicle.setColor, vid, NORMAL_VEHICLE)
        for vid in want - self._painted:
            self._safe(conn.vehicle.setColor, vid, CORRIDOR_VEHICLE)
        self._painted = want

    def follow(self, approach: Optional[str]) -> None:
        """Pan the frame toward the emergency vehicle on ``approach``.

        **This does not use ``gui.trackVehicle``, and that is the fix.**

        SUMO's own vehicle tracking zooms the rendered view right in --- to a
        scale bar reading ten metres, a window spanning less than two hundred
        metres, with the junction off screen entirely. It does so *without*
        reporting the change through TraCI: ``gui.getZoom`` returns the value
        :meth:`frame_junction` set, unchanged, before, during and after
        tracking, and ``gui.getBoundary`` likewise still describes the wide
        frame. Only the vehicle offset is written back.

        That discrepancy cost a round of debugging. Instrumenting a live run
        showed a constant zoom of 125.23 across 3670 frames and concluded the
        complaint must be about panning; a screenshot of the same behaviour
        showed the junction gone and a ten-metre scale bar. The renderer and
        the getters disagree, the getters are what a test can see, and the
        picture is what the viewer sees. So the camera is no longer asked
        politely to keep its zoom --- the zoom is no longer SUMO's to change.

        The frame is now set explicitly every step through
        :meth:`gui.setBoundary`, always ``view_radius_m`` about a centre this
        class chooses. The box handed over is the same size on every call, so
        the scale *cannot* drift: holding it fixed is now a property of the
        arithmetic rather than a hope about an API.

        What the centre does:

        * with no vehicle to follow it rests on the junction;
        * with one, it eases toward that vehicle, clamped by
          :attr:`follow_max_offset_m` so the junction never leaves the frame;
        * it moves by :attr:`follow_ease` of the remaining distance per step,
          so acquiring and releasing are glides rather than cuts.

        The vehicle is picked up only within :attr:`follow_range_m` and dropped
        as soon as it is no longer upstream on the approach being served ---
        crossing, crossed or gone --- which is what stops the camera chasing an
        already-served vehicle down the exit arm and out of the network.
        """
        if not self.enabled:
            return
        self._follow_approach = approach
        self.refresh_follow()

    def refresh_follow(self) -> None:
        """Advance the camera one step toward where it should be.

        Called every simulation step rather than once per decision epoch, for
        two reasons: the easing needs a per-frame tick to be smooth, and
        release has to be prompt --- a vehicle crosses the junction in about
        two seconds, and a camera re-checked only on epoch boundaries spends
        most of that still pointed at it.
        """
        if not self.enabled:
            return
        centre = self._junction_centre()
        if centre is None:
            return
        jx, jy = centre

        target = self._follow_target()
        goal = (jx, jy)
        if target is not None:
            pos = self._safe(self.env.conn.vehicle.getPosition, target)
            if pos is not None:
                # Clamp the offset so the junction stays in shot. Following is
                # meant to bias the view toward the action, not to replace the
                # subject of the demonstration with a patch of empty road.
                dx = float(pos[0]) - jx
                dy = float(pos[1]) - jy
                reach = max(math.hypot(dx, dy), 1e-9)
                limit = min(self.follow_max_offset_m, self.view_radius_m / 3.0)
                if reach > limit:
                    dx, dy = dx * limit / reach, dy * limit / reach
                goal = (jx + dx, jy + dy)

        if self._camera is None:
            self._camera = (jx, jy)
        cx, cy = self._camera
        k = float(np.clip(self.follow_ease, 0.0, 1.0))
        cx += (goal[0] - cx) * k
        cy += (goal[1] - cy) * k
        # Settle exactly rather than approaching forever, so a camera that has
        # arrived stops issuing indistinguishable boundary updates.
        if abs(cx - goal[0]) < 0.05 and abs(cy - goal[1]) < 0.05:
            cx, cy = goal
        self._camera = (cx, cy)
        self._tracking = target

        self._set_frame(cx, cy)

    def _set_frame(self, cx: float, cy: float) -> None:
        """Set the view, unless the viewer has taken it over.

        Following re-set the boundary on every frame, which meant the viewer
        could not zoom: the scroll wheel moved the camera and the next frame,
        a tenth of a second later, put it straight back. A demonstration you
        cannot lean into is worse than one framed slightly wrong, and the
        viewer knows what they want to look at better than this class does.

        So the frame SUMO is actually showing is compared against the one last
        asked for, and any difference means a hand on the mouse. From then on
        the camera belongs to them for the rest of the run --- not just until
        the next frame, which would be the same fight one step slower.
        """
        conn = self.env.conn
        if self._user_drove:
            return
        if self._last_fitted is not None:
            try:
                (x0, y0), (x1, y1) = conn.gui.getBoundary("View #0")
                now = (float(x0), float(y0), float(x1), float(y1))
                if max(abs(a - b) for a, b in zip(now, self._last_fitted)) > 1.0:
                    self._user_drove = True
                    return
            except Exception:
                pass
        self._safe(conn.gui.setBoundary, "View #0", *self._frame_box(cx, cy))
        try:
            (x0, y0), (x1, y1) = conn.gui.getBoundary("View #0")
            self._last_fitted = (float(x0), float(y0), float(x1), float(y1))
        except Exception:
            self._last_fitted = None

    def _follow_target(self) -> Optional[str]:
        """The emergency vehicle the camera should be biased toward, if any."""
        approach = self._follow_approach
        evs = self._evs()
        if approach is None or not evs:
            return None
        conn = self.env.conn
        present = set(self._safe(conn.vehicle.getIDList) or ())
        for vid in sorted(present & set(evs)):
            if self._safe(conn.vehicle.getRoadID, vid) != approach:
                continue
            gap = self._gap_to_stop_line(vid)
            # An unknown gap means a TraCI without one of these calls, not a
            # vehicle that is far away: follow it rather than degrade to
            # following nothing at all.
            if gap is None or gap <= min(self.follow_range_m, self.view_radius_m):
                return vid
            # Already held and still on its approach: keep it. The range test
            # is for *acquiring*; applying it to a vehicle the camera is
            # already on would drop it whenever the estimate wobbled across
            # the boundary.
            if self._tracking == vid:
                return vid
        return None

    def _gap_to_stop_line(self, vid: str) -> Optional[float]:
        """Metres from ``vid`` to the stop line ahead of it, or None.

        ``None`` for anything this cannot establish -- a vehicle inside the
        junction, or a TraCI without one of these calls. The whole body is
        guarded rather than each call, because :meth:`_safe` protects the
        *call* and an absent method fails at the attribute lookup before it is
        reached, which is how a stub without ``getLaneID`` took the painter
        down instead of degrading.
        """
        conn = self.env.conn
        try:
            lane = conn.vehicle.getLaneID(vid)
            if lane is None or str(lane).startswith(":"):
                return None
            gap = conn.lane.getLength(lane) - conn.vehicle.getLanePosition(vid)
        except Exception:
            return None
        return max(float(gap), 0.0)

    def screenshot(self, path) -> bool:
        """Save the current view, overlays included.

        Two details, both learned the hard way. TraCI's screenshot returns
        nothing on success, so the return value cannot be the test -- whether
        the file appeared is. And the view has to be *redrawn* after the
        overlay commands before the frame is captured: taking the shot in the
        same breath as adding the polygons produced pictures of the corridor
        the controller had opened, without the corridor drawn on them, which is
        precisely the failure this whole overlay exists to prevent.
        """
        if not self.enabled:
            return False
        conn = self.env.conn
        for _ in range(2):
            self._safe(conn.simulationStep)
        self._safe(conn.gui.screenshot, "View #0", str(path))
        for _ in range(2):
            self._safe(conn.simulationStep)
        from pathlib import Path as _P
        return _P(path).exists()

    def close(self) -> None:
        conn = getattr(self.env, "conn", None)
        if conn is None:
            return
        for pid in list(self._bands):
            self._safe(conn.polygon.remove, pid)
        for vid in list(self._badges):
            self._safe(conn.poi.remove, self._badges.pop(vid))
        self._bands.clear()
        self._band_colour.clear()


def _thicken(shape: Sequence[Sequence[float]], width: float):
    """Turn a lane centreline into a closed band polygon of the given width."""
    left, right = [], []
    n = len(shape)
    for i, (x, y) in enumerate(shape):
        if i == 0:
            dx, dy = shape[1][0] - x, shape[1][1] - y
        elif i == n - 1:
            dx, dy = x - shape[-2][0], y - shape[-2][1]
        else:
            dx = shape[i + 1][0] - shape[i - 1][0]
            dy = shape[i + 1][1] - shape[i - 1][1]
        norm = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / norm * width / 2.0, dx / norm * width / 2.0
        left.append((x + nx, y + ny))
        right.append((x - nx, y - ny))
    return left + right[::-1] + [left[0]]
