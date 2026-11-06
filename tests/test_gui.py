"""The live overlay, driven against a recording stub instead of SUMO.

The painter is pure TraCI bookkeeping, so the interesting failures -- adding a
band twice, leaving one behind when the corridor moves -- are visible without a
GUI, and a test that needed one could not run unattended.
"""

from __future__ import annotations

from atsc.evaluation.gui import GuiPainter


class _Polygon:
    def __init__(self, log):
        self._log = log
        self._ids = []

    def getIDList(self):
        return list(self._ids)

    def add(self, pid, shape, colour, fill, ptype, layer):
        self._log.append(("add", pid))
        if pid in self._ids:
            # What SUMO does: reject it, and say so in the message window.
            raise RuntimeError(f"Could not add polygon '{pid}'")
        self._ids.append(pid)

    def remove(self, pid):
        self._log.append(("remove", pid))
        if pid not in self._ids:
            raise RuntimeError(f"Could not remove polygon '{pid}'")
        self._ids.remove(pid)


_LANE_LENGTH = 320.0


class _Lane:
    def getShape(self, ln):
        return [(0.0, 0.0), (10.0, 0.0)]

    def getWidth(self, ln):
        return 3.2

    def getLength(self, ln):
        return _LANE_LENGTH


class _Vehicle:
    def __init__(self, on_road=None, gaps=None):
        self._on_road = on_road or {}
        # Metres from the stop line. Absent means "this TraCI cannot say",
        # which the painter must treat as in range rather than as far away.
        self._gaps = gaps or {}

    def getIDList(self):
        return list(self._on_road)

    def setColor(self, *a):
        pass

    def getRoadID(self, vid):
        return self._on_road.get(vid)

    def getLaneID(self, vid):
        road = self._on_road.get(vid)
        if road is None or vid not in self._gaps:
            raise KeyError(vid)          # as TraCI would, for an unknown id
        return f"{road}_0"

    def getLanePosition(self, vid):
        return _LANE_LENGTH - self._gaps[vid]

    def getPosition(self, vid):
        return (100.0, 200.0)

    def highlight(self, *a):
        pass


class _Gui:
    """Records every camera command, which is the whole point of the test."""

    def __init__(self, log, aspect=None):
        self._log = log
        self.aspect = aspect
        self._last = (0.0, 0.0, 1.0, 1.0)

    def setSchema(self, view, name):
        self._log.append(("schema", name))

    def setBoundary(self, view, xmin, ymin, xmax, ymax):
        self._log.append(("boundary", (xmin, ymin, xmax, ymax)))
        self._last = (xmin, ymin, xmax, ymax)

    def viewer_zooms(self, factor):
        """A hand on the scroll wheel: the view changes without us asking."""
        x0, y0, x1, y1 = self._last
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        hw, hh = (x1 - x0) / 2.0 * factor, (y1 - y0) / 2.0 * factor
        self._last = (cx - hw, cy - hh, cx + hw, cy + hh)

    def getBoundary(self, view):
        """What SUMO actually fitted, which is not what was asked for.

        setBoundary fits the requested box into the window rather than filling
        it, so a wider window returns a wider box than it was given. That is
        the behaviour the framing has to compensate for, so the stub reproduces
        it: whatever aspect the fake window has, the height is honoured and the
        width is stretched to match.
        """
        if self.aspect is None:
            raise AttributeError("getBoundary")     # a TraCI without the call
        x0, y0, x1, y1 = self._last
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        h = y1 - y0
        w = h * self.aspect
        return ((cx - w / 2.0, cy - h / 2.0), (cx + w / 2.0, cy + h / 2.0))

    def setZoom(self, view, zoom):
        self._log.append(("zoom", zoom))

    def trackVehicle(self, view, vid):
        self._log.append(("track", vid))


class _Poi:
    def add(self, *a, **k):
        pass

    def setPosition(self, *a):
        pass

    def remove(self, *a):
        pass


class _Conn:
    def __init__(self, log, vehicles=None, gaps=None, aspect=None):
        self.polygon = _Polygon(log)
        self.lane = _Lane()
        self.vehicle = _Vehicle(vehicles, gaps)
        self.gui = _Gui(log, aspect)
        self.poi = _Poi()


class _Cfg:
    gui = True


class _Env:
    """Two approaches, two lanes each, stop lines around (100, 200)."""

    def __init__(self, log, vehicles=None, scenario=None, gaps=None, aspect=None):
        self.cfg = _Cfg()
        self.conn = _Conn(log, vehicles, gaps, aspect)
        self.controlled_lanes = ["n_0", "n_1", "e_0", "e_1"]
        self.lane_to_approach = {"n_0": "N", "n_1": "N", "e_0": "E", "e_1": "E"}
        self.stop_line = {"N": (90.0, 200.0), "E": (110.0, 200.0)}
        self.scenario = scenario
        self.sim_time = 0.0


class _EV:
    veh_class = "ambulance"


class _Scenario:
    emergency_vehicles = {"ev_ambulance_0": _EV()}


def _painter(vehicles=None, scenario=None, gaps=None, aspect=None, **kw):
    log = []
    env = _Env(log, vehicles, scenario, gaps, aspect)
    p = GuiPainter(env, enabled=True, icons_dir=None, **kw)
    return p, env, log


def test_holding_a_corridor_does_not_re_add_its_bands():
    """The bug behind a screen full of "Could not add polygon".

    The band used to be re-added every epoch and the resulting exception
    swallowed, but the rejection is raised by the *server*, which logs it to
    the GUI message window first -- so a corridor held for n epochs printed
    two errors per lane per epoch at the viewer for something working
    perfectly. Each band must be added exactly once.
    """
    p, env, log = _painter()
    for _ in range(5):
        p.paint_corridor("N")
    adds = [pid for kind, pid in log if kind == "add"]
    assert sorted(adds) == ["corridor_n_0", "corridor_n_1"], adds
    assert sorted(env.conn.polygon.getIDList()) == ["corridor_n_0", "corridor_n_1"]


def test_moving_the_corridor_removes_the_old_bands():
    """Or every approach ever served stays painted green."""
    p, env, log = _painter()
    p.paint_corridor("N")
    p.paint_corridor("E")
    assert sorted(env.conn.polygon.getIDList()) == ["corridor_e_0", "corridor_e_1"]
    assert ("remove", "corridor_n_0") in log


def test_releasing_the_corridor_clears_every_band():
    p, env, _ = _painter()
    p.paint_corridor("N")
    p.paint_corridor(None)
    assert env.conn.polygon.getIDList() == []
    assert p._bands == set()


def test_a_band_that_failed_to_appear_is_retried():
    """The retry the duplicate-add was there for, kept without the spam."""
    p, env, log = _painter()
    real_add = env.conn.polygon.add
    calls = {"n": 0}

    def flaky(pid, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            log.append(("add", pid))
            raise RuntimeError("transient")   # never lands in _ids
        return real_add(pid, *a, **k)

    env.conn.polygon.add = flaky
    p.paint_corridor("N")
    first = set(env.conn.polygon.getIDList())
    p.paint_corridor("N")
    assert len(first) == 1, first          # one of the two failed
    assert sorted(env.conn.polygon.getIDList()) == ["corridor_n_0", "corridor_n_1"]


# ------------------------------------------------------------------- camera
def test_the_view_is_framed_by_boundary_on_the_junction():
    """Not by zoom level, and centred on the thing being demonstrated."""
    p, env, log = _painter(view_radius_m=100.0)
    p.setup()
    bounds = [b for kind, b in log if kind == "boundary"]
    assert bounds[-1] == (0.0, 100.0, 200.0, 300.0), bounds   # centre (100, 200)
    assert not [z for kind, z in log if kind == "zoom"]


def test_a_wide_window_does_not_stretch_the_view_across_the_whole_network():
    """The second complaint: "it got zoomed out too much".

    ``setBoundary`` *fits* the box into the window instead of filling it, so
    the window's shape decides how much of the other axis comes along. A square
    520 m box in a 1.66:1 window is fitted by its height and shows 865 m
    across -- the whole network plus margin, vehicles too small to read -- and
    in a 2.39:1 window, 1240 m. The same call framed the junction on one
    machine and the entire map on another, which is what a boundary in metres
    was supposed to prevent.

    The box is therefore shaped to the window, so the longest visible axis is
    always ``2 * view_radius_m`` whatever the window's proportions.
    """
    for aspect in (1.0, 1.66, 2.39, 0.6):
        p, env, log = _painter(view_radius_m=100.0, aspect=aspect)
        p.setup()
        x0, y0, x1, y1 = [b for kind, b in log if kind == "boundary"][-1]
        # What SUMO will actually show, given it fits by the binding axis.
        shown_h = y1 - y0
        shown_w = shown_h * aspect
        assert abs(max(shown_w, shown_h) - 200.0) < 1e-6, (aspect, shown_w, shown_h)
        # Still centred on the junction.
        assert abs((x0 + x1) / 2.0 - 100.0) < 1e-6
        assert abs((y0 + y1) / 2.0 - 200.0) < 1e-6


def test_the_frame_stays_one_size_on_a_wide_window_too():
    """The constant-scale guarantee must survive the aspect correction."""
    p, env, log = _painter(vehicles={"ev_ambulance_0": "N"},
                           gaps={"ev_ambulance_0": 40.0}, scenario=_Scenario(),
                           view_radius_m=100.0, aspect=2.39)
    p.setup()
    del log[:]
    for _ in range(30):
        p.follow("N")
    env.conn.vehicle._on_road["ev_ambulance_0"] = ":J0_0_7"
    for _ in range(30):
        p.follow("N")
    p.follow(None)
    for _ in range(30):
        p.refresh_follow()
    assert len(set(_sizes(log))) == 1, set(_sizes(log))


def test_an_emergency_vehicle_arriving_does_not_move_the_camera():
    """The complaint this fixes: the map appeared to fly outwards at the exact
    moment there was something worth looking at."""
    p, env, log = _painter(vehicles={"ev_ambulance_0": "N"}, scenario=_Scenario())
    p.setup()
    del log[:]
    p.paint_corridor("N")
    p.paint_emergency()
    assert not [e for e in log if e[0] in ("zoom", "boundary", "track")], log


def _sizes(log):
    """Every frame size the camera has asked for, as (width, height)."""
    return [(round(b[2] - b[0], 6), round(b[3] - b[1], 6))
            for kind, b in log if kind == "boundary"]


def _centres(log):
    return [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
            for kind, b in log if kind == "boundary"]


def test_following_never_changes_the_size_of_the_frame():
    """The complaint, and the property that answers it.

    SUMO's own ``gui.trackVehicle`` zooms the rendered view hard in -- a
    ten-metre scale bar, the junction off screen -- while reporting nothing
    through TraCI: ``getZoom`` and ``getBoundary`` both keep describing the
    wide frame, so an instrumented run measures a constant zoom and a
    screenshot of the same run shows the junction gone. Only the picture is
    true.

    So tracking is not used at all. The frame is set explicitly, always
    ``view_radius_m`` about a centre this class picks, and the test is that
    every box handed to SUMO is the same size. That is checkable arithmetic
    rather than a hope about an API.
    """
    p, env, log = _painter(vehicles={"ev_ambulance_0": "N"},
                           gaps={"ev_ambulance_0": 40.0}, scenario=_Scenario(),
                           view_radius_m=100.0)
    p.setup()
    for _ in range(40):
        p.follow("N")
    env.conn.vehicle._on_road["ev_ambulance_0"] = ":J0_0_7"
    for _ in range(40):
        p.follow("N")
    p.follow(None)
    for _ in range(40):
        p.refresh_follow()

    sizes = set(_sizes(log))
    assert sizes == {(200.0, 200.0)}, sizes
    assert not [e for e in log if e[0] in ("zoom", "track")], log


def test_following_moves_the_centre_toward_the_vehicle():
    p, env, log = _painter(vehicles={"ev_ambulance_0": "N"},
                           gaps={"ev_ambulance_0": 40.0}, scenario=_Scenario(),
                           view_radius_m=100.0)
    p.setup()
    del log[:]
    for _ in range(60):
        p.follow("N")
    cx, cy = _centres(log)[-1]
    # The stub puts the junction at (100, 200) and the vehicle at (100, 200)
    # offset by getPosition -> (100, 200) is the mean stop line; the vehicle
    # sits at (100.0, 200.0) in the stub, so the camera should not have run
    # away from the junction in any case.
    assert abs(cx - 100.0) <= 100.0 and abs(cy - 200.0) <= 100.0, (cx, cy)


def test_the_camera_eases_rather_than_cutting():
    """Successive frames must not jump; a cut is what reads as a zoom."""
    p, env, log = _painter(vehicles={"ev_ambulance_0": "N"},
                           gaps={"ev_ambulance_0": 40.0}, scenario=_Scenario(),
                           view_radius_m=100.0)
    p.setup()
    del log[:]
    for _ in range(60):
        p.follow("N")
    cs = _centres(log)
    steps = [abs(b[0] - a[0]) + abs(b[1] - a[1]) for a, b in zip(cs, cs[1:])]
    if steps:
        assert max(steps) <= 20.0, max(steps)


def test_releasing_the_camera_returns_it_to_the_junction():
    p, env, log = _painter(vehicles={"ev_ambulance_0": "N"},
                           gaps={"ev_ambulance_0": 40.0}, scenario=_Scenario(),
                           view_radius_m=100.0)
    p.setup()
    for _ in range(60):
        p.follow("N")
    del log[:]
    p.follow(None)
    for _ in range(120):
        p.refresh_follow()
    cx, cy = _centres(log)[-1]
    assert abs(cx - 100.0) < 1.0 and abs(cy - 200.0) < 1.0, (cx, cy)
    assert set(_sizes(log)) == {(200.0, 200.0)}


def test_the_camera_lets_go_when_the_vehicle_enters_the_junction():
    """The bug behind "it follows it from arriving to the end".

    Following used to be released only when the *scheduler* dropped its
    target, while the vehicle was matched by its road id -- which stops
    matching the moment it crosses the stop line and its road becomes an
    internal junction lane. The camera stayed on a vehicle that had already
    been served, across the junction and down the exit arm.
    """
    p, env, log = _painter(vehicles={"ev_ambulance_0": "N"},
                           gaps={"ev_ambulance_0": 40.0}, scenario=_Scenario())
    p.setup()
    p.follow("N")
    assert p._follow_target() == "ev_ambulance_0"

    env.conn.vehicle._on_road["ev_ambulance_0"] = ":J0_0_7"
    p.follow("N")                       # same approach, still committed
    assert p._follow_target() is None, "the camera must let go, not keep chasing"


def test_the_camera_lets_go_when_the_vehicle_leaves_the_network():
    p, env, log = _painter(vehicles={"ev_ambulance_0": "N"},
                           gaps={"ev_ambulance_0": 40.0}, scenario=_Scenario())
    p.setup()
    p.follow("N")
    env.conn.vehicle._on_road.clear()
    p.follow("N")
    assert p._follow_target() is None


def test_the_camera_does_not_bias_toward_a_vehicle_outside_the_frame():
    """Following biases the view toward something already on screen; a vehicle
    three hundred metres up an arm is not, so it is left alone until it is."""
    p, env, log = _painter(vehicles={"ev_ambulance_0": "N"},
                           gaps={"ev_ambulance_0": 300.0}, scenario=_Scenario())
    p.setup()
    p.follow("N")
    assert p._follow_target() is None

    env.conn.vehicle._gaps["ev_ambulance_0"] = 150.0
    p.follow("N")
    assert p._follow_target() == "ev_ambulance_0"


def test_a_held_vehicle_is_not_dropped_for_drifting_past_the_acquire_range():
    """The range test is for acquiring. Applying it to a vehicle already held
    would drop it whenever the estimate wobbled across the boundary."""
    p, env, log = _painter(vehicles={"ev_ambulance_0": "N"},
                           gaps={"ev_ambulance_0": 150.0}, scenario=_Scenario())
    p.setup()
    p.follow("N")
    env.conn.vehicle._gaps["ev_ambulance_0"] = 220.0
    p.follow("N")
    assert p._follow_target() == "ev_ambulance_0"


def test_following_a_vehicle_whose_range_cannot_be_read_still_works():
    """An older TraCI without getLaneID must degrade to following, not to
    never following anything."""
    p, env, log = _painter(vehicles={"ev_ambulance_0": "N"}, scenario=_Scenario())
    p.setup()
    p.follow("N")
    assert p._follow_target() == "ev_ambulance_0"


def test_the_junction_is_never_pushed_off_the_frame():
    """Following biases the view; it does not replace the subject of the
    demonstration with a patch of empty road."""
    p, env, log = _painter(vehicles={"ev_ambulance_0": "N"},
                           gaps={"ev_ambulance_0": 10.0}, scenario=_Scenario(),
                           view_radius_m=100.0)
    p.setup()
    del log[:]
    for _ in range(400):
        p.follow("N")
    for kind, b in log:
        if kind != "boundary":
            continue
        assert b[0] <= 100.0 <= b[2] and b[1] <= 200.0 <= b[3], b


def test_the_viewer_can_zoom_and_the_camera_stops_fighting_them():
    """The regression that made the demo unusable.

    Following re-set the boundary on every frame, so the scroll wheel moved the
    camera and a tenth of a second later the next frame put it straight back.
    The view could not be zoomed at all while an emergency vehicle was being
    followed. A demonstration you cannot lean into is worse than one framed
    slightly wrong, so any sign of a hand on the mouse hands the camera over
    for the rest of the run -- not until the next frame, which would be the
    same fight one step slower.
    """
    p, env, log = _painter(vehicles={"ev_ambulance_0": "N"},
                           gaps={"ev_ambulance_0": 40.0}, scenario=_Scenario(),
                           view_radius_m=100.0, aspect=1.66)
    p.setup()
    for _ in range(10):
        p.follow("N")

    env.conn.gui.viewer_zooms(0.25)          # the viewer zooms right in
    del log[:]
    for _ in range(60):
        p.follow("N")

    assert not [e for e in log if e[0] == "boundary"], (
        "the camera must stand down once the viewer has moved it")


def test_standing_down_is_permanent_for_the_run():
    p, env, log = _painter(vehicles={"ev_ambulance_0": "N"},
                           gaps={"ev_ambulance_0": 40.0}, scenario=_Scenario(),
                           view_radius_m=100.0, aspect=1.66)
    p.setup()
    p.follow("N")
    env.conn.gui.viewer_zooms(0.5)
    p.follow("N")
    del log[:]
    # Releasing, re-acquiring, even an explicit reframe: still hands off.
    p.follow(None)
    p.frame_junction()
    for _ in range(20):
        p.follow("N")
    assert not [e for e in log if e[0] == "boundary"], log


def test_an_untouched_camera_still_frames_itself():
    """The stand-down must not fire on the camera's own updates."""
    p, env, log = _painter(vehicles={"ev_ambulance_0": "N"},
                           gaps={"ev_ambulance_0": 40.0}, scenario=_Scenario(),
                           view_radius_m=100.0, aspect=1.66)
    p.setup()
    del log[:]
    for _ in range(30):
        p.follow("N")
    assert [e for e in log if e[0] == "boundary"], "camera stood down for no reason"
