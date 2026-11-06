"""Lane clearing, driven against a recording stub instead of SUMO.

The layer is geometry plus the junction's connection table, so the failures
that matter -- pushing a car into a lane its own turn cannot be made from,
leaving a vehicle under modified lane-change rules after the event, acting on a
vehicle that is still half a kilometre away -- are all visible without a
simulator, and a test that needed one could not run unattended.

The stub network is one approach edge ``A`` of three lanes, connected the way
``netgen`` connects them: lane 0 through and right, lane 1 through only, lane 2
left only. That asymmetry is the whole reason the route check exists.
"""

from __future__ import annotations

from atsc.control.clearance import (
    CLEARING_LC_MODE, DEFAULT_LC_MODE, CorridorClearance)

LANE_EXITS = {
    "A_0": ["THROUGH", "RIGHT"],
    "A_1": ["THROUGH"],
    "A_2": ["LEFT"],
}
LANE_LENGTH = 320.0


class _Veh:
    def __init__(self, vid, lane_idx, pos, route, route_i=0, speed=5.0):
        self.id = vid
        self.lane_idx = lane_idx
        self.pos = pos
        self.route = route
        self.route_i = route_i
        self.speed = speed


class _VehicleDomain:
    def __init__(self, world):
        self.w = world

    def getIDList(self):
        return list(self.w.vehicles)

    def getRoadID(self, vid):
        return "A"

    def getLaneID(self, vid):
        return f"A_{self.w.vehicles[vid].lane_idx}"

    def getLaneIndex(self, vid):
        return self.w.vehicles[vid].lane_idx

    def getLanePosition(self, vid):
        return self.w.vehicles[vid].pos

    def getSpeed(self, vid):
        return self.w.vehicles[vid].speed

    def getRoute(self, vid):
        return self.w.vehicles[vid].route

    def getRouteIndex(self, vid):
        return self.w.vehicles[vid].route_i

    def changeLane(self, vid, idx, duration):
        self.w.changes.append((vid, idx, duration))

    def setLaneChangeMode(self, vid, mode):
        self.w.modes.append((vid, mode))


class _LaneDomain:
    def __init__(self, world):
        self.w = world

    def getLength(self, lane):
        return LANE_LENGTH

    def getLinks(self, lane):
        # SUMO hands back the lane reached *after* the internal link, so the
        # stub does too: the code under test must read the edge off it.
        return [(f"{e}_0", True, True, False) for e in LANE_EXITS.get(lane, ())]

    def getEdgeID(self, lane):
        return lane.rsplit("_", 1)[0]


class _EdgeDomain:
    def __init__(self, world):
        self.w = world

    def getLaneNumber(self, edge):
        return 3

    def getLastStepVehicleIDs(self, edge):
        return list(self.w.vehicles)


class _Conn:
    def __init__(self, world):
        self.vehicle = _VehicleDomain(world)
        self.lane = _LaneDomain(world)
        self.edge = _EdgeDomain(world)


class _World:
    """One approach, three lanes, and whatever vehicles a test puts on it."""

    def __init__(self, vehicles):
        self.vehicles = {v.id: v for v in vehicles}
        self.changes = []
        self.modes = []
        self.conn = _Conn(self)
        self.sim_time = 100.0
        self.controlled_lanes = ["A_0", "A_1", "A_2"]
        self.lane_to_approach = {ln: "A" for ln in self.controlled_lanes}
        self.scenario = _Scenario()


class _Scenario:
    emergency_vehicles = {"ev_ambulance_0": object()}


def _clearance(vehicles, **kw):
    env = _World(vehicles)
    c = CorridorClearance(env, **kw)
    c.reset()
    return c, env


def _ev(lane_idx, pos, route=("A", "THROUGH"), **kw):
    return _Veh("ev_ambulance_0", lane_idx, pos, route, **kw)


def _lane_of(env, vid):
    """The lane index the layer last asked ``vid`` to move to."""
    for v, idx, _d in reversed(env.changes):
        if v == vid:
            return idx
    return None


# ------------------------------------------------------------------ triggering
def test_nothing_happens_while_the_vehicle_is_still_far_away():
    """The trigger is a distance, and it has to actually bound something.

    At 320 m from the stop line the approach is not this vehicle's problem yet,
    and holding traffic under modified lane-change rules for the length of an
    arm is the failure mode the range exists to prevent.
    """
    c, env = _clearance([_ev(1, 0.0), _Veh("car_0", 1, 100.0, ("A", "THROUGH"))],
                        trigger_m=220.0)
    c.update()
    assert env.changes == []
    assert env.modes == []
    assert c.stats()["clearance_events"] == 0.0


def test_a_car_in_the_way_is_moved_once_the_vehicle_is_close():
    c, env = _clearance([_ev(1, 200.0), _Veh("car_0", 1, 260.0, ("A", "THROUGH"))],
                        trigger_m=220.0)
    c.update()
    assert _lane_of(env, "car_0") == 0        # the only other through lane
    assert c.stats()["vehicles_moved_aside"] == 1.0
    assert c.stats()["clearance_events"] == 1.0


def test_cars_behind_and_beside_are_left_alone():
    """Only what is *in front, in the way*. A car behind is not blocking, and a
    car in the next lane over is where we are trying to send traffic."""
    c, env = _clearance([
        _ev(1, 200.0),
        _Veh("behind", 1, 150.0, ("A", "THROUGH")),
        _Veh("beside", 0, 260.0, ("A", "THROUGH")),
    ], trigger_m=220.0)
    c.update()
    assert [v for v, _i, _d in env.changes] == []


def test_a_car_too_far_ahead_to_matter_is_left_alone():
    c, env = _clearance([_ev(1, 100.0), _Veh("car_0", 1, 300.0, ("A", "THROUGH"))],
                        trigger_m=320.0, ahead_m=50.0)
    c.update()
    assert env.changes == []


# ------------------------------------------------------------- the route check
def test_a_right_turner_is_never_pushed_into_the_left_only_lane():
    """The check that stops this from jamming the approach it is clearing.

    A right-turning car sits in lane 0 and only lane 0 may turn right. Moving
    it sideways to satisfy "get out of the way" would leave it at the stop line
    unable to make its turn, stopped, with the emergency vehicle behind it --
    strictly worse than leaving it alone. It has nowhere to go, so it stays.
    """
    c, env = _clearance([_ev(0, 200.0, route=("A", "THROUGH")),
                         _Veh("right", 0, 260.0, ("A", "RIGHT"))],
                        trigger_m=220.0, move_ev=False)
    c.update()
    assert env.changes == []
    assert env.modes == []


def test_a_through_car_is_moved_only_to_a_through_lane():
    """Lane 2 is left-only and lane 0 takes through traffic, so 0 is the answer
    even though 2 is empty and equally adjacent."""
    c, env = _clearance([
        _ev(1, 200.0, route=("A", "THROUGH")),
        _Veh("car_0", 1, 250.0, ("A", "THROUGH")),
        _Veh("filler", 0, 300.0, ("A", "THROUGH")),
    ], trigger_m=220.0)
    c.update()
    assert _lane_of(env, "car_0") == 0


def test_a_vehicle_on_its_last_edge_is_not_moved():
    """No next edge means no way to check the turn is still makeable, and
    guessing is what the check exists to avoid."""
    c, env = _clearance([_ev(1, 200.0),
                         _Veh("car_0", 1, 250.0, ("A",), route_i=0)],
                        trigger_m=220.0, move_ev=False)
    c.update()
    assert env.changes == []


# --------------------------------------------------- the vehicle's own lane
def test_the_vehicle_moves_to_the_freer_of_the_lanes_its_turn_allows():
    c, env = _clearance([
        _ev(1, 200.0, route=("A", "THROUGH")),
        _Veh("a", 1, 240.0, ("A", "THROUGH")),
        _Veh("b", 1, 250.0, ("A", "THROUGH")),
        _Veh("c", 1, 260.0, ("A", "THROUGH")),
    ], trigger_m=220.0, move_ev=True)
    c.update()
    assert _lane_of(env, "ev_ambulance_0") == 0


def test_the_vehicle_stays_put_when_the_other_lane_is_no_better():
    """Or it weaves across the approach for nothing, which costs it the very
    seconds this is trying to save."""
    c, env = _clearance([
        _ev(1, 200.0, route=("A", "THROUGH")),
        _Veh("a", 1, 240.0, ("A", "THROUGH")),
        _Veh("b", 0, 250.0, ("A", "THROUGH")),
    ], trigger_m=220.0, move_ev=True)
    c.update()
    assert _lane_of(env, "ev_ambulance_0") is None


def test_a_left_turning_vehicle_is_never_moved_out_of_the_left_lane():
    """Only lane 2 may turn left, so there is no freer lane to move to."""
    c, env = _clearance([
        _ev(2, 200.0, route=("A", "LEFT")),
        _Veh("a", 2, 240.0, ("A", "LEFT")),
        _Veh("b", 2, 250.0, ("A", "LEFT")),
    ], trigger_m=220.0, move_ev=True)
    c.update()
    assert _lane_of(env, "ev_ambulance_0") is None


# ------------------------------------------------------------------- release
def test_every_moved_vehicle_gets_its_own_lane_change_model_back():
    """A car left under clearing rules drives the rest of its trip with keep
    right and speed gain switched off, long after the siren has gone."""
    c, env = _clearance([_ev(1, 200.0), _Veh("car_0", 1, 260.0, ("A", "THROUGH"))],
                        trigger_m=220.0)
    c.update()
    assert (("car_0", CLEARING_LC_MODE)) in env.modes
    c.close()
    assert env.modes[-1] == ("car_0", DEFAULT_LC_MODE)


def test_the_road_is_released_once_the_vehicle_has_passed():
    c, env = _clearance([_ev(1, 200.0), _Veh("car_0", 1, 260.0, ("A", "THROUGH"))],
                        trigger_m=220.0, request_s=0.0)
    c.update()
    env.vehicles.pop("ev_ambulance_0")
    env.sim_time += 30.0
    c.update()
    assert env.modes[-1] == ("car_0", DEFAULT_LC_MODE)


def test_the_mode_is_set_once_however_long_the_vehicle_is_followed():
    """The request is refreshed every step; the mode must not be."""
    c, env = _clearance([_ev(1, 200.0), _Veh("car_0", 1, 260.0, ("A", "THROUGH"))],
                        trigger_m=220.0)
    for _ in range(5):
        c.update()
    assert [m for m in env.modes if m[0] == "car_0"] == [("car_0", CLEARING_LC_MODE)]
    assert c.stats()["vehicles_moved_aside"] == 1.0


def test_disabled_is_inert():
    c, env = _clearance([_ev(1, 200.0), _Veh("car_0", 1, 260.0, ("A", "THROUGH"))],
                        enabled=False)
    c.update()
    assert env.changes == [] and env.modes == []
