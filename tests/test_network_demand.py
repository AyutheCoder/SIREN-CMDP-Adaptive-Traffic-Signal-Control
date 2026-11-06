"""Network generation, turn geometry, and demand generation."""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from atsc.config import DATA_DIR, DemandConfig, NetworkConfig
from atsc.demand import DemandGenerator, write_sumocfg
from atsc.netgen import (
    build_network, opposite, rotate_ccw, rotate_cw,
    turn_targets, write_actuated_program,
)

NETS = DATA_DIR / "networks"


# ------------------------------------------------------------------- geometry
def test_direction_algebra():
    assert opposite("N") == "S" and opposite("E") == "W"
    assert rotate_ccw("N") == "W" and rotate_cw("N") == "E"
    for d in ("N", "E", "S", "W"):
        assert opposite(opposite(d)) == d
        assert rotate_cw(rotate_ccw(d)) == d


def test_turn_targets_are_consistent():
    """Entering from the north means heading south: left is east, right is west."""
    t = turn_targets("N")
    assert t == {"through": "S", "left": "E", "right": "W"}
    for entry in ("N", "E", "S", "W"):
        t = turn_targets(entry)
        assert len({t["through"], t["left"], t["right"]}) == 3
        assert entry not in t.values(), "no movement may exit back up its own approach"


# -------------------------------------------------------------------- network
def test_single_intersection_builds():
    net, builder = build_network(NetworkConfig(name="test_single"), NETS, force=True)
    assert net.exists()
    assert builder.junctions == ["J0_0"]
    assert len(builder.incoming_edges("J0_0")) == 4
    assert len(builder.outgoing_edges("J0_0")) == 4


def test_generated_program_has_multiple_green_phases():
    net, _ = build_network(NetworkConfig(name="test_single"), NETS)
    root = ET.parse(str(net)).getroot()
    tls = root.findall("tlLogic")
    assert len(tls) == 1
    greens = [p.get("state") for p in tls[0].findall("phase")
              if any(c in "Gg" for c in p.get("state", "")) and "y" not in p.get("state", "")]
    assert len(greens) >= 2, greens
    # No two conflicting movements may be green in the same phase; check at
    # least that phases are distinct.
    assert len(set(greens)) == len(greens)


def test_grid_network_builds_and_connects():
    net, builder = build_network(
        NetworkConfig(name="test_grid", rows=1, cols=3), NETS, force=True)
    assert net.exists()
    assert builder.junctions == ["J0_0", "J0_1", "J0_2"]
    # The middle junction's eastern neighbour must be a junction, not a border.
    assert builder.neighbour["J0_1"]["E"] == "J0_2"
    assert builder.neighbour["J0_0"]["W"].startswith("B")


def test_actuated_program_is_written_for_every_tls():
    net, _ = build_network(NetworkConfig(name="test_single"), NETS)
    with tempfile.TemporaryDirectory() as d:
        out = write_actuated_program(net, Path(d) / "act.add.xml",
                                     min_dur=8.0, max_dur=55.0)
        root = ET.parse(str(out)).getroot()
        logics = root.findall("tlLogic")
        assert len(logics) == 1
        assert logics[0].get("type") == "actuated"
        greens = [p for p in logics[0].findall("phase") if p.get("minDur")]
        assert len(greens) >= 2
        for p in greens:
            assert float(p.get("minDur")) == 8.0
            assert float(p.get("maxDur")) == 55.0


# --------------------------------------------------------------------- demand
def _generator(rows=1, cols=1, **kwargs) -> DemandGenerator:
    _, builder = build_network(
        NetworkConfig(name=f"test_dem_{rows}x{cols}", rows=rows, cols=cols), NETS, force=True)
    return DemandGenerator(builder, DemandConfig(**kwargs))


def test_routes_are_contiguous_edge_chains():
    gen = _generator(rows=1, cols=3)
    builder = gen.builder
    borders = sorted(builder.border_pos)
    found = 0
    for src in borders[:4]:
        for dst in borders:
            edges = gen.route_edges(src, dst)
            if not edges:
                continue
            found += 1
            # Consecutive edges must share a node: "AtoB" then "BtoC".
            for a, b in zip(edges, edges[1:]):
                assert a.split("to")[1] == b.split("to")[0], (a, b)
    assert found > 0


def test_routes_never_pass_through_a_border_node():
    gen = _generator(rows=1, cols=3)
    borders = sorted(gen.builder.border_pos)
    for src in borders:
        for dst in borders:
            edges = gen.route_edges(src, dst)
            if not edges:
                continue
            interior = [e.split("to")[1] for e in edges[:-1]]
            assert all(not n.startswith("B") for n in interior), edges


def test_generated_demand_matches_requested_rate():
    with tempfile.TemporaryDirectory() as d:
        gen = _generator(horizon_s=3600, seed=3, base_flow_vph=600.0,
                         ev_rate_per_hour=20.0, regime="balanced")
        sc = gen.generate(Path(d) / "r.rou.xml")
        # 4 approaches x 600 veh/h for one hour, Bernoulli-thinned per second.
        assert 2000 < sc.n_background < 2800, sc.n_background
        assert 8 <= sc.n_emergency <= 36, sc.n_emergency


def test_emergency_metadata_is_complete():
    with tempfile.TemporaryDirectory() as d:
        gen = _generator(horizon_s=3600, seed=1, ev_rate_per_hour=60.0)
        sc = gen.generate(Path(d) / "r.rou.xml")
        assert sc.n_emergency > 0
        for vid, ev in sc.emergency_vehicles.items():
            assert ev.veh_id == vid
            assert ev.veh_class in ("ambulance", "fire", "police")
            assert 0.0 < ev.priority <= 1.0
            assert len(ev.route_edges) >= 2
            assert ev.entry_edge == ev.route_edges[0]
            assert ev.junctions, "an emergency vehicle must cross a signal"

        assert sc.meta_file is not None and sc.meta_file.exists()
        import json
        meta = json.loads(sc.meta_file.read_text(encoding="utf-8"))
        assert meta["n_emergency"] == sc.n_emergency
        assert set(meta["emergency_vehicles"]) == set(sc.emergency_vehicles)


def test_route_file_is_valid_xml_with_vtypes():
    with tempfile.TemporaryDirectory() as d:
        gen = _generator(horizon_s=600, seed=2, ev_rate_per_hour=60.0)
        sc = gen.generate(Path(d) / "r.rou.xml")
        root = ET.parse(str(sc.route_file)).getroot()
        vtypes = {v.get("id") for v in root.findall("vType")}
        assert {"car", "ambulance", "fire", "police"} <= vtypes
        for v in root.findall("vType"):
            if v.get("id") != "car":
                assert v.get("vClass") == "emergency"
        vehicles = root.findall("vehicle")
        assert len(vehicles) == sc.n_background + sc.n_emergency
        # Departures must be non-decreasing so SUMO accepts the file.
        departs = [float(v.get("depart")) for v in vehicles]
        assert departs == sorted(departs)


EV_TYPES = ("ambulance", "fire", "police")


def test_bluelight_device_is_off_by_default():
    """Yielding traffic would confound the measurement of signal-induced delay."""
    with tempfile.TemporaryDirectory() as d:
        gen = _generator(horizon_s=300, seed=1, ev_rate_per_hour=60.0)
        plain = gen.generate(Path(d) / "a.rou.xml")
        assert "bluelight" not in plain.route_file.read_text(encoding="utf-8")
        blue = gen.generate(Path(d) / "b.rou.xml", bluelight=True)
        assert "bluelight" in blue.route_file.read_text(encoding="utf-8")


def test_both_bluelight_settings_emit_parseable_vtypes():
    """Parse the file, rather than grep it for the word.

    Asserting only that "bluelight" appears in the text is what let the device
    ship broken: the template closed the element as ``</vType/>``, so SUMO
    refused the route file with "unterminated end tag" and ``bluelight=True``
    could never start a simulation, while a substring check passed happily.
    Parsing is the assertion that would have caught it.
    """
    with tempfile.TemporaryDirectory() as d:
        gen = _generator(horizon_s=300, seed=1, ev_rate_per_hour=60.0)
        for blue in (False, True):
            sc = gen.generate(Path(d) / f"r_{blue}.rou.xml", bluelight=blue)
            root = ET.parse(sc.route_file).getroot()   # raises on malformed XML
            types = {v.get("id"): v for v in root.iter("vType")}
            for name in EV_TYPES:
                assert name in types, (name, blue)
                keys = [p.get("key") for p in types[name].iter("param")]
                assert ("has.bluelight.device" in keys) == blue, (name, blue, keys)


def test_demand_regimes_shape_arrivals_differently():
    counts = {}
    with tempfile.TemporaryDirectory() as d:
        for regime in ("balanced", "asymmetric", "peak", "surge"):
            gen = _generator(horizon_s=1800, seed=4, regime=regime)
            counts[regime] = gen.generate(Path(d) / f"{regime}.rou.xml").n_background
    assert len(set(counts.values())) > 1, counts


def test_sumocfg_references_inputs():
    with tempfile.TemporaryDirectory() as d:
        net, _ = build_network(NetworkConfig(name="test_single"), NETS)
        gen = _generator(horizon_s=300, seed=1)
        sc = gen.generate(Path(d) / "r.rou.xml")
        cfg = write_sumocfg(Path(d) / "s.sumocfg", net, sc.route_file,
                            additional_files=[Path(d) / "extra.add.xml"])
        root = ET.parse(str(cfg)).getroot()
        assert root.find("input/net-file").get("value").endswith(".net.xml")
        assert root.find("input/additional-files") is not None


def test_spread_emergency_arrivals_keep_the_rate_but_lose_the_clumps():
    """The demonstration schedule: same count, gaps that are not exponential.

    Poisson is the right model and is what every reported number uses, but its
    gaps are exponential, so a demo at any given rate shows three vehicles in
    the first two minutes and then nothing. The spread schedule has to keep the
    *rate* honest -- it is not a different amount of traffic -- while making
    the run watchable end to end.
    """
    with tempfile.TemporaryDirectory() as d:
        gen = _generator(horizon_s=1200, seed=7, ev_rate_per_hour=24.0,
                         ev_schedule="spread")
        sc = gen.generate(Path(d) / "spread.rou.xml")
        times = sorted(ev.depart_s for ev in sc.emergency_vehicles.values())

        assert len(times) == 8, times                    # 24/h over 1200 s
        gaps = [b - a for a, b in zip(times, times[1:])]
        nominal = 1200.0 / 8.0
        # Every gap within the configured jitter of the nominal one, which is
        # the property Poisson does not have and this exists to provide.
        assert all(abs(g - nominal) <= 2 * 0.18 * nominal + 1e-6 for g in gaps), gaps
        # And spread over the whole horizon rather than bunched at either end.
        assert times[0] < 0.25 * 1200 and times[-1] > 0.75 * 1200, times


def test_the_spread_schedule_leaves_background_traffic_untouched():
    """It draws its randomness after the ordinary traffic has been drawn, so
    switching schedules must not silently re-roll the whole demand."""
    with tempfile.TemporaryDirectory() as d:
        out = []
        for schedule in ("poisson", "spread"):
            gen = _generator(horizon_s=900, seed=3, ev_rate_per_hour=24.0,
                             ev_schedule=schedule)
            out.append(gen.generate(Path(d) / f"{schedule}.rou.xml").n_background)
        assert out[0] == out[1], out


def test_an_unknown_emergency_schedule_is_refused():
    with tempfile.TemporaryDirectory() as d:
        gen = _generator(horizon_s=300, seed=1, ev_schedule="whenever")
        try:
            gen.generate(Path(d) / "r.rou.xml")
        except ValueError as exc:
            assert "whenever" in str(exc)
        else:
            raise AssertionError("a typo'd schedule must not silently mean poisson")
