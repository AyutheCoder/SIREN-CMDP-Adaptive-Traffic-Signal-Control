"""End-to-end environment behaviour against a live SUMO process.

These are slower than the unit tests but they are the ones that matter: they
check that phase discovery, the shield guarantee, the reward/cost split and the
metric pipeline all hold when SUMO is actually running, not just in isolation.
"""

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path

import numpy as np

from atsc.agents.baselines import GreedyQueue, MaxPressure, build_baseline, fixed_time_from_demand
from atsc.config import DATA_DIR, DemandConfig, EnvConfig, NetworkConfig
from atsc.env import SumoSignalEnv
from atsc.evaluation.metrics import jains_fairness, paired_bootstrap_pvalue, parse_tripinfo
from atsc.evaluation.runner import run_episode
from atsc.netgen import build_network
from atsc.sensing.sensor import cardioid_gain_db, endfire_pair_gain_db
from atsc.sensing.siren_synth import received_spl_db

NETS = DATA_DIR / "networks"


def _cfg(horizon: int = 500, ev_rate: float = 60.0, sensing: str = "surrogate") -> EnvConfig:
    cfg = EnvConfig()
    cfg.demand = DemandConfig(horizon_s=horizon, seed=11, ev_rate_per_hour=ev_rate)
    cfg.sensing_mode = sensing
    cfg.collect_emissions = False
    return cfg


def _env(cfg: EnvConfig, label: str, tripinfo: Path | None = None) -> SumoSignalEnv:
    net, _ = build_network(NetworkConfig(name="test_single"), NETS)
    return SumoSignalEnv(cfg, net_file=net, work_dir=DATA_DIR / "runs" / "tests",
                         label=label, tripinfo_path=tripinfo)


# ------------------------------------------------------------------ discovery
def test_phase_and_approach_discovery():
    env = _env(_cfg(), "t_discover")
    try:
        obs = env.reset(seed=1)
        assert env.n_actions >= 2
        assert len(env.approaches) == 4
        assert sorted(env.approach_direction[a] for a in env.approaches) == ["E", "N", "S", "W"]
        assert len(env.controlled_lanes) == 12
        # Every discovered green phase must actually green some lane.
        for p in env.phases:
            assert p.green_lanes and p.approaches
            assert "y" not in p.state.lower()
        assert obs.shape == (env.observation_dim,)
        assert np.isfinite(obs).all()
    finally:
        env.close()


def test_the_mast_array_separates_the_approaches_at_every_range():
    """The property the geometry exists for, asserted directly.

    This used to be a proxy -- "each mast is 20 to 200 m upstream and no two are
    within 50 m of each other" -- on the argument that masts near the junction
    would be equidistant from everything and could not localise. The proxy
    passed while the array was blind: at the 70 m offset it encoded, a vehicle
    at the stop line reached its own mast only 1.5 dB louder than the loudest
    wrong one, against several decibels of measurement noise, and the filter
    put a stopped ambulance on the wrong arm for sixty seconds at a time.

    Separation is what matters, so separation is what is measured: for a source
    anywhere on one arm, its own mast must hear it clearly louder than any
    other mast does. The far field is carried by the cardioid's pattern and the
    near field by range, and the geometry has to keep both.
    """
    env = _env(_cfg(), "t_mic")
    try:
        env.reset(seed=1)
        fb = env.cfg.acoustic.directivity_front_back_db
        dual = env.cfg.acoustic.dual_element_mast
        worst = float("inf")
        worst_near = float("inf")
        for approach in env.approaches:
            ax, ay = env.mic_axis[approach]
            sx, sy = env.stop_line[approach]
            for s_m in (300.0, 200.0, 120.0, 60.0, 25.0, 8.0, 1.0):
                px, py = sx + ax * s_m, sy + ay * s_m
                heard = {}
                for b in env.approaches:
                    bx, by = env.mic_pos[b]
                    ux, uy = env.mic_axis[b]
                    dx, dy = px - bx, py - by
                    d = max(math.hypot(dx, dy), 1.0)
                    theta = math.acos(max(-1.0, min(1.0, (dx * ux + dy * uy) / d)))
                    if dual:
                        f, bk = endfire_pair_gain_db(theta, fb)
                        gain = max(f, bk)
                    else:
                        gain = cardioid_gain_db(theta, fb)
                    heard[b] = received_spl_db("ambulance", d, gain)
                other = max(v for k, v in heard.items() if k != approach)
                sep = heard[approach] - other
                worst = min(worst, sep)
                if s_m <= 25.0:
                    worst_near = min(worst_near, sep)
        noise = env.cfg.acoustic.level_noise_db
        # Everywhere on every arm, the correct mast wins by enough to survive
        # the level noise. The floor is set by the cardioid's pattern and
        # measures 7.2 dB on the shipped geometry -- just under three standard
        # deviations, against 1.5 dB (0.6 sd) at the 70 m offset this replaced.
        assert worst > 2.5 * noise, worst
        # And near the stop line -- the case the whole change is about, where a
        # vehicle sits at the red waiting for its green -- by much more.
        assert worst_near > 4.0 * noise, worst_near

        # The masts must still be distinguishable positions on distinct arms.
        assert len({env.mic_pos[a] for a in env.approaches}) == len(env.approaches)
        for approach in env.approaches:
            mic = np.array(env.mic_pos[approach])
            stop = np.array(env.stop_line[approach])
            axis = np.array(env.mic_axis[approach])
            # Upstream of its own stop line, not beyond it.
            assert float(np.dot(mic - stop, axis)) > 0.0, approach
    finally:
        env.close()


def test_yellow_state_ambers_only_movements_losing_right_of_way():
    env = _env(_cfg(), "t_yellow")
    try:
        env.reset(seed=1)
        a, b = env.phases[0], env.phases[2 % env.n_actions]
        y = env._yellow_state(a, b)
        assert len(y) == len(a.state)
        for i, ch in enumerate(y):
            was_green = a.state[i] in "Gg"
            will_be_green = i in set(b.green_link_indices)
            if was_green and not will_be_green:
                assert ch == "y", (i, ch)
            elif was_green and will_be_green:
                assert ch in "Gg", (i, ch)
            else:
                assert ch == "r", (i, ch)
    finally:
        env.close()


# ----------------------------------------------------------------- step / mask
def test_action_mask_never_empty_and_step_respects_it():
    env = _env(_cfg(), "t_mask")
    try:
        obs = env.reset(seed=2)
        rng = np.random.default_rng(0)
        for _ in range(60):
            mask = env.action_mask()
            assert mask.any(), "shield must always leave a legal action"
            action = int(rng.choice(np.flatnonzero(mask)))
            obs, r, c, done, info = env.step(action)
            assert obs.shape == (env.observation_dim,)
            assert np.isfinite(obs).all()
            assert np.isfinite(r) and np.isfinite(c)
            assert c >= 0.0, "constraint cost must be non-negative"
            assert info["dt"] > 0
            if done:
                break
    finally:
        env.close()


def test_shield_override_forces_a_legal_action():
    """Proposing an illegal action must be silently corrected, not crash."""
    env = _env(_cfg(), "t_override")
    try:
        env.reset(seed=3)
        overrides = 0
        for _ in range(40):
            mask = env.action_mask()
            illegal = np.flatnonzero(~mask)
            action = int(illegal[0]) if illegal.size else 0
            _, _, _, done, info = env.step(action)
            overrides += bool(info["shield_overridden"])
            if done:
                break
        assert overrides > 0, "expected the shield to intervene at least once"
        assert env.shield.override_rate > 0
    finally:
        env.close()


def test_starvation_bound_holds_inside_sumo():
    """The S3 guarantee must survive contact with the simulator."""
    cfg = _cfg(horizon=900)
    env = _env(cfg, "t_starve")
    try:
        env.reset(seed=4)
        rng = np.random.default_rng(0)
        done = False
        while not done:
            mask = env.action_mask()
            # Adversarial: prefer whichever legal phase has waited least.
            legal = np.flatnonzero(mask)
            action = int(legal[np.argmin(env.time_since_served[legal])])
            _, _, _, done, _ = env.step(action)
        worst = env.episode_summary()["max_movement_wait_s"]
        assert worst <= cfg.control.max_starvation_s + 1e-6, worst
    finally:
        env.close()


def test_cost_is_zero_without_emergency_vehicles():
    cfg = _cfg(horizon=400, ev_rate=0.0)
    env = _env(cfg, "t_nocost")
    try:
        env.reset(seed=5)
        total = 0.0
        done = False
        while not done:
            mask = env.action_mask()
            _, _, c, done, _ = env.step(int(np.flatnonzero(mask)[0]))
            total += c
        assert total == 0.0, total
        assert env.episode_summary()["n_emergency"] == 0
    finally:
        env.close()


def test_cost_is_positive_when_emergency_vehicles_are_delayed():
    cfg = _cfg(horizon=900, ev_rate=120.0)
    env = _env(cfg, "t_cost")
    try:
        env.reset(seed=6)
        total = 0.0
        done = False
        while not done:
            # Deliberately serve the least useful phase to build delay.
            mask = env.action_mask()
            queues = env.phase_queues()
            legal = np.flatnonzero(mask)
            action = int(legal[np.argmin(queues[legal])])
            _, _, c, done, _ = env.step(action)
            total += c
        summary = env.episode_summary()
        assert summary["n_emergency"] > 0
        assert total > 0.0
        assert summary["ev_delay_s"] > 0.0
    finally:
        env.close()


def test_sensing_mode_none_zeroes_the_belief_features():
    cfg = _cfg(horizon=300, sensing="none")
    env = _env(cfg, "t_nosense")
    try:
        obs = env.reset(seed=7)
        n_belief = len(env.approaches) * 3 + env.n_actions
        for _ in range(10):
            obs, _, _, done, _ = env.step(int(np.flatnonzero(env.action_mask())[0]))
            assert np.allclose(obs[-n_belief:], 0.0)
            assert np.allclose(env.phase_urgency(), 0.0)
            if done:
                break
    finally:
        env.close()


def test_phase_urgency_tracks_the_approach_an_ev_is_on():
    """The phase-level feature must light up for phases serving the EV's approach."""
    cfg = _cfg(horizon=900, ev_rate=200.0, sensing="oracle")
    env = _env(cfg, "t_urgency")
    try:
        env.reset(seed=8)
        seen_match = False
        done = False
        while not done:
            _, _, _, done, _ = env.step(int(np.flatnonzero(env.action_mask())[0]))
            urgency = env.phase_urgency()
            assert urgency.shape == (env.n_actions,)
            assert (urgency >= 0).all() and (urgency <= 1.5).all()
            if urgency.max() > 0.3:
                hot = int(np.argmax(urgency))
                believed = max(env.belief.beliefs().items(), key=lambda kv: kv[1].presence)[0]
                if believed in env.phases[hot].approaches:
                    seen_match = True
        assert seen_match, "no phase ever became urgent for the believed approach"
    finally:
        env.close()


def test_oracle_sensing_detects_present_emergency_vehicles():
    cfg = _cfg(horizon=900, ev_rate=200.0, sensing="oracle")
    env = _env(cfg, "t_oracle")
    try:
        env.reset(seed=8)
        peak = 0.0
        done = False
        while not done:
            _, _, _, done, _ = env.step(int(np.flatnonzero(env.action_mask())[0]))
            peak = max(peak, max(b.presence for b in env.belief.beliefs().values()))
        assert peak > 0.9, peak
    finally:
        env.close()


# ------------------------------------------------------------------ controllers
def test_pressure_and_queue_helpers_have_right_shape():
    env = _env(_cfg(), "t_helpers")
    try:
        env.reset(seed=9)
        for _ in range(5):
            env.step(int(np.flatnonzero(env.action_mask())[0]))
        assert env.phase_pressure().shape == (env.n_actions,)
        assert env.phase_queues().shape == (env.n_actions,)
        assert (env.phase_queues() >= 0).all()
        for a in env.approaches:
            assert env.phases_serving_approach(a), a
    finally:
        env.close()


def test_webster_plan_is_within_the_timing_envelope():
    env = _env(_cfg(), "t_webster")
    try:
        cfg = env.cfg
        env.reset(seed=10)
        plan = fixed_time_from_demand(env, cfg.demand)
        assert len(plan.green_durations) == env.n_actions
        for g in plan.green_durations:
            assert cfg.control.min_green_s <= g <= cfg.control.max_green_s, g
    finally:
        env.close()


def test_full_episode_produces_tripinfo_metrics():
    cfg = _cfg(horizon=600, ev_rate=90.0)
    with tempfile.TemporaryDirectory() as d:
        trip = Path(d) / "trip.xml"
        env = _env(cfg, "t_trip", tripinfo=trip)
        summary = run_episode(env, lambda e: MaxPressure(), seed=12)

    assert summary["n_trips"] > 0
    assert summary["throughput"] > 0
    assert np.isfinite(summary["mean_time_loss_s"])
    assert summary["mean_time_loss_s"] >= 0
    assert 0.0 < summary["jain_fairness"] <= 1.0
    assert summary["n_emergency_trips"] > 0
    assert np.isfinite(summary["ev_mean_time_loss_s"])
    assert summary["max_movement_wait_s"] <= cfg.control.max_starvation_s + 1e-6


def test_actuated_baseline_runs_under_sumo_control():
    from atsc.netgen import write_actuated_program
    cfg = _cfg(horizon=500)
    net, _ = build_network(NetworkConfig(name="test_single"), NETS)
    with tempfile.TemporaryDirectory() as d:
        add = write_actuated_program(net, Path(d) / "act.add.xml",
                                     min_dur=cfg.control.min_green_s,
                                     max_dur=cfg.control.max_green_s)
        trip = Path(d) / "trip.xml"
        env = SumoSignalEnv(cfg, net_file=net, work_dir=DATA_DIR / "runs" / "tests",
                            label="t_act", tripinfo_path=trip, additional_files=[add])
        summary = run_episode(env, lambda e: build_baseline("actuated", e, cfg.demand), seed=13)
    assert summary["throughput"] > 0
    assert summary["controller"] == "actuated"
    # SUMO drove the signal, so the env must still have tracked phase changes.
    assert summary["phase_switches"] > 0


def test_two_controllers_are_comparable_on_the_same_seed():
    cfg = _cfg(horizon=600, ev_rate=60.0)
    results = {}
    with tempfile.TemporaryDirectory() as d:
        for name, factory in (("mp", lambda e: MaxPressure()),
                              ("gq", lambda e: GreedyQueue())):
            trip = Path(d) / f"{name}.xml"
            env = _env(cfg, f"t_cmp_{name}", tripinfo=trip)
            results[name] = run_episode(env, factory, seed=21)
    # Identical seed means identical demand, so both must see the same vehicles.
    assert results["mp"]["n_trips"] == results["gq"]["n_trips"]
    assert results["mp"]["departed"] == results["gq"]["departed"]


# --------------------------------------------------------------------- metrics
def test_jain_fairness_bounds():
    assert abs(jains_fairness([5.0, 5.0, 5.0, 5.0]) - 1.0) < 1e-9
    assert jains_fairness([1.0, 1.0, 1.0, 100.0]) < 0.5
    assert 0.0 < jains_fairness([3.0, 5.0, 4.0]) <= 1.0


def test_paired_bootstrap_detects_a_real_difference():
    a = [10.0, 11.0, 9.5, 10.2, 10.8, 9.9]
    b = [14.0, 15.0, 13.4, 14.3, 14.9, 13.8]
    out = paired_bootstrap_pvalue(a, b, n_boot=4000, seed=0)
    assert out["diff"] < 0
    assert out["p_value"] < 0.05
    same = paired_bootstrap_pvalue(a, a, n_boot=4000, seed=0)
    assert same["p_value"] > 0.5


def test_parse_tripinfo_handles_a_missing_file():
    out = parse_tripinfo(Path("does_not_exist.xml"))
    assert out["n_trips"] == 0


def test_served_approaches_track_the_live_signal_not_the_phase_index():
    """Cost attribution must follow the signal, not ``current_phase``.

    ``current_phase`` is only updated at the *end* of a switching epoch, after
    the incoming green has already been running, and during yellow/all-red no
    phase is serving at all. Deriving emergency exposure from the phase index
    therefore credits the wrong phase for part of every switch -- which
    un-credits the agent for exactly the action the constraint exists to
    encourage.
    """
    env = _env(_cfg(horizon=500), "t_serving")
    try:
        env.reset(seed=3)
        rng = np.random.default_rng(0)
        for _ in range(40):
            mask = env.action_mask()
            legal = np.flatnonzero(mask)
            _, _, _, done, _ = env.step(int(rng.choice(legal)))
            live = env.conn.trafficlight.getRedYellowGreenState(env.tls_id)
            expected = {
                env.lane_to_approach[env._link_in_lane[i]]
                for i, ch in enumerate(live)
                if ch in "Gg" and env._link_in_lane[i] is not None
            }
            assert env._serving_approaches == expected, (expected, env._serving_approaches)
            if done:
                break
    finally:
        env.close()


def test_anticipatory_cost_fires_only_for_unserved_ev_approaches():
    """The anticipatory term must be exactly the 'inbound and not served' signal.

    This is the invariant that makes the constraint learnable: if it charges
    cost while the emergency vehicle's approach *is* green, the cost critic
    cannot attribute anything to the action and the multiplier is just noise.
    Tested as an invariant rather than statistically, because total cost also
    contains realised delay from other vehicles and would confound the
    comparison.
    """
    cfg = _cfg(horizon=1200, ev_rate=200.0)
    env = _env(cfg, "t_anticip")
    try:
        env.reset(seed=5)
        rng = np.random.default_rng(0)
        saw_charged = saw_free = False
        done = False
        while not done:
            legal = np.flatnonzero(env.action_mask())
            _, _, _, done, _ = env.step(int(rng.choice(legal)))

            # Ground truth: is any emergency vehicle on an approach that is not
            # currently green? (Read after the step, matching the accumulator.)
            scene = env._acoustic_scene()
            inbound_unserved = any(
                approach not in env._serving_approaches
                and any(s.off_axis_rad < 0.6 and s.radial_velocity_ms < 0 for s in sources)
                for approach, sources in scene.items() if sources
            )
            charged = env._epoch_ev_unserved > 0.0

            if charged:
                saw_charged = True
            if inbound_unserved is False and not charged:
                saw_free = True
            # The accumulator may be non-zero from earlier in the epoch even if
            # the end-of-epoch snapshot is clear, so only the strong direction
            # is asserted: never charge when no emergency vehicle exists at all.
            if not env._ev_waiting:
                assert not charged, "charged with no emergency vehicle present"

        assert saw_charged, "anticipatory cost never fired"
        assert saw_free, "anticipatory cost never rested"
    finally:
        env.close()


def test_teleports_are_counted_not_assumed_zero():
    """Teleports must be measured, because they mask gridlock.

    SUMO removes vehicles stuck past --time-to-teleport and reinserts them
    downstream, which quietly rescues throughput in the runs whose failure
    matters most. An earlier version reported a hard-coded zero here.
    """
    cfg = _cfg(horizon=600, ev_rate=24.0)
    env = _env(cfg, "t_teleport")
    summary = run_episode(env, lambda e: MaxPressure(), seed=777, parse_trips=False)
    assert "teleports" in summary
    assert summary["teleports"] >= 0
    assert summary["teleports_per_1k_veh"] >= 0.0
    # A healthy, undersaturated run should not need teleporting at all.
    assert summary["teleports"] == 0, summary["teleports"]


def test_tripinfo_parser_does_not_invent_a_teleport_count():
    """Teleports are not in tripinfo; the parser must not claim otherwise."""
    cfg = _cfg(horizon=400, ev_rate=24.0)
    with tempfile.TemporaryDirectory() as d:
        trip = Path(d) / "trip.xml"
        env = _env(cfg, "t_trip_tel", tripinfo=trip)
        run_episode(env, lambda e: MaxPressure(), seed=5)
        parsed = parse_tripinfo(trip)
    assert "teleports" not in parsed, "tripinfo cannot report teleports"


# -------------------------------------------------------------------- playback
def _pacer(gui: bool, delay_ms: int) -> SumoSignalEnv:
    """A bare object carrying just what ``_pace`` reads.

    Built without SUMO on purpose: pacing is pure wall-clock arithmetic, and a
    test that had to start the GUI to check it could not run unattended.
    """
    stub = SumoSignalEnv.__new__(SumoSignalEnv)
    cfg = EnvConfig()
    cfg.gui = gui
    cfg.gui_delay_ms = delay_ms
    stub.cfg = cfg
    stub._frame_deadline = None
    return stub


def test_pace_is_inert_without_the_gui():
    """Batch runs must never sleep -- training would take days."""
    stub = _pacer(gui=False, delay_ms=100)
    t0 = time.monotonic()
    for _ in range(50):
        stub._pace()
    assert time.monotonic() - t0 < 0.05
    assert stub._frame_deadline is None


def test_pace_holds_the_requested_frame_rate():
    """The reason the demo was too fast: SUMO ignores --delay under TraCI, so
    the wait has to happen here. Ten 20 ms frames must take about 0.2 s."""
    stub = _pacer(gui=True, delay_ms=20)
    t0 = time.monotonic()
    for _ in range(11):        # the first call only arms the deadline
        stub._pace()
    elapsed = time.monotonic() - t0
    assert 0.15 <= elapsed <= 0.45, elapsed


def test_pace_does_not_repay_an_overrun_with_a_burst():
    """A frame that overruns resets the clock rather than leaving a deficit.

    Catching up after a hitch means running the following frames with no sleep
    at all, which shows on screen as exactly the too-fast motion the pacing
    exists to remove."""
    stub = _pacer(gui=True, delay_ms=10)
    stub._pace()                                   # arm
    stub._frame_deadline = time.monotonic() - 5.0  # pretend we stalled 5 s
    t0 = time.monotonic()
    stub._pace()
    assert time.monotonic() - t0 < 0.05            # no sleep, and no debt kept
    assert stub._frame_deadline > time.monotonic()


# ------------------------------------------------------ the sensing clock
def test_microphones_are_read_faster_than_the_controller_decides():
    """Every detector frame is consumed, not one in five.

    The belief used to be advanced once per 5 s decision epoch, so four of every
    five frames the detector produced were discarded and the posterior, range
    and level a corridor is priced against were constant across an epoch -- while
    the vehicle covered up to seventy metres. The corridor's 1 Hz re-check was
    re-pricing something that could not have changed.
    """
    cfg = _cfg(horizon=200)
    cfg.step_length_s = 1.0
    env = _env(cfg, "t_sense_rate")
    try:
        env.reset(seed=5)
        before = env.sensor.n_calls
        env.step(int(env.current_phase))
        calls = env.sensor.n_calls - before
        per_epoch = cfg.control.decision_interval_s / cfg.sensing_interval_s
        expected = per_epoch * len(env.approaches)
        assert calls >= 0.8 * expected, (calls, expected)
    finally:
        env.close()


def test_the_belief_moves_within_a_single_decision_epoch():
    """The point of the faster clock: something the corridor can react to.

    A vehicle closing at 14 m/s covers 70 m in one epoch. If the range estimate
    is identical at the start and the end of that epoch, the level and closeness
    the decision is supposed to turn on are not being measured.
    """
    cfg = _cfg(horizon=600, ev_rate=180.0)
    cfg.step_length_s = 1.0
    env = _env(cfg, "t_sense_within")
    try:
        env.reset(seed=3)
        moved = False
        for _ in range(60):
            before = {a: b.range_m for a, b in env.belief.beliefs().items()}
            env._advance(1.0, green=True)
            after = {a: b.range_m for a, b in env.belief.beliefs().items()}
            if any(abs(after[a] - before[a]) > 1e-6 for a in before):
                moved = True
                break
        assert moved, "the belief never changed inside an epoch"
    finally:
        env.close()


def test_sensing_runs_through_the_changeover():
    """Yellow plus all-red is five seconds the vehicle spends getting closer."""
    cfg = _cfg(horizon=300, ev_rate=180.0)
    cfg.step_length_s = 1.0
    env = _env(cfg, "t_sense_changeover")
    try:
        env.reset(seed=8)
        env._advance(cfg.control.min_green_s, green=True)
        env.green_elapsed_s = cfg.control.min_green_s
        before = env.sensor.n_calls
        other = (int(env.current_phase) + 1) % env.n_actions
        env._changeover(other)
        assert env.sensor.n_calls > before, "the filter was blind through the changeover"
    finally:
        env.close()


def test_the_corridor_commits_when_a_vehicle_is_really_inbound():
    """The regression this whole exercise exists to prevent.

    Measured before the corridor's economics were fixed: over a seeded hour, on
    the epochs where an emergency vehicle was genuinely inbound and in range,
    the scheduler committed on four of sixty-six -- the projected red collapsed
    to zero for any overdue phase, which is every phase an emergency vehicle is
    queued on, and the derived threshold became infinite. The approach carrying
    the vehicle was green half the time, which is chance for a four-phase
    signal.

    This asserts far less than the measured result, deliberately: enough to
    catch that failure returning, loose enough not to encode one seed's noise.
    """
    from atsc.control.corridor import CorridorScheduler

    cfg = _cfg(horizon=900, ev_rate=120.0)
    cfg.step_length_s = 1.0
    env = _env(cfg, "t_corridor_commits")
    try:
        env.reset(seed=17)
        sched = CorridorScheduler()
        sched.reset(env)
        inbound_epochs = committed_epochs = 0
        for _ in range(120):
            truth = _inbound_truth(env)
            base = int(env.current_phase)
            decision = sched.decide(env, env.action_mask(), base)
            if truth:
                inbound_epochs += 1
                committed_epochs += int(bool(decision.engaged))
            action = (decision.phase if decision.engaged and decision.phase is not None
                      else base)
            _, _, _, done, _ = env.step(int(action))
            if done:
                break
        assert inbound_epochs >= 10, f"test scenario produced too few events: {inbound_epochs}"
        rate = committed_epochs / inbound_epochs
        assert rate >= 0.20, (
            f"the corridor engaged on {committed_epochs}/{inbound_epochs} epochs with a "
            "vehicle genuinely inbound; before the economics were fixed this was 6%")
    finally:
        env.close()


def _inbound_truth(env) -> bool:
    """Is an emergency vehicle really upstream of a stop line right now?"""
    scenario = env.scenario
    if scenario is None or not scenario.emergency_vehicles:
        return False
    try:
        present = set(env.conn.vehicle.getIDList())
    except Exception:
        return False
    for vid in present & set(scenario.emergency_vehicles):
        try:
            if env.conn.vehicle.getRoadID(vid) in env.approaches:
                return True
        except Exception:
            continue
    return False


def test_sumo_start_retries_a_launch_failure():
    """Ordinary robustness: one failed subprocess launch must not end a stage."""
    import atsc.env as env_mod

    cfg = _cfg(horizon=100)
    net, _ = build_network(NetworkConfig(name="test_single"), NETS)
    env = SumoSignalEnv(cfg, net_file=net, work_dir=DATA_DIR / "runs" / "tests",
                        label="t_retry")
    calls = {"n": 0}
    real_start = env_mod.traci.start

    def flaky(cmd, label=None, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Could not connect.")
        return real_start(cmd, label=label, **kw)

    env_mod.traci.start = flaky
    try:
        env.reset(seed=1)
        assert calls["n"] == 2, f"expected one retry, saw {calls['n']} attempts"
        assert env.n_actions >= 2, "the retried run must be a working episode"
    finally:
        env_mod.traci.start = real_start
        env.close()


def test_a_too_long_scenario_path_is_named_as_the_cause():
    """The failure that looks like everything except what it is.

    Past 260 characters SUMO cannot open a file that Python has just written
    and that Path.exists() confirms is there, because the two use different
    Windows path APIs. It surfaces as "could not connect", which reads as a
    port clash, a race, or a permissions problem. It selects controllers by the
    length of their name, so it strikes partway through a long run. The error
    has to say so, or the next person loses the same hour we did.
    """
    import atsc.env as env_mod

    cfg = _cfg(horizon=100)
    net, _ = build_network(NetworkConfig(name="test_single"), NETS)
    env = SumoSignalEnv(cfg, net_file=net, work_dir=DATA_DIR / "runs" / "tests",
                        label="t_longpath")
    long_cfg = "C:\\" + "d" * (env_mod.WINDOWS_MAX_PATH + 8) + ".sumocfg"
    real_start = env_mod.traci.start
    env_mod.traci.start = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("Could not connect."))
    try:
        env._start_with_retry([env_mod.checkBinary("sumo"), "-c", long_cfg])
    except RuntimeError as exc:
        assert str(env_mod.WINDOWS_MAX_PATH) in str(exc), str(exc)
        assert "ATSC_RUNS_DIR" in str(exc), "the error must say how to fix it"
    else:
        raise AssertionError("an over-long scenario path must be diagnosed")
    finally:
        env_mod.traci.start = real_start


def test_sumo_start_gives_up_with_a_useful_message():
    """A genuinely broken launch must fail loudly, not retry forever."""
    import atsc.env as env_mod

    cfg = _cfg(horizon=100)
    net, _ = build_network(NetworkConfig(name="test_single"), NETS)
    env = SumoSignalEnv(cfg, net_file=net, work_dir=DATA_DIR / "runs" / "tests",
                        label="t_retry_fail")
    real_start = env_mod.traci.start
    env_mod.traci.start = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nope"))
    try:
        env.reset(seed=1)
    except RuntimeError as exc:
        assert "would not start after" in str(exc), str(exc)
        assert "nope" in str(exc), "the underlying cause must survive"
    else:
        raise AssertionError("a launch that always fails must raise")
    finally:
        env_mod.traci.start = real_start
        env.close()


def test_a_path_sumo_cannot_open_is_reported_not_retried():
    """The failure that cost an overnight evaluation.

    Past 260 characters SUMO cannot open a scenario Python has just written
    successfully, and `traci.start` does not raise: SUMO exits, TraCI reports
    "TraCI server already finished" and retries on a different port, over and
    over, launching and losing a process each time. The client sits at zero CPU
    with no SUMO alive, which looks like a hang rather than an error -- so the
    run stalls on whichever controller first has a long enough name and never
    reports anything. Here it selected the fifth of nineteen.

    The check therefore has to run *before* the first launch, and say which
    limit was hit and what to do about it.
    """
    from atsc.env import WINDOWS_MAX_PATH, SumoSignalEnv

    env = SumoSignalEnv.__new__(SumoSignalEnv)          # no SUMO, no network
    long_cfg = "C:\\" + "d" * (WINDOWS_MAX_PATH - 20) + "\scenario_x.sumocfg"
    assert len(long_cfg) >= WINDOWS_MAX_PATH

    try:
        env._start_with_retry(["sumo", "-c", long_cfg])
    except RuntimeError as exc:
        msg = str(exc)
        assert str(len(long_cfg)) in msg, msg
        assert str(WINDOWS_MAX_PATH) in msg, msg
        assert "ATSC_RUNS_DIR" in msg, msg
    else:
        raise AssertionError("a path over the limit must be refused up front")


def test_a_short_path_is_not_refused_by_the_length_check():
    """The guard must not reject paths SUMO can open; it only fails on the
    length, and everything shorter goes on to the ordinary launch path."""
    from atsc.env import SumoSignalEnv

    env = SumoSignalEnv.__new__(SumoSignalEnv)
    try:
        env._start_with_retry(["sumo", "-c", "C:\atsc\s.sumocfg"])
    except RuntimeError as exc:
        assert "character limit" not in str(exc), str(exc)
    except Exception:
        pass          # any other failure is the real launch, not this check
