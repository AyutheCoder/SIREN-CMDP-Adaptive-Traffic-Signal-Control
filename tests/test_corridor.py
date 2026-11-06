"""The risk-calibrated corridor: the threshold, the schedule, and the budget.

The claim these tests exist to protect is that the commitment threshold is
*derived* from the traffic state rather than tuned. That is easy to assert and
easy to break silently -- a stray clip or an economics bug turns
``theta = C / (gamma B)`` back into a constant without any test failing unless
one is written for it. So the first tests here drive the scheduler against a
stub intersection and check that the threshold moves in the direction the
arithmetic says it must.
"""

from __future__ import annotations

import numpy as np

from atsc.config import BeliefConfig, ControlConfig, EnvConfig
from atsc.control.corridor import CorridorScheduler, EVRequest


class StubBelief:
    """Minimal stand-in for one approach's posterior."""

    def __init__(self, presence, range_m, rate, eta, priority=1.0, eta_var=9.0,
                 signed=False):
        self.presence = presence
        self.range_m = range_m
        self.range_rate_ms = rate
        self.eta_s = eta
        self.expected_priority = priority
        self.eta_var = eta_var
        self.signed_range = signed
        """Is ``range_m`` signed about the stop line? The joint filter's is, and
        the screening test differs for the two -- see CorridorScheduler."""


class StubTracker:
    def __init__(self, beliefs):
        self._b = beliefs

    def beliefs(self):
        return self._b


class StubEnv:
    """A four-phase intersection with settable queues and beliefs.

    Phases 0 and 2 serve approach N and S respectively; 1 and 3 serve E and W.
    The mapping does not have to be realistic, only consistent.
    """

    def __init__(self, queues=(0, 0, 0, 0), approach_queues=None,
                 beliefs=None, current_phase=1, green_elapsed=20.0):
        self.cfg = EnvConfig()
        self.cfg.control = ControlConfig()
        self.cfg.belief = BeliefConfig()
        self.n_actions = 4
        self.approaches = ["n", "e", "s", "w"]
        self.approach_direction = {a: a.upper() for a in self.approaches}
        self.current_phase = current_phase
        self.green_elapsed_s = green_elapsed
        self.time_since_served = np.array([30.0, 0.0, 30.0, 30.0])
        self.sim_time = 100.0
        self._queues = np.asarray(queues, dtype=float)
        self._approach_queues = approach_queues or {a: 0.0 for a in self.approaches}
        self.belief = StubTracker(beliefs or {})

    def phase_queues(self):
        return self._queues

    def approach_queues(self):
        return dict(self._approach_queues)

    def phases_serving_approach(self, approach, primary_only=False):
        """One primary phase per approach, plus a protected-left sibling.

        Mirrors the real four-phase program, where every approach is greened by
        an all-movements phase *and* by a left-turn-only phase that discharges
        one lane in three. The corridor must never target the sibling, so the
        stub has to be able to offer it.
        """
        primary = {"n": [0], "e": [1], "s": [2], "w": [3]}[approach]
        if primary_only:
            return list(primary)
        return list(primary) + [self.left_sibling[approach]]

    left_sibling = {"n": 1, "e": 0, "s": 3, "w": 2}


def _sched(**kw):
    s = CorridorScheduler(**kw)
    return s


def _fresh(env, **kw):
    s = _sched(**kw)
    s.reset(env)
    return s


ALL = np.ones(4, dtype=bool)


def test_threshold_rises_with_conflicting_queue():
    """The core claim: the same evidence is worth acting on less when it costs more."""
    beliefs = {"n": StubBelief(0.5, 120.0, -12.0, 6.0)}
    thetas = []
    for cross in (0.0, 10.0, 40.0):
        env = StubEnv(queues=(0.0, cross, 0.0, 0.0),
                      approach_queues={"n": 4.0, "e": cross, "s": 0.0, "w": 0.0},
                      beliefs=beliefs)
        s = _fresh(env)
        reqs = s._build_requests(env, base_action=1)
        assert reqs, "no request built"
        thetas.append(reqs[0].threshold)
    assert thetas[0] < thetas[1] < thetas[2], thetas
    assert len(set(thetas)) == 3, "threshold is not responding to the state"


def test_free_preemption_collapses_the_threshold():
    """When the base policy already wants that phase, preemption costs nothing."""
    beliefs = {"e": StubBelief(0.5, 100.0, -12.0, 5.0)}
    env = StubEnv(queues=(20.0, 3.0, 20.0, 20.0),
                  approach_queues={"n": 0.0, "e": 3.0, "s": 0.0, "w": 0.0},
                  beliefs=beliefs, current_phase=1)
    s = _fresh(env)
    reqs = s._build_requests(env, base_action=1)   # policy already holding phase 1
    assert reqs[0].cost_s == 0.0
    assert reqs[0].threshold == s.theta_min


def test_distant_vehicle_is_seen_but_not_acted_on_yet():
    """Anticipation and timing are separate questions, and both are derived.

    A vehicle 40 s out is *worth* a corridor -- it will meet a red when it
    arrives, and the projected-timeline benefit says so -- but acting on it now
    would hold the corridor for forty seconds to buy exactly what holding it for
    ten seconds buys. So the request is priced, and deferred, and the thing that
    defers it is ``slack``, not a lead-time window.
    """
    env_kw = dict(queues=(0.0, 5.0, 0.0, 0.0),
                  approach_queues={"n": 0.0, "e": 5.0, "s": 0.0, "w": 0.0})
    far = StubEnv(beliefs={"n": StubBelief(0.9, 290.0, -14.0, 40.0)}, **env_kw)
    near = StubEnv(beliefs={"n": StubBelief(0.9, 60.0, -14.0, 4.0)}, **env_kw)
    r_far = _fresh(far)._build_requests(far, base_action=1)[0]
    r_near = _fresh(near)._build_requests(near, base_action=1)[0]

    assert r_far.slack_s > r_near.slack_s
    assert not r_far.due, "a vehicle 40 s out does not need the corridor yet"
    assert not r_far.committed
    assert r_near.committed, "an imminent vehicle with a clear road must commit"


def test_benefit_survives_range():
    """The bug this replaced: benefit collapsing to zero at any useful lead.

    Under the old model the delay avoided was ``residual_red_now + clearance -
    eta``, so any vehicle further away than the current residual red scored
    exactly zero and the corridor could not open until it was seconds from the
    stop line. The benefit must instead be governed by the red the vehicle meets
    *when it arrives*, which does not go away just because it is far off.
    """
    env_kw = dict(queues=(0.0, 5.0, 0.0, 0.0),
                  approach_queues={"n": 0.0, "e": 5.0, "s": 0.0, "w": 0.0})
    far = StubEnv(beliefs={"n": StubBelief(0.9, 290.0, -14.0, 40.0)}, **env_kw)
    r = _fresh(far)._build_requests(far, base_action=1)[0]
    assert r.benefit_s > 0.0, "a distant vehicle facing red must still be priced"
    assert r.residual_red_s > 0.0


def test_corridor_holds_once_opened():
    """Hysteresis: an open corridor is not dropped on a one-epoch dip."""
    beliefs = {"n": StubBelief(0.9, 40.0, -14.0, 3.0)}
    env = StubEnv(queues=(0.0, 6.0, 0.0, 0.0),
                  approach_queues={"n": 1.0, "e": 6.0, "s": 0.0, "w": 0.0},
                  beliefs=beliefs, current_phase=1)
    s = _fresh(env)
    first = s.decide(env, ALL, base_action=1)
    assert first.engaged, first.reason
    # The posterior dips below the derived threshold but stays above 0.6 of it.
    theta = first.target.threshold
    beliefs["n"].presence = 0.7 * theta
    second = s.decide(env, ALL, base_action=1)
    assert second.engaged, "an open corridor must survive a marginal dip"
    # Evidence gone *and* the clearance the corridor was priced with already
    # served: only then is it released.
    beliefs["n"].presence = 0.2 * theta
    s._held_s = s._hold_target_s + 1.0
    third = s.decide(env, ALL, base_action=1)
    assert not third.engaged, "and must be released when the evidence really goes"


def test_open_corridor_is_seen_through_a_posterior_collapse():
    """A corridor already open survives the filter losing its own vehicle.

    This is not hysteresis, which is about a marginal dip. The acoustic filter
    can lose an emergency vehicle outright at close range -- it begins moving
    its mass past the stop line before the vehicle is actually across, and when
    that tips over, the presence marginal goes to zero and the range track
    resets to its prior. Without a hold the green is handed to a conflicting
    approach with the vehicle metres short of the stop line, which is the worst
    possible moment to do it and costs more than never having preempted.
    """
    beliefs = {"n": StubBelief(0.95, 40.0, -14.0, 3.0)}
    env = StubEnv(queues=(0.0, 6.0, 0.0, 0.0),
                  approach_queues={"n": 4.0, "e": 6.0, "s": 0.0, "w": 0.0},
                  beliefs=beliefs, current_phase=1)
    s = _fresh(env)
    opened = s.decide(env, ALL, base_action=1)
    assert opened.engaged, opened.reason
    assert s._hold_target_s > 0.0, "an opened corridor must owe its vehicle a hold"
    beliefs["n"].presence = 0.0        # the filter drops the track entirely
    beliefs["n"].range_m = 400.0
    beliefs["n"].range_rate_ms = -8.0
    assert s.decide(env, ALL, base_action=1).engaged,         "the corridor must be seen through, not abandoned mid-crossing"


def test_the_clearance_hold_is_bounded():
    """It is a hold, not a latch: it expires, and it is capped."""
    beliefs = {"n": StubBelief(0.95, 40.0, -14.0, 3.0)}
    env = StubEnv(queues=(0.0, 6.0, 0.0, 0.0),
                  approach_queues={"n": 4.0, "e": 6.0, "s": 0.0, "w": 0.0},
                  beliefs=beliefs, current_phase=1)
    s = _fresh(env)
    assert s.decide(env, ALL, base_action=1).engaged
    assert s._hold_target_s <= s.max_hold_s
    beliefs["n"].presence = 0.0
    beliefs["n"].range_m = 400.0
    # Twenty epochs is far past any hold the arithmetic can produce.
    for _ in range(20):
        last = s.decide(env, ALL, base_action=1)
    assert not last.engaged, "the hold must expire on its own"


def test_a_vehicle_in_the_junction_still_has_a_request():
    """Signed range goes negative as the vehicle crosses; it is not gone.

    The screening test used to be ``range > 0``, so the request disappeared at
    the instant the vehicle entered the intersection -- exactly when the green
    has to be held for it to finish crossing.
    """
    env = StubEnv(queues=(0.0, 6.0, 0.0, 0.0),
                  approach_queues={"n": 0.0, "e": 6.0, "s": 0.0, "w": 0.0},
                  beliefs={"n": StubBelief(0.9, -4.0, -12.0, 0.0, signed=True)},
                  current_phase=1)
    reqs = _fresh(env)._build_requests(env, base_action=1)
    assert reqs, "a vehicle inside the junction must still be a request"
    assert reqs[0].in_junction
    assert reqs[0].eta_s == 0.0, "there is no arrival left to predict"


def test_a_vehicle_well_past_the_junction_is_dropped():
    """The tolerance is one junction, not an open-ended grace period."""
    env = StubEnv(queues=(0.0, 6.0, 0.0, 0.0),
                  approach_queues={"n": 0.0, "e": 6.0, "s": 0.0, "w": 0.0},
                  beliefs={"n": StubBelief(0.9, -80.0, -12.0, 0.0, signed=True)},
                  current_phase=1)
    assert _fresh(env)._build_requests(env, base_action=1) == []


def test_receding_vehicle_is_ignored():
    """Presence decays slowly; without a closing test the corridor holds for a ghost."""
    env = StubEnv(beliefs={"n": StubBelief(0.99, 40.0, +9.0, 5.0)},
                  approach_queues={"n": 2.0, "e": 0.0, "s": 0.0, "w": 0.0})
    s = _fresh(env)
    assert s._build_requests(env, base_action=1) == []


def test_calibrator_is_applied_to_presence():
    """The rule must see the calibrated probability, not the raw filter output."""
    env = StubEnv(beliefs={"n": StubBelief(0.99, 80.0, -12.0, 5.0)},
                  approach_queues={"n": 2.0, "e": 0.0, "s": 0.0, "w": 0.0})
    raw = _fresh(env)._build_requests(env, base_action=1)[0].presence
    cal = _fresh(env, calibrator=lambda p: 0.25 * p)
    got = cal._build_requests(env, base_action=1)[0].presence
    assert abs(raw - 0.99) < 1e-9
    assert abs(got - 0.2475) < 1e-9


def test_conflict_schedule_serves_both_and_orders_by_urgency():
    """Two rivals on conflicting phases: plan both, most urgent first."""
    # Both are due: the scheduler only sequences requests that are ready to be
    # acted on this epoch, so a conflict test has to present two of those.
    beliefs = {
        "n": StubBelief(0.9, 90.0, -13.0, 7.0, priority=0.7),
        "e": StubBelief(0.9, 40.0, -13.0, 3.0, priority=1.0),
    }
    env = StubEnv(queues=(2.0, 2.0, 2.0, 2.0),
                  approach_queues={"n": 2.0, "e": 2.0, "s": 0.0, "w": 0.0},
                  beliefs=beliefs, current_phase=3)
    s = _fresh(env)
    decision = s.decide(env, ALL, base_action=3)
    assert decision.conflict, "two committed requests must register as a conflict"
    assert len(decision.schedule) >= 2, decision.schedule
    assert set(decision.schedule[:2]) == {0, 1}, decision.schedule
    # The ambulance 3 s away outranks the police car 12 s away.
    assert decision.schedule[0] == 1, decision.schedule


def test_greedy_horizon_plans_only_one_phase():
    """The ablation must actually ablate: horizon 1 cannot sequence."""
    beliefs = {
        "n": StubBelief(0.9, 150.0, -13.0, 12.0),
        "e": StubBelief(0.9, 40.0, -13.0, 3.0),
    }
    env = StubEnv(queues=(2.0, 2.0, 2.0, 2.0),
                  approach_queues={"n": 2.0, "e": 2.0, "s": 0.0, "w": 0.0},
                  beliefs=beliefs, current_phase=3)
    s = _fresh(env, plan_horizon=1)
    decision = s.decide(env, ALL, base_action=3)
    assert len(decision.schedule) == 1, decision.schedule


def test_shared_phase_serves_two_vehicles_at_once():
    """Rivals on one phase are not a conflict to be sequenced."""
    beliefs = {
        "n": StubBelief(0.9, 60.0, -13.0, 4.0),
        "s": StubBelief(0.9, 60.0, -13.0, 4.0),
    }
    env = StubEnv(queues=(2.0, 2.0, 2.0, 2.0),
                  approach_queues={"n": 1.0, "e": 0.0, "s": 1.0, "w": 0.0},
                  beliefs=beliefs, current_phase=1)

    class Shared(StubEnv):
        def phases_serving_approach(self, approach, primary_only=False):
            return [0] if approach in ("n", "s") else [1]

    shared = Shared(queues=(2.0, 2.0, 2.0, 2.0),
                    approach_queues={"n": 1.0, "e": 0.0, "s": 1.0, "w": 0.0},
                    beliefs=beliefs, current_phase=1)
    s = _fresh(shared)
    decision = s.decide(shared, ALL, base_action=1)
    assert decision.engaged and decision.phase == 0
    assert len(decision.schedule) == 1, "one phase clears both; do not sequence"


def test_shield_mask_is_respected():
    """The scheduler proposes only what the shield permits."""
    env = StubEnv(beliefs={"n": StubBelief(0.99, 60.0, -13.0, 4.0)},
                  approach_queues={"n": 1.0, "e": 0.0, "s": 0.0, "w": 0.0},
                  current_phase=1)
    s = _fresh(env)
    forbidden = np.array([False, True, True, True])
    decision = s.decide(env, forbidden, base_action=1)
    assert not decision.engaged
    assert decision.reason == "shield defers the corridor"
    assert decision.committed, "the request itself should survive to the next epoch"


def test_false_preemption_budget_binds():
    """Spending the budget on weak evidence must raise the bar, not stop the rule.

    The budget is spent per *engagement*, so each iteration here drops the
    scheduler's target first -- which is what a new vehicle on a new approach
    looks like to it. Polling one open corridor forty times is one corridor,
    and charging it forty times is the bug this shape of test used to hide:
    with the calibrated posterior bounded below 0.6, a per-poll charge put the
    bucket permanently into debt within two minutes of any real episode and
    the near-certainty gate then blocked every corridor for the rest of it.
    """
    env = StubEnv(beliefs={"n": StubBelief(0.30, 60.0, -13.0, 3.0)},
                  approach_queues={"n": 1.0, "e": 0.0, "s": 0.0, "w": 0.0},
                  current_phase=1)
    # Enough for exactly one engagement at 1 - p = 0.70, and no more.
    s = _fresh(env, false_preempt_per_hour=0.8)
    engaged = []
    for _ in range(40):
        s._target = None            # a fresh vehicle, not the same one again
        engaged.append(s.decide(env, ALL, base_action=1).engaged)
    assert engaged[0], "the first call must be affordable"
    assert not all(engaged), "an exhausted budget must eventually bind"
    assert s.n_budget_blocks > 0

    # Overwhelming evidence still gets through once the budget is gone.
    env.belief = StubTracker({"n": StubBelief(0.99, 60.0, -13.0, 3.0)})
    s._target = None
    assert s.decide(env, ALL, base_action=1).engaged


def test_holding_one_corridor_is_charged_once():
    """Re-polling an open corridor is not a second chance to be wrong."""
    env = StubEnv(beliefs={"n": StubBelief(0.30, 60.0, -13.0, 3.0)},
                  approach_queues={"n": 1.0, "e": 0.0, "s": 0.0, "w": 0.0},
                  current_phase=1)
    s = _fresh(env)
    for _ in range(30):
        assert s.decide(env, ALL, base_action=1).engaged
    assert s.n_engagements == 1
    assert s.spent_tokens <= 1.0, s.spent_tokens
    assert s.n_budget_blocks == 0


def test_the_bucket_never_goes_into_debt():
    """A bound on a rate, not a latch: an empty bucket is empty, not negative."""
    env = StubEnv(beliefs={"n": StubBelief(0.10, 60.0, -13.0, 3.0)},
                  approach_queues={"n": 1.0, "e": 0.0, "s": 0.0, "w": 0.0},
                  current_phase=1)
    s = _fresh(env, false_preempt_per_hour=0.8)
    for _ in range(50):
        s._target = None
        s.decide(env, ALL, base_action=1)
    assert s._tokens >= 0.0, s._tokens


def test_lambda_scales_willingness_to_act():
    """The CMDP dual still controls the trade-off after priority left the critic."""
    beliefs = {"n": StubBelief(0.5, 120.0, -12.0, 6.0)}
    env = StubEnv(queues=(0.0, 25.0, 0.0, 0.0),
                  approach_queues={"n": 3.0, "e": 25.0, "s": 0.0, "w": 0.0},
                  beliefs=beliefs)
    lo = _fresh(env)
    hi = _fresh(env)
    hi.set_lambda(20.0)
    t_lo = lo._build_requests(env, base_action=1)[0].threshold
    t_hi = hi._build_requests(env, base_action=1)[0].threshold
    assert t_hi < t_lo, (t_lo, t_hi)


def test_value_ratio_of_zero_never_preempts():
    """gamma = 0 is the no-priority control point of the frontier sweep."""
    env = StubEnv(beliefs={"n": StubBelief(0.99, 60.0, -13.0, 3.0)},
                  approach_queues={"n": 1.0, "e": 5.0, "s": 0.0, "w": 0.0},
                  queues=(0.0, 5.0, 0.0, 0.0))
    s = _fresh(env, ev_value_ratio=0.0)
    assert not s.decide(env, ALL, base_action=1).engaged


def test_stats_are_reported():
    env = StubEnv(beliefs={"n": StubBelief(0.99, 60.0, -13.0, 3.0)},
                  approach_queues={"n": 1.0, "e": 0.0, "s": 0.0, "w": 0.0})
    s = _fresh(env)
    for _ in range(3):
        s.decide(env, ALL, base_action=1)
    st = s.stats()
    for key in ("priority_engagements", "priority_epochs", "priority_duty_cycle",
                "conflict_epochs", "expected_false_preemptions"):
        assert key in st, key
    assert st["priority_epochs"] == 3


# ------------------------------------------------- emergency vehicles at a stop
class StoppedBelief(StubBelief):
    """A joint-filter posterior for a vehicle held in the queue.

    ``signed_range`` marks the range as signed about the stop line, which is
    what the joint filter reports and what makes the closing test inapplicable.
    ``eta_s`` sits at the filter's ceiling because a stationary source has no
    estimable arrival time -- that is the whole difficulty.
    """

    def __init__(self, presence, range_m, rate=0.0):
        super().__init__(presence, range_m, rate, BeliefConfig().max_eta_s)
        self.signed_range = True


def test_a_stopped_emergency_vehicle_still_raises_a_request():
    """The bug behind an ambulance sitting through a red.

    Arrival time is estimated as range over closing speed. A vehicle that has
    come to a stop has no closing speed, so that ratio returned tens of
    thousands of seconds and the request was screened out as "not arriving" --
    silently discarding exactly the vehicle the corridor exists to serve.
    """
    env = StubEnv(beliefs={"n": StoppedBelief(0.95, 40.0)},
                  approach_queues={"n": 6.0, "e": 1.0, "s": 0.0, "w": 0.0},
                  queues=(6.0, 1.0, 0.0, 0.0))
    s = _fresh(env)
    reqs = s._build_requests(env, base_action=1)
    assert [r.approach for r in reqs] == ["n"], reqs
    # It is waiting on the signal, not travelling, so its arrival is now.
    assert reqs[0].eta_s == 0.0
    assert reqs[0].benefit_s > 0.0


def test_a_stopped_vehicle_far_upstream_is_still_ignored():
    """The screen is kept for tracks that carry no usable arrival time.

    Beyond the queue range a non-closing track is a stale or crawling one, and
    nothing about the corridor decision depends on it yet.
    """
    env = StubEnv(beliefs={"n": StoppedBelief(0.95, 260.0)},
                  approach_queues={"n": 6.0, "e": 1.0, "s": 0.0, "w": 0.0})
    s = _fresh(env)
    assert s._build_requests(env, base_action=1) == []


def test_a_closing_vehicle_still_uses_range_over_speed():
    """The ordinary path is unchanged: a moving vehicle gets a real ETA."""
    env = StubEnv(beliefs={"n": StoppedBelief(0.95, 60.0, rate=-12.0)},
                  approach_queues={"n": 2.0, "e": 1.0, "s": 0.0, "w": 0.0})
    s = _fresh(env)
    reqs = s._build_requests(env, base_action=1)
    assert len(reqs) == 1
    assert 4.0 < reqs[0].eta_s < 6.0, reqs[0].eta_s   # 60 m at 12 m/s


def test_an_imminent_vehicle_on_a_green_approach_is_still_worth_holding():
    """The green a vehicle is approaching is only guaranteed if we guarantee it.

    Pricing the do-nothing baseline as though the current green persists makes
    the projected red zero for every vehicle whose approach happens to be green
    at the decision epoch, hence the benefit zero and the derived threshold
    infinite. The corridor then declines, never sets the call that suspends
    gap-out, and the phase is released to a conflicting approach as the vehicle
    arrives -- the failure this test exists to prevent. Measured on a
    demonstration run before the fix: 45 of 78 emergency-vehicle events priced
    out this way, every one reading a projected red of zero on a green approach.
    """
    env = StubEnv(queues=(4.0, 6.0, 0.0, 0.0),
                  approach_queues={"n": 4.0, "e": 6.0, "s": 0.0, "w": 0.0},
                  beliefs={"n": StubBelief(0.9, 60.0, -13.0, 4.0, signed=True)},
                  current_phase=0)          # phase 0 serves approach "n"
    req = _fresh(env)._build_requests(env, base_action=0)
    assert req, "a vehicle approaching a green must still raise a request"
    r = req[0]
    assert np.isfinite(r.threshold), (
        "an infinite threshold means the corridor declined and will not hold "
        "the green it is sitting on")
    assert r.benefit_s > 0.0, r.benefit_s
    assert r.residual_red_s > 0.0, (
        "released now, the vehicle meets the red that follows the changeover")


def test_holding_a_green_the_policy_already_wants_is_free():
    """Protecting it costs cross traffic nothing, so the threshold floors."""
    env = StubEnv(queues=(4.0, 0.0, 0.0, 0.0),
                  approach_queues={"n": 4.0, "e": 0.0, "s": 0.0, "w": 0.0},
                  beliefs={"n": StubBelief(0.9, 60.0, -13.0, 4.0, signed=True)},
                  current_phase=0)
    r = _fresh(env)._build_requests(env, base_action=0)[0]
    assert r.cost_s == 0.0
    assert r.threshold == 0.02, r.threshold


def test_a_distant_vehicle_does_not_hold_a_green_it_will_not_use():
    """The protective hold is bounded by the green's own residual.

    Holding a green for a vehicle that arrives after it would have ended buys
    that vehicle nothing and costs cross traffic every second, because an
    active call suspends the shield's gap-out rule -- the mechanism that stops
    the cross street spilling back. Applied without this bound it gridlocked
    two seeds in twelve, at corridor duty cycles of 0.35 and 0.20 against
    0.12-0.17 on the healthy ones, with 79 and 11 teleports.
    """
    near = StubEnv(queues=(4.0, 6.0, 0.0, 0.0),
                   approach_queues={"n": 4.0, "e": 6.0, "s": 0.0, "w": 0.0},
                   beliefs={"n": StubBelief(0.9, 60.0, -13.0, 4.0, signed=True)},
                   current_phase=0)
    far = StubEnv(queues=(4.0, 6.0, 0.0, 0.0),
                  approach_queues={"n": 4.0, "e": 6.0, "s": 0.0, "w": 0.0},
                  beliefs={"n": StubBelief(0.9, 290.0, -13.0, 40.0, signed=True)},
                  current_phase=0)
    r_near = _fresh(near)._build_requests(near, base_action=0)[0]
    r_far = _fresh(far)._build_requests(far, base_action=0)[0]
    assert r_near.benefit_s > 0.0, "an imminent vehicle's green must be protected"
    assert r_near.committed, "and protecting it is free when the policy agrees"
    assert not r_far.committed, (
        "a vehicle 40 s out must not pin the green now; gap-out has to be able "
        "to serve the cross street in the meantime")


def test_the_queued_fallback_is_bounded_by_the_queue_that_is_there():
    """A stationary track well past the back of the queue is not "arriving now".

    The fallback that rescues a vehicle stopped at a red used to be a flat
    150 m radius, which is further back than this intersection ever queues. On
    a demonstration run that produced a request reading ``eta=0.0s rng=117m``:
    the corridor held a green for a vehicle two hundred metres away while the
    cross street queued nine deep behind it. The extent is now derived from the
    queue the environment actually reports.
    """
    # Empty approach: nothing is queued, so a stationary track is a lost track.
    env = StubEnv(beliefs={"n": StoppedBelief(0.95, 117.0)},
                  approach_queues={"n": 0.0, "e": 9.0, "s": 0.0, "w": 0.0},
                  queues=(0.0, 9.0, 0.0, 0.0))
    assert _fresh(env)._build_requests(env, base_action=1) == []

    # Same track, but now there really is a queue long enough to reach it.
    env2 = StubEnv(beliefs={"n": StoppedBelief(0.95, 117.0)},
                   approach_queues={"n": 33.0, "e": 1.0, "s": 0.0, "w": 0.0},
                   queues=(33.0, 1.0, 0.0, 0.0))
    reqs = _fresh(env2)._build_requests(env2, base_action=1)
    assert [r.approach for r in reqs] == ["n"], reqs
    assert reqs[0].eta_s == 0.0


# ------------------------------------------------------- the distance trigger
def test_the_distance_trigger_is_off_unless_asked_for():
    """Every number in the paper is produced with the economics alone, so the
    default has to leave the deferral above intact."""
    env = StubEnv(queues=(0.0, 5.0, 0.0, 0.0),
                  approach_queues={"n": 0.0, "e": 5.0, "s": 0.0, "w": 0.0},
                  beliefs={"n": StubBelief(0.9, 100.0, -14.0, 40.0)})
    r = _fresh(env)._build_requests(env, base_action=1)[0]
    assert not r.due and not r.committed


def test_the_distance_trigger_commits_a_vehicle_the_deadline_would_defer():
    """The operator's rule: close enough is its own reason.

    Same state as above -- a vehicle whose arrival is far enough off that the
    just-in-time deadline defers it -- but now inside the trigger range. It is
    committed, and the economics are not consulted.
    """
    env = StubEnv(queues=(0.0, 5.0, 0.0, 0.0),
                  approach_queues={"n": 0.0, "e": 5.0, "s": 0.0, "w": 0.0},
                  beliefs={"n": StubBelief(0.9, 100.0, -14.0, 40.0)})
    r = _fresh(env, commit_range_m=150.0)._build_requests(env, base_action=1)[0]
    assert r.due and r.committed


def test_the_distance_trigger_still_respects_the_evidence_floor():
    """It removes the *cost* side of the test, not the sensor.

    A posterior below the level at which the approach is owed service at all is
    not preempted for, however close the track claims to be -- otherwise this
    would be a way to preempt on detector noise.
    """
    s = _sched(commit_range_m=150.0)
    floor = s.service_request_presence * s._attainable_presence()
    # A cross street queued deep enough that the economics decline on their
    # own, so what is left to test is the floor rather than the arithmetic.
    env_kw = dict(queues=(0.0, 90.0, 0.0, 0.0),
                  approach_queues={"n": 0.0, "e": 90.0, "s": 0.0, "w": 0.0})
    weak = StubEnv(beliefs={"n": StubBelief(0.9 * floor, 40.0, -14.0, 3.0)}, **env_kw)
    ok = StubEnv(beliefs={"n": StubBelief(1.1 * floor, 40.0, -14.0, 3.0)}, **env_kw)

    r_weak = _fresh(weak, commit_range_m=150.0)._build_requests(weak, base_action=1)[0]
    r_ok = _fresh(ok, commit_range_m=150.0)._build_requests(ok, base_action=1)[0]
    assert r_weak.presence < r_weak.threshold, "the economics must decline both"
    assert r_ok.presence < r_ok.threshold, r_ok
    assert not r_weak.committed, r_weak
    assert r_ok.committed, r_ok


def test_the_distance_trigger_ignores_a_vehicle_beyond_its_range():
    env = StubEnv(queues=(0.0, 5.0, 0.0, 0.0),
                  approach_queues={"n": 0.0, "e": 5.0, "s": 0.0, "w": 0.0},
                  beliefs={"n": StubBelief(0.9, 260.0, -14.0, 40.0)})
    r = _fresh(env, commit_range_m=150.0)._build_requests(env, base_action=1)[0]
    assert not r.committed


# ----------------------------------------------- serving the right movement
def test_the_corridor_never_targets_a_protected_left_sibling():
    """The bug that made the guarantee vacuous.

    Every approach on the real four-phase program is greened by two phases: one
    that releases all its movements, and one that releases the protected left
    and nothing else -- one lane in three. The acoustic layer cannot hear which
    way a vehicle intends to turn, so treating the second as "serving" the
    approach let the corridor commit to a protected left for an ambulance going
    straight. Observed end to end before this fix: the overlay drew its green
    band over the approach, S6 recorded the request as served, and the vehicle
    sat at a red until the signal moved to the phase that actually carried it,
    paying a second full changeover to get there. That changeover is where the
    stop came from.

    The stub offers the sibling; the request must contain only the primary.
    """
    env = StubEnv(beliefs={"n": StubBelief(0.9, 60.0, -13.0, 4.0)},
                  approach_queues={"n": 4.0, "e": 1.0, "s": 0.0, "w": 0.0})
    reqs = _fresh(env)._build_requests(env, base_action=1)
    assert len(reqs) == 1
    assert reqs[0].phases == (0,), reqs[0].phases
    assert env.left_sibling["n"] not in reqs[0].phases


def test_the_service_set_handed_to_the_shield_excludes_the_sibling():
    """S6 must not be satisfiable by greening a movement the vehicle is not
    making -- that is the difference between a guarantee and a formality."""
    env = StubEnv(beliefs={"n": StubBelief(0.9, 60.0, -13.0, 4.0)},
                  approach_queues={"n": 4.0, "e": 1.0, "s": 0.0, "w": 0.0})
    decision = _fresh(env).decide(env, ALL, base_action=1)
    assert decision.service_phases == (0,), decision.service_phases


# ---------------------------------------------------- the windowed closing test
class TrendEnv(StubEnv):
    """Replays a sequence of tracked ranges, one per poll."""

    def __init__(self, ranges, **kw):
        self._seq = list(ranges)
        self._i = 0
        super().__init__(**kw)

    def advance(self):
        self._i = min(self._i + 1, len(self._seq) - 1)
        self.sim_time += 1.0
        for b in self.belief.beliefs().values():
            b.range_m = self._seq[self._i]


def _replay(ranges, rate, queue=0.0):
    """Poll the scheduler once a second down a range sequence."""
    env = TrendEnv(ranges,
                   beliefs={"n": StubBelief(0.95, ranges[0], rate, 1e9,
                                            eta_var=1e6)},
                   approach_queues={"n": queue, "e": 0.0, "s": 0.0, "w": 0.0})
    s = _fresh(env)
    reqs = []
    for _ in range(len(ranges)):
        reqs = s._build_requests(env, base_action=1, poll_dt=1.0)
        env.advance()
    return reqs


def test_a_closing_track_the_rate_head_cannot_resolve_is_still_served():
    """Range MAE beyond 150 m is 83 m, so the per-second rate reads near zero
    for a vehicle doing 14 m/s and the request used to be dropped outright.
    Differenced over the window the same track is plainly closing."""
    ranges = [260.0, 260.0, 216.0, 216.0, 216.0, 155.0, 155.0, 111.0]
    reqs = _replay(ranges, rate=-0.4)
    assert len(reqs) == 1, "a closing track must survive a mute rate head"
    assert 0.0 < reqs[0].eta_s < 30.0, reqs[0].eta_s


def test_a_genuinely_stationary_track_is_still_dropped():
    """The other half of the same test: the window must not rescue a track that
    is not going anywhere, or every stale detection raises a service request."""
    reqs = _replay([260.0] * 8, rate=-0.4)
    assert reqs == [], reqs


def test_a_track_reset_is_not_read_as_closing():
    """The filter reinstates its prior at close range, which looks like the
    range jumping outwards. Differencing across that reports a large closing
    speed for a vehicle that has not moved."""
    ranges = [40.0, 40.0, 300.0, 300.0, 300.0, 300.0]
    reqs = _replay(ranges, rate=-0.4)
    assert reqs == [], reqs
