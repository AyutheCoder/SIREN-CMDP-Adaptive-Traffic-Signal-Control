"""Safety shield: the S1/S2/S3 guarantees, including the starvation bound.

The starvation test is the important one. An earlier version of the shield
derived its bound without the ``n_phases`` factor and was violated in
simulation (115 s observed against a claimed 110 s), so the bound is now
re-derived adversarially here: a policy that deliberately tries to starve one
movement is run against the shield and the worst observed wait is compared to
the guarantee.
"""

from __future__ import annotations

import numpy as np

from atsc.control.shield import SafetyShield


def _shield(n_phases: int = 4, max_starvation: float = 180.0,
            **kwargs) -> SafetyShield:
    return SafetyShield(
        n_phases=n_phases,
        min_green_s=8.0,
        max_green_s=55.0,
        yellow_s=3.0,
        all_red_s=2.0,
        decision_interval_s=5.0,
        max_starvation_s=max_starvation,
        **kwargs,
    )


def _shield_with_recovery(**kwargs) -> SafetyShield:
    """S5 is off by default (it did not pay for itself), so enable it here."""
    return _shield(gridlock_queue=90.0, **kwargs)


def test_rejects_infeasible_bound():
    """A guarantee the timing envelope cannot deliver must not be accepted."""
    try:
        _shield(n_phases=4, max_starvation=50.0)
    except ValueError:
        return
    raise AssertionError("shield accepted a starvation bound below n*(g_min+dt+T)")


def test_s1_min_green_holds_current_phase():
    sh = _shield()
    report = sh.allowed_actions(current_phase=2, green_elapsed_s=3.0,
                                time_since_served_s=[10, 10, 0, 10])
    assert report.allowed == [2], report.allowed
    assert report.forced_reason == "min_green"


def test_s2_max_green_forces_release():
    sh = _shield()
    report = sh.allowed_actions(current_phase=1, green_elapsed_s=60.0,
                                time_since_served_s=[10, 0, 10, 10])
    assert 1 not in report.allowed
    assert report.forced_reason == "max_green"
    assert len(report.allowed) == 3


def test_free_choice_when_unconstrained():
    sh = _shield()
    report = sh.allowed_actions(current_phase=0, green_elapsed_s=20.0,
                                time_since_served_s=[0, 5, 5, 5])
    assert sorted(report.allowed) == [0, 1, 2, 3]
    assert report.forced_reason is None


def test_projection_records_overrides():
    sh = _shield()
    report = sh.project(3, current_phase=0, green_elapsed_s=2.0,
                        time_since_served_s=[0, 9, 9, 9])
    assert report.applied == 0 and report.overridden
    assert sh.n_overrides == 1 and sh.n_decisions == 1

    report = sh.project(1, current_phase=0, green_elapsed_s=20.0,
                        time_since_served_s=[0, 9, 9, 9])
    assert report.applied == 1 and not report.overridden
    assert sh.override_rate == 0.5


def test_mask_matches_allowed_set():
    sh = _shield()
    for elapsed in (2.0, 20.0, 60.0):
        allowed = sh.allowed_actions(1, elapsed, [30, 0, 30, 30]).allowed
        mask = sh.mask(1, elapsed, [30, 0, 30, 30])
        assert sorted(np.flatnonzero(mask).tolist()) == sorted(allowed)


def test_s3_starvation_bound_under_adversarial_policy():
    """Simulate the shield against a policy that tries hard to starve phase 3."""
    for n_phases in (2, 3, 4, 6):
        for w_max in (150.0, 180.0, 240.0):
            service = 8.0 + 5.0 + 3.0 + 2.0
            if w_max <= n_phases * service:
                continue
            sh = _shield(n_phases=n_phases, max_starvation=w_max)

            current = 0
            green_elapsed = 8.0
            waits = np.zeros(n_phases)
            worst = 0.0
            rng = np.random.default_rng(0)

            for _ in range(4000):
                # Adversary: never voluntarily serve the starving phases.
                victim = int(np.argmax(waits))
                proposal = int(rng.choice([p for p in range(n_phases) if p != victim]))
                report = sh.project(proposal, current, green_elapsed, waits)
                applied = int(report.applied)

                if applied != current:
                    dt = 3.0 + 2.0 + 5.0  # yellow + all-red + one interval
                    current = applied
                    green_elapsed = 5.0
                else:
                    dt = 5.0
                    green_elapsed += 5.0

                waits += dt
                # The phase is green from the end of the changeover onward, so
                # its true wait ends before the epoch does.
                worst = max(worst, float(waits.max()))
                waits[current] = 0.0

            assert worst <= w_max + 1e-6, (
                f"S3 violated for n_phases={n_phases}, W_max={w_max}: "
                f"observed worst wait {worst:.1f}s"
            )


def test_starvation_threshold_derivation():
    sh = _shield(n_phases=4, max_starvation=180.0)
    assert abs(sh.service_cost_s - 18.0) < 1e-9
    assert abs(sh.starvation_threshold_s - (180.0 - 4 * 18.0)) < 1e-9
    assert sh.worst_case_wait_s == 180.0


def test_gap_out_releases_a_cleared_phase():
    """A phase with no queue must yield to one that has a queue."""
    sh = _shield()
    report = sh.allowed_actions(current_phase=0, green_elapsed_s=20.0,
                                time_since_served_s=[0, 5, 5, 5],
                                phase_queues=[0.0, 6.0, 0.0, 0.0])
    assert report.forced_reason == "gap_out"
    assert report.allowed == [1]


def test_gap_out_does_not_evict_an_emergency_call():
    """The corridor clears the queue; gap-out must not then take the green.

    This interaction is the whole reason the shield is told a call is active. A
    corridor's purpose is to have the served approach *cleared* by the time the
    emergency vehicle arrives -- and the instant it succeeds, that phase's queue
    is empty, which is exactly gap-out's trigger. The green is then handed to a
    conflicting approach seconds before the vehicle gets there, and the
    preemption recovers nothing. Measured on the hard-preemption baseline,
    suppressing gap-out during a call moved emergency stopped delay from 15.3 s
    to 11.2 s and the clean-pass rate from 0.18 to 0.36.

    Suppression is safe because gap-out only ever *removes* the current phase
    from the permitted set: S1, S2 and S3 are decided before it and are
    untouched, so maximum green and the starvation bound still force release.
    """
    sh = _shield()
    args = dict(current_phase=0, green_elapsed_s=20.0,
                time_since_served_s=[0, 5, 5, 5],
                phase_queues=[0.0, 6.0, 0.0, 0.0])
    assert sh.allowed_actions(**args).forced_reason == "gap_out"
    held = sh.allowed_actions(emergency=True, **args)
    assert held.forced_reason != "gap_out"
    assert 0 in held.allowed, "the corridor's own phase must remain available"


def test_emergency_call_cannot_defeat_maximum_green():
    """An emergency call suspends gap-out; it does not suspend S2 or S3."""
    sh = _shield()
    over_max = sh.allowed_actions(current_phase=0, green_elapsed_s=999.0,
                                  time_since_served_s=[0, 5, 5, 5],
                                  phase_queues=[0.0, 6.0, 0.0, 0.0],
                                  emergency=True)
    assert 0 not in over_max.allowed, "maximum green still forces release"

    starved = sh.allowed_actions(current_phase=0, green_elapsed_s=20.0,
                                 time_since_served_s=[0, 5, 5, 10_000.0],
                                 phase_queues=[0.0, 6.0, 0.0, 0.0],
                                 emergency=True)
    assert starved.allowed == [3], starved.allowed


def test_gap_out_is_suspended_under_saturation():
    """Above the load threshold, holding green beats paying changeover cost.

    Applying gap-out unconditionally took a 2.4x demand surge from 63 s to
    105 s mean delay and doubled teleports, because every changeover costs
    yellow + all-red of pure lost time and saturated intersections cannot
    afford them. The rule is therefore gated on total queue.
    """
    sh = _shield()
    heavy = [0.0, 40.0, 30.0, 20.0]          # total 90, far above the threshold
    report = sh.allowed_actions(current_phase=0, green_elapsed_s=20.0,
                                time_since_served_s=[0, 5, 5, 5],
                                phase_queues=heavy)
    assert report.forced_reason != "gap_out"
    assert sorted(report.allowed) == [0, 1, 2, 3]


def test_gap_out_ignores_single_arrivals():
    sh = _shield()
    report = sh.allowed_actions(current_phase=0, green_elapsed_s=20.0,
                                time_since_served_s=[0, 5, 5, 5],
                                phase_queues=[0.0, 1.0, 0.0, 0.0])
    assert report.forced_reason is None


def test_gap_out_never_overrides_minimum_green_or_starvation():
    """S4 must not weaken S1 or S3."""
    sh = _shield()
    early = sh.allowed_actions(0, 3.0, [0, 5, 5, 5], phase_queues=[0.0, 9.0, 0, 0])
    assert early.forced_reason == "min_green" and early.allowed == [0]

    starving = sh.allowed_actions(0, 20.0, [0, 5, 5, 200.0],
                                  phase_queues=[0.0, 9.0, 0, 0])
    assert starving.forced_reason == "starvation" and starving.allowed == [3]


def test_queue_free_callers_still_work():
    """phase_queues is optional; omitting it must not change S1-S3."""
    sh = _shield()
    assert sh.allowed_actions(0, 20.0, [0, 5, 5, 5]).forced_reason is None
    assert sh.mask(0, 20.0, [0, 5, 5, 5]).all()


def test_gridlock_recovery_engages_and_clears_with_hysteresis():
    """S5: sustained jam hands control to round-robin until the queue drains.

    Rare degenerate states let a policy sit while queues spill back and never
    clear. They are infrequent -- about one seed in twelve -- but catastrophic:
    one such run reached 403 s mean delay and 62 teleports against a ~22 s
    median for the same controller.
    """
    sh = _shield_with_recovery()
    jam = [30.0, 30.0, 20.0, 20.0]            # total 100, above gridlock_queue

    for _ in range(sh.gridlock_epochs - 1):
        r = sh.allowed_actions(0, 20.0, [0, 5, 5, 5], phase_queues=jam)
        assert r.forced_reason != "gridlock_recovery", "engaged too early"
    assert not sh.in_recovery

    r = sh.allowed_actions(0, 20.0, [0, 5, 5, 200.0], phase_queues=jam)
    assert sh.in_recovery
    assert r.forced_reason == "gridlock_recovery"
    assert r.allowed == [3], "recovery must serve the longest-waiting phase"

    # Hysteresis: still recovering just under the jam threshold.
    sh.allowed_actions(0, 20.0, [0, 5, 5, 5], phase_queues=[20.0, 20.0, 20.0, 20.0])
    assert sh.in_recovery

    # Clears only once well below.
    sh.allowed_actions(0, 20.0, [0, 5, 5, 5], phase_queues=[5.0, 5.0, 5.0, 5.0])
    assert not sh.in_recovery
    assert sh.n_recoveries == 1


def test_gridlock_recovery_still_honours_minimum_green():
    """S5 must not truncate a green below the safety minimum."""
    sh = _shield_with_recovery()
    jam = [30.0, 30.0, 20.0, 20.0]
    for _ in range(sh.gridlock_epochs + 2):
        sh.allowed_actions(0, 20.0, [0, 5, 5, 5], phase_queues=jam)
    assert sh.in_recovery
    early = sh.allowed_actions(0, 2.0, [0, 5, 5, 200.0], phase_queues=jam)
    assert early.forced_reason == "min_green" and early.allowed == [0]


def test_no_recovery_on_a_healthy_intersection():
    sh = _shield()
    for _ in range(50):
        sh.allowed_actions(0, 20.0, [0, 5, 5, 5], phase_queues=[4.0, 3.0, 2.0, 1.0])
    assert not sh.in_recovery and sh.n_recoveries == 0


def test_gridlock_recovery_is_disabled_by_default():
    """S5 ships off: measurement did not support it.

    On the seed where max-pressure gridlocks, recovery engaged 14 times and
    finished slightly worse than leaving the policy alone (454 s against 412 s).
    Once queues have spilled back that far the jam is not something a
    round-robin can unwind.
    """
    sh = _shield()
    assert sh.gridlock_queue == float("inf")
    jam = [40.0, 40.0, 40.0, 40.0]
    for _ in range(50):
        r = sh.allowed_actions(0, 20.0, [0, 5, 5, 5], phase_queues=jam, discharge=0.0)
        assert r.forced_reason != "gridlock_recovery"
    assert not sh.in_recovery and sh.n_recoveries == 0


def test_recovery_does_not_fire_on_a_busy_but_moving_intersection():
    """Saturated is not jammed: the discriminator is discharge, not queue.

    Gating on queue alone fired recovery during normal operation under a 2.4x
    surge, where round-robin is much worse than the policy.
    """
    sh = _shield_with_recovery()
    jam = [30.0, 30.0, 20.0, 20.0]
    for _ in range(30):
        sh.allowed_actions(0, 20.0, [0, 5, 5, 5], phase_queues=jam, discharge=6.0)
    assert not sh.in_recovery, "recovery fired while traffic was still discharging"


# ---------------------------------------------------------------------------
# S6: bounded emergency service latency
# ---------------------------------------------------------------------------
def test_s6_rejects_a_deadline_the_change_interval_forbids():
    """A guarantee physics cannot deliver must be refused, not silently kept."""
    try:
        _shield(ev_service_deadline_s=5.0)
    except ValueError:
        return
    raise AssertionError("accepted a deadline below emergency green + change interval")


def test_s6_forces_service_of_a_standing_request():
    sh = _shield()
    ages = np.zeros(4)
    ages[2] = sh.ev_deadline_threshold_s + 0.1
    rep = sh.allowed_actions(0, 20.0, np.zeros(4), np.full(4, 5.0),
                             ev_request_age_s=ages)
    assert rep.allowed == [2], rep.allowed
    assert rep.forced_reason == "ev_service_deadline"


def test_s6_does_nothing_before_its_deadline():
    """Below the threshold the request is the priority layer's business, not the
    shield's -- otherwise S6 would be hard preemption wearing a proof."""
    sh = _shield()
    ages = np.zeros(4)
    ages[2] = sh.ev_deadline_threshold_s - 0.1
    rep = sh.allowed_actions(0, 20.0, np.zeros(4), np.full(4, 5.0),
                             ev_request_age_s=ages)
    assert rep.allowed == [0, 1, 2, 3]


def test_s6_never_widens_the_action_set():
    """The regression that cost 3.3 vehicles of mean queue.

    Written as an early return, S6 answered *instead of* S2 and S4 rather than
    after them, so a request that named every phase re-permitted a green that
    maximum green had already ended. S6 may only ever remove options.
    """
    sh = _shield()
    ages = np.full(4, sh.ev_deadline_threshold_s + 1.0)   # every phase due
    rep = sh.allowed_actions(0, 999.0, np.zeros(4), np.full(4, 5.0),
                             ev_request_age_s=ages)
    assert 0 not in rep.allowed, "S2 (max green) was overridden by S6"


def test_s6_cannot_defeat_minimum_green():
    sh = _shield()
    ages = np.zeros(4)
    ages[2] = 999.0
    rep = sh.allowed_actions(0, 1.0, np.zeros(4), np.full(4, 5.0),
                             ev_request_age_s=ages)
    assert rep.allowed == [0] and rep.forced_reason == "min_green"


def test_s6_cannot_defeat_the_starvation_bound():
    """S3 outranks S6 by design: sensor noise must not be able to starve an arm."""
    sh = _shield()
    waits = np.zeros(4)
    waits[1] = sh.starvation_threshold_s + 1.0
    ages = np.zeros(4)
    ages[3] = 999.0
    rep = sh.allowed_actions(0, 20.0, waits, np.full(4, 5.0),
                             ev_request_age_s=ages)
    assert rep.allowed == [1] and rep.forced_reason == "starvation"


def test_s6_serves_only_the_oldest_of_conflicting_requests():
    """Forcing the union lets two conflicting approaches alternate at the
    emergency minimum green so neither ever discharges."""
    sh = _shield()
    ages = np.zeros(4)
    ages[1] = sh.ev_deadline_threshold_s + 30.0
    ages[3] = sh.ev_deadline_threshold_s + 0.5
    rep = sh.allowed_actions(0, 20.0, np.zeros(4), np.full(4, 5.0),
                             ev_request_age_s=ages)
    assert rep.allowed == [1], rep.allowed


def test_s6_bound_holds_against_a_policy_that_refuses_to_serve():
    """The guarantee, run adversarially -- the same treatment S3 gets.

    A request is raised on one phase and a policy that never wants it drives
    the signal. The shield alone must deliver service within the stated bound.
    """
    sh = _shield()
    dt = sh.ev_poll_interval_s
    target, current, green, age = 2, 0, 20.0, 0.0
    waits = np.zeros(4)
    worst = 0.0
    for _ in range(4000):
        ages = np.zeros(4)
        ages[target] = age
        rep = sh.allowed_actions(current, green, waits, np.full(4, 5.0),
                                 emergency=True, ev_request_age_s=ages)
        # adversary: pick anything legal that is not the requested phase
        choice = next((p for p in rep.allowed if p != target), rep.allowed[0])
        if choice != current:
            green = 0.0
            waits += sh.yellow_s + sh.all_red_s
            current = choice
        else:
            green += dt
        waits += dt
        waits[current] = 0.0
        if current == target:
            worst = max(worst, age)
            age = 0.0
        else:
            age += dt
        worst = max(worst, age)
    assert worst <= sh.worst_case_ev_wait_with_starvation_s, (
        f"S6 bound violated: waited {worst:.1f}s against a guarantee of "
        f"{sh.worst_case_ev_wait_with_starvation_s:.1f}s")


def test_guarantees_are_reported_as_numbers():
    g = _shield().guarantees()
    assert g["S6_ev_service_deadline_s"] == 16.0
    assert g["S3_max_movement_wait_s"] == 180.0
    assert g["S6_threshold_s"] < g["S6_ev_service_deadline_s"]


def test_s6_falls_through_to_a_grantable_request():
    """The starvation inversion that produced our worst measured wait.

    A request whose phase is never legal is never served, so its age never
    resets and grows without bound -- after which it outranks every genuine
    request forever. Taking only the oldest due phase then makes S6 do nothing
    at all in a state where another waiting vehicle could have been released
    immediately. Traced on one seed, an emergency vehicle stood on the west
    approach for 145 s behind exactly this.
    """
    sh = _shield()
    ages = np.zeros(4)
    ages[1] = 40.0        # stale request, oldest
    ages[2] = 5.0         # genuine request, younger
    # Every phase carries a queue, so gap-out stays out of the way and the only
    # rule that can remove the current phase is maximum green.
    queues = np.full(4, 5.0)

    # While the oldest request is grantable it keeps priority: phase 1 is the
    # current phase, inside its green, so it stays permitted and S6 holds it.
    rep = sh.allowed_actions(1, 20.0, np.zeros(4), queues, ev_request_age_s=ages)
    assert rep.allowed == [1], rep.allowed

    # Now make the oldest genuinely ungrantable -- phase 1 has run to maximum
    # green, so S2 removes it. S6 must fall through to the next-oldest due
    # phase rather than give up and hand the choice back to the policy.
    rep2 = sh.allowed_actions(1, 999.0, np.zeros(4), queues, ev_request_age_s=ages)
    assert rep2.allowed == [2], rep2.allowed
    assert rep2.forced_reason == "ev_service_deadline"
