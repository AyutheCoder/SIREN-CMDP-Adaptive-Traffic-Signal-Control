"""Safety shield: a correct-by-construction filter on the policy's action set.

The learned policy proposes a target green phase; the shield projects that
proposal onto the set of actions that keep the signal legal and live. This
decouples *performance* (learned, and therefore only statistically good) from
*safety and liveness* (enforced, and therefore guaranteed). It is what lets the
paper claim a hard worst-case bound on service despite an unconstrained neural
policy sitting underneath.

Guarantees
----------
Let ``g_min`` be the minimum green, ``dt`` the decision interval, and
``T = yellow + all_red`` the changeover time. The shield enforces:

**S1 (minimum green).** A green phase, once started, is held for at least
``g_min`` seconds. Phase changes always traverse yellow then all-red, so no two
conflicting movements are ever simultaneously green.

**S2 (maximum green).** No phase is held beyond ``g_max``; the current phase is
removed from the action set once its green has run that long (unless it is the
only legal action, which cannot happen while at least two phases exist).

**S3 (bounded starvation).** Every phase is served at least once every
``W_max`` seconds, where the shield forces the longest-waiting phase as soon as
any phase's wait reaches

    tau = W_max - n_phases * (g_min + dt + T).

*Proof.* Let phase ``p`` cross ``tau`` at epoch ``t_p``. Once above ``tau`` the
shield forces service of ``argmax`` wait, so any phase served before ``p``
must itself have crossed ``tau`` strictly earlier than ``p`` did; there are at
most ``n_phases - 1`` of those. Servicing one phase costs at most
``g_min + dt + T``: at worst S1 obliges the shield to hold the current green for
another ``g_min``, the next decision epoch is at most ``dt`` away, and the
changeover consumes ``T = yellow + all_red``. Adding ``p``'s own service gives
at most ``n_phases`` such intervals, so ``p`` is served by
``tau + n_phases * (g_min + dt + T) = W_max``. QED

The ``n_phases`` factor matters: an earlier version of this shield omitted it
and its 110 s bound was violated (115 s observed) precisely because several
approaches can sit above the threshold at once. The bound is now asserted in
``tests/test_shield.py`` and ``max_movement_wait_s`` is reported for every
evaluation run as a runtime check.

**S6 (bounded emergency service latency).** While a preemption *request* names a
set of phases, the shield forces one of them within ``ev_service_deadline_s`` of
the request being raised. That figure assumes the shield is consulted every
second, as it is while the corridor polls it; when it is consulted only at the
controller's 5 s decision points with the normal 8 s minimum green, the bound is
4 + 5 + 10 + 5 = 24 s on the shipped timings (96 s composed with S3).

This is the guarantee the system did not have, and its absence was the single
largest source of emergency delay. Everything above the shield -- posterior,
calibrated threshold, Bayes-risk arithmetic, conflict schedule -- is an
*option*: at every step it may decline, defer, or be outbid by the cost term.
Measured per-second against ground truth over a seeded hour, the consequences
were exactly what an unbounded option predicts. Eleven of twelve emergency
vehicles came to a full stop; the mean stopped delay was 18.4 s; and the worst
case was a vehicle that sat 15 m short of the stop line for 30 consecutive
seconds while its approach's posterior read 0.00 and no request existed at all.

Deployed preemption (NEMA, Opticom) does not work that way. A detected call
enters a preemption sequence and the sequence *runs*; it is not re-priced
against cross-street delay every second. S6 restores that property without
giving up anything the risk calculus buys, because it is a floor rather than a
replacement: the corridor is still free to open earlier and to choose *which*
serving phase, and the token bucket still bounds how often a spurious request
may be raised. S6 only guarantees that a request which has been raised cannot
be starved.

*Proof.* Let a request naming phase set ``P`` be raised at ``t_0`` and persist.
Once ``t - t_0`` reaches ``tau_ev = ev_service_deadline_s - (g_ev + dt + T)``
the shield restricts the action set to ``P``, where ``g_ev`` is the emergency
minimum green and ``T = yellow + all_red``. At worst S1 then obliges the current
green to run another ``g_ev``, the next poll is at most ``dt`` away, and the
changeover costs ``T``. So service begins by ``tau_ev + g_ev + dt + T =
ev_service_deadline_s``. QED

The bound is stated *against S3*, not above it: S3 is evaluated first, so a
movement about to breach its 180 s starvation bound is still served ahead of an
emergency call. That ordering is deliberate. Inverting it would let a stream of
emergency requests -- including spurious ones, which a probabilistic detector
will always produce -- starve an approach without limit, which is the failure
mode that makes hard preemption unsafe to deploy. When S3 does intervene it
costs the emergency call a service interval per starving phase -- the same
``n_phases`` factor S3's own proof carries, and for the same reason: several
movements can sit above the starvation threshold at once, and S3 walks all of
them. So the composed worst case is ``ev_service_deadline_s + n_phases *
(g_min + dt + T)``. This said "at most one further service interval" and
therefore stated a bound of 34 s that the shield does not meet; measured on a
seed where three phases starved together, an emergency request stood for 49 s
while behaving exactly as specified. Both numbers are reported by
:meth:`SafetyShield.guarantees` and asserted in ``tests/test_shield.py``.

To shorten the *typical* case without touching that worst case, S3 breaks its
own ties toward a phase that also carries a due emergency request. Every
starving phase still leaves the starving set when it is served, so each pass
serves a distinct one and the ``n_phases`` charge is unchanged; only the order
within the set moves, and it moves toward the vehicle that is waiting.

The shield remains unaware of *sensing*: S6 consumes a request, not a
posterior, so the safety argument still does not depend on the detector being
right. A request that is wrong costs cross traffic one bounded service; it
cannot violate S1, S2 or S3, all of which are evaluated before it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class ShieldReport:
    """Why the shield allowed what it allowed at one decision epoch."""

    allowed: List[int]
    forced_reason: Optional[str] = None
    overridden: bool = False
    proposed: Optional[int] = None
    applied: Optional[int] = None


@dataclass
class SafetyShield:
    n_phases: int
    min_green_s: float
    max_green_s: float
    yellow_s: float
    all_red_s: float
    decision_interval_s: float
    max_starvation_s: float

    emergency_min_green_s: float = 6.0
    """Reduced minimum green honoured while a preemption call is active.

    Defaults to the normal minimum if left at or above it, so a shield built
    without emergency awareness behaves exactly as before. The change interval
    is never shortened: an emergency call may cut a green short, but yellow and
    all-red are served in full, so the clearance guarantee that S1 exists to
    protect is unaffected.

    The starvation bound S3 is also unaffected, and in the safe direction. Its
    proof charges each phase ``min_green + decision_interval + yellow +
    all_red`` of service before the starved movement is reached; a rule that
    only ever makes greens *shorter* cannot make that worst case longer, so the
    guaranteed bound is still ``W_max``."""

    ev_poll_interval_s: float = 1.0
    """How often a standing emergency request is re-checked, seconds.

    The corridor polls inside a green once per simulated second
    (``SumoSignalEnv._hold_green``), so this is 1 s rather than the 5 s decision
    interval, and S6's deadline is that much tighter than it would be if
    emergency service could only begin on an epoch boundary."""

    ev_service_deadline_s: float = 16.0
    """S6: seconds within which a standing emergency request must be served.

    Not a tuned performance knob -- it is the service-level guarantee the
    operator is buying, and the only reason it is not smaller is physics. The
    floor is ``g_ev + dt + T`` (emergency minimum green, one poll, one change
    interval), which on the shipped timings is ``6 + 1 + 5 = 12`` s; anything
    below that is unachievable and :meth:`__post_init__` rejects it rather than
    silently promising what the change interval alone forbids. 16 s leaves one
    poll of headroom above the floor.

    Raising it weakens the guarantee; lowering it toward the floor makes S6 fire
    sooner and take more from cross traffic. ``scripts/value_sweep.py`` reports
    the trade-off."""

    gap_out_queue: float = 0.0
    """A phase counts as cleared when its served queue is at or below this."""

    gap_out_min_waiting: float = 1.0
    """Another phase must have at least this many waiting vehicles before a
    gap-out is forced, so the signal does not churn on single arrivals."""

    gap_out_max_load: float = 30.0
    """Total halting vehicles above which gap-out is suspended. Beyond this the
    intersection is saturated and changeover lost time dominates, so holding
    green is the better move."""

    congestion_queue: float = 0.0
    """Total halting vehicles at which the green ceiling reaches
    ``congested_max_green_s``. Zero disables green-ceiling scaling entirely,
    which is the default: we implemented the scaling on the intuition that long
    greens cause spillback and measured the opposite -- shortening green under
    load made a demand surge substantially worse, because it multiplies the
    number of changeovers exactly when lost time is most expensive."""

    congested_max_green_s: float = 20.0
    """Green ceiling at ``congestion_queue``. Only used when that is non-zero."""

    gridlock_queue: float = float("inf")
    """Total halting vehicles above which the intersection is treated as jammed.

    Infinite by default: S5 is implemented and tested but **disabled**, because
    measurement did not support it. On the seed where max-pressure gridlocks,
    recovery engaged 14 times and ended slightly worse than leaving the policy
    alone (454 s against 412 s). Once queues have spilled back that far the jam
    is not a scheduling artefact a round-robin can unwind. Retained because the
    discharge gate makes it harmless when enabled, and it may help on networks
    with more storage."""

    gridlock_epochs: int = 6
    """Consecutive epochs above ``gridlock_queue`` before recovery engages."""

    gridlock_clear_queue: float = 45.0
    """Recovery holds until total queue falls below this (hysteresis, so the
    controller does not oscillate in and out of recovery at the threshold)."""

    gridlock_min_discharge: float = 1.0
    """Vehicles discharged per decision epoch below which the intersection
    counts as stalled rather than merely busy. This is what separates gridlock
    from saturation."""

    n_overrides: int = field(default=0, init=False)
    n_decisions: int = field(default=0, init=False)
    n_gap_outs: int = field(default=0, init=False)
    n_ev_forced: int = field(default=0, init=False)
    """How many times S6 had to force emergency service. A high count means the
    layers above are leaving the guarantee to do their job for them."""
    n_recovery_epochs: int = field(default=0, init=False)
    n_recoveries: int = field(default=0, init=False)
    _jam_epochs: int = field(default=0, init=False)
    _in_recovery: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.n_phases < 2:
            raise ValueError("shield requires at least two green phases")
        # An "emergency minimum" longer than the normal one would be a
        # contradiction; clamp rather than surprise the caller.
        self.emergency_min_green_s = min(self.emergency_min_green_s, self.min_green_s)
        service_cost = (
            self.min_green_s + self.decision_interval_s + self.yellow_s + self.all_red_s
        )
        slack = self.n_phases * service_cost
        if self.max_starvation_s <= slack:
            raise ValueError(
                f"max_starvation_s ({self.max_starvation_s}) must exceed "
                f"n_phases * (min_green + decision_interval + yellow + all_red) "
                f"= {self.n_phases} * {service_cost} = {slack}; otherwise S3 is "
                "unachievable and the guarantee would be vacuous."
            )
        self.service_cost_s = service_cost
        self.starvation_threshold_s = self.max_starvation_s - slack

        # S6. The emergency service cost uses the *emergency* minimum green and
        # the poll interval the corridor actually runs at, because a standing
        # request is re-checked every simulation step rather than once an epoch.
        ev_service_cost = (
            self.emergency_min_green_s + self.ev_poll_interval_s
            + self.yellow_s + self.all_red_s
        )
        if self.ev_service_deadline_s < ev_service_cost:
            raise ValueError(
                f"ev_service_deadline_s ({self.ev_service_deadline_s}) is below "
                f"emergency_min_green + ev_poll_interval + yellow + all_red = "
                f"{ev_service_cost}. No scheduling rule can meet it: the change "
                "interval alone forbids it, so the guarantee would be a promise "
                "the signal cannot keep."
            )
        self.ev_service_cost_s = ev_service_cost
        self.ev_deadline_threshold_s = self.ev_service_deadline_s - ev_service_cost

    # ------------------------------------------------------------------ queries
    def allowed_actions(
        self,
        current_phase: int,
        green_elapsed_s: float,
        time_since_served_s: Sequence[float],
        phase_queues: Optional[Sequence[float]] = None,
        discharge: Optional[float] = None,
        emergency: bool = False,
        ev_request_age_s: Optional[Sequence[float]] = None,
    ) -> ShieldReport:
        """Legal target phases at this decision epoch.

        ``phase_queues`` (halting vehicles served by each phase) enables the
        gap-out rule S4. It is optional so the shield still works for callers
        that have no queue information, but supplying it is strongly
        recommended: without S4 the shield tolerates a phase holding green over
        an empty approach for up to ``max_green_s`` while other approaches spill
        back, which is the mechanism by which every adaptive controller we
        tested collapsed under saturated demand.

        ``ev_request_age_s[p]`` is how long an emergency request naming phase
        ``p`` has been standing and unserved, or 0 where there is none. It
        enables S6. Like ``phase_queues`` it is optional, so a caller with no
        priority layer gets exactly the shield that was proved before.
        """
        self._update_jam_state(phase_queues, discharge)

        # S1: minimum green not yet satisfied -> the only legal action is to hold.
        #
        # A preemption call may truncate green to the shorter emergency minimum
        # instead. This mirrors NEMA preemption practice, where entry into a
        # preemption sequence is allowed to cut a green short of its normal
        # minimum provided a reduced minimum and the full change interval are
        # still served; the change interval (yellow + all-red) is never
        # shortened here, so the clearance guarantee is untouched. It matters
        # because the normal 8 s floor was the binding constraint on how quickly
        # a corridor could open: an emergency vehicle closing at 14 m/s covers
        # 110 m while the shield waits out a green that has nothing left to
        # discharge.
        floor = self.emergency_min_green_s if emergency else self.min_green_s
        if green_elapsed_s < floor:
            return ShieldReport(allowed=[current_phase], forced_reason="min_green")

        # S5 (gridlock recovery). Rare degenerate states exist in which a policy
        # settles into a pattern that lets queues spill back and never clears
        # them. They are infrequent -- roughly one seed in twelve -- but
        # catastrophic when they happen: a single such run reached 403 s mean
        # delay and 62 teleports against a ~22 s median for the same controller,
        # which is enough to destroy the mean of an otherwise healthy set. They
        # affect different controllers on different seeds, including plain
        # max-pressure, so this is a property of the state the signal reaches
        # rather than of any one policy.
        #
        # Once jammed, the shield stops asking the policy and serves the
        # longest-waiting phase until the queue drains. Strict round-robin is
        # not optimal, but it is guaranteed to discharge every approach, which
        # is what a jammed intersection needs.
        if self._in_recovery:
            self.n_recovery_epochs += 1
            waits = np.asarray(time_since_served_s, dtype=np.float64)
            return ShieldReport(allowed=[int(np.argmax(waits))],
                                forced_reason="gridlock_recovery")

        # S6's due set, built before S3 so that S3 can break its own ties
        # toward an emergency call. It is only ever consulted as a *narrowing*
        # of a set some other rule has already chosen; see _narrow_to_ev.
        ev_due: List[int] = []
        if ev_request_age_s is not None:
            ages = np.asarray(ev_request_age_s, dtype=np.float64)
            if ages.size == self.n_phases:
                due = np.flatnonzero(ages >= self.ev_deadline_threshold_s)
                if due.size > 0:
                    # Every due phase, oldest first. Not just the oldest: the
                    # oldest may be one the shield cannot legally grant right
                    # now, and taking only that one makes S6 do *nothing* in a
                    # state where it could have served a different waiting
                    # vehicle immediately.
                    #
                    # That is not a corner case, it is a starvation inversion,
                    # and it produced the worst emergency wait we measured. A
                    # request whose phase is never legal is also never served,
                    # so its age never resets and it grows without bound --
                    # after which it outranks every genuine request forever.
                    # Traced on one seed: an emergency vehicle stood on the west
                    # approach (phases 2 and 3) for 145 s while a stale request
                    # on phase 1, aged past 20 s and never grantable, held the
                    # top of the ordering and suppressed the phases that would
                    # have released it.
                    #
                    # The candidate filter below walks this list in order and
                    # takes the first phase that is actually permitted, so the
                    # oldest request still has priority whenever it *can* be
                    # served, and cannot block the others when it cannot.
                    ev_due = [int(p) for p in due[np.argsort(-ages[due])]]

        # S3: a movement is about to breach its service bound -> force it.
        waits = np.asarray(time_since_served_s, dtype=np.float64)
        starving = np.flatnonzero(waits >= self.starvation_threshold_s)
        if starving.size > 0:
            # Among the phases that are starving, prefer one that also carries a
            # due emergency request. This does not weaken S3 and it is not a
            # priority inversion: whichever starving phase is served, it leaves
            # the starving set, so each pass still serves a *distinct* starving
            # phase and every one of them is reached within the ``n_phases``
            # service intervals the bound already charges for. The ``argmax``
            # was never load-bearing in that proof -- what is load-bearing is
            # that nothing outside the starving set is served while the set is
            # non-empty, which is unchanged.
            #
            # What it fixes is the composition of S3 with S6, which was costing
            # far more than the composed bound admitted. Several phases cross
            # the starvation threshold together on this network -- the two
            # protected lefts in particular, which the policy rarely chooses --
            # and S3 then walks all of them ahead of the emergency call.
            # Measured on one seed: a fire appliance stood at the east stop
            # line with its request aged to 49 s while S3 served phases 0, 1
            # and 3 in turn, against a stated composed bound of 34 s.
            starve = [int(p) for p in starving[np.argsort(-waits[starving])]]
            pick = next((p for p in ev_due if p in starve), starve[0])
            return ShieldReport(allowed=[int(pick)], forced_reason="starvation")

        others = [p for p in range(self.n_phases) if p != current_phase]

        # S6 is evaluated as a *narrowing* of whatever S2 and S4 leave legal,
        # never as a branch that returns ahead of them. Written as an early
        # return it silently widened the action set instead: a request naming
        # every phase -- which the acoustic cross-talk produces on 16% of steps
        # -- returned all four as "allowed" and skipped maximum green and
        # gap-out altogether. Measured, that cost 3.3 vehicles of mean queue and
        # made emergency delay worse, not better, because the congestion it
        # created was congestion the emergency vehicle then had to sit in.
        #
        # The candidate set is therefore built first and filtered afterwards, so
        # S6 can only ever remove options. It cannot re-permit a phase that S2
        # has ended or that S4 has gapped out.
        # S2: maximum green reached -> the current phase must yield.
        effective_max_green = self.effective_max_green(phase_queues)
        if green_elapsed_s >= effective_max_green:
            allowed = self._narrow_to_ev(others, ev_due)
            return ShieldReport(allowed=allowed,
                                forced_reason="max_green" if allowed == others
                                else "ev_service_deadline")

        # S4 (gap-out): the current phase has cleared its queue while another
        # phase still has one, so holding green buys nothing and costs the
        # waiting approach directly.
        #
        # Crucially this applies only while the intersection is *under*
        # saturated. Once queues are long, every changeover costs
        # yellow + all-red of pure lost time, and a controller that gaps out
        # eagerly spends its capacity on changeovers rather than discharge --
        # the classical result that oversaturated intersections want longer
        # cycles, not shorter. We measured the mistake: applying gap-out
        # unconditionally took a 2.4x demand surge from 63 s to 105 s mean
        # delay and from 2 to 6 teleports, while on balanced demand the same
        # rule is mildly beneficial (16.5 -> 15.2 s emergency stopped delay at
        # no cost). It is therefore gated on load.
        #
        # It is also suspended while an emergency call is active, and that
        # exception is not a courtesy -- without it the two mechanisms fight and
        # the priority layer loses. The corridor's whole purpose is to have the
        # queue on the served approach *cleared* by the time the emergency
        # vehicle arrives; the moment it succeeds, that phase's queue is empty,
        # gap-out fires, and the green is taken away seconds before the vehicle
        # gets there. Measured, the corridor was recovering essentially none of
        # the available emergency-delay reduction for exactly this reason.
        #
        # Suspending it costs nothing in safety or liveness: gap-out is a
        # performance rule that *removes* the current phase from the permitted
        # set, so suppressing it only leaves that phase available. S1, S2 and S3
        # all return before this point and are untouched, so maximum green and
        # the starvation bound still force release on schedule.
        if phase_queues is not None and not emergency:
            q = np.asarray(phase_queues, dtype=np.float64)
            if q.size == self.n_phases and float(q.sum()) < self.gap_out_max_load:
                waiting = [p for p in others if q[p] > self.gap_out_min_waiting]
                if q[current_phase] <= self.gap_out_queue and waiting:
                    allowed = self._narrow_to_ev(waiting, ev_due)
                    return ShieldReport(allowed=allowed,
                                        forced_reason="gap_out" if allowed == waiting
                                        else "ev_service_deadline")

        allowed = self._narrow_to_ev(list(range(self.n_phases)), ev_due)
        if allowed != list(range(self.n_phases)):
            return ShieldReport(allowed=allowed, forced_reason="ev_service_deadline")
        return ShieldReport(allowed=allowed)

    def _narrow_to_ev(self, candidates: Sequence[int],
                      ev_due: Sequence[int]) -> List[int]:
        """Restrict ``candidates`` to the most urgent grantable emergency phase.

        ``ev_due`` arrives ordered oldest request first. This walks it in that
        order and returns the first phase that is *also* currently permitted, so
        priority follows waiting time while an ungrantable request cannot veto
        the ones that could be served instead.

        Returns ``candidates`` unchanged when no request is due, and when none
        of the due phases is currently legal -- S6 may never re-permit something
        S2 or S4 has ruled out, since those carry maximum green and the
        anti-spillback guarantee. The request then keeps ageing and is served at
        the next poll at which one of its phases becomes legal, bounded by S1's
        residual minimum green.

        Exactly one phase is returned rather than the set of equally-aged ones.
        Handing back several lets the policy pick among them, and the phases in
        a due set generally belong to *different* approaches, so the policy is
        free to alternate between them at the emergency minimum green and
        discharge neither.
        """
        cand = list(candidates)
        if not ev_due:
            return cand
        allowed = set(cand)
        for p in ev_due:
            if p in allowed:
                # Counted only when the narrowing actually removes something.
                # ``allowed_actions`` is re-entered on every mask query, every
                # projection and every one-second poll inside a green, so an
                # unconditional increment here counts polls rather than
                # interventions: on a 30-minute episode with twelve emergency
                # vehicles it read 33, of which most were epochs where the
                # policy had asked for that phase anyway and S6 removed no
                # option at all. The number is reported as "how often the
                # layers above left the guarantee to do their job for them",
                # so it has to mean that.
                if len(cand) > 1:
                    self.n_ev_forced += 1
                return [int(p)]
        return cand

    def _update_jam_state(self, phase_queues: Optional[Sequence[float]],
                          discharge: Optional[float] = None) -> None:
        """Track whether the intersection is *jammed*, with hysteresis.

        Jammed is not the same as busy, and the difference is discharge. A
        saturated intersection has long queues and keeps moving vehicles; a
        gridlocked one has long queues and has stopped. Triggering on queue
        alone conflates the two, and the cost of that confusion is large: an
        earlier version fired recovery whenever total queue passed 90 vehicles,
        which under a 2.4x demand surge is simply the normal operating state,
        and round-robin is markedly worse there than the learned policy
        (104 s against 48 s mean delay). Recovery therefore also requires
        discharge to have collapsed.
        """
        if phase_queues is None:
            return
        q = np.asarray(phase_queues, dtype=np.float64)
        if q.size != self.n_phases:
            return
        total = float(q.sum())

        if self._in_recovery:
            # Leave recovery once the queue drains *or* traffic is moving again.
            moving = discharge is not None and discharge > self.gridlock_min_discharge
            if total <= self.gridlock_clear_queue or moving:
                self._in_recovery = False
                self._jam_epochs = 0
            return

        stalled = discharge is None or discharge <= self.gridlock_min_discharge
        if total >= self.gridlock_queue and stalled:
            self._jam_epochs += 1
            if self._jam_epochs >= self.gridlock_epochs:
                self._in_recovery = True
                self.n_recoveries += 1
        else:
            self._jam_epochs = 0

    @property
    def in_recovery(self) -> bool:
        return self._in_recovery

    def effective_max_green(self, phase_queues: Optional[Sequence[float]] = None) -> float:
        """Maximum green for the current conditions.

        Held at ``max_green_s`` while the intersection is coping and shortened
        toward ``min_green_s`` as total queue rises. Long greens maximise
        discharge only when downstream capacity exists; once queues approach
        the storage available on an approach, a long green fills the receiving
        link and the extra time is spent creating spillback rather than moving
        anyone. Measured: with a fixed 55 s ceiling every adaptive controller in
        our comparison gridlocked under a 2.4x demand surge, while SUMO's
        actuated controller -- which gaps out and holds a short ceiling --
        did not.
        """
        if phase_queues is None or self.congestion_queue <= 0:
            return self.max_green_s
        q = np.asarray(phase_queues, dtype=np.float64)
        if q.size != self.n_phases:
            return self.max_green_s
        load = float(np.clip(q.sum() / self.congestion_queue, 0.0, 1.0))
        return float(self.max_green_s - load * (self.max_green_s - self.congested_max_green_s))

    def mask(
        self,
        current_phase: int,
        green_elapsed_s: float,
        time_since_served_s: Sequence[float],
        phase_queues: Optional[Sequence[float]] = None,
        discharge: Optional[float] = None,
        emergency: bool = False,
        ev_request_age_s: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Boolean mask over actions, for masking Q-values before the argmax."""
        report = self.allowed_actions(current_phase, green_elapsed_s,
                                      time_since_served_s, phase_queues, discharge,
                                      emergency, ev_request_age_s)
        m = np.zeros(self.n_phases, dtype=bool)
        m[report.allowed] = True
        return m

    def project(
        self,
        proposed_action: int,
        current_phase: int,
        green_elapsed_s: float,
        time_since_served_s: Sequence[float],
        phase_queues: Optional[Sequence[float]] = None,
        discharge: Optional[float] = None,
        emergency: bool = False,
        ev_request_age_s: Optional[Sequence[float]] = None,
    ) -> ShieldReport:
        """Project a proposed action onto the legal set, recording any override."""
        report = self.allowed_actions(current_phase, green_elapsed_s,
                                      time_since_served_s, phase_queues, discharge,
                                      emergency, ev_request_age_s)
        if report.forced_reason == "gap_out":
            self.n_gap_outs += 1
        report.proposed = proposed_action
        self.n_decisions += 1
        if proposed_action in report.allowed:
            report.applied = proposed_action
        else:
            report.applied = report.allowed[0]
            report.overridden = True
            self.n_overrides += 1
        return report

    @property
    def override_rate(self) -> float:
        return self.n_overrides / max(self.n_decisions, 1)

    def reset_counters(self) -> None:
        self.n_overrides = 0
        self.n_decisions = 0
        self.n_gap_outs = 0
        self.n_ev_forced = 0
        self.n_recovery_epochs = 0
        self.n_recoveries = 0
        self._jam_epochs = 0
        self._in_recovery = False

    @property
    def worst_case_wait_s(self) -> float:
        """The value of ``W_max`` that S3 guarantees."""
        return self.max_starvation_s

    @property
    def worst_case_ev_wait_s(self) -> float:
        """The value of ``W_ev`` that S6 guarantees when S3 does not intervene."""
        return self.ev_service_deadline_s

    @property
    def worst_case_ev_wait_with_starvation_s(self) -> float:
        """``W_ev`` composed with the S3 interventions that outrank it.

        ``n_phases`` service intervals, not one. This carried the same omission
        the S3 proof itself was corrected for: several phases can be above the
        starvation threshold simultaneously -- on this network the two
        protected lefts routinely are, because the policy rarely chooses them
        -- and S3 then serves each in turn ahead of the emergency call. One
        interval is the cost of a single intervention; the reachable worst case
        is one per phase, which is exactly why S3's own bound carries the
        factor.

        Measured before the correction, on a seed where three phases starved
        together: an emergency request stood for 49 s against a stated bound of
        34. The bound was wrong; the shield was behaving as specified.

        S3 now breaks its own ties toward a due emergency phase where it can
        (see :meth:`allowed_actions`), which shortens the typical case without
        changing this worst case -- the guarantee has to hold when the starving
        phase and the requested one are different, and then the call still
        waits for the phases ahead of it.
        """
        return self.ev_service_deadline_s + self.n_phases * self.service_cost_s

    def guarantees(self) -> dict:
        """Every bound this shield enforces, as numbers rather than prose.

        Reported by the evaluation runner and asserted in the tests, so the
        claims in the paper are checked against the object that makes them
        rather than restated by hand.
        """
        return {
            "S1_min_green_s": self.min_green_s,
            "S1_emergency_min_green_s": self.emergency_min_green_s,
            "S2_max_green_s": self.max_green_s,
            "S3_max_movement_wait_s": self.max_starvation_s,
            "S3_threshold_s": self.starvation_threshold_s,
            "S6_ev_service_deadline_s": self.ev_service_deadline_s,
            "S6_threshold_s": self.ev_deadline_threshold_s,
            "S6_with_starvation_contention_s":
                self.worst_case_ev_wait_with_starvation_s,
        }
