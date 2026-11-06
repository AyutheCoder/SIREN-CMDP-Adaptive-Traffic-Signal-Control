"""Risk-calibrated emergency corridors: when to preempt, and in what order.

The problem this solves
-----------------------
Two things break every emergency-vehicle preemption scheme we could find in the
literature, and both are visible in our own earlier version of this code.

**1. The commit threshold is a free parameter.** Detection is probabilistic, so
something has to turn a posterior into a yes/no. Published systems -- and our
own first overlay -- pick a number (0.5, 0.6, 0.58) and tune it. That number
cannot be right, because the right answer depends on the state: a 0.55 posterior
is worth acting on when the cross street is empty and the vehicle faces 40 s of
red, and is not worth acting on when forty cars are queued on the conflicting
approach. A fixed threshold answers a decision problem with a constant.

**2. Concurrent requests are unhandled.** Deployed preemption systems
(Opticom-style infrared, GPS/V2X, siren detection) and most published RL work
grant the corridor to one vehicle and either ignore or first-come-first-serve
the rest. Reviews of the area name simultaneous conflicting requests as an open
problem. It is not a rare corner: at the emergency rates used in our demo we log
concurrent committed requests in roughly one preemption epoch in six.

What this module does instead
-----------------------------
Both are treated as one decision-theoretic problem, in the units the operator
actually cares about -- vehicle-seconds of delay.

*Risk-calibrated commitment.* For approach ``a`` with posterior presence
``p_a``, preempting buys an expected ``p_a * B_a`` seconds of prioritised delay
avoided and costs ``C_a`` seconds of background delay, which is paid whether or
not the vehicle is real. The Bayes rule is therefore

    preempt  iff  p_a * (1 + lambda) * B_a  >  C_a,

equivalently ``p_a > theta_a* = C_a / ((1 + lambda) B_a)``. The threshold is
*derived* from the current queue state at every epoch rather than tuned, it
rises automatically when preemption is expensive, and ``lambda`` -- the dual
variable of the CMDP the policy was trained under -- is exactly the knob that
converts the operator's emergency-delay budget into willingness to act on weak
evidence. Nothing here is fitted to the evaluation set.

*Anticipation falls out for free.* ``B_a`` is the delay actually avoided, which
is ``max(0, residual_red + queue_clearance - eta)``. A vehicle 40 s away facing
25 s of red loses nothing if we wait, so ``B_a`` is zero and the rule declines;
as it closes, ``B_a`` rises and ``theta*`` falls, and the corridor opens at the
last moment that still clears the queue in front of it. We did not have to add a
lead-time window -- the previous version of this file had two of them, both hand
set, and they were the source of most of its misfires.

*Conflict scheduling.* When several approaches commit at once the scheduler
solves the sequencing problem exactly: order the phases so as to minimise
priority-weighted emergency lateness plus background delay, subject to minimum
green and changeover lost time. With at most four phases this is a search over
at most 64 orderings, microseconds per epoch, and it is optimal for the horizon
it plans over rather than a heuristic. Two vehicles whose approaches share a
phase are served together automatically, because the schedule reasons about the
set of requests a phase clears rather than about vehicles one at a time.

*A chance constraint on spurious preemption.* Acting on a posterior means
sometimes acting on nothing. A token bucket bounds the expected rate of false
preemptions: each engagement spends ``(1 - p_a)`` tokens and the bucket refills
at the operator's budget, so over any window the expected number of corridors
granted to vehicles that were not there is bounded by that budget. When the
bucket is empty the rule still fires, but only on near-certain evidence. This
replaces the duty-cycle cap of the previous version, which limited the wrong
quantity -- it throttled the overlay hardest exactly when emergency traffic was
heaviest and real.

Safety is unaffected
--------------------
The scheduler only ever selects among actions the shield already permits, so
S1 (minimum green), S2 (maximum green) and S3 (bounded starvation) hold
unconditionally, with or without preemption, for any belief and any adversarial
detector output. That is the point of keeping the shield underneath: this module
is free to be aggressive because it cannot be unsafe.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SATURATION_FLOW_PER_LANE = 0.5
"""Saturation headway per lane, veh/s. 1800 veh/h/lane is the HCM default for a
through movement under ideal conditions, and SUMO's Krauss model with the
car-following parameters used here discharges at very close to it."""

STARTUP_LOST_TIME_S = 2.0
"""Time from green onset to the first vehicle crossing the stop line."""

EV_PASS_TIME_S = 4.0
"""Green that must remain after the queue clears for the emergency vehicle
itself to cross. One vehicle length plus the junction at approach speed."""

PLAN_CYCLES = 3
"""How many projected green windows ahead the arrival is tested against."""


@dataclass
class EVRequest:
    """One approach the belief says is carrying an emergency vehicle."""

    approach: str
    direction: str
    phases: Tuple[int, ...]
    presence: float
    priority: float
    eta_s: float
    eta_sd: float
    range_m: float
    range_rate_ms: float
    queue: float
    clearance_s: float
    residual_red_s: float
    """Projected seconds of red the vehicle meets *at its arrival time*, under
    the signal timeline the base policy is running. Not the residual red now."""
    benefit_s: float
    cost_s: float
    threshold: float
    committed: bool
    delay_no_preempt_s: float = 0.0
    delay_if_preempt_s: float = 0.0
    open_time_s: float = 0.0
    """Seconds from acting to a cleared green: changeover plus queue discharge."""

    hold_s: float = 0.0
    """Seconds of green this request was *priced* at: the changeover, the queue
    in front of the vehicle, the wait for the vehicle to arrive, and the time
    it takes to cross.

    Carried on the request because it is the number the corridor owes. Cross
    traffic was charged for exactly this much green when the decision was
    taken (:meth:`CorridorScheduler._preempt_cost_s`), so holding for less is
    not restraint -- it is paying the bill and leaving without the goods."""
    slack_s: float = 0.0
    """How much longer we may wait and still open in time. The corridor acts at
    the moment this reaches zero, not before -- see :meth:`CorridorScheduler.decide`."""
    due: bool = True
    """Is this the last epoch at which acting still delivers the full benefit?"""

    snr_db: float = float("nan")
    """Loudest level above ambient on this approach's mast, dB.

    The measurement the posterior is a decision *about*, and it is carried here
    because it is monotone in closeness where the posterior is not: a detector
    that saturates reads 1.00 at 200 m and at 20 m alike, and the twenty
    decibels between those two are the only thing separating them."""

    snr_rate_db_s: float = float("nan")
    """Rate of change of that level. Positive means the source is closing.

    A siren that is getting louder is approaching, whatever the range-rate head
    says, and the level trend survives both situations that break the Kalman
    rate: a vehicle stopped in a queue (rate zero, still there) and a track
    just re-acquired (rate at its prior)."""

    in_junction: bool = False
    """Has the vehicle crossed the stop line but not yet cleared the junction?

    The signed filter reports a small negative range for a vehicle that is *on*
    the junction, and the previous screening test -- ``range > 0`` -- dropped
    the request outright at exactly that moment, handing the green back with
    the emergency vehicle still inside the intersection."""

    @property
    def weight(self) -> float:
        """Scheduling weight: triage priority scaled by how sure we are."""
        return self.presence * self.priority

    def describe(self) -> str:
        eta = "  n/a" if not np.isfinite(self.eta_s) else f"{self.eta_s:5.1f}"
        thr = "  inf" if not np.isfinite(self.threshold) else f"{self.threshold:.2f}"
        # The level and its trend go first because they are the measurement --
        # everything after them is inference from it. An arrow that is rising
        # while the range falls is the whole mechanism in two characters.
        if np.isfinite(self.snr_db):
            trend = ("^" if np.isfinite(self.snr_rate_db_s) and self.snr_rate_db_s > 0.15
                     else "v" if np.isfinite(self.snr_rate_db_s)
                     and self.snr_rate_db_s < -0.15 else "-")
            level = f"snr={self.snr_db:4.0f}dB{trend} "
        else:
            level = ""
        where = " IN-JUNCTION" if self.in_junction else ""
        return (f"{self.direction}: {level}p={self.presence:.2f} (need {thr}) "
                f"eta={eta}s rng={self.range_m:5.0f}m{where} "
                f"red@arrival={self.residual_red_s:4.0f}s "
                f"save={self.delay_no_preempt_s - self.delay_if_preempt_s:5.1f}s "
                f"slack={self.slack_s:5.1f}s cost={self.cost_s:6.1f}s")


@dataclass
class CorridorDecision:
    """What the scheduler did this epoch, and why -- used by the demo HUD."""

    engaged: bool
    phase: Optional[int] = None
    target: Optional[EVRequest] = None
    requests: List[EVRequest] = field(default_factory=list)
    committed: List[EVRequest] = field(default_factory=list)
    schedule: Tuple[int, ...] = ()
    service_phases: Tuple[int, ...] = ()
    """Phases carrying a standing emergency request, for the shield's S6 clock.

    Deliberately *not* the same set as ``schedule``. The schedule is what the
    risk calculus decided to do; this is what the evidence says is owed. They
    differ exactly when the economics decline a vehicle that is really there --
    which, before S6 existed, is where the emergency delay came from."""
    reason: str = ""
    conflict: bool = False
    imminent: bool = False
    """Is the served vehicle close enough that gap-out must be held off?

    Engaging the corridor and suspending the shield's gap-out rule were one
    decision until this flag existed, and they should never have been. Gap-out
    is what releases a phase whose queue has cleared, and on a near-capacity
    intersection it is the mechanism that prevents spillback; switching it off
    for as long as a corridor is open means switching it off whenever an
    emergency vehicle exists anywhere on a green arm. Measured, that gridlocked
    two seeds in twelve. Wanting the phase and needing the stability rule
    disabled are different claims, and only the second needs the vehicle to be
    seconds away."""

    @property
    def n_committed(self) -> int:
        return len(self.committed)


@dataclass
class CorridorScheduler:
    """Bayes-risk preemption with exact conflict scheduling.

    Every field below is either a physical constant, a quantity the operator
    sets as policy, or a screening bound that exists to keep the search cheap.
    None of them is a tuned decision threshold -- that is computed per epoch.
    """

    calibrator: object = None
    """Monotone map from raw filter presence to an actual probability.

    The Bayes rule below multiplies ``p`` by a value ratio and compares against
    a delay cost, which is arithmetic on a probability. The recursive presence
    filter does not produce one: it ranks well (AUC 0.90-0.95) and is
    over-confident by roughly a factor of three in the band that matters --
    fitted against ground truth, a raw reading of 0.97 corresponds to a true
    probability of 0.16. Supplying a calibrator fixes the units; omitting one
    leaves the raw score in place, which is what the ablation in the paper
    measures. See ``atsc/sensing/calibration.py``."""

    # --- operator policy ---
    ev_value_ratio: float = 50.0
    """Background vehicle-seconds that one priority-weighted emergency
    vehicle-second is worth. This is the one genuinely normative number in the
    method and it is deliberately exposed rather than buried: preemption is a
    transfer of delay from an emergency vehicle to everybody else, and somebody
    has to say the exchange rate.

    It is not tuned on our results. 50 sits inside the range implied by practice
    -- signal-priority schemes routinely weight an emergency vehicle at tens of
    passenger-car equivalents, and the gap between the value of ordinary travel
    time and of emergency response time is wider still. Section
    ``scripts/frontier.py`` sweeps it, and the sweep is the headline
    operating-point figure: every value of this ratio yields a different,
    predictable point on the delay / emergency-delay frontier, which is what an
    operator actually needs to choose from."""

    false_preempt_per_hour: float = 10.0
    """Chance constraint: expected spurious preemptions per hour. The token
    bucket enforces it in expectation over any window.

    It has to be set against what the sensor can achieve, and that is
    arithmetic rather than taste. An engagement on a request whose calibrated
    posterior is ``p`` contributes ``1 - p`` expected spurious corridors, and
    ``p`` is bounded above by the detector's precision -- 0.59 on the shipped
    map. So a controller that opens a corridor for every emergency vehicle
    that is genuinely there, at the 24/h the demonstration runs at and roughly
    one engagement each, spends about ``24 * 0.45 = 11`` per hour *while being
    right every time*. Any budget below that is not a constraint on spurious
    preemption; it is a cap on service that happens to be spelled in the units
    of spurious preemption.

    This was 2.0, and that is what it did. Measured over a 30-minute episode at
    24 emergency vehicles an hour, the bucket emptied inside the first two
    minutes and the near-certainty gate then applied to every subsequent
    corridor: 112 polls blocked, twelve of twelve vehicles brought to a full
    stop, mean stopped delay 15.5 s and a worst case of 61 s. Nothing in the
    logs distinguished it from weak evidence, because the two share a branch.

    Ten leaves the constraint doing the job it was written for -- bounding a
    detector that has stuck on, or a burst of corridors opened on scraps of
    evidence -- without standing between the controller and the vehicles that
    are really there. ``scripts/frontier.py`` sweeps it; the count actually
    spent is reported as ``expected_false_preemptions`` after every run, so
    whether the budget is binding is visible rather than assumed."""

    high_confidence: float = 0.90
    """Confidence required once the false-preemption budget is exhausted,
    **as a fraction of the largest posterior the sensor can produce**.

    It was an absolute probability, and that was a latent failure rather than a
    conservative choice. Calibration maps the filter's raw score onto a true
    probability, and the largest value the map can emit is bounded by the
    detector's precision -- 0.63 on the shipped map. A gate at an absolute 0.90
    is therefore unreachable by construction: the first time the token bucket
    ran dry the corridor stopped opening for anything at all, permanently,
    until the bucket refilled at the operator's two per hour. Nothing in the
    logs said so, because "budget exhausted" and "evidence too weak" are the
    same branch.

    Expressed as a fraction of the attainable maximum the gate means what it
    was meant to mean -- near-certainty *for this sensor* -- and stays a real
    constraint rather than an off switch."""

    stopline_tolerance_m: float = 14.0
    """How far past the stop line a vehicle is still treated as ours to serve.

    The joint filter's range is signed about the stop line, and the screening
    test was ``range > 0``. A vehicle physically inside the junction has a
    small negative range, so the request vanished at the precise moment the
    green had to be *held* -- the vehicle is committed, in the intersection,
    and the corridor stops asking for the phase that is carrying it. One
    junction width plus a vehicle length is the distance over which it is still
    the signal's problem."""

    snr_commit_db: float = 8.0
    """Level above ambient at which a mast is treated as certainly hearing a
    siren nearby, whatever the filter's track is doing.

    Not a second detection threshold -- the posterior still has to clear its
    derived threshold before anything is preempted. This governs only whether
    the *level trend* is trusted as a closing test, which matters because the
    Kalman range rate is zero for a vehicle stopped in a queue and at its prior
    for a track just re-acquired, and in both cases the vehicle is plainly
    approaching."""

    level_window_s: float = 6.0
    """Window over which the level trend is differenced."""

    range_window_s: float = 6.0
    """Window over which the tracked range is differenced for a closing speed.

    Sized against the sensor's resolution rather than against the traffic: the
    hypothesis grid is log-spaced and its bins beyond 150 m are 60-85 m apart,
    so a one-second difference is dominated by quantisation. Six seconds is a
    vehicle-at-the-limit displacement of 84 m, which is larger than the coarsest
    bin, and still short enough to be current."""

    range_reset_jump_m: float = 60.0
    """Upward jump in tracked range treated as the track resetting rather than
    the vehicle receding. Differencing across a reset reports a large closing
    speed for a vehicle that merely had its prior reinstated."""

    gapout_hold_eta_s: float = 8.0
    """Arrival time within which the shield's gap-out rule is held off.

    One changeover plus a little: the window in which releasing the phase would
    actually strand the vehicle, rather than merely be a phase it will not use.
    Outside it the corridor may still hold the phase against the policy -- that
    is priced -- but gap-out keeps its authority to release a green whose queue
    has cleared, which is what keeps the cross street moving."""

    gapout_hold_range_m: float = 60.0
    """Distance within which gap-out is held off regardless of arrival time.

    A second, independent way into the same window, because the two estimates
    fail in different places and the consequence of missing the window is a
    vehicle stranded metres short of a green it had already been granted.

    Arrival time is range over closing speed and is therefore a ratio of two
    estimates, either of which can be wrong on its own: a vehicle decelerating
    onto the back of a queue has a closing speed heading for zero and an
    arrival time heading for infinity while it is fifty metres out and
    committed. Range is the estimate the filter is actually good at near the
    stop line, where the mast is. Sixty metres is about four seconds at
    approach speed and comfortably inside one changeover, so a vehicle this
    close is one whose phase must not be gapped out from under it whatever the
    rate head says."""

    just_in_time: bool = True
    """Wait until the last instant at which the corridor can still be opened in
    time, rather than acting as soon as the risk rule is satisfied.

    On by default and exposed only so the ablation can turn it off: holding a
    corridor buys nothing extra for starting early and costs cross traffic one
    denied vehicle-second for every vehicle-second it is held, so the optimal
    onset is the deadline. Setting this False reproduces the act-on-sight rule
    the previous version used."""

    commit_range_m: float = 0.0
    """Distance at which a supported request is committed outright, metres.

    A distance trigger rather than an economic one, and deliberately outside
    the Bayes rule: inside this range the corridor opens because the vehicle is
    close, whatever the cross-street queue is worth. It exists for the
    deployment case the risk calculus cannot express -- an operator who has
    decided that an emergency vehicle within a block of the junction is served,
    full stop -- and it is what makes the road ahead actually clear before the
    vehicle gets there rather than as it arrives.

    The evidence floor still applies (:attr:`service_request_presence`), so
    this is not a way to preempt on noise; it only removes the *cost* side of
    the test for a vehicle the sensor is already confident about. Zero disables
    it, which is the research default: every number in the paper is produced
    with the economics alone.

    What it is measured to do, over twelve seeds at 24 emergency vehicles an
    hour with a 150 m range: full stops per emergency vehicle 1.04 -> 0.93 and
    a marked drop in how often S6 has to force service, bought with 0.8 veh of
    extra standing background queue (10.7 -> 11.4) and **no** change in mean
    emergency stopped delay (+0.8 s, s.e. 3.2). That is the trade the Bayes
    rule exists to make properly, and it is why this is not the default here --
    it is exposed because "a vehicle within a block of the junction is served"
    is a rule an operator may simply want to state, and because seeing it run
    next to the economics is the clearest way to see what the economics do.
    """

    hold_presence: float = 0.6
    """Fraction of its own derived threshold an *already-open* corridor must
    still meet to be kept. Hysteresis, not a second threshold: the corridor is
    opened on ``p >= theta`` and closed on ``p < 0.6 theta``, so a posterior
    that flickers around the boundary does not toggle the signal. The closing
    test is still derived from the state, because ``theta`` is."""

    background_weight: float = 1.0
    """Relative price of one background vehicle-second against one emergency
    vehicle-second inside the conflict schedule. At 1.0 an emergency second is
    worth ``priority`` background seconds, which is what the triage weights
    already express; raising it makes the schedule protect cross traffic more."""

    # --- screening bounds (cheap rejects, not decision thresholds) ---
    screen_presence: float = 0.02
    """Below this the posterior is not worth evaluating.

    Deliberately far below anything the risk rule can produce. It has to be:
    once the posterior is calibrated its *maximum* attainable value is bounded
    by the precision of the sensor -- on our detector a raw 1.0 maps to about
    0.46 -- so a screening floor set for raw scores would quietly become the
    decision threshold and undo the whole point of deriving one."""

    max_range_m: float = 300.0
    """Detection radius. Beyond it the tracker is extrapolating."""

    max_eta_s: float = 120.0
    """Arrival times beyond this are not acted on.

    Not a lead-time window -- the just-in-time rule already decides *when* to
    act. This is a validity bound on the estimate itself: at the 320 m arm and
    the minimum closing speed the tracker will accept, an arrival time of two
    minutes means the vehicle is crawling, and range over closing speed is then
    a ratio of two small numbers rather than a prediction."""

    max_hold_s: float = 40.0
    """Longest a single corridor may be held on its clearance guarantee alone.

    Bounds the hold below, never above, what the shield permits: S2 (maximum
    green) and S3 (the starvation bound) are evaluated first and are untouched,
    so this cannot extend a green past either. It exists so a corridor that has
    been opened is seen through even if the posterior collapses mid-crossing,
    which the filter does routinely -- at close range it moves its mass past
    the stop line, declares the vehicle departed and resets the track while the
    vehicle is still short of the junction. Measured on a seeded hour, that
    reset fired with the vehicle between one and sixteen metres from the stop
    line on nine separate approaches."""

    max_receding_ms: float = 4.0
    """Range rate above which an upstream track is treated as lost rather than
    as a queued vehicle. Used only when the range is signed about the stop
    line; see the inbound test in :meth:`_build_requests`."""

    queued_range_m: float = 150.0
    """Hard ceiling on the "stopped in the queue" ETA fallback, metres.

    Arrival time is estimated as range over closing speed, so a vehicle that has
    come to a stop -- closing speed zero -- produces an arrival time of tens of
    thousands of seconds and is screened out as "not arriving". That is exactly
    backwards: the vehicles the corridor exists for are the ones stopped at the
    red it is supposed to lift. Measured on this network, the scheduler built no
    request at all on 60% of the epochs in which an emergency vehicle was
    sitting on a red approach.

    So a stopped source *inside the queue* is treated as arriving now. The
    binding test is :attr:`queue_extent_per_veh_m` below, which asks whether the
    track is close enough to be in the queue that is actually there; this
    ceiling only bounds that test from above for the case where the queue
    estimate is itself unreliable.

    It used to be the only test, at this value, and 150 m is far past any queue
    this intersection forms. The effect was to declare ``eta = 0`` for tracks
    over a hundred metres out: observed on a demonstration run, a request read
    ``eta=0.0s rng=117m`` and the corridor held a green for a vehicle two
    hundred metres and twenty seconds away while the cross street queued nine
    deep behind it."""

    queue_extent_per_veh_m: float = 7.0
    """Road length one queued vehicle occupies, metres. Vehicle plus gap.

    Turns "is it stopped in the queue?" into a question about the queue that is
    measurably there rather than about a fixed radius: ``queue / lanes``
    vehicles back from the stop line, each taking this much road. A track
    further out than that is not waiting at this signal, whatever its closing
    speed says."""

    queue_extent_margin_m: float = 40.0
    """Slack added to the computed queue extent, metres.

    Covers two things, and is sized by the larger of them. A vehicle
    decelerating onto the back of the queue is already committed to stopping
    and should be served with it, which is worth a vehicle length or two. The
    binding term is the range estimate's own error: measured against ground
    truth by distance band, the filter's absolute range error runs to several
    tens of metres, so a margin tighter than that would reject genuinely queued
    vehicles on estimator noise alone.

    It is deliberately *not* large enough to re-admit the case this test was
    added for. On an empty approach the extent collapses to this margin, so a
    stationary track 117 m out -- which is what the flat 150 m ceiling used to
    accept, holding a green for a vehicle twenty seconds away while the cross
    street queued nine deep -- is still rejected."""

    min_closing_ms: float = 1.5
    """Required closing speed. Presence decays slowly by design, so without a
    closing test the corridor stays open for a vehicle that has already gone --
    measured on the previous version of this code, removing the test cost 3.8 s
    of emergency stopped delay, more than the entire benefit of preempting."""

    service_request_presence: float = 0.55
    """Evidence at which an approach is *owed* service, as a fraction of the
    largest calibrated posterior this sensor can emit.

    This is the input to the shield's S6 deadline, and it is deliberately an
    **evidence** test rather than an economic one. The Bayes rule above answers
    "is opening a corridor worth it right now?", and its correct answer is
    sometimes no -- for a vehicle that is genuinely there. Feeding S6 the same
    condition would make the guarantee vacuous, because it would only ever
    promise to do what the economics had already chosen to do.

    So the request is raised on the posterior alone, and from then on the
    vehicle is owed a green within the deadline whatever the queue on the cross
    street is doing. The economics keep their remaining job, which is to open
    the corridor *earlier* than the deadline when that is cheap, and to decide
    which of several requests to serve first.

    Expressed as a fraction of the attainable maximum for the same reason
    :attr:`high_confidence` is: calibration bounds the largest producible
    posterior well below 1, so an absolute figure here would be either
    unreachable or trivial depending on the detector, rather than meaning
    "more likely than not"."""

    theta_min: float = 0.02
    theta_max: float = 0.95
    """Clip on the derived threshold.

    The floor stops a free preemption (cost zero because the base policy wanted
    that phase anyway) from firing on detector noise. The ceiling is a
    numerical guard, not a decision: it sits above the largest posterior this
    sensor can produce (0.63 on the fitted map), so a request clipped to it is
    a request the arithmetic has already declined, and the clip only keeps the
    number finite and comparable. That is deliberate -- unlike
    :attr:`high_confidence`, which *was* an absolute confidence and therefore
    an unsatisfiable rule, nothing here is supposed to be reachable at the
    ceiling."""

    # --- horizon / model ---
    lanes_per_approach: int = 3
    plan_horizon: int = 4
    """Phases planned ahead when resolving conflicts."""

    # --- runtime state ---
    _tokens: float = field(default=0.0, init=False)
    _capacity: float = field(default=0.0, init=False)
    _service_interval: Optional[np.ndarray] = field(default=None, init=False)
    _green_duration: Optional[np.ndarray] = field(default=None, init=False)
    _green_start_at: Optional[np.ndarray] = field(default=None, init=False)
    _last_phase: Optional[int] = field(default=None, init=False)
    _last_served_at: Optional[np.ndarray] = field(default=None, init=False)
    _target: Optional[str] = field(default=None, init=False)
    _held_s: float = field(default=0.0, init=False)
    _hold_phase: Optional[int] = field(default=None, init=False)
    _hold_request: Optional[EVRequest] = field(default=None, init=False)
    _hold_target_s: float = field(default=0.0, init=False)
    """Seconds the open corridor owes its vehicle: the changeover-plus-clearance
    it was priced with, plus the time to cross. See :attr:`max_hold_s`."""
    _levels: Dict[str, List[Tuple[float, float]]] = field(
        default_factory=dict, init=False)
    """Per-approach ``(sim_time, snr_db)`` history, for the level trend."""
    _ranges: Dict[str, List[Tuple[float, float]]] = field(
        default_factory=dict, init=False)
    """Per-approach ``(sim_time, range_m)`` history, for the closing speed."""
    lam: float = field(default=0.0, init=False)

    n_epochs: int = field(default=0, init=False)
    n_preempt_epochs: int = field(default=0, init=False)
    n_engagements: int = field(default=0, init=False)
    n_conflict_epochs: int = field(default=0, init=False)
    n_budget_blocks: int = field(default=0, init=False)
    spent_tokens: float = field(default=0.0, init=False)
    lead_times_s: List[float] = field(default_factory=list, init=False)
    """Arrival time remaining when each corridor was opened.

    The quantity the just-in-time rule is supposed to control, recorded so the
    claim can be checked rather than asserted: acting on sight puts this in the
    tens of seconds and pays for every one of them in cross-traffic delay, while
    acting at the deadline should concentrate it just above the time needed to
    change over and discharge the queue."""
    open_times_s: List[float] = field(default_factory=list, init=False)
    """The changeover-plus-clearance the corridor needed, i.e. the deadline it
    was working back from. ``lead - open`` is the slack actually left on the
    table, and should be small and non-negative."""

    # ------------------------------------------------------------------ setup
    def reset(self, env=None) -> None:
        self._capacity = max(self.false_preempt_per_hour, 0.0)
        self._tokens = self._capacity
        self._target = None
        self._held_s = 0.0
        self._hold_target_s = 0.0
        self._hold_phase = None
        self._hold_request = None
        self._levels = {}
        self._ranges = {}
        self.n_epochs = 0
        self.n_preempt_epochs = 0
        self.n_engagements = 0
        self.n_conflict_epochs = 0
        self.n_budget_blocks = 0
        self.spent_tokens = 0.0
        self.lead_times_s = []
        self.open_times_s = []
        if env is not None:
            n = env.n_actions
            self.lanes_per_approach = int(env.cfg.network.lanes_per_approach)
            # Seeded at a plausible cycle so the first epochs are sane; the
            # estimate is replaced by measurement within one cycle.
            cyc = n * (env.cfg.control.min_green_s + env.cfg.control.yellow_s
                       + env.cfg.control.all_red_s)
            self._service_interval = np.full(n, float(cyc), dtype=np.float64)
            self._green_duration = np.full(
                n, float(env.cfg.control.min_green_s
                         + env.cfg.control.decision_interval_s), dtype=np.float64)
            self._green_start_at = np.zeros(1, dtype=np.float64)
            self._last_phase = int(getattr(env, "current_phase", 0))
            self._last_served_at = np.zeros(n, dtype=np.float64)
        else:
            self._service_interval = None
            self._green_duration = None
            self._green_start_at = None
            self._last_phase = None
            self._last_served_at = None

    def set_lambda(self, lam: float) -> None:
        """CMDP dual variable: the price the operator puts on emergency delay."""
        self.lam = float(np.clip(lam, 0.0, 30.0))

    # ------------------------------------------------------- inter-service EMA
    def observe_service(self, env) -> None:
        """Measure the timeline the base policy is actually running.

        The benefit of preempting is the delay it removes, and the delay it
        removes depends on what the signal would have done *at the moment the
        vehicle arrives* -- not on what it is doing now. That future is not
        knowable, but it is estimable: for each phase, how often it is served and
        for how long. Both are measured here as exponential moving averages
        rather than assumed from a cycle plan, because there is no cycle plan --
        the base policy is a learned controller and its effective cycle changes
        with demand.
        """
        if self._service_interval is None:
            return
        p = int(env.current_phase)
        t = float(env.sim_time)
        gap = t - float(self._last_served_at[p])
        if gap > 1.0:
            self._service_interval[p] = 0.8 * self._service_interval[p] + 0.2 * gap
        if self._green_start_at is not None:
            if self._last_phase is not None and int(self._last_phase) != p:
                dur = t - float(self._green_start_at[0])
                if 0.0 < dur < 300.0:
                    q = int(self._last_phase)
                    self._green_duration[q] = (0.8 * self._green_duration[q]
                                               + 0.2 * dur)
                self._green_start_at[0] = t
            self._last_phase = p
        self._last_served_at[p] = t

    def _green_windows(self, env, phase: int,
                       assume_release: bool = False) -> List[Tuple[float, float]]:
        """Projected [start, end] green windows for ``phase``, from now.

        A renewal projection: the phase is served every ``m`` seconds for ``g``
        seconds, both measured. This is an estimate and is treated as one --
        it enters only through an expectation, and the shield does not depend on
        it being right.

        ``assume_release`` drops the residual of a green that is running *now*,
        and it is what the "do nothing" baseline has to be evaluated under.
        Counting that residual asserts the phase will still be green when the
        vehicle arrives, and nothing guarantees that: absent a priority call the
        policy is free to switch away and the shield's gap-out rule will release
        the phase the moment its queue clears -- which is exactly what a
        corridor arranges to happen just before the vehicle gets there.

        Assuming the green persists is therefore self-defeating, and measurably
        so. It made the projected red zero for every vehicle whose approach was
        green at the decision epoch, hence the benefit zero and the derived
        threshold infinite, so the corridor declined to act and never set the
        call that would have suspended gap-out. On a demonstration run 45 of 78
        emergency-vehicle events priced out this way, each reading
        ``red@arrival = 0`` while the approach was green -- and the vehicles
        then watched that green handed to a conflicting approach as they
        arrived. The delay a corridor avoids by *holding* a green is the red
        that follows releasing it.
        """
        ctrl = env.cfg.control
        changeover = ctrl.yellow_s + ctrl.all_red_s
        if self._service_interval is None:
            m = env.n_actions * (ctrl.min_green_s + changeover)
            g = ctrl.min_green_s
        else:
            m = float(self._service_interval[phase])
            g = float(self._green_duration[phase])
        g = float(np.clip(g, ctrl.min_green_s, ctrl.max_green_s))
        m = float(np.clip(m, g + changeover, ctrl.max_starvation_s))

        out: List[Tuple[float, float]] = []
        if int(env.current_phase) == int(phase) and not assume_release:
            rem = max(0.0, g - float(env.green_elapsed_s))
            out.append((0.0, rem))
            nxt = rem + (m - g)
        elif int(env.current_phase) == int(phase):
            # Released now: the phase goes to yellow, and the next green on it
            # is a full service interval away.
            first = changeover + max(m - g, 0.0)
            out.append((first, first + g))
            nxt = first + m
        else:
            waited = float(env.time_since_served[phase])
            # Earliest physically possible green on a phase that is not running:
            # the current phase must finish its minimum green and the full
            # change interval must be served. Without this floor the renewal
            # estimate is free to say a phase that is overdue -- ``waited >=
            # m``, which is the normal state of the approach an emergency
            # vehicle is on, because that is why it is queued -- goes green at
            # t = 0. The projected red a vehicle meets is then zero, the delay
            # the corridor would avoid is zero, and the derived threshold is
            # C / 0 = infinity: no posterior can clear it.
            #
            # That was the single largest failure in the measured pipeline. On
            # a seeded hour with an emergency vehicle genuinely inbound, 30 of
            # 66 such epochs priced the corridor at an infinite threshold and
            # 25 more were screened out before pricing, leaving four epochs in
            # sixty-six on which the corridor could commit at all. The vehicle
            # sitting at a red stop line is exactly the case the projection was
            # declaring cost-free.
            floor = max(0.0, ctrl.min_green_s - float(env.green_elapsed_s)) + changeover
            expected = m - waited
            if expected <= 0.0:
                # The phase is *overdue*: it has already waited longer than its
                # own measured inter-service interval, so the renewal
                # projection has been falsified and carries no information about
                # when it will next be served. Reading that as "immediately" is
                # the optimistic extreme, and it is not a harmless one -- it is
                # the state every approach an emergency vehicle is queued on is
                # in, precisely because being unserved is why there is a queue.
                # It made the projected red equal to the changeover, so opening
                # a corridor appeared to buy nothing at all.
                #
                # Fall back to the same steady-state residual this method
                # already uses beyond its horizon: a vehicle arriving at a
                # uniformly random moment of a cycle of mean length m with mean
                # green g waits (1 - g/m) * m / 2.
                expected = max((1.0 - g / m) * m / 2.0, 0.0)
            first = max(expected, floor)
            out.append((first, first + g))
            nxt = first + m
        for _ in range(PLAN_CYCLES - 1):
            out.append((nxt, nxt + g))
            nxt += m
        return out

    def _projected_red_at(self, env, phases: Sequence[int], eta: float,
                          assume_release: bool = False) -> float:
        """Seconds of red a vehicle arriving in ``eta`` would still have to wait.

        This is the correction that makes anticipation work at all. The previous
        version compared the vehicle's arrival time against the residual red
        *now*: a vehicle 40 s out facing 20 s of red was scored as losing
        nothing, because the light would have gone green long before it got
        there. That is only true if the green is still on when it arrives, and
        for a 15 s green inside a 45 s cycle it usually is not. Projecting the
        timeline to the arrival time instead recovers the delay the vehicle
        actually faces -- and with it the whole anticipation horizon, which
        collapsed to about five seconds under the old model.
        """
        best = float("inf")
        for p in phases:
            windows = self._green_windows(env, int(p), assume_release)
            for start, end in windows:
                if eta <= end:
                    best = min(best, max(0.0, start - eta))
                    break
            else:
                # The arrival is beyond the projection. Charging the starvation
                # bound here is not conservative, it is wrong: it says a vehicle
                # three minutes away will meet three minutes of red, which
                # scores an enormous benefit for acting now on something the
                # projection knows nothing about. Beyond the horizon the only
                # defensible estimate is the steady-state one -- a vehicle
                # arriving at a uniformly random moment in a cycle of mean
                # length m with mean green g waits (1 - g/m) * m / 2 -- and it
                # is bounded, which the previous answer was not.
                span = windows[-1][0] - windows[0][0]
                cycle = max(span / max(len(windows) - 1, 1), 1e-6)
                green = max(windows[0][1] - windows[0][0], 0.0)
                best = min(best, max(0.0, (1.0 - green / cycle) * cycle / 2.0))
        return float(np.clip(best, 0.0, env.cfg.control.max_starvation_s))

    # ------------------------------------------------------------- the levels
    def _observe_levels(self, env) -> Dict[str, Tuple[float, float]]:
        """Record this poll's per-approach level and return ``(snr, d snr/dt)``.

        Loudness is the part of the acoustic evidence that keeps rising all the
        way to the stop line. The posterior does not: it saturates well before
        the vehicle arrives, and at close range the filter starts moving mass
        past the stop line and the posterior *falls* while the siren is at its
        loudest. So the level is tracked separately and differenced over a few
        seconds, which gives a closing test that works in the two places the
        Kalman range rate does not -- a vehicle stopped in a queue, and a track
        that has just been re-acquired.
        """
        now = float(env.sim_time)
        window = max(self.level_window_s, 1e-3)
        out: Dict[str, Tuple[float, float]] = {}
        readings = getattr(env, "last_readings", None) or {}
        for approach, r in readings.items():
            snr = getattr(r, "snr_db", None)
            spl = getattr(r, "spl_db", None)
            if snr is None or not np.isfinite(snr):
                continue
            # The gate is on the signal-to-noise ratio -- whether the mast can
            # hear anything at all -- but the *trend* is taken on the absolute
            # level, which is the quantity the source's motion moves. Differencing
            # the ratio would fold the noise floor's own drift into the answer.
            level = float(spl) if spl is not None and np.isfinite(spl) else float(snr)
            hist = self._levels.setdefault(approach, [])
            if not hist or now > hist[-1][0] + 1e-6:
                hist.append((now, level))
            # Keep one sample older than the window so the difference spans it.
            while len(hist) > 2 and hist[1][0] < now - window:
                hist.pop(0)
            if len(hist) >= 2 and hist[-1][0] > hist[0][0] + 1e-6:
                rate = (hist[-1][1] - hist[0][1]) / (hist[-1][0] - hist[0][0])
            else:
                rate = float("nan")
            out[approach] = (float(snr), float(rate))
        return out

    def _calibrated(self, presence: float) -> float:
        """Raw filter output -> probability the approach really has a vehicle."""
        if self.calibrator is None:
            return float(presence)
        return float(self.calibrator(float(presence)))

    def _observe_ranges(self, env, beliefs) -> Dict[str, float]:
        """Record this poll's tracked range and return the closing speed, m/s.

        Positive means the range is falling, i.e. the source is approaching.
        NaN means there is not yet enough history to say.

        This exists for the same reason :meth:`_observe_levels` does, and
        answers the question that one cannot: *how fast*. The Kalman rate head
        is differenced against a range whose own resolution beyond 150 m is one
        bin in 60-85 m, so second to second it reads near zero for a vehicle
        doing 14 m/s. Over the window the displacement is larger than the bin
        spacing and the estimate is usable.
        """
        now = float(env.sim_time)
        window = max(self.range_window_s, 1e-3)
        out: Dict[str, float] = {}
        for approach, b in beliefs.items():
            r = float(getattr(b, "range_m", float("nan")))
            if not np.isfinite(r):
                continue
            hist = self._ranges.setdefault(approach, [])
            # A reset track is a new track: differencing across the reset
            # reports a huge closing speed for a vehicle that simply had its
            # prior reinstated.
            if hist and r > hist[-1][1] + self.range_reset_jump_m:
                hist.clear()
            if not hist or now > hist[-1][0] + 1e-6:
                hist.append((now, r))
            while len(hist) > 2 and hist[1][0] < now - window:
                hist.pop(0)
            if len(hist) >= 2 and hist[-1][0] > hist[0][0] + 1e-6:
                out[approach] = float(
                    (hist[0][1] - hist[-1][1]) / (hist[-1][0] - hist[0][0]))
            else:
                out[approach] = float("nan")
        return out

    def _nominal_speed_ms(self, env) -> float:
        """Speed to credit an approaching vehicle when the rate head is mute.

        The arm's own limit, not the emergency vehicle's own top speed: it is
        the speed a vehicle on this road can be *assumed* to make without
        knowing anything about it, and over-estimating it would pull the
        arrival time in and open the corridor too early.
        """
        try:
            return float(max(env.cfg.network.speed_limit_ms, self.min_closing_ms))
        except AttributeError:
            return 13.89

    def _attainable_presence(self) -> float:
        """The largest calibrated posterior this sensor can actually produce.

        Every absolute confidence level in this module has to be expressed
        against this rather than against 1.0. The calibration map is monotone
        and bounded by the detector's precision -- the shipped map tops out at
        0.59 -- so a rule written as "require 0.90" is not a strict rule, it is
        an unsatisfiable one, and the branch it guards silently becomes dead.
        """
        return float(np.clip(self._calibrated(1.0), 1e-3, 1.0))

    # ----------------------------------------------------------- the economics
    def _queued_extent_m(self, queue: float) -> float:
        """How far back from the stop line the queue on this approach reaches.

        The distance a stopped, non-closing track has to be inside before it is
        read as "waiting at this red" rather than as an unusable or stale track.
        Derived from the queue the environment is reporting, so it shrinks to
        the margin alone on an empty approach -- where a stationary source is
        not queued, it is a track that has been lost.
        """
        per_lane = queue / max(self.lanes_per_approach, 1)
        extent = per_lane * self.queue_extent_per_veh_m + self.queue_extent_margin_m
        return float(min(extent, self.queued_range_m))

    def _clearance_s(self, queue: float) -> float:
        """Seconds of green needed to discharge the queue ahead of the vehicle."""
        if queue <= 0.0:
            return 0.0
        rate = max(self.lanes_per_approach * SATURATION_FLOW_PER_LANE, 1e-6)
        return STARTUP_LOST_TIME_S + queue / rate

    def _preempt_cost_s(self, env, phase: int, hold_s: float,
                        base_action: int) -> float:
        """Background vehicle-seconds added by taking ``phase`` for ``hold_s``.

        Zero when the base policy already intends that phase: preemption that
        agrees with the policy is free, and the threshold correctly collapses to
        its floor in that case.
        """
        if int(phase) == int(base_action) and int(phase) == int(env.current_phase):
            return 0.0
        queues = env.phase_queues()
        denied = float(queues.sum() - queues[phase])
        ctrl = env.cfg.control
        changeover = 0.0 if int(phase) == int(env.current_phase)             else ctrl.yellow_s + ctrl.all_red_s
        # Vehicles that would otherwise have been discharged wait for the
        # corridor and for the changeover that opens it.
        return float(denied * (hold_s + changeover))

    # ------------------------------------------------------------ the requests
    def _build_requests(self, env, base_action: int,
                        poll_dt: Optional[float] = None) -> List[EVRequest]:
        """Price every approach the belief says might be carrying a vehicle.

        Two quantities per approach, both in seconds of delay:

        ``B`` -- what preempting *buys*. The vehicle's projected delay if we do
        nothing, minus its delay if we open the corridor now. The first term is
        evaluated against the signal timeline at the vehicle's arrival time
        (:meth:`_projected_red_at`), which is the only version of it that is
        true; the second is non-zero only when we are already too late to clear
        the queue before it gets here.

        ``C`` -- what preempting *costs*. Background vehicle-seconds denied
        while the corridor is held. This is where the timing enters: holding the
        corridor from now until the vehicle passes costs more the earlier we
        start, and ``B`` does not improve for starting early. Acting late is
        therefore strictly better, right up to the last instant at which the
        queue can still be cleared in time -- which is what ``slack`` measures
        and what :meth:`decide` acts on.
        """
        ctrl = env.cfg.control
        dt = float(poll_dt if poll_dt is not None else ctrl.decision_interval_s)
        beliefs = env.belief.beliefs()
        approach_queues = env.approach_queues()
        levels = self._observe_levels(env)
        trends = self._observe_ranges(env, beliefs)
        gain_scale = self.ev_value_ratio * (1.0 + self.lam)
        # The evidence floor the distance trigger shares with S6. Computed once:
        # it depends only on the calibrator, not on the approach.
        service_floor = self.service_request_presence * self._attainable_presence()
        out: List[EVRequest] = []

        for approach, b in beliefs.items():
            presence = self._calibrated(b.presence)
            if presence < self.screen_presence:
                continue
            snr, snr_rate = levels.get(approach, (float("nan"), float("nan")))
            trend = float(trends.get(approach, float("nan")))
            trend_speed = trend if np.isfinite(trend) else 0.0
            # "Getting louder" is a closing test the range-rate head cannot
            # give: it is zero for a vehicle stopped in the queue the corridor
            # exists to clear, and at its prior for a track just re-acquired.
            loud_and_closing = bool(
                np.isfinite(snr) and snr >= self.snr_commit_db
                and np.isfinite(snr_rate) and snr_rate > 0.0)
            # Is the vehicle still upstream of the stop line? The right test
            # depends on which filter produced the belief, and getting it wrong
            # costs in both directions.
            #
            # With the joint filter the range is signed about the stop line, so
            # range > 0 is exactly the question. Adding a closing test on top of
            # it is actively harmful: a vehicle stopped in a queue has a range
            # rate of zero, fails the closing test, and the corridor releases
            # the green for precisely the vehicle it exists to serve. With the
            # per-approach filter the range is distance to the mast and
            # unsigned, so the sign of the range rate is the only signal that a
            # vehicle has gone, and the closing test is required.
            signed = bool(getattr(b, "signed_range", False))
            if not np.isfinite(b.range_m):
                continue
            in_junction = False
            if signed:
                # The state is signed distance to the stop line, so a small
                # negative range is a vehicle *inside the junction*, not one
                # that has gone. Screening it out at ``range > 0`` dropped the
                # request at exactly the moment the green had to be held, and
                # the phase could then be handed to a conflicting approach with
                # the emergency vehicle still crossing.
                if not (-self.stopline_tolerance_m < b.range_m <= self.max_range_m):
                    continue
                in_junction = b.range_m <= 0.0
                if (np.isfinite(b.range_rate_ms)
                        and b.range_rate_ms > self.max_receding_ms
                        and not loud_and_closing):
                    # Signed range still leaves one case: a track moving away
                    # from the stop line while nominally upstream, which is a
                    # lost track rather than a queued vehicle. A mast that is
                    # hearing the siren get *louder* says otherwise, and it is
                    # measuring the thing directly.
                    continue
            else:
                if not (0.0 < b.range_m <= self.max_range_m):
                    continue
                # Three ways to establish that the source is closing, and the
                # screen needs any one of them. The Kalman rate head is the
                # obvious one and is mute over the band where the range grid is
                # coarsest; the level trend answers it when the mast can hear
                # the siren getting louder; and the *windowed* range difference
                # answers it whenever neither can, which is the case this
                # branch used to drop outright.
                #
                # That was a real inconsistency rather than a conservative
                # choice: the windowed trend was already trusted a few lines
                # below, where it supplies the arrival time the rate head
                # cannot, so a track was being screened out here for failing a
                # test the code went on to answer for it. A genuinely
                # stationary track still fails all three, because its range
                # does not fall over the window either.
                closing_now = (np.isfinite(b.range_rate_ms)
                               and b.range_rate_ms <= -self.min_closing_ms)
                if not (closing_now or loud_and_closing
                        or trend_speed >= self.min_closing_ms):
                    continue

            # Primary phases only. A phase that greens one movement out of the
            # approach does not serve a vehicle whose movement we cannot hear,
            # and treating it as though it does is how the corridor came to
            # commit to a protected left for an ambulance going straight --
            # green band drawn, request recorded as served, vehicle at a red.
            # It also removes the sibling swap: with one phase per approach the
            # planner can no longer pay a changeover to arrive where it was.
            phases = tuple(env.phases_serving_approach(approach, primary_only=True))
            if not phases:
                continue

            queue = float(approach_queues.get(approach, 0.0))
            # A vehicle already inside the junction has the approach's queue
            # *behind* it, not in front, so there is nothing left to discharge
            # on its account -- only a green to hold while it finishes
            # crossing. Charging the clearance here would overstate what the
            # corridor buys for it, on the one request whose real requirement
            # is simply "do not change the signal yet".
            clearance = 0.0 if in_junction else self._clearance_s(queue)

            # Time to arrival. When the Kalman ETA sits at its ceiling it
            # carries no information, so fall back on range over closing speed,
            # which is what the ceiling was hiding.
            eta = float(b.eta_s)
            ceiling = env.cfg.belief.max_eta_s - 1e-3
            if not np.isfinite(eta) or eta >= ceiling:
                closing = -float(b.range_rate_ms) if np.isfinite(b.range_rate_ms) else 0.0
                if in_junction:
                    # Already across the stop line: there is no arrival left to
                    # predict, only a junction to finish crossing.
                    eta = 0.0
                elif closing >= self.min_closing_ms:
                    eta = float(b.range_m / closing)
                elif b.range_m <= self._queued_extent_m(queue):
                    # Detected, upstream, and not closing, at a range the queue
                    # actually reaches: it is stopped *in* that queue. Dividing
                    # by its closing speed here is what produced arrival times
                    # of tens of thousands of seconds and screened out the
                    # vehicle the corridor exists to serve. It is not arriving
                    # late -- it is already here and waiting on the signal.
                    eta = 0.0
                elif trend_speed >= self.min_closing_ms:
                    # The instantaneous rate head is mute, but the range has
                    # been *falling across a window*, which is the same
                    # question asked at a resolution the sensor can answer.
                    #
                    # Measured against ground truth over three seeds: of the
                    # vehicle-seconds in which an emergency vehicle was really
                    # travelling faster than 5 m/s, 32% reported an estimated
                    # closing speed below the 1.5 m/s the test above requires,
                    # and the same 32% had the Kalman eta pinned at its
                    # ceiling. The cause is resolution rather than noise: the
                    # hypothesis grid is log-spaced, so beyond 150 m its bins
                    # are 60-85 m apart and a vehicle can cover 50 m without
                    # moving the posterior mean at all -- range MAE in that
                    # band is 83 m. Differenced over one bin spacing the signal
                    # is buried; differenced over six seconds, during which a
                    # vehicle at the limit covers 84 m, it is not.
                    #
                    # "Not measurable this second" is not "not approaching",
                    # and the request that was being dropped here is the one
                    # the corridor exists to open early. A track that is
                    # genuinely stationary still fails this test, because its
                    # range does not fall over the window either -- which is
                    # what keeps a stale or crawling track out.
                    eta = float(b.range_m / trend_speed)
                else:
                    # Not closing by either measure, and beyond the queue that
                    # is actually there: the track carries no usable arrival
                    # time and nothing depends on it yet.
                    continue
            if not np.isfinite(eta) or eta > self.max_eta_s:
                continue
            eta = float(max(eta, 0.0))
            eta_sd = float(np.sqrt(max(b.eta_var, 0.0))) if np.isfinite(b.eta_var) else 5.0

            # --- what it buys ---
            # Priced against *releasing* the phase rather than holding it: the
            # green a vehicle is approaching is only guaranteed if the corridor
            # guarantees it (see _green_windows). But that counterfactual is
            # only the vehicle's business while the green running now is the one
            # it will actually use. Past that, holding buys the vehicle nothing
            # -- it arrives after this green ends either way -- and costs cross
            # traffic every second of it, because an active call suspends the
            # shield's gap-out rule.
            #
            # Applied unconditionally it gridlocks. Measured over twelve seeds:
            # ten improved, and two ran the corridor at a duty cycle of 0.35 and
            # 0.20 against 0.12-0.17 on the healthy ones, held gap-out off long
            # enough for the cross street to spill back, and finished with 79
            # and 11 teleports. Gap-out is what prevents that, and it cannot be
            # suspended merely because an emergency vehicle exists somewhere on
            # a green arm.
            #
            # The horizon is derived, not chosen: it is the residual of the
            # green itself, from the same renewal estimate used everywhere else.
            protect = True
            if int(env.current_phase) in phases:
                held = self._green_windows(env, int(env.current_phase))
                protect = eta <= held[0][1] + EV_PASS_TIME_S
            red_at_arrival = self._projected_red_at(env, phases, eta,
                                                    assume_release=protect)
            # Queue discharge is charged whenever there is a queue, not only
            # when the vehicle meets a red. A vehicle arriving at a green with
            # fifteen cars stopped in front of it still waits for them to move;
            # gating the clearance term on ``red > 0`` wrote that wait out of
            # the benefit and, on the epochs where the projection returned a
            # green, priced the whole corridor at zero.
            delay_none = red_at_arrival + (clearance if (red_at_arrival > 0.0
                                                         or queue > 0.0) else 0.0)

            best = None
            for ph in phases:
                changeover = 0.0 if int(ph) == int(env.current_phase)                     else ctrl.yellow_s + ctrl.all_red_s
                open_time = changeover + clearance
                delay_pre = max(0.0, open_time - eta)
                # Green that must be held if we commit *now*: until the vehicle
                # is through, which is at least until it arrives.
                hold = max(ctrl.ev_min_green_s,
                           max(eta, open_time) + EV_PASS_TIME_S)
                cost = self._preempt_cost_s(env, ph, hold, base_action)
                # The shield can refuse for up to the residual minimum green on
                # the phase running now, so the deadline has to be brought
                # forward by exactly that much. Derived, not tuned.
                latency = 0.0 if int(ph) == int(env.current_phase) else                     max(0.0, ctrl.min_green_s - float(env.green_elapsed_s))
                slack = eta - open_time - latency
                cand = (cost, ph, open_time, delay_pre, slack, hold)
                if best is None or cand[0] < best[0]:
                    best = cand
            cost, _ph, open_time, delay_pre, slack, hold_s = best

            benefit = max(0.0, delay_none - delay_pre)                 * float(b.expected_priority) * gain_scale
            if benefit <= 1e-9:
                theta = float("inf")
            else:
                theta = float(np.clip(cost / benefit, self.theta_min, self.theta_max))

            worth_it = presence >= theta
            # The deadline is not waived for a zero-cost engagement. It used to
            # be: preemption that agrees with the base policy denies no cross
            # traffic, so acting on it immediately looked free. It is not.
            # Engaging raises the emergency call, and an active call suspends
            # the shield's gap-out rule for as long as it is held -- so a
            # vehicle forty seconds out would pin the green for forty seconds,
            # the cross street would never be released, and the cost of that is
            # nowhere in ``_preempt_cost_s``. Measured, it gridlocked two seeds
            # in twelve at corridor duty cycles of 0.35 and 0.20.
            #
            # Deferring costs nothing, because the scheduler is polled every
            # simulated second and will engage the moment the request is
            # genuinely due. Which is the just-in-time rule's whole argument,
            # applied to the one case that had been exempted from it.
            #
            # The waiver survives for a vehicle that is arriving into *this*
            # green, which is the case it was written for: there the engagement
            # really does deny nobody, because the phase is already running and
            # is the one about to be used. Deleting it outright rather than
            # bounding it costs more than the bug did -- measured over twelve
            # seeds, emergency stopped delay 14.17 s -> 16.00 s and clean pass
            # 0.222 -> 0.210 against leaving the original behaviour alone.
            free = cost <= 1e-9 and protect
            # Act one poll before the deadline, plus whatever the arrival-time
            # estimate is worth doubting by. Acting at ``slack <= dt`` exactly
            # is acting at the median, which is late half the time -- measured
            # over twelve seeds the slack actually left on the table averaged
            # -1.7 s, i.e. the corridor was typically opening after the moment
            # it needed to. The margin is the filter's own standard deviation
            # on the arrival time, bounded by one changeover: past that the
            # estimate is not uncertain, it is uninformative, and buying
            # insurance against it would just mean opening every corridor on
            # sight, which is the rule the just-in-time argument replaced.
            margin = 0.0
            if self.just_in_time:
                changeover = ctrl.yellow_s + ctrl.all_red_s
                margin = float(np.clip(eta_sd, 0.0, changeover))
            due = bool(slack <= dt + margin or free) if self.just_in_time else True

            # Distance trigger. Close enough, and supported by enough evidence
            # to be owed service at all, is its own reason to open the corridor
            # -- see ``commit_range_m``. It bypasses the economics and the
            # deadline, not the sensor: an approach below the service floor
            # raises nothing here either.
            near = bool(self.commit_range_m > 0.0
                        and presence >= service_floor
                        and (in_junction or b.range_m <= self.commit_range_m))
            if near:
                due = True

            out.append(EVRequest(
                approach=approach,
                direction=env.approach_direction.get(approach, approach),
                phases=phases,
                presence=float(presence),
                priority=float(b.expected_priority),
                eta_s=eta,
                eta_sd=eta_sd,
                range_m=float(b.range_m),
                range_rate_ms=float(b.range_rate_ms),
                queue=queue,
                clearance_s=clearance,
                residual_red_s=float(red_at_arrival),
                benefit_s=float(benefit),
                cost_s=float(cost),
                threshold=theta,
                committed=bool((worth_it and due) or near),
                delay_no_preempt_s=float(delay_none),
                delay_if_preempt_s=float(delay_pre),
                open_time_s=float(open_time),
                hold_s=float(hold_s),
                slack_s=float(slack),
                due=due,
                snr_db=float(snr),
                snr_rate_db_s=float(snr_rate),
                in_junction=bool(in_junction),
            ))
        return out

    # ----------------------------------------------------- conflict scheduling
    def _schedule_cost(self, seq: Sequence[int], requests: Sequence[EVRequest],
                       env, total_queue: float) -> float:
        """Priority-weighted emergency lateness plus background delay for a plan."""
        ctrl = env.cfg.control
        queues = env.phase_queues()
        changeover = ctrl.yellow_s + ctrl.all_red_s
        pending = list(requests)
        prev = int(env.current_phase)
        t = 0.0
        cost = 0.0

        for phase in seq:
            switch = phase != prev
            if switch:
                t += changeover
            served = [r for r in pending if phase in r.phases]
            green = ctrl.ev_min_green_s
            for r in served:
                lateness = max(0.0, (t + r.clearance_s) - r.eta_s)
                cost += self.ev_value_ratio * r.weight * lateness
                green = max(green, r.clearance_s + STARTUP_LOST_TIME_S)
            pending = [r for r in pending if r not in served]
            # Everyone not served by this phase waits through it.
            denied = float(total_queue - queues[phase])
            cost += self.background_weight * denied * (green + (changeover if switch else 0.0))
            t += green
            prev = phase
            if not pending:
                break

        # Anything the plan never serves is charged at the horizon, at double
        # weight so a plan that abandons a vehicle can never look cheapest.
        for r in pending:
            cost += (self.ev_value_ratio * r.weight
                     * max(0.0, (t + r.clearance_s) - r.eta_s) * 2.0)
        return cost

    def _plan(self, requests: Sequence[EVRequest], env) -> Tuple[int, ...]:
        """Exact minimum-cost phase ordering for the committed requests.

        The candidate set is the union of phases serving a committed approach --
        at most four -- so enumerating every ordering of every prefix length is
        at most 64 evaluations. There is no need for a heuristic at this size,
        and being exact is what lets us call the conflict resolution optimal
        rather than merely reasonable.
        """
        candidates = sorted({p for r in requests for p in r.phases})
        if not candidates:
            return ()
        total_queue = float(env.phase_queues().sum())
        best: Tuple[int, ...] = ()
        best_cost = float("inf")
        depth = min(self.plan_horizon, len(candidates))
        for k in range(1, depth + 1):
            for seq in itertools.permutations(candidates, k):
                c = self._schedule_cost(seq, requests, env, total_queue)
                if c < best_cost - 1e-9:
                    best, best_cost = seq, c
        return best

    # ------------------------------------------------------------------ decide
    def decide(self, env, mask: np.ndarray, base_action: int,
               rank_phases=None, poll_dt: Optional[float] = None) -> CorridorDecision:
        """Choose this epoch's phase, or decline to interfere.

        ``poll_dt`` is how long it will be before this method is called again --
        one decision interval at an epoch boundary, one second when polled
        inside a green by :meth:`CorridorPolicy.interrupt`. It is the resolution
        of the just-in-time deadline: a request is acted on when waiting one
        more poll would make the corridor late, so a finer poll defers action
        closer to the vehicle and costs cross traffic less.
        """
        dt = float(poll_dt if poll_dt is not None
                   else env.cfg.control.decision_interval_s)
        self.n_epochs += 1
        self.observe_service(env)

        # Token bucket refill (chance constraint on spurious preemption).
        if self._capacity > 0.0:
            self._tokens = min(self._capacity,
                               self._tokens + self._capacity * dt / 3600.0)

        requests = self._build_requests(env, base_action, poll_dt=dt)
        committed = [r for r in requests if r.committed]

        # The evidence-based service set, which is what S6's deadline runs on.
        # Computed from every request that survived screening, before any of the
        # economics are consulted, so a vehicle the risk rule declines is still
        # owed a green. See `service_request_presence`.
        floor = self.service_request_presence * self._attainable_presence()
        eligible = [r for r in requests if r.presence >= floor]
        # At most one approach is ever owed service at a time, and it is the one
        # the evidence most strongly supports. The four masts sit within a
        # couple of hundred metres of one another, so a single siren is audible
        # at all of them and the screening stage routinely returns two or three
        # approaches at once -- measured, the unrestricted set named every phase
        # in the signal on 16% of steps, which is not a request, it is the
        # cross-talk this project's joint filter exists to resolve.
        #
        # Granting a union of conflicting phases is worse than granting none:
        # the shield alternates between them at the emergency minimum green and
        # neither approach ever discharges. Ranking by weight (posterior times
        # triage priority) and taking the single best one is what the older
        # fixed-threshold overlay did for the same reason, and it is the one
        # guard from that design worth keeping.
        if eligible:
            best = max(eligible, key=lambda r: r.weight)
            service_phases = tuple(sorted({int(p) for p in best.phases}))
        else:
            service_phases = ()
        # Once a corridor is open it stays open until the vehicle is through.
        # Dropping it because the posterior dipped for one epoch would leave the
        # intersection having paid the changeover and taken none of the benefit,
        # and the vehicle then meets a red it would not have met at all.
        owing = self._target is not None and self._held_s < self._hold_target_s
        if self._target is not None:
            held = [r for r in requests
                    if r.approach == self._target and r not in committed
                    and (owing or r.presence >= self.hold_presence * r.threshold)]
            committed = committed + held

        # Chance constraint: with the budget spent, only near-certain evidence
        # may open a *new* corridor. The gate is a fraction of the largest
        # posterior this sensor can emit, not an absolute probability -- see
        # ``high_confidence``. A corridor already open and still inside the
        # clearance it was priced with is not a new engagement and is not
        # gated: the cost of abandoning a vehicle mid-crossing has already been
        # incurred, and stopping halfway spends it for nothing.
        # Affordability, not emptiness, is the test. The bucket is floored at
        # zero and refills continuously, so ``tokens <= 0`` is true for exactly
        # one poll after a spend and false ever after -- a gate that can never
        # bind. What the constraint actually says is that a corridor may be
        # opened when the bucket holds the ``1 - p`` it would cost.
        if self._capacity > 0.0 and committed and not owing:
            price = sum(max(0.0, 1.0 - r.presence)
                        for r in committed if r.approach != self._target)
            if price > self._tokens + 1e-9:
                floor = self.high_confidence * self._attainable_presence()
                kept = [r for r in committed if r.presence >= floor]
                if len(kept) < len(committed):
                    self.n_budget_blocks += 1
                committed = kept

        if not committed:
            # The request can vanish outright rather than merely weaken, and
            # the moment it is most likely to is the worst one: at close range
            # the filter starts moving its mass past the stop line, the
            # presence marginal dips and the range track can reset to its prior
            # with the vehicle still short of the line. A corridor that was
            # opened and paid for
            # is seen through its own clearance regardless -- bounded by
            # ``max_hold_s``, and still only ever among phases the shield
            # permits.
            if (owing and self._hold_phase is not None
                    and mask[int(self._hold_phase)]):
                self._held_s += dt
                self.n_preempt_epochs += 1
                return CorridorDecision(
                    True, phase=int(self._hold_phase), target=self._hold_request,
                    requests=requests, committed=[], imminent=True,
                    service_phases=service_phases,
                    reason="holding the corridor through its clearance")
            if self._target is not None:
                self._target = None
                self._held_s = 0.0
                self._hold_target_s = 0.0
                self._hold_phase = None
                self._hold_request = None
            return CorridorDecision(False, requests=requests,
                                    service_phases=service_phases,
                                    reason="no request clears its risk threshold")

        if len(committed) > 1:
            self.n_conflict_epochs += 1

        seq = self._plan(committed, env)
        legal = [p for p in seq if mask[p]]
        if not legal:
            # The shield forbids every phase the plan wants -- almost always a
            # minimum green elsewhere. Hold the target and try again next epoch
            # rather than abandoning a vehicle mid-approach.
            self._held_s += dt
            return CorridorDecision(False, requests=requests, committed=committed,
                                    schedule=seq, conflict=len(committed) > 1,
                                    service_phases=service_phases,
                                    reason="shield defers the corridor")


        phase = legal[0]
        # Among phases the plan ranks equally the learned policy still picks, so
        # the agent keeps the through/protected-left choice it is good at.
        if rank_phases is not None and len(legal) > 1:
            head_cost = self._schedule_cost(seq, committed, env,
                                            float(env.phase_queues().sum()))
            tied = [p for p in legal
                    if abs(self._schedule_cost((p,) + tuple(seq[1:]), committed, env,
                                               float(env.phase_queues().sum()))
                           - head_cost) < 1e-6]
            if len(tied) > 1:
                ordered = [p for p in rank_phases(tied) if p in tied]
                if ordered:
                    phase = ordered[0]

        target = max((r for r in committed if phase in r.phases),
                     key=lambda r: r.weight, default=committed[0])

        new_engagement = self._target != target.approach
        if new_engagement:
            self.n_engagements += 1
            self._held_s = 0.0
            # What the corridor owes this vehicle, in the same arithmetic that
            # priced it: open the phase, discharge the queue in front of the
            # vehicle, wait for it to arrive, then let it cross. Holding for
            # that long is not extra aggression -- it is the plan whose cost
            # was already charged to cross traffic when the decision was taken.
            #
            # It used to be ``open_time + EV_PASS_TIME_S``, which drops the
            # waiting term, and that is the whole of it: on an approach already
            # green with no queue the open time is zero, so the corridor owed
            # four seconds, released at four, and the shield's next epoch took
            # the phase away. Traced end to end, an ambulance closing at 17 m/s
            # lost its green 28 m short of the stop line -- one and a half
            # seconds from crossing -- and then stood at the resulting red for
            # sixty. Cross traffic had already been charged for the full hold;
            # releasing early paid the bill and left without the goods.
            #
            # Honouring it also matters because the posterior does not survive
            # the approach: at close range the filter moves its mass past the
            # stop line, calls the vehicle departed and resets the track, and
            # without a hold the green is handed away with the vehicle metres
            # short of it.
            self._hold_target_s = float(np.clip(
                max(target.hold_s, target.open_time_s + EV_PASS_TIME_S),
                0.0, self.max_hold_s))
            self.lead_times_s.append(float(target.eta_s))
            self.open_times_s.append(float(target.open_time_s))
        self._target = target.approach
        self._hold_phase = int(phase)
        self._hold_request = target
        self._held_s += dt
        self.n_preempt_epochs += 1
        # The chance constraint bounds the expected number of corridors granted
        # to vehicles that were not there, so it is charged **per engagement**:
        # one corridor that turns out to be spurious is one spurious corridor
        # whether it is held for five seconds or thirty.
        #
        # It was charged per poll instead, pro-rated by ``dt/60``, and that is
        # not a smaller version of the same constraint -- it is a different and
        # unsatisfiable one. The scheduler polls at 1 Hz, so a corridor held
        # twenty seconds cost 20 * (1 - p) / 60 tokens; with the calibrated
        # posterior bounded above by the detector's precision (0.49 on the
        # shipped map) even a certain detection charges 0.17 per engagement,
        # against a refill of two per hour. Measured over a 30-minute episode
        # the bucket went permanently negative inside the first two minutes and
        # stayed there, after which every new corridor had to clear the
        # near-certainty gate -- 0.9 * 0.49 = 0.44, above all but the single
        # topmost calibration bin. The corridor was blocked on 112 polls and
        # the emergency vehicles waited: "budget exhausted" and "evidence too
        # weak" are the same branch, so nothing said so.
        #
        # The bucket is also floored at zero. Debt is not part of the
        # constraint -- a bucket that can be overdrawn takes hours to climb
        # back to its own empty, which turns a bound on the long-run rate into
        # a one-way latch.
        if new_engagement:
            spend = sum(max(0.0, 1.0 - r.presence) for r in committed)
            self._tokens = max(0.0, self._tokens - spend)
            self.spent_tokens += spend

        return CorridorDecision(
            True, phase=int(phase), target=target, requests=requests,
            committed=committed, schedule=seq, conflict=len(committed) > 1,
            service_phases=service_phases,
            reason="risk-calibrated corridor",
            imminent=bool(target.eta_s <= self.gapout_hold_eta_s
                          or target.range_m <= self.gapout_hold_range_m
                          or target.in_junction),
        )

    # ------------------------------------------------------------------- stats
    def stats(self) -> Dict[str, float]:
        return {
            "priority_engagements": float(self.n_engagements),
            "priority_epochs": float(self.n_preempt_epochs),
            "priority_duty_cycle": self.n_preempt_epochs / max(self.n_epochs, 1),
            "conflict_epochs": float(self.n_conflict_epochs),
            "conflict_rate": self.n_conflict_epochs / max(self.n_preempt_epochs, 1),
            "budget_blocks": float(self.n_budget_blocks),
            "expected_false_preemptions": float(self.spent_tokens),
            "mean_lead_time_s": float(np.mean(self.lead_times_s))
            if self.lead_times_s else float("nan"),
            "median_lead_time_s": float(np.median(self.lead_times_s))
            if self.lead_times_s else float("nan"),
            "mean_open_time_s": float(np.mean(self.open_times_s))
            if self.open_times_s else float("nan"),
            "mean_slack_left_s": float(np.mean(
                np.asarray(self.lead_times_s) - np.asarray(self.open_times_s)))
            if self.lead_times_s else float("nan"),
        }
