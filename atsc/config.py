"""Typed configuration objects for the whole pipeline.

Everything is a plain dataclass so configs are serialisable to JSON (no external
YAML dependency) and every experiment can record the exact config that produced it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Tuple

import os

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
MODELS_DIR = REPO_ROOT / "models"
FIGURES_DIR = REPO_ROOT / "figures"

RUNS_DIR = Path(os.environ.get("ATSC_RUNS_DIR") or (RESULTS_DIR / "runs"))
"""Where per-run SUMO working files go: route files, sumocfg, tripinfo XML.

All of it is regenerable and none of it is read after the run that produced it,
but there is a lot: a full study writes well over a gigabyte. If the repository
lives inside a synced folder, the sync client reads every file back as it is
written, and measured here that alternated episode wall time between 12 s and
45 s depending on whether it had caught up. Set ``ATSC_RUNS_DIR`` to somewhere
outside the synced tree and the variance disappears. Nothing else moves --
results, figures and models stay in the repository."""


# --------------------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------------------
@dataclass
class NetworkConfig:
    """Geometry of the generated SUMO network."""

    name: str = "single"
    rows: int = 1
    cols: int = 1
    arm_length_m: float = 320.0
    """Length of each approach. Must exceed the acoustic detection radius so an
    emergency vehicle is audible before it is committed to the intersection."""
    lanes_per_approach: int = 3
    speed_limit_ms: float = 13.89  # 50 km/h
    left_turn_lane: bool = True
    right_turn_on_lane0: bool = True
    junction_type: str = "traffic_light"


# --------------------------------------------------------------------------------------
# Demand
# --------------------------------------------------------------------------------------
@dataclass
class DemandConfig:
    """Stochastic traffic demand, including emergency-vehicle injection."""

    horizon_s: int = 3600
    seed: int = 0

    regime: str = "balanced"
    """One of ``balanced``, ``asymmetric``, ``peak``, ``surge``. Controls the
    per-approach arrival intensity profile."""

    base_flow_vph: float = 700.0
    """Baseline arrivals per hour per approach before the regime multiplier."""

    turn_ratios: Tuple[float, float, float] = (0.15, 0.70, 0.15)
    """(left, through, right) routing proportions."""

    ev_rate_per_hour: float = 12.0
    """Expected emergency-vehicle arrivals per hour across the whole network."""

    ev_schedule: str = "poisson"
    """How emergency arrivals are placed in time: ``poisson`` or ``spread``.

    Poisson is the right model and is the default everywhere a number is
    reported: arrivals are memoryless, so the gaps are exponential and both
    clumps and long empty stretches are real properties of the process being
    studied.

    They are also what makes a *demonstration* unwatchable. At the same mean
    rate an exponential gap distribution puts two vehicles ten seconds apart
    and then nothing for six minutes, so a viewer sees the intersection
    saturate once and then watches ordinary traffic for the rest of the run --
    measured on the previous demo default, four of the eight vehicles arrived
    inside one ninety-second window. ``spread`` keeps the same *count* over the
    horizon and places the arrivals at even intervals with a little jitter, so
    the corridor is exercised from the beginning of the run to the end of it.

    Nothing evaluated uses it. See ``ev_spread_jitter``."""

    ev_spread_jitter: float = 0.18
    """Fraction of the nominal gap that a ``spread`` arrival may slide by.

    Enough that the arrivals are not metronomic -- and, more usefully, that
    they do not all land at the same point in the signal cycle, which would
    make every event in the run the same event."""

    ev_class_weights: Dict[str, float] = field(
        default_factory=lambda: {"ambulance": 0.55, "fire": 0.2, "police": 0.25}
    )

    ev_priority: Dict[str, float] = field(
        default_factory=lambda: {"ambulance": 1.0, "fire": 0.9, "police": 0.7}
    )
    """Triage weight in [0, 1]. Higher means a tighter delay budget."""

    car_following_sigma: float = 0.5
    depart_speed: str = "max"


# --------------------------------------------------------------------------------------
# Acoustic sensing
# --------------------------------------------------------------------------------------
@dataclass
class AcousticConfig:
    """Physical siren synthesis + microphone front-end."""

    sample_rate: int = 16000
    frame_s: float = 1.0
    """Duration of one analysis frame handed to the detector. One second rather
    than a half gives the range head enough of the siren's sweep to read the
    absorption-induced spectral tilt, and is still far shorter than the 5 s
    control decision interval."""

    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 64
    fmin: float = 50.0
    fmax: float = 7500.0

    speed_of_sound: float = 343.0
    detection_radius_m: float = 300.0
    """Beyond this range the siren is assumed inaudible above the noise floor."""

    mic_offset_m: float = 6.0
    """Distance upstream of the stop line at which each approach's microphone is
    mounted -- i.e. on the signal pole, which is where there is a pole.

    This was 70 m, on the argument that a mast at the junction would be
    roughly equidistant from every approaching vehicle and so could not
    localise. That argument is right about *range* and wrong about the array,
    because it leaves out the only cue that operates at long range: the
    cardioid's own directivity. A source 300 m up the west arm is on-axis to
    the west mast and broadside (-6 dB) to the north and south ones whatever
    the offset is, so the far-field separation is set by the pattern, not by
    the geometry.

    What the offset does control is the *near* field, and there it works
    backwards. Measured on this network as the level at the correct mast minus
    the level at the loudest wrong one, for a source at signed distance ``s``
    from the stop line:

        s (m)             300   150    60    25     8     1
        70 m offset       5.3   7.7  23.9  11.0   2.6   1.5
         6 m offset       7.0   7.5   9.6  13.1  23.2  11.9

    At 70 m the array is blind over the last twenty metres of every approach --
    1.5 dB against a level noise of several -- which is the stretch where a
    vehicle is stopped at the red the corridor exists to lift. That is exactly
    where it was failing: traced per-second over a seeded hour, an ambulance
    standing 1 m short of the west stop line put the posterior on the *north*
    arm for sixty consecutive seconds while its own approach read 0.00, and the
    corridor spent the time forcing a phase serving nobody.

    Six metres puts the worst case anywhere on the arm at 7.0 dB and the last
    twenty metres at 12-36 dB. It is also the mounting a deployment would
    actually use."""

    directivity_front_back_db: float = 14.0
    """Front-to-back ratio of the cardioid response, aimed up its own approach.
    Consistent with a compact two-element endfire array; this plus the offset is
    what separates the approaches acoustically."""

    dual_element_mast: bool = False
    """Run each mast as a *steerable* two-element endfire pair rather than as a
    single cardioid aimed upstream.

    This was introduced to rescue the 70 m mast, whose rear null swallowed
    every vehicle between it and the stop line, and it did rescue the level:
    reading the louder of the two lobes made a vehicle at the line audible
    again. What it also did, and what was not noticed, is throw away the
    pattern. ``max(front, back)`` of two opposed cardioids is 0 dB on both
    axes and -6 dB broadside, so a steered mast is very nearly omnidirectional
    -- the front-to-back ratio that separates the four approaches at long range
    collapses from 14 dB to 6, and the far-field separation with it (measured,
    5.3 dB against the single element's 11.8 at 300 m).

    With the mast at the signal pole (:attr:`mic_offset_m`) there is no rear
    null to rescue: an inbound vehicle is on-axis all the way in. The rear null
    then does useful work instead, because it is aimed at the junction, so a
    vehicle that has *crossed* goes quiet by 14 dB -- which is the
    inbound-versus-departed cue the signed state space in
    :mod:`atsc.sensing.joint_belief` is there to consume.

    Set True to recover the steered mast, which is the sensing ablation."""

    score_level: bool = True
    """Whether the joint filter's likelihood reads the measured level.

    On. Off is the ablation, and it is the state this project was in: the
    predicted level was computed for every hypothesis and used only to select
    an SNR bin, so the one observable that separates a source on the east arm
    from one on the south arm was never scored. See
    :meth:`atsc.sensing.joint_belief.JointBeliefTracker._log_likelihood`."""

    level_noise_db: float = 2.5
    """Measurement noise on the reported sound pressure level, dB (std).

    The mast reports a level, and until now it reported the propagation model's
    answer exactly. That was harmless while nothing consumed it as a
    measurement -- the corridor only ever differenced it for a trend -- and is
    not harmless now that the joint filter scores it against a predicted level,
    because a noiseless observable makes the likelihood arbitrarily sharp and
    the filter arbitrarily confident about a position it inferred from one
    number.

    Two and a half decibels is what a calibrated outdoor Class 1 measurement
    integrated over a one-second frame is worth in traffic: microphone
    tolerance and windshield insertion loss are a decibel between them, and the
    rest is the frame-to-frame variation of the source itself as the siren
    sweeps through its pattern and the vehicle's own body shadows the speaker.
    The same figure is used by the filter's likelihood, so the sensor's
    sampling distribution and the model that reads it cannot drift apart."""

    front_back_noise_db: float = 3.0
    """Measurement noise on the front-minus-back ratio, dB (std).

    Only consulted when :attr:`dual_element_mast` is set, since a single
    cardioid has one lobe and reports no ratio.

    Larger than the level noise on either lobe alone would suggest, and
    deliberately so: the two lobes come from the same two capsules, so capsule
    mismatch, wind on the windshield and the finite element spacing all appear
    in the difference rather than cancelling from it. Against a geometric
    separation of ``directivity_front_back_db`` (14 dB between a source
    upstream of the mast and one downstream of it), 3 dB still leaves the cue
    worth about 4.7 standard deviations, which is why it resolves the
    side-of-mast ambiguity that the level alone cannot."""

    reference_spl_db: float = 110.0
    """Siren SPL at 1 m, per EN 7010 / SAE J1849 (typ. 105-120 dB)."""

    ambient_noise_db_range: Tuple[float, float] = (55.0, 80.0)
    """Urban intersection ambient noise floor."""

    ambient_drift_db_per_sqrt_s: float = 1.2
    """Volatility of that floor, dB per root second.

    The floor is a reflected random walk with this increment rather than an
    independent draw per frame: its stationary law is still the uniform above,
    so anything fitted against the marginal is unaffected, but consecutive
    frames now differ by about a decibel instead of by up to twenty-five. That
    matters because the *level* is a measurement the controller uses -- a siren
    getting louder is a siren getting closer -- and a floor redrawn each frame
    buries a real 1-2 dB/s approach trend under 25 dB of noise. At 1.2 the
    floor wanders the full 25 dB range over roughly seven minutes, which is the
    timescale urban background noise actually moves on."""

    siren_types: Tuple[str, ...] = ("wail", "yelp", "hilo")
    """wail = slow sweep (ambulance/fire), yelp = fast sweep (police),
    hilo = European two-tone."""

    class_map: Dict[str, int] = field(
        default_factory=lambda: {"none": 0, "ambulance": 1, "fire": 2, "police": 3,
                                 "horn": 4}
    )
    """Detector output classes. ``horn`` is not an emergency class -- it is the
    **confuser**, and giving it its own label rather than folding it into
    ``none`` is deliberate.

    A road horn is the one urban sound genuinely confusable with a siren: it is
    loud (100-112 dB at 1 m, overlapping the siren range), harmonic, and its
    tones sit on the low end of the siren band. Trained against broadband
    traffic noise alone -- which is what this detector was, because horns did
    not exist anywhere in the model -- the network never had to learn the
    distinction, and a detector that has never seen a confuser cannot be said to
    reject it.

    Folding horns into ``none`` would train the same rejection but discard the
    evidence that the rejection is *working*: with its own class the confusion
    matrix reports horn-as-siren directly, which is the number that decides
    whether the corridor is safe to run in traffic that honks."""

    horn_rate_per_veh_hour: float = 15.0
    """Horn presses per vehicle-hour of *queued* time.

    Honking is a function of frustration, not of distance travelled, so it is
    modelled per unit of time spent stopped rather than per vehicle or per
    kilometre. At 15 per standing hour, a junction holding fifteen queued
    vehicles produces a press roughly every fifteen seconds -- a busy urban
    intersection, not a pathological one, and dense enough that the detector is
    rejecting a hard negative almost continuously while it waits for a siren
    that arrives every two minutes.

    The first value tried here was 3, which produced nine horn-seconds in an
    eleven-minute run. That is not a test of anything: a confuser that appears
    once every two minutes cannot distinguish a detector that rejects horns from
    one that has simply never met one. ``scripts/demo.py --horn-rate`` sweeps
    it, and ``0`` recovers the silent-traffic model this project used
    previously."""

    reverb_rt60_s: float = 0.8
    max_radial_speed_ms: float = 30.0
    """Normalisation constant for the Doppler regression head."""


@dataclass
class DetectorConfig:
    """CNN that maps a log-mel frame to (siren class, radial velocity)."""

    conv_channels: Tuple[int, ...] = (32, 64, 64)
    dense_units: int = 128
    dropout: float = 0.25
    learning_rate: float = 1e-3
    batch_size: int = 128
    epochs: int = 25
    train_clips: int = 12000
    val_clips: int = 2000
    test_clips: int = 4000
    velocity_loss_weight: float = 1.0
    seed: int = 0


@dataclass
class BeliefConfig:
    """Recursive Bayesian filter over emergency-vehicle presence per approach."""

    prior_presence: float = 0.05
    """Prior probability that an emergency vehicle is inbound on a given
    approach at any decision epoch. Set to the chain's own stationary value
    below, so the filter starts where it will settle instead of climbing to it."""

    survival_prob: float = 0.92
    """P(EV still *inbound* on the approach next epoch | inbound now, and still
    closing). Encodes that vehicles eventually clear the approach."""

    survival_prob_receding: float = 0.45
    """Survival probability once the tracker says the source is moving away.

    The hidden state is "an emergency vehicle is inbound on this approach", and
    a vehicle that has crossed the stop line has left that state -- but it is
    still audible, often for another twenty seconds, so the detector keeps
    reporting a siren and a single survival probability keeps the posterior
    pinned near 1. Measured against ground truth, that stale tail was 72 % of
    all approach-epochs with presence > 0.9 and nothing inbound; cross-talk was
    26 % and outright detector hallucination only 2 %.

    Direction is exactly the information that separates the two cases, and the
    Kalman range-rate already carries it: a closing source has negative range
    rate, a departing one positive. Conditioning survival on the sign is the
    minimal correct fix -- no new sensor, no new threshold, just the transition
    model being told what the state actually means."""

    birth_prob: float = 0.0045
    """P(EV appears on the approach | absent now).

    Not a free parameter: it is pinned to the deployment's actual emergency
    traffic. A two-state chain with birth b and survival s has
    stationary occupancy b / (b + 1 - s), and that number is the prior the
    filter falls back on when the microphone says nothing. The true occupancy is
    the fraction of decision epochs an approach has an inbound emergency vehicle
    inside the detection radius: at 40 EV/h over four approaches, a 300 m radius
    and 50 km/h, that is (40/4) * (300/13.9) / 3600 = 0.06, and we measure
    0.053-0.060 directly against ground truth.

    It was originally 0.02, giving a stationary prior of 0.20 -- nearly four
    times the truth. That is a large part of why the presence posterior looked
    saturated: absent an emergency vehicle, mean presence sat at 0.22 by
    construction, and only a modest amount of cross-talk evidence was needed to
    carry it past any threshold. Pinning it to the measured rate costs nothing
    in sensitivity and is the difference between a posterior that ranks well and
    one that can be read as a probability."""

    transition_reference_s: float = 5.0
    """Interval the survival and birth probabilities above are quoted for.

    ``survival_prob`` and ``birth_prob`` are per-*call* transition
    probabilities, and until the sensing clock was decoupled from the decision
    epoch there was only ever one call per 5 s epoch, so the distinction did
    not arise. It does now: the detector emits a frame every
    ``AcousticConfig.frame_s`` (one second), the environment consumes every one
    of them, and a per-call probability applied five times as often is a
    different Markov chain -- same stationary occupancy, but a presence that
    decays five times faster in wall-clock time, which drops a corridor through
    any two-frame detection dropout.

    Scaling to ``s(dt) = s ** (dt / reference)`` and
    ``b(dt) = 1 - (1 - b) ** (dt / reference)`` makes the chain a function of
    elapsed time rather than of how often it is asked, so the filter behaves
    identically however finely it is sampled. At ``dt == reference`` the
    arithmetic is the identity, so every number fitted under the old
    once-per-epoch schedule still holds."""

    evidence_per_reference_interval: float = 1.0
    """How many independent detector looks one ``transition_reference_s`` is worth.

    The presence filter multiplies likelihood ratios, which is correct for
    conditionally independent measurements and wrong for a stream sampled
    faster than the scene changes. Reading the microphones once per second
    instead of once per five-second epoch does not deliver five times the
    information: the source has barely moved, the noise floor has barely
    drifted, and -- decisively -- the likelihood is a *fitted* model, so any
    misspecification in it compounds with every extra look rather than
    averaging out.

    Measured, that is not a subtlety. Consuming five frames per epoch raised
    the joint filter's false detections from 122/h to 1020/h at a fixed raw
    threshold and cut precision from 0.50 to 0.43, and in closed loop it took
    hard preemption -- a baseline containing none of this project's other
    machinery, and which reads the raw posterior against a fixed 0.6 -- from
    10.2 s of emergency delay and no gridlock to 27.2 s with a seed jammed.

    Each update's log-likelihood is therefore scaled by
    ``evidence_per_reference_interval * dt / transition_reference_s``, so a
    reference interval carries the evidence of one look however finely it is
    sampled. What the fast clock still buys is everything that is *not*
    evidence accumulation: the range track, the arrival estimate and the
    measured level are all refreshed every second, which is what lets a
    corridor be priced against a vehicle's real closeness rather than against a
    five-second-old snapshot of it.

    Raise it above 1.0 to trust the extra frames as genuinely independent.
    """

    eta_process_noise_s: float = 1.5
    eta_measurement_noise_s: float = 4.0
    max_eta_s: float = 45.0
    commit_threshold: float = 0.6
    """Belief above which the controller treats the approach as EV-bearing."""


# --------------------------------------------------------------------------------------
# Environment / control
# --------------------------------------------------------------------------------------
@dataclass
class ControlConfig:
    """Signal timing envelope enforced by the safety shield."""

    min_green_s: float = 8.0
    max_green_s: float = 55.0
    yellow_s: float = 3.0
    all_red_s: float = 2.0
    decision_interval_s: float = 5.0
    """Green extension granularity: how long a 'keep current phase' action lasts."""

    max_starvation_s: float = 180.0
    """Guaranteed worst-case time any phase may be denied green (``W_max`` in the
    shield's S3 proof). 180 s matches the maximum cycle practitioners design to.
    It must exceed ``n_phases * (min_green + decision_interval + yellow +
    all_red)``, which is 72 s for the four-phase intersection used here."""

    ev_service_deadline_s: float = 16.0
    """S6: guaranteed worst-case latency from a standing emergency request to
    green, when S3 does not intervene (``W_ev`` in the shield's S6 proof).

    Must be at least ``ev_min_green_s + ev_poll_interval_s + yellow + all_red``,
    which is 14 s on these timings; the shield refuses a smaller value rather
    than promising what the change interval forbids. Set to ``inf`` to disable
    S6 entirely, which is the ablation reported in ``scripts/ablation_ev.py``."""

    ev_poll_interval_s: float = 1.0
    """How often a standing emergency request is re-checked. The corridor is
    polled once per simulated second inside a green, not once per decision
    epoch, and S6's deadline is tightened accordingly."""

    ev_request_grace_s: float = 6.0
    """How long a standing emergency request survives after the evidence for it
    last cleared its floor, seconds.

    Without it, S6's clock restarts on every flicker of the posterior -- and the
    posterior does flicker, hardest at close range, where the filter routinely
    moves its mass past the stop line for an epoch and back again. The deadline
    would then never be reached in exactly the situation it exists to bound.

    Sized as slightly more than one changeover (5 s), so a request survives a
    dropout that spans a phase transition, and well short of the deadline
    itself, so a request that has genuinely cleared is released long before it
    could force a spurious service."""

    gap_out_queue: float = 0.0
    """Served queue at or below which a phase counts as cleared (gap-out)."""
    gap_out_min_waiting: float = 1.0
    """Vehicles another phase must have waiting before a gap-out is forced."""
    gap_out_max_load: float = 30.0
    """Total halting vehicles above which gap-out is suspended; beyond this the
    intersection is saturated and changeover lost time dominates."""
    congestion_queue: float = 0.0
    """Non-zero enables scaling the green ceiling with load. Disabled by
    default: measured, it made a demand surge markedly worse."""
    congested_max_green_s: float = 20.0
    """Green ceiling at ``congestion_queue``, when that scaling is enabled."""

    gridlock_queue: float = float("inf")
    """Total halting vehicles above which the intersection counts as jammed.

    Infinite by default, i.e. S5 recovery is **off**. We built it to rescue the
    rare seed that gridlocks and measured that it does not: on the seed where
    max-pressure jams, recovery engaged 14 times and left the run slightly worse
    (454 s against 412 s without it). By the time queues have spilled back this
    far the jam is not a scheduling artefact that round-robin can unwind. Set a
    finite value to re-enable; the mechanism is implemented and tested."""
    gridlock_epochs: int = 6
    """Consecutive jammed epochs before the shield takes over with round-robin."""
    gridlock_clear_queue: float = 45.0
    """Recovery holds until total queue falls below this (hysteresis)."""
    gridlock_min_discharge: float = 1.0
    """Vehicles cleared per epoch below which the intersection counts as
    stalled rather than merely busy -- what separates gridlock from saturation."""

    ev_min_green_s: float = 6.0
    """Minimum green honoured while an emergency preemption call is active.

    NEMA practice allows a preemption call to truncate a green below its normal
    minimum, and the shield implements that: it takes this value as
    ``emergency_min_green_s`` and applies it to S1 only, never to the change
    interval, so the clearance yellow and all-red are served in full.

    This was 8 s -- equal to ``min_green_s``, i.e. no dispensation at all -- on
    a measurement that said the dispensation cost more than it bought
    (emergency stopped delay 16.2 s against 17.3 s at 6 s, background 24.0
    against 25.8). That measurement is not wrong, but it was taken while the
    mast array could not localise a vehicle near the stop line, so a large
    share of the greens being cut short were cut for a phantom on the wrong
    arm. The reasoning recorded at the time says as much -- "every cut costs
    yellow plus all-red of pure lost time, *including on the cuts triggered by
    a detection that turns out to be wrong*" -- and once the detections stopped
    turning out wrong the arithmetic reversed.

    Re-measured over eight seeds of half an hour each with the corrected
    geometry, 8 s against 6 s:

        emergency stopped delay   15.3 s -> 7.5 s   (mean over seeds)
        background mean queue     13.9   -> 11.4    veh
        seeds that gridlocked      1/8   -> 0/8

    The gridlock is the striking part and it is not an artefact of the mean: on
    that seed the 8 s floor produced 22 teleports and a 552 s emergency
    vehicle, while the same seed at 6 s finished with none and 5.5 s. Both
    numbers improve, which is the signature of removing an obstruction rather
    than of trading one cost against another. Set it equal to ``min_green_s``
    to recover the no-dispensation behaviour; the mechanism and its test are
    unchanged either way."""


@dataclass
class RewardConfig:
    """Reward and constraint-cost shaping."""

    w_queue: float = 0.25
    w_wait: float = 0.30
    w_throughput: float = 0.20
    w_switch: float = 0.10
    queue_norm: float = 40.0
    wait_norm: float = 200.0
    throughput_norm: float = 20.0

    # Constraint signal (CMDP cost)
    ev_delay_norm: float = 20.0
    ev_stop_cost: float = 1.0
    """Extra cost charged whenever a prioritised vehicle comes to a full stop."""

    w_ev_unserved: float = 1.0
    """Weight on the *anticipatory* cost: an emergency vehicle is inbound and the
    phase serving its approach is not green.

    This term exists because the realised-delay terms above are unusable as a
    learning signal on their own. They only become non-zero once the vehicle has
    already stopped, which is several decision epochs after the action that
    caused it -- beyond the n-step return, and swamped by a reward signal that
    is dense where this one is sparse. Measured directly: with realised delay
    alone, driving the multiplier to 20-30 did not reduce emergency delay below
    an unconstrained agent's at any budget tried; it only inflated queues,
    because the cost critic could not attribute delay to actions and the policy
    was following noise. The anticipatory term is non-zero exactly while the
    controller can still do something about it, which is what makes the
    constraint learnable."""

    ev_unserved_norm: float = 6.0
    """Normaliser for the anticipatory term, in EV-seconds of unserved approach
    per decision epoch."""

    ev_imminence_range_m: float = 220.0
    """Range within which an approaching emergency vehicle starts to count as
    imminent. Cost weight ramps linearly from 0 at this range to 1 at the stop
    line."""


@dataclass
class EnvConfig:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    demand: DemandConfig = field(default_factory=DemandConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    belief: BeliefConfig = field(default_factory=BeliefConfig)
    acoustic: AcousticConfig = field(default_factory=AcousticConfig)

    step_length_s: float = 1.0
    gui: bool = False
    gui_delay_ms: int = 0
    """Wall-clock milliseconds one simulation step is allowed to take in GUI mode.

    Playback speed is ``step_length_s / (gui_delay_ms / 1000)``, so with a 0.05 s
    step ``100`` is half real time and ``25`` is twice real time. Zero means "as
    fast as possible", which is right for batch runs and unwatchable in a demo.

    The wait is taken by the client (``SumoSignalEnv._pace``), not by SUMO: the
    ``--delay`` SUMO is launched with governs its own play button and is never
    consulted while TraCI is stepping the simulation."""

    bluelight: bool = False
    """Give emergency vehicles SUMO's ``bluelight`` device, so ordinary traffic
    pulls aside for them.

    Off by default, for two independent reasons. With it on, part of the
    emergency-delay reduction comes from drivers yielding rather than from
    signal control, which confounds the quantity the study is about. And the
    device is not safe to leave on unattended: making room for the vehicle can
    deadlock a saturated junction. Measured over six runs it cut emergency
    delay roughly in half on five of them and gridlocked the sixth -- 24 s of
    background delay to 213 s, with a 95-vehicle queue that never cleared."""
    gui_quit_on_end: bool = True
    """Close the GUI when the episode finishes. Turn off to inspect the final
    state."""
    sensing_interval_s: float = 1.0
    """How often the microphones are read and the presence filter advanced.

    This is a property of the *sensor*, not of the controller, and the two were
    conflated: the belief used to be updated once per 5 s decision epoch, so
    four of every five detector frames were discarded and the posterior a
    corridor was priced against could be five seconds stale. An emergency
    vehicle closing at 14 m/s covers 70 m in that time, which is most of the
    approach -- so the loudness and closeness that are supposed to drive the
    decision were, within an epoch, constant.

    One second is the detector's own frame length (``AcousticConfig.frame_s``),
    i.e. every frame the front end produces is now consumed. The filter's
    transition model is scaled to the elapsed time
    (``BeliefConfig.transition_reference_s``) so the dynamics are unchanged;
    what changes is that five independent measurements per epoch are used
    instead of one. Set equal to ``ControlConfig.decision_interval_s`` to
    restore the previous once-per-epoch behaviour."""

    sensing_mode: str = "surrogate"
    """``surrogate``  - sample detections from the detector's fitted error model (fast,
                        used for training);
    ``end_to_end``    - synthesise audio and run the CNN every decision epoch (slow,
                        used to validate that the surrogate is faithful);
    ``oracle``        - perfect V2I detection (upper bound / ablation);
    ``none``          - no emergency-vehicle awareness at all (ablation)."""

    belief_mode: str = "joint"
    """``joint``       - one exclusive multi-Bernoulli filter over which approach
                         carries the siren, updated from all four microphones at
                         once (default; see ``atsc/sensing/joint_belief.py``);
    ``independent``   - four per-approach two-state HMMs with the geometric
                         association test in front of them. This is the previous
                         design and is retained as the ablation that measures
                         what modelling cross-talk jointly is worth."""

    use_association: bool = True
    """Run the cross-microphone geometric association test before the presence
    filter. Off is the ablation that measures what treating cross-talk as noise
    rather than as a consistency constraint costs; see
    ``atsc/sensing/association.py``."""

    collect_emissions: bool = True
    teleport_s: int = 300
    seed: int = 0


# --------------------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------------------
@dataclass
class AgentConfig:
    """Dueling Double DQN with prioritised n-step replay and a Lagrangian dual."""

    hidden_units: Tuple[int, ...] = (256, 256)
    learning_rate: float = 3e-4
    gamma: float = 0.95
    batch_size: int = 64
    buffer_size: int = 60000
    n_step: int = 3
    target_sync_steps: int = 500
    train_every: int = 4
    warmup_transitions: int = 1500
    grad_clip_norm: float = 10.0

    epsilon_start: float = 1.0
    epsilon_final: float = 0.02
    epsilon_decay_steps: int = 20000

    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_final: float = 1.0
    per_eps: float = 1e-3

    # --- Constrained MDP ---
    constrained: bool = True
    cost_budget: float = 0.35
    """Per-decision-epoch discounted emergency-vehicle delay budget d0. The dual
    variable rises whenever the running estimate of J_c exceeds this."""
    lambda_init: float = 1.0
    lambda_lr: float = 0.01
    """Dual step size, applied to the *relative* constraint violation
    ``(J_c - d0)/d0``, so the dual dynamics do not change when the budget does."""
    lambda_max: float = 30.0
    lambda_warmup_steps: int = 6000
    """Environment steps before dual ascent begins. At a tight budget the early
    policy violates by a wide margin, so ascent from step zero drives lambda to
    its ceiling while the cost critic is still noise -- the policy then chases a
    meaningless gradient and never recovers. Holding lambda fixed until the
    critics are usable avoids this; observed directly, a run at d0=0.030
    converged to a *lower* cost (0.012) than a run started at d0=0.012, which
    diverged."""
    cost_gamma: float = 0.95
    cost_ema: float = 0.99

    seed: int = 0


@dataclass
class TrainConfig:
    episodes: int = 120
    eval_every: int = 20
    eval_episodes: int = 3
    save_dir: str = "models"
    run_name: str = "siren_cmdp"
    log_every: int = 1


# --------------------------------------------------------------------------------------
# (de)serialisation helpers
# --------------------------------------------------------------------------------------
def to_json(cfg, path: Path) -> None:
    """Record the exact configuration that produced a run, beside its results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")


def ensure_dirs() -> None:
    for d in (DATA_DIR, RESULTS_DIR, MODELS_DIR, FIGURES_DIR, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)
