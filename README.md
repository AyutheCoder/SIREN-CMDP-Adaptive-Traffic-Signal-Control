# SIREN-ARC / SIREN-CMDP

**Acoustically-informed, risk-constrained adaptive traffic signal control.**

An intersection controller that *hears* emergency vehicles coming, works out
which arm each one is on, opens a corridor in time to clear it — and does none
of that at the cost of wrecking the intersection for everyone else, with a
provable bound on how long any movement can be made to wait.

Two configurations ship, and they are two points on one frontier:

| | policy trained for | priority comes from | balance |
|---|---|---|---|
| **SIREN-ARC** | efficiency alone | the corridor overlay | lower background delay |
| **SIREN-CMDP** | efficiency *subject to* an emergency-delay budget | the constraint **and** the corridor | lower emergency delay |

```bash
python scripts/demo.py                          # watch it run, with the sirens audible
python scripts/demo.py --controller siren_cmdp  # the constrained variant
python scripts/run_all.py                       # reproduce every number here
```

## Paper

**SIREN-ARC: Hearing the Ambulance — Traffic Signals that Listen for Sirens and
Clear the Way, with Guaranteed Waiting Limits.** Ayushi Shukla, Anshuman Ravi and
Dinesh Dash, National Institute of Technology Patna. Accepted at ICISS 2027
(Springer LNNS). A link to the published paper will be added here once it is
available.

---

## What this is

Reinforcement learning for traffic signals is a crowded field. Eight things here
are not standard practice, and together they are the contribution.

> **Revision note.** Points **6**, **7** and **8** exist because the controller
> was instrumented against simulator ground truth rather than against its own
> summary statistics — which is how every defect below survived a green test
> suite. Per second, for every emergency vehicle, the log records where it
> actually was, what the filter believed, and what the signal did.
>
> The first round found three things: the microphone array was **deaf over the
> last 70 m of every approach**, which is exactly where a vehicle queues at a
> red; nothing anywhere **guaranteed** that a detected emergency vehicle would
> ever be served — every layer above the shield was free to decline; and the
> detector had **never been shown a car horn**, the one urban sound genuinely
> confusable with a siren. Eleven of twelve emergency vehicles came to a full
> stop and the mean stopped delay was 18.4 s.
>
> The second round found that the first round's sensing fix had not worked, for
> a reason it could not have detected: the mast had been made *audible* at the
> stop line without being made *able to localise* there, and the filter was not
> reading the level anyway. Twelve of twelve emergency vehicles still stopped;
> one stood at a red for 61 s while the corridor held a green for a phantom on
> a different arm. Four defects, each measured (§6, and the docstrings in
> `atsc/config.py`, `atsc/sensing/joint_belief.py` and
> `atsc/control/corridor.py`):
>
> 1. **The array could not separate the arms near the stop line** — 1.5 dB at
>    1 m, against several dB of measurement noise.
> 2. **The joint filter never scored the measured level**, so the one cue that
>    was still working was discarded.
> 3. **The corridor released its own green early**, holding for the changeover
>    it had priced instead of the wait it had *charged cross traffic for* — a
>    vehicle lost its green 28 m short of the line and then waited 60 s.
> 4. **The false-preemption budget was unsatisfiable**, so the chance
>    constraint silently became an off switch two minutes into every episode.
>
> Mean emergency stopped delay over eight seeds: **15.5 s → 7.5 s**, with
> background queue also down (12.1 → 11.4 veh) and one gridlocked seed removed.
> Both sides improving is the signature of removing an obstruction rather than
> of trading one delay against another.
>
> The four are **not** equal contributors, and §6 reports the split rather than
> the sum. Ablated on its own, **scoring the level is worth −8.4 s of emergency
> stopped delay (s.e. 2.76, better on 8 seeds out of 8)**; the mast siting is
> worth −0.29 s (s.e. 1.49, i.e. nothing measurable) and −1.33 veh of background
> queue. The previous revision reached for the geometry as the explanation and
> stopped there; this one measures each piece separately, because that is
> exactly how the mistake was made. See
> [`docs/RESULTS.md`](docs/RESULTS.md) for the measured effect of each fix and
> [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) for how the result sits against
> twenty-one papers.

### 1. The emergency vehicle is *heard*, not teleported into the state vector

Almost all emergency-preemption work assumes the controller is handed the
vehicle's exact position by V2I or GPS. That assumption does a lot of quiet
work: it removes false positives, missed detections, unknown range and unknown
intent from the problem, and it requires equipment on every vehicle and in every
cabinet.

Here the only emergency-vehicle sensor is a **calibrated microphone on each
approach**. A CNN reads a one-second log-mel frame and returns a siren class
posterior and a range estimate; everything the sensing stack cannot resolve
stays unresolved — most importantly, **a microphone cannot hear which way the
vehicle intends to turn**, so the controller must commit to serving a whole
approach on partial information.

Three findings from building it are reported honestly rather than hidden:

- **Single-frame Doppler regression does not work.** A 6 % frequency shift at
  road speeds is well below one mel band, and the siren's own sweep moves the
  fundamental far more than that. The velocity head reaches only ~12 m/s error
  on a ±30 m/s range. Range rate is instead recovered by *tracking* range across
  decision epochs, which works because the unknown directivity offset is roughly
  constant along an approach and therefore cancels out of the trend.
- **Range is estimable but only to tens of metres, and the limit is the
  microphone's own directivity.** Level alone cannot be inverted for range — a
  near source off-axis and a far source on-axis give the same reading.
  Frequency-dependent atmospheric absorption breaks the tie via spectral tilt,
  which is why absorption has to be modelled per harmonic rather than as a flat
  coefficient (doing the latter first cost us 13 points of range accuracy).
- **Mel filterbanks are usually built in a way that destroys absolute level.**
  Slaney (unit-area) normalisation shifts the output ~15 dB off true SPL. That
  is harmless for speech but fatal when level *is* the measurement, so the
  filterbank here is power-preserving and validated against known tones.

### 2. Cross-talk between the microphones is the localisation cue, not the noise

Four masts sit within a couple of hundred metres of one another, so a siren on
the eastbound arm is plainly audible at all four. Each detector correctly answers
the question it was asked — *is a siren audible here?* — with **yes**. Run four
independent per-approach filters on those answers and all four approaches
conclude the vehicle is on **them**.

That is not a tuning problem, it is structural, and it is large. Measured
against ground truth over six hours: four independent filters light **2.5 arms**
when exactly one is occupied, at a precision of **0.27**.

`atsc/sensing/joint_belief.py` replaces them with **one filter over the joint
state** — which arm holds the siren, and where on that arm — truncated at two
simultaneous sources. Two things make it work:

- **The likelihood is the detector's own measured detection curve.** For each
  hypothesised source position, geometry plus the propagation model give the
  level and range every mast should be reporting; the CNN's fitted
  logit-vs-SNR distribution then says how likely the readings actually observed
  are. A mast reporting a range that no hypothesised source explains is scored
  against a uniform clutter density, in the standard multi-target way. Nothing
  is fitted separately — the filter inherits whatever reliability the detector
  turned out to have.
- **The state space extends *past* the stop line.** A vehicle that has crossed
  stays audible for another twenty seconds and, having cleared the junction,
  drives straight up the opposite arm's axis past that mast, on-axis, at a range
  indistinguishable from an inbound vehicle. With nowhere in the state space for
  that vehicle to be, the only available explanation is an inbound one — so the
  filter invents an emergency vehicle on an arm it has just cleared. Truncating
  the downstream extent at 70 m instead of 300 m cost nine points of precision.

Inference is **exact** over the 2731 hypotheses — no sampling, no gating
heuristic — and costs about a millisecond against a five-second decision epoch.
The predict step is factorised, the update is joint, and the posterior is
projected back by marginalising: assumed-density filtering, the same
approximation that makes multi-Bernoulli filters tractable.

The geometric association heuristic this replaces is retained as the ablation.
It had the right idea and the wrong form: greedy (one host at a time, never
revisited), scoring consistency only in range and never in level, and producing
a temper weight rather than a probability — so it could not say *how likely* it
is that the east arm is the one.

### 3. The sensor runs on its own clock, not the controller's

The detector produces a frame every second. Until recently the environment
consumed one of every five of them, because the belief filter was advanced once
per 5 s decision epoch -- so four fifths of the evidence was discarded, and
within an epoch the posterior, the range and the level were *constant* while
the vehicle covered up to seventy metres.

That made the corridor's 1 Hz re-check a re-check of nothing. It re-priced the
queue, which does change, against a posterior that could not. The two clocks
are now separate: `EnvConfig.sensing_interval_s` governs the microphones and
`ControlConfig.decision_interval_s` governs the controller, and sensing runs
*through the changeover* as well, which is another five seconds the filter used
to spend blind while the vehicle kept coming.

Decoupling them required the filter's transition model to be a function of
elapsed time rather than of how often it is asked. `survival_prob` and
`birth_prob` are per-epoch probabilities; applying them five times as often is
the same chain run five times as fast -- same stationary occupancy, but a
presence that decays five times quicker in wall-clock time, which drops a
corridor through any two-frame dropout. They are now scaled to the interval
(`BeliefConfig.transition_reference_s`), which is the identity at the old rate.

**And more looks are not more evidence.** This is the part we got wrong first,
and it is worth stating plainly because the failure was invisible in the
sensing metrics we were watching. The filter multiplies likelihood ratios,
which is correct for conditionally independent measurements and wrong for a
stream sampled faster than the scene changes: between one frame and the next
the source has barely moved and the noise floor has barely drifted, and --
decisively -- the likelihood is a *fitted* model, so its misspecification
compounds with every extra look instead of averaging out. More data through a
slightly wrong likelihood buys more confident wrong answers.

Measured, at a fixed raw threshold, consuming five frames per epoch drove the
joint filter's precision from 0.50 down to 0.43 and its AUC from 0.910 to
0.892 — the filter got *worse* at the thing more data is supposed to improve.
We judged that acceptable on the grounds that nothing downstream reads the raw
score: the corridor consumes the calibrated posterior against a derived
threshold, and held-out calibration error was 0.010. The reasoning was sound
and the conclusion was wrong, because one consumer does read the raw score —
the hard-preemption **baseline**, at a fixed 0.6. In closed loop, over twelve
seeds, it went to 27.2 s of emergency delay with a seed gridlocked; tempered,
on the identical seeds, it is 14.4 s with none. A controller containing none of
this project's machinery, which we had no reason to re-examine, is what caught
a defect in our sensing stack.

Tempering did not merely undo the damage. The joint filter now reaches AUC
**0.925** and 104 false detections an hour, against 0.910 and 122/h for the
original once-per-epoch schedule — better than either of the alternatives,
because the fast clock really does carry more information about *position*
once it stops double-counting the same evidence about *presence*.

Each update's likelihood is therefore raised to
`evidence_per_reference_interval * dt / transition_reference_s`, so a reference
interval carries the evidence of one look however finely it is sampled -- `L**w`
being the evidence of `w` independent looks. What the fast clock still buys is
everything that is *not* evidence accumulation: range, arrival time and level
are refreshed every second, which is what lets a corridor be priced against a
vehicle's real closeness instead of a five-second-old snapshot of it.

On identical seeds, that distinction is the whole result:

| sensing rate | background delay | EV stopped | EV stops | clean pass |
|---|---|---|---|---|
| 5 s (one look per decision) | 23.29 s | 16.72 s | 1.13 | 0.204 |
| **1 s, tempered** | 23.63 s | **14.47 s** | **1.01** | **0.231** |

Thirteen per cent of emergency stopped delay for a third of a second of
background delay. The same comparison run on hard preemption goes the other
way -- 12.6 s to 15.5 s -- because that controller reads a saturating presence
flag against a fixed threshold and never looks at range or level at all. Which
is the argument for deriving thresholds from state rather than fixing them,
made from the opposite direction: a fixed threshold is not merely suboptimal,
it is *sensitive to the sampling rate of the sensor underneath it*.

**Loudness is the part of the evidence that keeps working all the way in.** The
posterior saturates hundreds of metres out and then, at close range, *falls* --
the mast sits 70 m upstream aimed away from the junction, so the last stretch
of every approach is in its own microphone's rear null, and the filter starts
moving its mass past the stop line while the siren is at its loudest. The level
has no such shape: it rises monotonically, about 6 dB per halving of distance,
right up to the stop line. So the reading now carries the measured level and
its floor (`SensorReading.spl_db`, `.snr_db`), the corridor differences it over
a few seconds as a closing test that survives a vehicle stopped in a queue, and
the demo can play the whole thing out loud (`--sound`).

Making the level usable also meant fixing the noise floor, which was redrawn
independently over its full 25 dB range at every frame. An approaching siren
gains one or two decibels a second; under 25 dB of per-frame noise that trend
is invisible. It is now a random walk reflected in the same interval, so the
marginal the detector's error model was fitted against is unchanged and
consecutive frames differ by about a decibel.

**How loudness becomes a green light.** The chain is worth stating end to end,
because it is the whole system in one line and no step of it is a heuristic:

```
siren gets louder  ->  every hypothesised source position predicts a level at
                       every mast, and the detector's own fitted logit-vs-SNR
                       curve says how likely the observed readings are under
                       each                              (joint_belief.py)
                   ->  posterior shifts onto nearer positions on one arm:
                       presence up, range down, arrival time down
                   ->  the projected red that vehicle meets at its arrival, and
                       the queue standing in front of it, are priced in seconds
                       of delay avoided                       (corridor.py)
                   ->  threshold  theta* = C / (gamma B)  falls as B grows
                   ->  posterior crosses it, and the corridor opens at the
                       latest instant it can still clear the queue in time
                   ->  the shield vets the phase; cross traffic is still
                       guaranteed service within W_max          (shield.py)
```

Loudness is not a side channel bolted on beside the detector: it is the
measurement the whole inference is built from, and the level is where closeness
enters. What changed is that the level is now sampled often enough to move
inside a decision, reported alongside the reading so the controller can also
use its trend directly, and audible.

### 4. Preemption is a deadline problem, not a threshold test

Two things break every emergency-preemption scheme we could find, and both were
present in earlier versions of this controller.

**The commitment threshold is a free parameter.** Detection is probabilistic, so
something has to turn a posterior into a yes or no, and published systems pick a
number and tune it. That number cannot be right, because the right answer
depends on the state: a posterior of 0.3 is worth acting on when the cross
street is empty and the vehicle faces forty seconds of red, and is not worth
acting on when forty cars are queued on the conflicting approach.

`atsc/control/corridor.py` derives it instead. Opening a corridor buys `p·B`
seconds of prioritised delay and costs `C` seconds of background delay, paid
whether or not the vehicle is real, so the Bayes rule is `p·γ·B > C` —
equivalently `p > θ* = C / (γB)`. The threshold is recomputed every epoch from
the queue state, and `γ` — background vehicle-seconds per priority-weighted
emergency vehicle-second — is the single normative number, deliberately exposed
rather than buried.

We also tried scaling `γ` by the CMDP's converged dual `λ`, on the reasoning
that both express willingness to pay for emergency delay. They do, but not in
the same units, and it makes the controller measurably worse — see the ablation
row in [Results](#results). `λ` governs the policy, `γ` governs the overlay,
and neither reads the other.

**The delay avoided must be evaluated at the vehicle's arrival time.** Our first
version wrote it as `residual_red_now + clearance − eta`. That is wrong in a way
that quietly destroys anticipation: it assumes that once the light goes green it
*stays* green until the vehicle arrives. A vehicle forty seconds out facing
twenty seconds of red scored as losing nothing — when in truth a fifteen-second
green inside a forty-five-second cycle comes and goes and the vehicle arrives at
red anyway. Under that model the corridor could not commit until the vehicle was
seconds from the stop line, and measured, it opened corridors at a median ETA of
about four seconds. There is nothing left to anticipate at four seconds.

The fix is to project the signal timeline forward — each phase carries a
measured mean inter-service interval and mean green duration — and ask what red
the vehicle meets *when it gets there*.

**The projection must not assume the phase is about to be served.** The
benefit of a corridor is the red the vehicle meets at its arrival, and that
comes from a renewal projection: each phase is served every `m` seconds for `g`
seconds, both measured. The projection read `first_green = max(0, m - waited)`,
which says that a phase that has already waited longer than its own mean
inter-service interval goes green *now*. Every approach an emergency vehicle is
queued on is in exactly that state -- being unserved is why there is a queue --
so the projected red collapsed to zero, the delay avoided collapsed to zero,
and the derived threshold became `C / 0`. **No posterior can clear an infinite
threshold.**

Measured on a seeded hour with an emergency vehicle genuinely inbound: of 66
decision epochs, 30 priced the corridor at an infinite threshold and 25 more
were screened out before pricing. The corridor could commit on **four epochs in
sixty-six**, and the approach carrying the vehicle was green half the time,
which is chance for a four-phase signal.

Three things were wrong and all three are structural rather than tuning:

- The earliest green on a phase that is not running is bounded below by the
  residual minimum green plus the full change interval. It is never zero.
- An overdue phase has falsified the renewal model, so the projection falls
  back to the steady-state residual `(1 - g/m)·m/2` -- the same estimator the
  method already used beyond its horizon, for the same reason.
- Queue discharge was charged only when the vehicle met a red. A vehicle
  arriving at a green with fifteen cars stopped in front of it still waits for
  them to move.

**A vehicle inside the junction has not gone away.** The joint filter's range is
signed about the stop line, and the screening test was `range > 0` -- so the
request vanished at the instant the vehicle entered the intersection, which is
precisely when the green has to be *held*. There is now a one-junction
tolerance, and a request inside it reports `eta = 0`: there is no arrival left
to predict, only a crossing to finish.

**A corridor that has been opened is seen through.** Hysteresis on the
posterior is not enough, because the posterior does not merely dip -- at close
range it collapses outright and the range track resets to its prior. Without a
hold the green is handed to a conflicting approach with the vehicle metres short
of the stop line, which costs more than never having preempted. The corridor
therefore holds for the green it was *already priced with*, capped at
`max_hold_s` and, as always, only ever among actions the shield permits.

**Priced with** is load-bearing, and it was not what the code did. The cost
charged to cross traffic used `hold = max(eta, open_time) + pass_time`; the hold
actually taken was `open_time + pass_time`, which drops the term for *waiting
for the vehicle to arrive*. On an approach already green with an empty queue
that is four seconds. Traced end to end: an ambulance closing at 17 m/s lost
its green 28 m short of the line — a second and a half from crossing — and then
stood at the resulting red for sixty. Cross traffic had been charged for the
full hold either way, so the corridor was paying the bill and leaving without
the goods.

**The chance constraint must be satisfiable.** With the budget spent, only
near-certain evidence could open a corridor: `p >= 0.90`. Calibration bounds the
attainable posterior by the detector's precision, and the fitted map tops out at
0.63, so that gate was unreachable *by construction* -- the first time the token
bucket ran dry the corridor stopped opening for anything, permanently, until the
bucket refilled at two per hour. Nothing in the logs said so, because "budget
exhausted" and "evidence too weak" are the same branch. The gate is now a
fraction of the attainable maximum, which is what it was always meant to mean.

Two further things about that bucket were wrong in the same direction, and both
made the constraint an off switch rather than a constraint. It was charged **per
poll**, pro-rated by `dt/60`, where the constraint it implements is specified
per *engagement* -- one corridor that turns out to be spurious is one spurious
corridor whether it is held for five seconds or thirty -- so a corridor held
twenty seconds cost 0.17 tokens against a refill of two an hour, and the bucket
went permanently negative (with no floor, so it could not climb back to its own
empty) two minutes into every episode. And the budget itself was below what the
sensor can achieve: with the calibrated posterior bounded above by the
detector's precision, opening a corridor for every emergency vehicle that is
genuinely there, at 24/h, spends about **eleven** expected spurious corridors an
hour *while being right every time*. A budget of two is not a bound on spurious
preemption at that point; it is a cap on service written in the units of
spurious preemption. It is now ten, the bucket is charged once per engagement
and floored at zero, and `expected_false_preemptions` is reported after every
run so that whether the budget is binding is visible rather than assumed.

**And then the corridor should open as late as possible.** Recovering the
benefit answers *whether*; it does not answer *when*, and the two are not the
same. `B` does not improve for acting early — once the queue is cleared and the
light is green on arrival, the delay is already zero. But the corridor must be
*held* from the moment it opens until the vehicle is through, so the cost grows
linearly with how early it starts. Acting Δ seconds early buys nothing and costs
Δ seconds of denied cross traffic.

So the corridor opens at the **latest instant at which the queue can still be
cleared in time**:

```
slack = eta − (changeover + clearance) − shield_latency
act when slack ≤ (time until the next poll) + margin
```

The `margin` is the arrival-time estimate's own standard deviation, bounded by
one changeover. Acting at `slack ≤ poll` exactly is acting at the *median* of a
noisy estimate, which is late half the time — measured over twelve seeds, the
slack actually left on the table averaged **−1.7 s**, i.e. the corridor was
typically opening after the moment it needed to. Bounding the margin at a
changeover matters as much as having one: past that the estimate is not
uncertain, it is uninformative, and insuring against it would mean opening every
corridor on sight, which is the rule the just-in-time argument replaced.

Every term is measured or a property of the timing envelope. There is no
lead-time window and nothing to fit. `shield_latency` is the minimum green still
owed to the phase running now — exactly how long the shield can refuse the
request — so the deadline is brought forward by precisely the amount the safety
layer can delay it. The scheduler is polled every simulated second inside a
green, so it can hold off to within a second of the deadline; the dual-rate
design and the just-in-time rule are the same mechanism seen from two sides.

**Simultaneous requests are scheduled exactly.** When several approaches commit
at once the scheduler minimises priority-weighted emergency lateness plus
background delay over every ordering of every prefix — at most 64 evaluations for
a four-phase signal, microseconds per epoch. Reviews of the area name
simultaneous conflicting requests as an open problem; at the emergency rate used
here we log a rival committed request in roughly one request-epoch in six.

**A chance constraint bounds spurious preemption.** A token bucket refilling at
the operator's budget spends `(1 − p)` per engagement, so the expected number of
corridors granted to vehicles that were not there is bounded over any window.
This replaces a duty-cycle cap, which limited the wrong quantity — it throttled
hardest exactly when emergency traffic was heaviest and real.

### 5. Emergency delay is a *constraint*, and safety is *enforced*

The usual approach scalarises: `reward = −delay − k·ev_delay`, then tunes `k`.
That gives a controller whose emergency behaviour is whatever the tuning
happened to produce, and `k` has to be retuned for every demand level. This is
posed instead as a **Constrained MDP** — maximise efficiency subject to a bound
on priority-weighted emergency delay — solved with a two-timescale Lagrangian:
two dueling critics on a shared trunk, greedy action `argmax_a [Q_r − λQ_c]`,
and projected dual ascent on the constraint violation. The requirement is the
input and the trade-off weight is discovered.

A **safety shield** then filters the policy's proposed action to the legal set.
It guarantees minimum green, a full yellow → all-red changeover (so conflicting
movements are never simultaneously green by construction), maximum green, and a
hard bound `W_max` on how long any movement may be denied service:

```
τ = W_max − n_phases · (g_min + Δt + yellow + all_red)
```

The `n_phases` factor is not decoration — the first version of this shield
omitted it, and its claimed 110 s bound was violated (115 s observed) because
several approaches can sit above the threshold simultaneously. The corrected
bound is proved in `atsc/control/shield.py`, checked adversarially in
`tests/test_shield.py`, verified inside SUMO in `tests/test_env.py`, and reported
as `max_movement_wait_s` on every evaluation run.

**The corridor cannot break it.** The scheduler only ever selects among actions
the shield already permits, at decision epochs and at 1 Hz interrupts alike, so
S1–S3 hold for any belief and any adversarial detector output. That is the point
of keeping the shield underneath: the priority layer is free to be aggressive
because it cannot be unsafe.

### 6. The mast array is sited and read so it can *localise a vehicle at the stop line*

The single hardest moment for an acoustic array is the one that matters most:
an emergency vehicle standing at a red, metres from the junction, waiting for
the green this whole system exists to give it. Getting that moment wrong is not
a degradation, it is an inversion — the controller holds a green for an arm
that has nothing on it while the vehicle it is trying to serve sits at a red.

Two separate defects produced exactly that, and both are worth stating because
each looked like a fix for the other.

**The masts were in the wrong place.** They sat 70 m upstream of their stop
lines, on the argument that a mast at the junction would be roughly equidistant
from every approaching vehicle and so could not localise. That argument is
right about *range* and leaves out the cue that actually operates at long
range: the cardioid's own pattern. A source 300 m up the west arm is on-axis to
the west mast and broadside to the north and south ones **whatever the offset
is**, so the far field is separated by directivity, not by geometry. What the
offset controls is the near field — and there it worked backwards. Measured as
the level at the correct mast minus the level at the loudest wrong one:

| source distance from the stop line | 300 m | 150 m | 60 m | 25 m | 8 m | 1 m |
|---|---|---|---|---|---|---|
| mast 70 m upstream (steered pair) | 5.3 dB | 7.7 | 23.9 | 11.0 | 2.6 | **1.5** |
| **mast 6 m upstream (cardioid)** | **7.0 dB** | **7.5** | **9.6** | **13.1** | **23.2** | **11.9** |

At 70 m the array was blind over the last twenty metres of every approach —
1.5 dB against a level noise of several — which is precisely where a vehicle
queues at a red.

**And the filter was not reading the level at all.** ``pred_spl`` was computed
for every hypothesis and then used only to select an SNR bin; what separated
one hypothesis from another was the detector logit and the range head. That
works while the detector is unsaturated, and at the ranges that matter it is
not: every mast within earshot reports ``p_siren = 1.00``, so the logit term is
*identical* under "the source is on the east arm" and "the source is on the
south arm", and the range head's error is tens of metres. The levels were never
identical — traced on a stalled fire appliance nine metres short of the east
stop line, the four masts read 28, 17, 12 and −6 dB SNR — and all fifteen
decibels of that were being discarded.

The two defects hid each other. Reading only the louder lobe of a *steered*
two-element pair does remove the 70 m mast's rear null, which is what it was
added for; it also makes the mast very nearly omnidirectional (0 dB on both
axes, −6 dB broadside), collapsing the pattern separation from 14 dB to 6. That
cost nothing observable only because nothing was reading the level.

So: the mast moves to the signal pole where a pole already is, reads as a single
cardioid aimed up its approach, and its measured level is scored against the
level each hypothesis predicts (with a 2.5 dB measurement noise applied once, in
the sensor, so the likelihood and the sampling distribution cannot drift apart).
The rear null now does useful work rather than harm: it is aimed at the
junction, so a vehicle that has **crossed** goes quiet by 14 dB, which is the
inbound-versus-departed cue the signed state space is there to consume.

**What each half is worth, measured separately.** Eight seeds of half an hour
at 24 emergency vehicles an hour, each half ablated on its own with everything
else at the shipped setting, paired by seed:

| ablation | emergency stopped delay | background queue |
|---|---|---|
| level not scored | 15.85 s → **7.47 s** (**−8.38 s**, s.e. 2.76, **8/8 seeds**) | 13.59 → 11.38 veh (6/8) |
| 70 m steered mast | 7.76 s → 7.47 s (−0.29 s, s.e. 1.49) | 12.70 → **11.38 veh** (−1.33, s.e. 0.51, 7/8) |

**The level term is the fix; the siting is a refinement.** Reading the measured
level takes emergency stopped delay from 15.9 s to 7.5 s on *every one of eight
seeds*, and the worst single vehicle from 70 s to 25 s, while also taking two
vehicles off the background queue. The siting on its own does nothing to
emergency delay we can distinguish from noise; what it does is let the filter
localise with fewer corridors, which shows up as cross-street queue.

That ordering is the opposite of the one the previous revision assumed, and the
reason is worth stating: the geometry was the *visible* defect, so it got the
fix and the credit, while a computed quantity that was simply never consumed
sat underneath it for a whole revision — and it survived because `pred_spl`
*was* being used, to pick an SNR bin, so nothing read as dead code.

`mic_offset_m`, `dual_element_mast` and `AcousticConfig.score_level` are all
exposed, so each can be measured on its own.

### 7. The detector is tested against the sound that actually fools it

Every siren detector in the literature reports how well it separates sirens from
*background noise*. Background noise is an easy negative: it is broadband, and a
harmonic-template detector rejects it almost regardless of level. The hard
negative is a **road horn** — and until this revision there were no horns
anywhere in this project's acoustic model.

That is not a small omission, because the horn wins on every axis a level-based
detector could use:

| | siren | road horn |
|---|---|---|
| source level at 1 m | 109–116 dB | 100–112 dB (UNECE R28) |
| spectrum | harmonic | harmonic |
| band | 420–1750 Hz | 320–620 Hz fundamentals + harmonics |
| **where it is** | still 200 m away | **in the queue, metres from the mast** |
| **level at the mast** | 15.5 dB SNR at 200 m | **23 dB SNR at 30 m** |

**The confuser is the louder signal.** No level threshold separates them, and a
detector trained only against pink noise has never been asked to try.

**What does separate them is modulation.** A siren *sweeps* its fundamental —
`wail` glides an octave in 4–6 s, `yelp` in 0.2–0.4 s, `hilo` alternates about
once a second. A horn is a pair of fixed-pitch diaphragms and does not sweep at
all. Over a one-second frame a siren draws a moving ridge on the spectrogram and
a horn draws horizontal lines. That cue is invariant to everything that makes
level unusable — distance, directivity, the ambient floor — because it is a
property of the source's modulation rather than its amplitude. It is asserted
directly in `test_horn_holds_its_pitch_while_a_siren_sweeps`.

So horns are now first-class throughout:

- **Synthesised physically** — dual tones a third apart, both diaphragms
  rendered so they beat against each other, with press-and-release envelopes so
  a frame can catch an onset, a tail, or a gap.
- **Sounded by the simulation.** Stopped vehicles honk at
  `AcousticConfig.horn_rate_per_veh_hour`, so honking rises and falls with
  congestion — the detector faces its hardest negatives exactly when the
  intersection is busiest, and an emergency vehicle arriving into a jam arrives
  into a chorus.
- **A distinct detector class**, not folded into `none`. Folding it in would
  train the same rejection but discard the evidence that it works: with its own
  label the confusion matrix reports horn-as-siren directly.
- **Its own confusion curve in the surrogate.** `SensorErrorModel.horn_logit`
  is fitted from the CNN's held-out predictions per SNR bin. An error model
  fitted before horns existed falls back to the *siren* curve — the pessimistic
  reading, so an un-updated model treats every horn as maximally confusable
  rather than silently as free.
- **Audible in the demo.** `scripts/demo.py --sound` now plays the horns too. A
  demo in which the only sound is the siren makes the detection problem sound
  far easier than it is.

**The bug this uncovered is the most instructive part.** Evaluated through the
existing readout, the retrained detector called a horn a siren with probability
**1.000 at every SNR above 0 dB** — which reads as a network that cannot tell
them apart. It was not. `p_siren` was computed as `1 - P(none)`, which is
correct only while every non-silent class is an emergency class; once the
confuser has its own label, a detector that has confidently and *correctly*
identified a horn puts its mass on `horn` and `1 - P(none)` reads 1.0. The
network had learned the distinction and the readout was throwing it away.
Summing the siren classes instead:

| SNR (dB) | true siren | horn, `1 - P(none)` | horn, siren-sum |
|---|---|---|---|
| −5..0 | 0.999 | 0.997 | **0.019** |
| 0..5 | 1.000 | 1.000 | **0.001** |
| > 5 | 1.000 | 1.000 | **0.000** |

Closed loop over 8 shared seeds with traffic honking, honking costs a blind
detector 4.7 points of precision and horn-awareness recovers **all** of it,
landing within 0.11 points of the silent-traffic reference (−4.62 pp,
**p = 0.0014**). The effect on emergency delay is *not* established (+2.79 s,
p = 0.11) and is reported as such in `docs/RESULTS.md` §5.3.

This matters more here than it would in a pure detection paper, because of §8:
the S6 guarantee acts on a request *decisively*, so a spurious request is not a
missed opportunity but a forced green on the wrong approach. `docs/RESULTS.md`
measures exactly what an imprecise detector does to a system that trusts it.

```bash
python scripts/horn_robustness.py --seeds 8     # silent vs blind vs horn-aware
python scripts/demo.py --horn-rate 30           # hear it in heavy traffic
```

### 8. Emergency service is *guaranteed*, not merely optimised (S6)

Every layer above the shield is an **option**. The posterior may fall short of
its derived threshold; the Bayes rule may be outbid by the cross-street queue;
the token bucket may be empty. Nothing promised that a vehicle which *is* there
gets a green — and measured per second against ground truth, that is where the
delay came from: a vehicle sat 15 m short of the stop line for 30 consecutive
seconds while its approach's posterior read 0.00 and no request existed at all.

Deployed preemption does not work that way. A detected call *enters a sequence
and the sequence runs*; it is not re-priced against cross-street delay every
second. **S6** restores that property with a proof:

> A standing emergency request naming phase set `P` is granted one of those
> phases within `ev_service_deadline_s` of being raised, provided it persists.
> With `g_ev + poll + yellow + all_red = 12 s`, the deadline is 16 s; the shield
> **refuses** any smaller value rather than promising what the change interval
> forbids.

**When each bound holds.** The 16 s figure assumes the shield is consulted every
second with the 6 s emergency minimum green, which is the case while the corridor
itself has called the emergency (1 s polls). Between corridor calls the shield is
consulted only at the 5 s decision points and the normal 8 s minimum green
applies, so the guaranteed bound is `4 + T + ceil(g_min/dt)*dt + T` = **24 s**
(4 s request age, 5 s changeover twice, 10 s for the minimum green to reach a
decision point). The shield stress test (`scripts/shield_stress.py`, results in
`results/camera_ready/`) never exceeded either bound: 21 s without S3 contention.
`SafetyShield.guarantees()` reports the 1 s-poll figures.

Three things make it more than hard preemption with a proof attached:

- **It sits *below* S3, not above it.** A movement about to breach its 180 s
  starvation bound is still served first. Inverting that is exactly what makes
  classical preemption unsafe on a busy network: a detector with any
  false-positive rate could otherwise starve an approach without limit.

  The composed worst case is `deadline + n_phases x (g_min + dt + T)`: **88 s**
  with 1 s polling, as reported by `SafetyShield.guarantees()`, and
  `24 + 4 x 18` = **96 s** at the 5 s decision points (the stress test's longest
  request under S3 contention was 36 s). It was stated as 34 s — one
  starvation service rather than one per phase — and that was simply wrong:
  several movements sit above the starvation threshold together on this network,
  S3 walks all of them, and a request was measured standing 49 s against the
  34 s promise while the shield behaved exactly as specified. The `n_phases`
  factor is the same one S3's own proof carries and had already been corrected
  for once. To shorten the typical case without moving the worst one, S3 now
  breaks its own ties toward a due emergency phase where it can: every starving
  phase still leaves the starving set when served, so the charge is unchanged
  and only the order within the set moves.
- **It consumes a *request*, never a posterior**, so the safety argument still
  does not depend on the detector being right. A wrong request costs cross
  traffic one bounded service and cannot violate S1, S2 or S3.
- **It is raised on evidence, not on economics.** Feeding S6 the corridor's own
  commit condition would make it vacuous — it would promise only what the
  economics had already chosen to do. The request is raised on the calibrated
  posterior alone, and the vehicle is then owed a green whatever the queue is
  doing.

S6 may only ever *narrow* the action set. Written as an early return it widened
it instead — a request naming every phase re-permitted a green that maximum
green had already ended, silently disabling S2 and gap-out on 16% of steps, and
the congestion that created cost more emergency delay than the rule saved. That
regression is pinned by `test_s6_never_widens_the_action_set`.

---

## Results

> **These numbers predate the sensing and corridor revision described in §6 and
> in the revision note above, and every one of them is expected to move.** The
> closed-loop diagnostic they are being replaced from measures emergency stopped
> delay at 15.5 s → 7.5 s over eight seeds under the same corridor, so the
> `EV stopped` column below is the one to distrust most. Regenerate the table
> with
>
> ```bash
> python scripts/run_all.py --only evaluate --force
> python scripts/make_tables.py
> ```
>
> and note the second caveat while doing so: the four policy checkpoints in
> `models/` were **trained under an earlier revision of the sensing model**. We
> measured what that costs rather than leaving it as a caveat. Retraining the
> same configuration under the current sensing moved median emergency stopped
> delay by 1.3 s (p = 0.36) and left the corridor's contribution qualitatively
> unchanged (−7.50 s against −8.65 s). Because this test uses only twelve seeds, it is primarily sensitive to relatively large effects; thus, a non-significant result should be interpreted as no detectable effect rather than evidence of no effect.

Balanced demand, **12 seeds × 3600 s**, 30 emergency vehicles/h, every
controller on identical demand realisations. Read the **median** first: the
distribution is heavy-tailed, because this intersection runs near capacity and a
seed that tips into spillback reaches several hundred seconds against a ~23 s
median for the same controller. `jam` counts seeds that gridlocked. **Clean
pass** is the fraction of emergency vehicles that cross without ever stopping.

<!-- Generated by scripts/make_tables.py from results/evaluation_balanced.json. Do not edit. -->

| controller | median (s) | mean (s) | jam | EV stopped (s) | EV stops | clean pass |
|---|---|---|---|---|---|---|
| **SIREN-ARC (ours)** | 24.41 | 24.35 | 0/12 | 8.58 | 0.88 | 0.309 |
| **SIREN-CMDP (ours, constrained)** | 24.15 | 57.88 | 1/12 | 49.17 | 1.16 | 0.308 |
| SIREN-CMDP, overlay coupled to λ (ablation) | 24.16 | 24.21 | 0/12 | 9.66 | 0.96 | 0.264 |
| Learned policy alone | 23.23 | 23.38 | 0/12 | 16.43 | 1.12 | 0.129 |
| Constrained policy alone | 23.05 | 23.49 | 0/12 | 13.55 | 1.06 | 0.197 |
| *Ceiling:* ours + perfect detection | 23.56 | 23.85 | 0/12 | 8.63 | 0.69 | 0.467 |
| Max-pressure + hard preemption | 27.42 | 27.29 | 0/12 | 9.99 | 0.81 | 0.315 |
| *Ceiling:* preemption + perfect detection | 24.90 | 89.94 | 2/12 | 26.38 | 1.07 | 0.421 |
| Fixed-time (Webster) | 26.59 | 26.78 | 0/12 | 19.01 | 1.16 | 0.143 |
| Actuated (SUMO) | 27.49 | 27.46 | 0/12 | 19.52 | 0.89 | 0.248 |
| Max-pressure | 22.71 | 61.44 | 2/12 | 38.76 | 1.75 | 0.190 |
| Greedy queue | 24.14 | 60.54 | 1/12 | 53.48 | 1.94 | 0.173 |
| Ours, independent per-approach filters | 24.52 | 24.51 | 0/12 | 15.23 | 1.17 | 0.183 |
| Ours, act-on-sight instead of just-in-time | 25.11 | 87.75 | 1/12 | 43.66 | 1.63 | 0.296 |
| Ours, uncalibrated posterior | 24.72 | 57.04 | 2/12 | 40.74 | 1.63 | 0.306 |
| Ours, greedy conflict horizon | 24.51 | 29.24 | 1/12 | 12.48 | 1.03 | 0.247 |

### What to read out of that

**Read the median, and read `jam` next to it.** This intersection runs near
capacity, so a seed that tips into spillback reaches several hundred seconds
against a ~24 s median for the same controller. Means in this table are
descriptions of how often that happened, not of typical behaviour, and the gap
can be enormous: the constrained corridor's median emergency delay is 8.75 s
and its mean is 49.17 s, and the entire difference is one seed out of twelve.
Quoting the mean alone would call that controller catastrophic; quoting the
median alone would hide a real tail. Both columns are there because neither is
sufficient.

**The corridor earns its place, and this is the strongest result here.**
Against the identical policy weights with no priority layer, median emergency
stopped delay falls from **17.47 s to 8.83 s** and clean pass rises from
**0.108 to 0.305** — nearly threefold. Paired over the twelve shared seeds, both
are significant at p = 0.0001. It is paid for: background delay rises 1.18 s
(p = 0.024). That is the trade the method exists to make, and at roughly eight
seconds of emergency delay bought per second of background delay it is a
favourable one.

**It dominates hard preemption on both objectives.** Max-pressure with
Opticom-style preemption reaches 9.53 s emergency delay for 27.42 s of
background delay; the corridor reaches 8.83 s for 24.41 s. Better on both axes
at once, and both differences hold up paired (p = 0.0091 and p = 0.0001).
Clean pass is now a *tie* — 0.305 against 0.328, p = 0.86 — where the previous
run showed preemption clearly ahead. Seizing the intersection outright is still
the most direct way to never make a vehicle stop, but on this run it no longer
buys a measurable advantage at it, and it still costs three seconds of
background delay to everybody else.

**Perfect detection buys clean passes, not time.** Running the identical
corridor on *oracle* sensing — exact position, range and class, the assumption
almost all preemption work makes — gives 6.72 s median emergency delay against
our 8.83 s from four microphones, but paired across seeds that difference is
not significant (p = 0.95; the means are 8.63 against 8.58, i.e. identical).
Clean pass is a different story: 0.491 against 0.305, p = 0.0001, a large and
solid gain. So the sensor is not the bottleneck on *how long* an emergency
vehicle waits, and it is the bottleneck on *whether it stops at all*. The
previous revision of this file claimed perfect detection would buy nothing; on
these numbers that is true of delay and false of clean pass, and the distinction
is the useful part. The oracle also reaches its result while engaging 30.8 times
an hour against our 52.1, which is what a sufficient sensor looks like from the
inside.

**The constrained configuration is a tail-risk story, not a central-tendency
one.** SIREN-CMDP was introduced as the second point on a frontier. On medians
it is marginally *better* than SIREN-ARC on emergency delay — 8.75 s against
8.83 s — which is well inside noise. What separates them is that it gridlocks
one seed in twelve where SIREN-ARC gridlocks none, and that one seed takes its
mean to 49.17 s. The reason to prefer **SIREN-ARC remains, but it is now about
the tail rather than about typical performance**, which is a weaker and more
honest claim than the previous run supported.

**The dual variable is still not an exchange rate, but this run cannot show
it.** Feeding the CMDP's converged λ into the corridor as a multiplier on γ
gives 9.58 s median emergency delay against the decoupled 8.75 s — a difference
of 0.83 s at p = 0.59, and 0.01 s of background delay at p = 0.20. The two are
indistinguishable here. The previous run showed a large gap (24.82 against
15.36 s) and that gap has gone. The *argument* is unchanged and is a priori:
γ is in vehicle-seconds, λ is whatever scale balances two critics, and neither
reads the other. But it is now an argument without a measurement behind it, and
the row stays in the table as the ablation it is rather than as evidence.

**What the ablations support, and what they do not.** The joint filter is
clearly load-bearing: running the corridor on independent per-approach filters
costs 6.21 s of median emergency delay and 0.148 of clean pass (p = 0.0001),
and the separately trained independent policy costs 6.33 s (p = 0.0001). The
exact conflict schedule is worth defending too — the greedy horizon costs
0.077 of clean pass (p = 0.021) and gridlocks a seed. The other two do *not*
pay for themselves on these numbers: acting on sight rather than at the
deadline costs 0.58 s at p = 0.18, and dropping posterior calibration
*improves* the median by 0.56 s at p = 0.19 — though it gridlocks two seeds
against none, so what calibration buys here is tail behaviour rather than
central performance, exactly as with the constrained configuration. Two of
these rows changed sign between runs. An ablation table cannot separate
"unnecessary" from "redundant with something else that is also load-bearing",
and at twelve seeds on a heavy-tailed distribution single rows should not be
asked to carry much.

**The shield never failed.** Across every controller and seed in this run —
including the runs that gridlocked and the runs where the corridor was engaging
on noise — the longest any movement waited was **143 s** against the guaranteed
180 s bound.

**Off design demand, a much simpler controller still beats this one.** Agents
train only on `balanced`; `asymmetric`, `peak` and `surge` are held out. On all
three, SUMO's stock actuated controller (28–36 s median, no gridlock in 18
seeds) beats our corridor (32–79 s, three jams in eighteen) on both objectives.
On `balanced`, actuated is the *slowest* stable controller. The whole advantage
is specific to the regime we optimised for, and a rule with no learning, no
sensing model and no constraint is more robust than the entire apparatus here
away from it.

> **Consistency note.** The `balanced` table, the ablation, the sensor-loss
> study, the γ sweep and the trajectory trace were all re-measured on the
> current code, on seed ranges disjoint from one another and from training. Two
> sub-studies still predate them: the sensing benchmark and the three held-out
> demand regimes, so the off-design paragraph compares a current controller
> against an older one. That result's direction has been large and stable across
> every revision, but its specific figures should not be read to the same
> precision as the `balanced` table. Re-running them is `python
> scripts/run_all.py --only evaluate --force`.

> **Provenance.** Every learned checkpoint in `models/` was trained under an
> earlier revision of the sensing model. We measured what that costs rather than
> leaving it as a caveat. Retraining the same configuration under the current
> sensing moved median emergency stopped delay by 1.3 s (p = 0.36) and left the
> corridor's contribution qualitatively unchanged (−7.50 s against −8.65 s).
> Because this test uses only twelve seeds, it is primarily sensitive to relatively large effects; thus, a non-significant result should be interpreted as no detectable effect rather than evidence of no effect.

### Sensing

Six held-out hours, both filters consuming byte-identical detector output on
identical seeds. The only difference is whether the four readings are explained
jointly.

<!-- Generated by scripts/make_tables.py from results/sensing_benchmark.json. Do not edit. -->

| | independent filters | joint filter (ours) | |
|---|---|---|---|
| Precision at p ≥ 0.5 | 0.294 | **0.532** | higher is better |
| Recall at p ≥ 0.5 | 0.900 | **0.636** | higher is better |
| F1 | 0.444 | **0.579** | higher is better |
| False detections per hour | 401 | **104** | lower is better |
| Arms above 0.5 when one is occupied | 2.26 | **0.94** | 1.0 is ideal |
| Correct arm is the argmax | 0.645 | **0.749** | higher is better |
| Expected calibration error | 0.188 | **0.066** | lower is better |
| Brier score | 0.171 | **0.072** | lower is better |
| Area under ROC | 0.908 | **0.925** | unchanged — it was never the ranking |
| Median range error (m) | 73.9 | **40.9** | lower is better |
| Arrival-time MAE (s) | 32.6 | **29.7** | lower is better |

The gap in area under the ROC curve is small — 0.908 to 0.925 — and that is
worth dwelling on, because for a long time it was *zero* to three decimals. The
per-approach filter was never badly **ranked**; an evaluation that stopped at
AUC would have concluded there was nothing to fix. What it was, was badly
localised and badly scaled: it lights 2.26 arms when one is occupied, puts the
right arm at the argmax 65 % of the time against the joint filter's 75 %, and
is miscalibrated by a factor that no threshold can repair. Location and scale
are what a decision rule is made of, and neither of them is AUC.

The trade is real and we report it rather than choosing an operating point that
hides it: the independent filter has much the higher per-epoch recall — 0.900
against 0.636 — because a filter that lights every arm will not miss the right
one. Per *arrival* the gap is far smaller: 93 % of emergency vehicles are still
detected before they arrive, at a median of 148 m out, so most of what is lost
is redundant confirmation of vehicles already detected.

**What the joint filter buys in closed loop is a better outcome from fewer
interventions.** The independent configuration reaches 15.04 s median emergency
stopped delay against our 8.83 s, and clean pass 0.158 against 0.305
(p = 0.0001), while opening **73.4 corridors an hour to our 52.1**. Worse
outcome, a third *more* interventions. On the previous run the two were
statistically indistinguishable on delay and the argument for the joint filter
rested on economy of intervention alone; on these numbers it is ahead on the
outcome as well. Preemption is a public act: a controller that stops cross
traffic for a vehicle that is not there spends credibility as well as green
time.

### Degradation under sensor loss

The study a transponder-based scheme cannot run — an infrared or V2I link has no
partial failure mode, so there is no row between working and not working.

| condition | EV stopped (s) | clean pass | jam |
|---|---|---|---|
| all four masts | 13.6 | 0.214 | 0/6 |
| ambient floor +5 dB | 15.0 | 0.215 | 0/6 |
| ambient floor +10 dB | 28.7 | 0.202 | 1/6 |
| ambient floor +15 dB | 73.3 | 0.226 | 1/6 |
| one mast dead | 15.2 | 0.208 | 0/6 |
| two masts dead | 15.0 | 0.160 | 0/6 |
| *no acoustic sensing at all* | 16.2 | 0.109 | 0/6 |

**Losing masts degrades the controller gracefully.** With half the array gone
emergency delay is 15.0 s against 13.6 s nominal, and clean pass holds at 0.160
against 0.109 with no acoustic sensing at all — so two microphones still deliver
about half the benefit of four. That is the property a transponder scheme cannot
offer, and it is the strongest row in this table.

**A rising noise floor is different, and much worse than we previously
reported.** At +10 dB emergency delay is 28.7 s and a seed gridlocks; at +15 dB,
73.3 s. Both are worse than running with *no sensor at all* — the controller is
not merely blind, it is actively misled.

Those numbers used to read 15.8 s and 19.0 s, and the improvement was an
artefact of the noise model rather than of the controller. The ambient floor was
formerly redrawn independently at every frame, so even a "+15 dB" configuration
drew a quiet floor roughly half the time and the filter simply detected through
the gaps. Urban noise does not do that. It now drifts, so a loud patch is
genuinely loud for a minute at a time, and the filter — correctly — cannot hear
through it. The earlier figures overstated robustness; these are the honest
ones. Mast loss removes information; noise adds misinformation, and only the
second can make a controller worse than having no sensor at all.

### Which configuration to use

- **Design demand** → SIREN-ARC, at whatever γ prices your priorities. It is
  the better configuration on this run — 8.83 s median emergency delay against
  the bare policy's 17.47 s — and γ moves it along the frontier predictably,
  with no gridlock at any of the nine values swept.
- **Design demand, and clean pass is the metric you are held to** → either.
  Hard preemption reaches 0.328 against the corridor's 0.305, and that
  difference is not significant (p = 0.86); preemption pays 3.0 s of extra
  background delay for it. On the previous run preemption won this outright,
  and it no longer does.
- **Saturated or off-design demand** → SUMO actuated. None of the adaptive
  controllers here are ready for it, ours included.
- **SIREN-CMDP only if you can tolerate the tail.** On medians it matches
  SIREN-ARC (8.75 s against 8.83 s, inside noise), but it gridlocks one seed in
  twelve where ARC gridlocks none. The constrained formulation is sound and the
  budget binds; what is unproven is that the controller built on it is safer,
  and on this run it is not. See the results narrative above.

---

## Install

Requires **SUMO ≥ 1.20** with `SUMO_HOME` set, and Python 3.10+.

```bash
python -m venv venv
venv/Scripts/activate          # Windows;  source venv/bin/activate elsewhere
pip install -r requirements.txt
```

Verify:

```bash
python tests/run_tests.py
```

---

## Watch it run

```bash
python scripts/demo.py                       # SUMO GUI, 1.5x real time, sirens on
python scripts/demo.py --no-sound            # silence them
python scripts/demo.py --speed 0.5           # half real time, to read the commentary
python scripts/demo.py --bluelight           # traffic pulls aside (can deadlock; see below)
python scripts/demo.py --speed 2             # twice real time
python scripts/demo.py --follow-ev           # camera pans with the vehicle being served
python scripts/demo.py --no-clear-lane       # without the lane clearing, for contrast
python scripts/demo.py --commit-range 0      # economics alone, as every reported number is
python scripts/demo.py --ev-schedule poisson # the arrival process the evaluations use
python scripts/demo.py --screenshot-ev       # save a frame on each commitment
python scripts/demo.py --no-gui              # headless, commentary only
python scripts/demo.py --sound-out run.wav   # record the soundtrack, no device needed
python scripts/demo.py --belief independent  # the previous per-approach filter
python scripts/demo.py --no-jit              # act on sight instead of at the deadline
python scripts/demo.py --controller max_pressure_preempt
```

**The demo is not the evaluation.** Three of its defaults are chosen to make
the mechanism watchable and are off everywhere a number is reported: emergency
vehicles arrive evenly spread rather than as a Poisson process
(`--ev-schedule`), the lane ahead of one is cleared (`--no-clear-lane`), and a
vehicle within 150 m of the stop line is served whatever the cross-street queue
is worth (`--commit-range 0`). Each has its own flag so you can watch the
difference, and the honest measurement of each is below.

**What you are hearing.** The demo plays the acoustic scene by default whenever
the GUI is up -- the controller's only sensor is a microphone, and a silent
demonstration of a system that hears is missing its input. It is not a sound
effect: the waveform comes from the same physical model the detector is
trained on and the joint filter's likelihood is built from
(`atsc/evaluation/audio.py`). Pattern and timbre are the wail, yelp and hi-lo
sweeps with their per-class horn characteristics; the level is
`received_spl_db` at the vehicle's true distance -- spherical spreading plus
ISO 9613-1 absorption summed over harmonics -- so it really does rise about
6 dB per halving of distance; the pitch carries the Doppler shift, and each
vehicle is panned by its true bearing, so which arm a siren is on is audible as
well as visible. The whole argument of this project is an argument about a
sound, and it is much easier to follow when you can hear it: the siren gets
louder, the corridor opens, the vehicle crosses without stopping.

It plays through the Windows `waveOut` API (with `winsound` as a fallback) or
`aplay`/`paplay`/`afplay` elsewhere; no dependency is added for it, and with no
device at all the mix is still rendered. `--no-sound` turns it off; headless
runs are silent already, because there is nothing for the audio to be in step
with at forty times real time. `--sound-out FILE.wav` records it, in the *simulation's* time base
rather than the wall clock, so a headless run at forty times real time still
produces a soundtrack that lines up with the episode.

Deliberately, the audio is the scene's **ground truth**, not the controller's
belief -- so when the filter is wrong about which arm a siren is on, you can
hear that it is wrong.

**What you are looking at.** Emergency vehicles carry a class badge — a red
cross for an ambulance, a flame for a fire appliance, a chequer-and-beacon for
police — and flash between their two livery colours, so which vehicle the
controller is prioritising, and what triage weight that implies, is visible on
the map rather than only in the log. The approach the scheduler has committed to
is overlaid with a green band and its ordinary traffic turns green; the band
appears when the *decision* is taken, which is before the signal changes, so the
corridor visibly opens ahead of the vehicle rather than around it.

An approach the filter is *tracking* a siren on but that the scheduler has not
committed to is banded **amber**. That half of the picture was missing, and it
is the half a viewer needs: a green band only appears once the controller has
decided to act, so everything before that — hearing a vehicle, tracking it,
pricing the corridor, deliberately waiting for the last moment at which opening
it still works — was invisible, and an emergency vehicle approaching a red gave
a viewer no way to distinguish "the controller has not heard it" from "the
controller has heard it and is timing the opening". Those are the two
hypotheses the whole demonstration is about. Amber, then green, then the
vehicle crosses.

**Where the camera is.** Framed on the junction, and it stays there. This took
three attempts, and the two failed ones are worth recording because they are the
same mistake twice: *the TraCI getters and the rendered picture disagree, and
only the picture is true.*

**Attempt one — SUMO's own vehicle tracking.** `--follow-ev` used
`gui.trackVehicle`, which zooms the rendered view hard in: a scale bar reading
ten metres, a window spanning under two hundred, the junction off screen
entirely. It does this **without reporting it**. Instrumented over 3670 frames
of a live run, `gui.getZoom` returned a constant 125.23 before, during and after
tracking, and `gui.getBoundary` kept describing the full wide frame; only the
offset was written back. So a measurement said "the zoom never changes" while a
screenshot of the same run showed the junction gone. Tracking is no longer used
at all. The camera is driven by explicit `gui.setBoundary` calls, always the
same size about a centre this code picks, so a constant scale is arithmetic
rather than a hope about an API.

**Attempt two — a square boundary.** `setBoundary` *fits* the box it is given
into the window instead of filling it, so the window's aspect ratio decides how
much of the other axis comes along. Measured against a live sumo-gui, a square
520 m box in a 2.39:1 window is fitted by its height and shows **1241.5 × 520 m**
— the whole 640 m network plus green margin, vehicles too small to read. This is
exactly the failure the old docstring claimed a boundary in metres had *solved*:
it is the same box on every machine and a different picture on each. The frame
is now shaped to the window, measured once at startup, so the longest visible
axis is always `2 × --view-radius` whatever the window's proportions. At the
190 m default that is **380 m across**, and the fitted size is identical on every
frame of a run:

| | requested | actually shown |
|---|---|---|
| before | 520 × 520 m | 1241.5 × 520 m |
| after | aspect-shaped | 300 m across the long axis |

`--view-radius` takes half the long axis: 100 gives 200 m across, 200 gives 400.

**Attempt three — the camera would not let go of the wheel.** Fixing the two
above left it setting the boundary on *every frame*, which meant the view could
not be zoomed at all: the scroll wheel moved the camera and a tenth of a second
later the next frame put it straight back. A demonstration you cannot lean into
is worse than one framed slightly wrong. The frame SUMO is showing is now
compared against the one last asked for, and any difference is taken as a hand
on the mouse — after which the camera is the viewer's for the rest of the run,
not until the next frame, which would be the same fight one step slower.

**And the camera lets go.** Following was released only when the *scheduler*
dropped its target, while the vehicle was matched by its road id — which stops
matching the moment it crosses the stop line and its road becomes an internal
junction lane. So the release never fired on time: the view stayed locked to a
vehicle that had already been served and followed it out of the network. The
camera now takes hold of a vehicle only within `GuiPainter.follow_range_m`, lets
go the moment it is no longer upstream on the approach being served, and eases
toward its target by `follow_ease` of the remaining distance per frame, so
acquiring and releasing are glides rather than cuts. The offset from the
junction is clamped by `follow_max_offset_m` (55 m, and no more than a third of
the view radius), so **the junction stays near the middle of the frame** — at
the 110 m it started at, following a vehicle on the west approach put the
junction hard against the right-hand edge with two thirds of the window showing
empty road, which is not a demonstration of an intersection — following biases the view toward the arm where something is
happening, it does not replace the subject of the demonstration with a patch of
empty road. It is re-evaluated every frame rather than every decision epoch,
because a vehicle crosses the junction in about two seconds and a camera that
checks every five spends most of that pointed at the wrong thing.

Twenty-one tests in `tests/test_gui.py` cover this. The load-bearing one collects
every box handed to SUMO across acquire, junction-crossing and release and
asserts there is exactly **one distinct size**; another checks the visible span
is correct at four window aspect ratios (1.0, 1.66, 2.39, 0.6); and three more
check that a viewer who zooms is left alone, permanently, without the camera
mistaking its own updates for theirs.

`--follow-ev` is **off by default**, and that is the right default: the junction
is the thing worth watching, and a camera that chases vehicles spends most of a
wide frame on empty road.

**Clearing the lane ahead.** While an emergency vehicle is within 220 m of a
stop line, every ordinary car ahead of it in its own lane is asked to move to
another lane of the same approach (`atsc/control/clearance.py`). Only sideways —
nothing is stopped or slowed — and only into a lane the car's own turn can still
be made from, checked against the junction's connection table. That last check
is what separates this from SUMO's `bluelight` device, which yields regardless of
whether there is anywhere to yield to and deadlocks this intersection outright on
about one seed in six.

*And it does not reduce emergency delay here.* Over twelve seeds at 24 emergency
vehicles an hour: 15.5 s of stopped delay per emergency vehicle without it, 15.6 s
with it — a paired difference of +0.1 s against a standard error of 2.3, which is
no effect rather than a small one. Background delay was unchanged (10.9 → 10.7 veh
mean queue), nothing gridlocked, and the starvation bound held. The reason is
geometry: three lanes at 700 veh/h put two or three cars in front of an arriving
vehicle in any one lane, and a lane change takes about as long as those cars take
to discharge once the green arrives. The mechanism is real and visible; the margin
it works on is not. It is on in the demo because "clear a path" is half of what
emergency priority means and it belongs on the screen next to the other half — and
it is off in every evaluation run.

**Serving on distance.** `--commit-range 150` (on by default in the demo) serves
any emergency vehicle the sensor is confident about once it is within 150 m of the
stop line, bypassing the Bayes rule that otherwise decides whether the corridor is
worth its cost. This is an operator's rule, not the paper's: the evidence floor
still applies, so it is not a way to preempt on noise, but the cross-street queue
no longer gets a vote. `--commit-range 0` is the behaviour every reported number is
produced with.

Measured over the same twelve seeds, it does about what you would expect a blunter
rule to do: full stops per emergency vehicle 1.04 → 0.93 and noticeably fewer S6
interventions, paid for with 0.8 veh more standing background queue (10.7 → 11.4)
and no change in mean emergency delay (+0.8 s, s.e. 3.2). It buys a cleaner-looking
run, not a faster one.

**What actually bounds the latency, and what does not.** The number that governs
worst-case service is the shield's S6 deadline (with 1 s polling; 24 s between
corridor calls, as explained in section 8), and it is 16 s because that is
about as tight as this intersection can carry: at 14 s — its arithmetic floor of
emergency-minimum-green + poll + yellow + all-red — the guarantee starts thrashing,
firing hundreds of times an episode instead of a dozen, because a forced changeover
costs the whole budget and the request comes due again the moment it lands. Measured
at 14 s: mean queue 39 veh against 11, emergency delay 68 s against 14. Lowering the
evidence floor (`--service-floor 0.35`) changed essentially nothing, because the
calibrated posterior rarely lands in the band between the two thresholds.

So the emergency delay this controller delivers here — mean ~15 s of stopped delay
per vehicle, with a seed-to-seed spread from 6 s to 33 s — is not sitting behind a
knob that was set conservatively. The spread across seeds is far larger than any of
these settings moves the mean.

**Why it looks smooth now.** SUMO redraws once per simulation step. At the 1 s
step the controller is trained on, a car at 50 km/h jumps fourteen metres
between frames and then freezes — which reads as stutter, not as traffic. The
demo runs the *simulation* at `--step 0.1` (ten frames a simulated second) while
leaving the 5 s decision epoch untouched. The controller is unchanged; it still
decides every 5 s and still re-checks the corridor every step. `--speed` is
playback relative to real time, and `--delay-ms` overrides it directly.

**Why it is no longer too fast.** Playback is paced by the client, in
`SumoSignalEnv._pace`. SUMO's own `--delay` paces the window's play button and
is *not* consulted while TraCI is stepping the simulation, so the demo used to
run at whatever rate the machine managed — measured here at 3.3× real time when
0.5× had been asked for. The step loop now sleeps to the requested rate, so
`--speed 0.5` really is half real time.

The commentary prints, per decision epoch, the queue, the per-approach belief,
and for each corridor the red the vehicle is projected to meet at its arrival,
the delay the corridor saves, the slack left before it must open, and what it
costs cross traffic:

```
  time  ph          queue N/E/S/W         belief N/E/S/W       nearest EV  act  event
    68   0       0/   2/   0/   1    1.00/0.00/0.00/0.00 N   120m  14.9m/s    0  CORRIDOR N <- N: snr= -10dB- p=0.59 (need 0.02) eta=  3.3s rng=   92m red@arrival=  20s save= 20.3s slack=  3.3s cost=   0.0s [GREEN]
   358   2       1/   2/   2/   0    0.00/0.00/0.00/1.00 W   107m  17.0m/s    0  CORRIDOR W <- W: snr=   3dB^ p=0.59 (need 0.23) eta= 25.2s rng=  106m red@arrival=   5s save=  5.1s slack= 25.2s cost=  58.5s [GREEN]
```

The `watching` line is the just-in-time rule visible: the request is priced,
found worth acting on, and deliberately deferred until it is due.

The **nearest EV** column is the caption's one piece of ground truth, read from
the simulator rather than from the belief — the closest emergency vehicle's arm,
its true distance to the stop line, and its speed, with `STOPPED` called out
rather than left to be inferred from a zero. It is there because the question a
viewer keeps asking of the belief to its left ("is there really one out there,
and is it moving?") is a question the belief cannot answer about itself, and
because a stopped emergency vehicle is the failure the whole system exists to
prevent, so it should be impossible to miss in the log.

---

## Reproducing everything

```bash
python scripts/run_all.py            # full study, several hours
python scripts/run_all.py --quick    # reduced settings, about an hour
python scripts/run_all.py --only evaluate
```

> **`--only evaluate` skips a regime whose results file already exists.** All
> four exist in a fresh clone, so the stage is a no-op and the tables are then
> rebuilt from whatever numbers were already there. Pass `--force` to re-measure,
> or call `scripts/evaluate.py` directly.

### Two things that will waste a night if you hit them

**Keep `ATSC_RUNS_DIR` short on Windows.** Scenario files are written to
`<runs dir>/<regime>_<controller>/scenario_<controller>_<seed>.sumocfg`, and the
controller names in the full set are long. Past 260 characters — Windows'
`MAX_PATH` — SUMO cannot open a file that Python has just written perfectly
happily, because Python resolves paths through the long-path API and SUMO is a
C++ binary using the legacy one. `Path.exists()` returns `True` and SUMO reports
*"Could not access configuration"* naming the file that is provably there.

Worse, TraCI does not fail on it: SUMO exits, TraCI reports *"TraCI server
already finished"* and retries on a **different port**, dozens of times,
launching and losing a process each time. The client sits at 0 % CPU with no
SUMO alive, which reads as a hang rather than an error. The failure therefore
selects controllers by the length of their name — a 100-character run directory
puts `max_pressure` at 248 characters (fine) and `max_pressure_preempt` at 264
(broken) — so a full evaluation stops dead on its fifth controller and reports
nothing at all.

`SumoSignalEnv._start_with_retry` now checks the length *before* the first
launch and raises with the measurement and the remedy, so this is a clear error
in a few seconds rather than an overnight silence. The repository default
(`results/runs`) is short enough; set `ATSC_RUNS_DIR` to something near the
drive root if you relocate it.

**The evaluation is embarrassingly parallel and `evaluate.py` is not.** It runs
one episode at a time, so a 19-controller × 12-seed × 3600 s run takes about
nine hours on four cores. Split it by controller across concurrent processes and
merge, which `scripts/merge_evaluation.py` exists for — it refuses the merge
unless regime, flow, EV rate, horizon, seeds and sensing all match, then
recomputes the paired bootstrap over the union:

```bash
# group A carries the baselines and the significance reference (max_pressure)
ATSC_RUNS_DIR=/short/runs_A python scripts/evaluate.py --seeds 12 --horizon 3600 \
    --ev-rate 30 --regime balanced --out /tmp/eval_A.json \
    --agent base_no_sensing=models/no_sensing.h5 &

# groups B and C add learned controllers only -- 'none' suppresses the
# baselines, which would otherwise be measured three times over
ATSC_RUNS_DIR=/short/runs_B python scripts/evaluate.py --seeds 12 --horizon 3600 \
    --ev-rate 30 --regime balanced --baselines none --reference base_arc \
    --out /tmp/eval_B.json --corridor siren_arc=models/siren_arc.h5 ... &
wait

python scripts/merge_evaluation.py --into /tmp/eval_A.json \
    --add /tmp/eval_B.json --add /tmp/eval_C.json --reference max_pressure
```

Give every group its **own** `ATSC_RUNS_DIR`. Per-controller work directories do
not collide, but `evaluate.py` runs a one-off probe episode in a fixed directory
under a fixed TraCI label, and concurrent processes will fight over it. Pin
TensorFlow to one thread per process (`OMP_NUM_THREADS=1`,
`TF_NUM_INTRAOP_THREADS=1`, `TF_NUM_INTEROP_THREADS=1`) or three unconstrained
processes will oversubscribe every core and give back most of the parallelism.
Measured on four cores, three groups took 2 h 15 against about nine hours
serially.

Stage order is not arbitrary — the detector's fitted error model *is* the joint
filter's likelihood, the calibration map has to be fitted under the phase
behaviour it will be used with, and the constraint-cost sanity check runs before
any training because a cost that cannot rank fixed controllers of known quality
will not teach a policy either. Individual stages:

```bash
# 1. acoustic detector and the error model fitted to its held-out predictions
python scripts/train_detector.py --epochs 22 --train-clips 20000

# 2. base policies (the sensing ablation is trained, not swapped at test time)
python scripts/train_agent.py --episodes 170 --unconstrained --belief-mode joint \
    --run-name siren_arc

# 3. one calibration map per filter, on seeds disjoint from everything else
python scripts/fit_calibration.py --episodes 8 --belief-mode joint

# 4. score the filters against simulator ground truth
python scripts/sensing_benchmark.py --episodes 6

# 5. check the fast surrogate sensor really matches the real detector
python scripts/validate_surrogate.py

# 6. evaluate every controller on identical seeded demand
python scripts/evaluate.py --seeds 12 --regime balanced --ev-rate 30 \
    --corridor siren_arc=models/siren_arc.h5

# 7. degradation under a rising noise floor and under mast failure
python scripts/robustness.py --seeds 6

# 8. traces, sweeps, audits
python scripts/record_trajectory.py     # time-space diagram, one vehicle, four controllers
python scripts/record_trace.py          # belief and corridor state through one approach
python scripts/threshold_audit.py       # is the threshold really derived?
python scripts/value_sweep.py           # the operating-point frontier
python scripts/frontier.py --agents budget_tight,siren_cmdp,budget_loose

# 9. everything in the paper, and the tables in this file
python scripts/make_tables.py && python scripts/make_figures.py
python scripts/update_readme.py
```

Nothing in the paper is typed by hand: every figure and table is a pure function
of a file in `results/`. That now includes the two tables in this README, which
were the one place a stale number could survive a re-run --
`scripts/update_readme.py` splices them from the same generated file the paper
reads, and `--check` fails if they have drifted.

### A note on which emergency metric to read

The headline emergency metric is **stopped delay** (and stop count), not SUMO's
`timeLoss`. `timeLoss` is measured against each vehicle's *own desired* speed,
and emergency vehicles are configured to want to travel faster than surrounding
traffic — so they accrue time loss simply by following law-abiding cars, a
component no signal controller can remove. Reporting `timeLoss` as the headline
would mask most of the effect under study. Both are in the results table.

---

## Layout

```
atsc/
  config.py            all configuration as typed dataclasses, serialised with every run
  netgen.py            builds the SUMO network from node/edge/connection XML via netconvert
  demand.py            stochastic demand + emergency-vehicle injection, four demand regimes
  env.py               the CMDP environment; step() returns (obs, reward, cost, done, info)
  control/
    shield.py          safety shield and its guarantees
    corridor.py        risk-calibrated just-in-time corridors and exact conflict scheduling
    priority.py        the fixed-threshold overlay this replaced, kept as the ablation
  sensing/
    dsp.py             power-preserving log-mel front end, pure NumPy
    siren_synth.py     physical siren synthesis: Doppler, spreading, ISO 9613-1 absorption
    detector.py        three-headed CNN (class / range-rate / range)
    sensor.py          four masts, each a steerable two-element endfire pair;
                       oracle / surrogate / end-to-end modes
    joint_belief.py    the exclusive multi-Bernoulli filter over which arm holds the siren
    belief.py          the previous four independent per-approach filters (ablation)
    association.py     the geometric association heuristic (ablation)
    calibration.py     isotonic map from filter output to probability
  agents/
    dueling_ddqn.py    Lagrangian dueling Double DQN with two critics
    replay.py          prioritised n-step replay carrying reward *and* cost
    baselines.py       Webster, SUMO-actuated, max-pressure, greedy, hard preemption
  evaluation/
    metrics.py         tripinfo parsing, Jain fairness, paired bootstrap, per-class EV delay
    runner.py          episode and multi-seed experiment runners
    plots.py           every paper figure
    gui.py             the live overlay: badges, beacons, corridor bands
    icons.py           the class badges, generated rather than shipped as binaries
```

```
docs/
  RESULTS.md           what was wrong, what was fixed, and what it actually
                       bought -- including the two things that did not work
  RELATED_WORK.md      twenty-one papers, compared on assumptions and
                       guarantees rather than on incommensurable benchmarks

scripts/               (the ones added with this revision)
  ablation_ev.py       four-arm ablation: mast and S6, apart and together
  ev_latency.py        the per-vehicle delay *distribution*, not its mean --
                       the right statistic for a worst-case bound
  request_quality.py   request recall by distance to the stop line, against
                       simulator ground truth; the diagnostic that found both bugs
  s6_sweep.py          the service-deadline trade-off
    audio.py           the sirens, played out loud from the same physical model
scripts/               one entry point per experiment stage; run_all.py chains them
                       update_readme.py splices the generated tables into this file
tests/                 dependency-free runner: python tests/run_tests.py
models/                trained detector and policy weights used for every result
results/camera_ready/  additional experiments: 30 seeds, sweeps, stress test,
                       latency benchmark, mixed-traffic retraining
```

### Design notes worth knowing

- **The controller is network-agnostic.** Nothing hard-codes four approaches or
  two phases. At reset it reads the signal program, classifies phases as green or
  transitional, maps each to its controlled lanes, derives approach bearings from
  lane geometry, and builds the action space from what it found.
- **SUMO's `bluelight` device is off by default.** With it on, part of the
  measured emergency-delay reduction would come from other drivers pulling aside
  rather than from signal control, confounding exactly the quantity under study.
  Enable it for a realism ablation with `DemandGenerator.generate(bluelight=True)`,
  `EnvConfig.bluelight`, or `scripts/demo.py --bluelight` — but do not leave it
  on unattended. Over six runs at 30 EV/h it roughly halved emergency delay on
  five of them and *deadlocked* the sixth: background delay went from 24 s to
  213 s behind a 95-vehicle queue that never cleared. Making room for the
  vehicle is not free on a saturated junction.
- **The microphones and the controller run on different clocks.** The detector
  emits a one-second frame and every one of them is consumed
  (`EnvConfig.sensing_interval_s`), while the policy still decides every 5 s.
  The filter's transition model is scaled to elapsed time so the belief
  dynamics are unchanged by the sampling rate; what changes is that the
  corridor's 1 Hz re-check now has something new to look at. Set the interval
  equal to the decision interval to restore the previous behaviour.
- **The RL loop trains against a fitted sensor error model, not live audio.**
  Rendering four microphone frames and running the CNN every decision epoch is
  ~30× slower. `scripts/validate_surrogate.py` demonstrates the substitution is
  sound at both the sensor and closed-loop level.
- **A policy is always evaluated with the filter it was trained on.** The two
  presence filters produce differently scaled features; swapping them at test
  time measures a train/test mismatch, not an ablation. The checkpoint records
  its own `belief_mode` and evaluation honours it.
- **Per-vehicle metrics come from SUMO's `tripinfo`**, not from step-loop
  accumulation, so `timeLoss` uses each vehicle's own free-flow reference
  including its individual speed factor — the standard definition of delay.

---


