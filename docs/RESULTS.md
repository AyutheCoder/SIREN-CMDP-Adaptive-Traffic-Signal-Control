# What was wrong, what was fixed, and what it actually bought

This document reports **two** revisions, both of which began from the same
observation — emergency vehicles sitting through reds while the controller did
nothing — and it reports the negative results as carefully as the positive
ones, because several of the things measured here did not do what they were
expected to do, and one of them made the intersection worse.

The second revision exists because the first one's headline fix did not work,
and could not have been shown not to work by the evidence the first one
gathered. That is the most useful thing in this document, so it is stated at the
top rather than buried in §2b: the first revision found that the mast could not
*hear* a vehicle at the stop line, fixed that, confirmed the fix on audibility,
and shipped. The controller does not consume audibility. It consumes
*separability* — which mast heard it loudest — and that had not been fixed, was
not measured, and had in fact been made slightly worse by the same change.
Twelve of twelve emergency vehicles still came to a full stop.

Every number below comes from running the code in this repository. Nothing is
quoted from another paper's benchmark; see
[`RELATED_WORK.md`](RELATED_WORK.md) for why cross-paper numbers in this field
are not commensurable.

---

## 1. How the defects were found

Not by the test suite. **All 188 tests passed** against the broken code, and
they still pass now — the defects were invisible at unit level because every
unit was behaving as specified. The second round is worse than that: the test
that should have caught the sensing defect asserted a *proxy* for the property
that mattered ("each mast is 20-200 m upstream and no two are within 50 m"),
and the proxy passed while the array was blind. It now asserts the separation
directly, over the whole arm, which is a test the broken geometry fails. They were also invisible in the aggregate sensing metrics
the project already reported.

What found them was instrumenting the closed loop against **simulator ground
truth**: logging, once per simulated second and for every emergency vehicle, the
true edge and true distance to the stop line alongside what the filter believed,
what request the controller raised, and what the signal did.

The single most important methodological point in this revision is about how
that log is *summarised*:

> **Recall averaged over the whole approach hides the failure.** A missed
> detection 300 m out costs nothing — the just-in-time rule is deliberately not
> acting yet. A missed detection at 20 m is the vehicle stopping at a red.
> Averaged together, a collapse in the near bands is diluted by the far bands
> where a miss is *correct behaviour*, and the mean barely moves.

Conditioned on distance to the stop line, the failure was unmistakable.
`scripts/request_quality.py` is that diagnostic, kept as a script.

---

## 2. Defect one: the mast is deaf where the vehicle queues

Each mast sits `mic_offset_m = 70 m` upstream of its stop line as a single
cardioid **aimed away from the junction**. Everything between the mast and the
line is therefore behind it, in the rear null. Evaluating the propagation model
along the arm (`figures/fig_mast_response.pdf`, reproducible with
`plots.plot_mast_response`, which reads no result file and so cannot drift from
the code):

| EV distance from stop line | single cardioid | steered two-element pair |
|---|---|---|
| 300 m | 63.5 dB | 63.5 dB |
| 71 m (at the mast) | 112.0 dB | 112.0 dB |
| 30 m | 65.7 dB | **79.7 dB** |
| **1 m (at the stop line)** | **60.8 dB** | **74.8 dB** |

Ambient is 55–80 dB. The single-element response is not merely weak at the stop
line, it is **inverted**: a vehicle *at* the line is quieter than the same
vehicle 300 m away, and sits at or below the noise floor.

Downstream, logged per second against ground truth:

- the joint filter's presence collapses from 1.00 to 0.00 with the vehicle still
  **15 m short of the stop line**;
- its range estimate reads **65 m with the vehicle 1 m away**;
- the corridor, having no request, hands the green to a conflicting approach
  with the ambulance stationary at the line;
- **11 of 12 emergency vehicles came to a full stop**; mean stopped delay 18.4 s.

### The fix, and what it cost

A front-to-back ratio of 14 dB is already *"consistent with a compact two-element
endfire array"* — two capsules, read once. An endfire pair is steered by the
**sign** of the delay between its capsules, so inverting that sign yields a
second cardioid aimed at the junction, from the same hardware, in the same
frame. The mast then reports the **steered level** (the louder lobe) and the
**front-minus-back ratio**, in which source level, range and the ambient floor
cancel, leaving only which side of the mast the source is on.

That second quantity matters because an offset mast makes level **non-monotone**
in distance to the stop line: 85 dB is equally consistent with 90 m and with
50 m, and no amount of level accuracy resolves it.

Measured effect on the request signal, 6 seeds, 1800 s
(`scripts/request_quality.py`, `figures/fig_request_quality.pdf`):

| | single cardioid | two-element mast |
|---|---|---|
| spurious-request rate (no EV anywhere) | 11.7% | **6.3%** |
| recall, 25–50 m | 90.6% | 60.5% |
| recall, 0–10 m | 66.1% | 41.9% |
| mean EV stopped delay | 28.33 s | **15.81 s** |

**The trade is not the one we expected, and it is worth stating plainly.**
Making the mast able to hear behind itself does not simply raise recall. The
second lobe and the front-back ratio give the joint filter enough geometry to
*reject* hypotheses it previously accepted, so the filter becomes more
conservative: it raises fewer requests, and recall falls in most bands while the
spurious rate roughly halves.

That direction is the favourable one here, and the reason is S6. A guarantee
acts on a request *decisively*, so a wrong request is not a wasted opportunity
— it is a forced green on the wrong approach. **Precision, not recall, is what
the guarantee consumes**, and §4.1 measures what happens when it is given the
imprecise version.

The physics, unlike the closed-loop statistics, is not in doubt: the level curve
above is arithmetic on the propagation model rather than a noisy measurement,
and it is asserted in
`tests/test_sensing.py::test_two_element_mast_hears_a_vehicle_at_the_stop_line`.

### 2b. What the *next* revision found: audible is not the same as locatable

Everything above is still true and the fix still does what it says: the mast
became audible at the stop line. It did not become able to say *which arm* the
sound was on there, and that is the question the controller actually asks.

Two things were wrong, and each concealed the other.

**The pattern was traded away.** Reading the louder of two opposed cardioids
gives 0 dB on both axes and −6 dB broadside, so a steered mast is very nearly
omnidirectional. The front-to-back ratio that separates the four approaches at
long range collapses from 14 dB to 6. Measured as the level at the correct mast
minus the level at the loudest wrong one, for a source at signed distance `s`
from its own stop line:

| `s` | 300 m | 150 m | 60 m | 25 m | 8 m | 1 m |
|---|---|---|---|---|---|---|
| 70 m upstream, steered pair | 5.3 dB | 7.7 | 23.9 | 11.0 | 2.6 | **1.5** |
| 6 m upstream (signal pole), cardioid | **7.0 dB** | 7.5 | 9.6 | 13.1 | 23.2 | **11.9** |

At the stop line the array was separating the arms by 1.5 dB, against several
decibels of measurement noise.

**And the level was never scored.** `pred_spl` was computed for every
hypothesis and used only to select an SNR bin. What discriminated between
hypotheses was the detector logit and the range head — and at the ranges that
matter the detector is saturated, so every audible mast reports `p_siren =
1.00` and the logit term is *identical* whichever arm the hypothesis puts the
source on. Traced on a fire appliance stalled 9 m short of the east stop line,
the four masts read 28, 17, 12 and −6 dB SNR: fifteen decibels of answer,
discarded. The filter spent 57 consecutive seconds holding a green for a phantom
on the south arm while the vehicle stood at a red on the east one, and 12 of 12
emergency vehicles came to a full stop.

**The split, measured.** Eight seeds, 1800 s, 24 EV/h, each half ablated on its
own with everything else at the shipped setting. Paired by seed.

*Scoring the level* (`AcousticConfig.score_level=False` is the ablation):

| | level not scored | level scored | paired difference |
|---|---|---|---|
| emergency stopped delay | 15.85 s | **7.47 s** | **−8.38 s** (s.e. 2.76, **8/8 seeds**) |
| worst single vehicle | 69.8 s | **24.8 s** | **−45.0 s** (s.e. 20.9, 8/8) |
| background mean queue | 13.59 veh | **11.38 veh** | −2.21 veh (s.e. 1.10, 6/8) |

*Mast siting* (70 m steered pair with its own calibration map, level scored in
both arms):

| | 70 m steered | 6 m cardioid | paired difference |
|---|---|---|---|
| emergency stopped delay | 7.76 s | 7.47 s | −0.29 s (s.e. 1.49) |
| background mean queue | 12.70 veh | **11.38 veh** | **−1.33 veh** (s.e. 0.51, 7/8) |
| worst single vehicle | 32.1 s | 24.8 s | −7.4 s (s.e. 10.8) |

**The level term is the fix; the siting is a refinement.** Reading the measured
level takes emergency stopped delay from 15.9 s to 7.5 s and does so on *every
one of eight seeds*, while also taking two vehicles off the background queue --
both objectives at once, which is what removing an obstruction looks like. The
siting on its own does nothing to emergency delay that can be distinguished
from noise; what it does is let the filter localise with fewer corridors, and
that shows up as 1.3 veh of cross-street queue.

That ordering is the opposite of the one the first revision assumed, and it is
worth being blunt about why: the geometry was the visible defect, so it got the
fix and the credit, and the invisible one -- a computed quantity that was never
consumed -- sat underneath it for a whole revision. It survived because
`pred_spl` *was* being used, to pick an SNR bin, so nothing read as dead code.

This is the same mistake as §1, one level up: the first revision found a real
defect, fixed it, confirmed the fix on the quantity it had been reasoning about
(audibility), and did not check the quantity the controller consumes
(separability). `AcousticConfig.score_level`, `mic_offset_m` and
`dual_element_mast` are each exposed so the three can be measured apart.

### 2c. Three control-layer defects found in the same pass

Each was found the same way — per-second logging against ground truth — and
each is a case of a mechanism doing something other than what its own
documentation said.

1. **The corridor released its own green early.** The cost charged to cross
   traffic was priced with `hold = max(eta, open_time) + pass_time`; the hold
   actually taken was `open_time + pass_time`, which drops the waiting term. On
   an approach already green with an empty queue that is four seconds. Traced:
   an ambulance closing at 17 m/s lost its green 28 m short of the line — a
   second and a half from crossing — and then stood at the resulting red for
   60 s. Cross traffic had already been charged for the full hold.
2. **The false-preemption budget was unsatisfiable.** The chance constraint is
   specified per *engagement* and was charged per *poll*, pro-rated by `dt/60`.
   With the calibrated posterior bounded above by the detector's precision, a
   corridor held twenty seconds cost 0.17 tokens against a refill of two per
   hour; the bucket went permanently negative two minutes into every episode,
   after which the near-certainty gate (0.9 × 0.49 = 0.44) blocked all but the
   topmost calibration bin. 112 polls blocked per 30-minute episode, and
   nothing in the logs said so, because "budget exhausted" and "evidence too
   weak" are the same branch. The budget itself was also below what the sensor
   can achieve: serving every genuine vehicle at 24/h spends about 11 expected
   spurious corridors an hour *while being right every time*.
3. **The composed S6/S3 bound was wrong.** `worst_case_ev_wait_with_starvation_s`
   charged one starvation service where the reachable worst case is one per
   phase — the same `n_phases` factor S3's own proof carries, and which S3 had
   already been corrected for once. Measured against the stated 34 s: an
   emergency request stood 49 s while S3 served three other phases in turn. The
   bound is now `deadline + n_phases * service_cost`, and S3 breaks its own ties
   toward a due emergency phase where it can, which shortens the typical case
   without touching that worst case.

A fourth was found and **rejected on measurement**: the protected-left phases
are greened permissively under the through phases, so their starvation clocks
run while their movements are moving, and S3 forces a phase nobody is waiting
for once a cycle. Resetting the clock when a phase's movements are all green
with nothing halted on them looks obviously right and gridlocked the
intersection outright (mean queue 98 veh, 22 teleports, one emergency vehicle at
837 s). The permissive left cannot in fact discharge against 700 veh/h of
opposing flow; those forced services are load-bearing. Reverted, and recorded
here rather than deleted.

---

## 3. Defect two: nothing guaranteed service

Every layer above the shield is an **option**. The calibrated posterior may fall
short of the derived threshold; the Bayes rule may be outbid by the cross-street
queue; the token bucket may be empty. Each refusal is individually correct, and
their conjunction is a system in which nothing ever promises that a vehicle
which *is* there gets a green. The worst instance in the log: a vehicle sat
**15 m short of the stop line for 30 consecutive seconds** while its approach's
posterior read 0.00 and no request existed anywhere in the controller.

**S6** (`atsc/control/shield.py`) bounds the latency from a standing request to
green, with a proof and an adversarial test. Three design points, each of which
was a bug first:

1. **It may only ever *narrow* the action set.** Written as an early return it
   widened it: cross-talk produces a request naming every phase on 16% of steps,
   and such a request re-permitted a green that maximum green had already ended,
   disabling S2 and gap-out outright. Cost: +3.3 vehicles of mean queue and
   emergency delay 26.8 → 33.7 s. Pinned by
   `test_s6_never_widens_the_action_set`.
2. **It must fall through to a *grantable* request.** Taking only the oldest due
   phase produced a starvation inversion: a request whose phase is never legal
   is never served, so its age never resets and grows without bound, after which
   it outranks every genuine request forever. Traced: a vehicle stood on the
   west approach for 145 s behind a stale phase-1 request aged past 20 s.
   Pinned by `test_s6_falls_through_to_a_grantable_request`.
3. **A request must survive a dropout.** The posterior flickers hardest at close
   range. Deleting the request the instant it dips resets the deadline clock, so
   it would never be reached in the one situation it exists to bound. A grace
   period slightly longer than one changeover raised effective recall 62% → 75%.

---

## 4. What it bought — including where it did not

### 4.1 The two changes interact, and neither works alone

Four arms, 12 shared seeds, 1800 s, 30 EV/h (`scripts/ablation_ev.py`):

| metric | baseline | + mast only | + S6 only | + both |
|---|---|---|---|---|
| EV stopped delay / veh (s) | 15.49 | 17.09 | **21.01** | 15.60 |
| EV full stops | 13.17 | 14.25 | 14.75 | 13.50 |
| background mean queue (veh) | 10.70 | 10.65 | **15.05** | 10.72 |
| worst movement wait (s) | 129.0 | 127.6 | 129.1 | **127.2** |
| teleports (gridlock) | 0 | 0 | **1** | 0 |

Paired against the baseline on identical seeds, the combined arm is
**+0.11 s, p = 0.96** — indistinguishable.

Two findings sit in that table, and the second is the interesting one.

**S6 on a bad detector is actively harmful.** Given the single-cardioid mast,
which raises a request with no emergency vehicle anywhere on 11.7% of steps, the
guarantee makes everything worse: emergency delay 15.5 → 21.0 s (+36%),
background queue 10.7 → 15.1 vehicles (+41%), and the only gridlocked seed in
the entire study. The mechanism is not subtle. A guarantee acts *decisively*: a
wrong request is not merely a wasted opportunity, it is a forced green on the
wrong approach, and the shield will not be argued out of it.

> **A service guarantee is only as good as the detector it consumes.** This is
> the result we would most want a reader to take away, because the instinct
> when emergency vehicles are waiting is to add a rule that forces the issue.
> Adding that rule on top of perception that is 88% precise made this
> intersection worse on every axis at once.

**The corrected mast is what makes the guarantee safe.** It halves the spurious
rate (11.7% → 6.3%) and the damage disappears entirely: queue returns to
baseline, the gridlock does not occur, and delay returns to 15.6 s.

### 4.2 With the mast fixed, the guarantee does pay — and the tight deadline is right

Sweeping the deadline on the two-element mast, 12 seeds
(`scripts/s6_sweep.py`, `results/s6_sweep.json`):

| S6 deadline | EV delay (s) | queue (veh) | worst (s) | P(wait > 30 s) | jam |
|---|---|---|---|---|---|
| **16 s** (shipped) | **15.60** | 10.72 | **85** | **0.176** | 0 |
| 25 s | 16.93 | 10.72 | 109 | 0.181 | 0 |
| 40 s | 17.06 | 10.73 | 91 | 0.207 | 0 |
| 60 s | 17.09 | 10.65 | 91 | 0.207 | 0 |
| off | 17.09 | 10.65 | 91 | 0.207 | 0 |

Against S6 disabled, the shipped 16 s deadline is better on **every** emergency
statistic — mean −1.49 s (−8.7%), worst case −6 s, P(wait > 30 s) 0.207 → 0.176
— at a background queue that is identical to within 0.07 vehicles, and with no
gridlock. Deadlines of 40 s and above are indistinguishable from off, which is
the expected behaviour of a bound so loose it never binds, and is a useful
check that the mechanism does what it claims.

So the default was not tuned to this table: 16 s is the smallest value the
timing envelope admits, and it turns out to be the best one.

### 4.3 Against the shipped baseline, the net is a wash

The honest bottom line: **+0.11 s, p = 0.96**. The two-element mast alone is
*worse* than baseline (17.09 s) because it is more conservative and raises fewer
requests; S6 recovers exactly that loss by guaranteeing the requests it does
raise. The pair lands back where it started on the mean.

An earlier draft of this document claimed a 38% reduction. That was measured on
one seed — 2029, the *demo* seed, chosen for exposition because it puts an
emergency vehicle in early — at 45 EV/h. Generalising from it was a mistake, and
the 12-seed paired test is what corrects it.

### 4.4 The demonstration seed, and why it is not evidence

Running the demo configuration head to head (seed 2029, 24 EV/h, 1270 s,
headless so the two arms are identical apart from the code):

| | EV stopped delay | EV stops | queue | jam |
|---|---|---|---|---|
| as shipped (single mast, no S6) | 14.44 s | 6 | 10.69 | 0 |
| SIREN-ARC v2 (mast + S6) | **9.78 s** | 6 | 11.05 | 0 |

A 32% reduction — on **one seed**. The paired standard deviation across the
twelve evaluation seeds is 5.79 s, so a single-seed difference of 4.7 s is well
inside the noise of this system, and §4.3 is the number that should be quoted.
It is recorded here only because it is the configuration the GUI demo runs, and
because a reader who watches the demo and then reads §4.3 deserves to know that
both are true and why they differ.

### 4.5 Retraining the policy on the corrected sensing

This project's own rule is that a policy is evaluated with the sensing it was
trained on: swapping at test time measures a train/test mismatch, not an
ablation. The shipped `siren_arc` predates the two-element mast, so we retrained
it under the corrected mast (200 episodes, 49.6 min) as `siren_arc_v2`.

| | policy `siren_arc` | policy `siren_arc_v2` |
|---|---|---|
| baseline sensing, no S6 | 15.15 | 18.11 |
| two-element mast + S6 | 15.78 | 16.79 |
| **effect of the fixes** | **+0.63** | **−1.32** |

The fixes help under the matched policy and hurt under the mismatched one,
which is the direction the rule predicts. But the retrained policy is *worse in
absolute terms* than the original, so we ship the original. One 200-episode run
is a single sample of a noisy process and we do not read more into it than that;
`models/siren_arc_v2.h5` is kept for inspection.

### 4.6 Why the residual delay is not a control problem

Decomposing the worst vehicle's 138 on-approach seconds by what the signal was
actually doing:

| | seconds |
|---|---|
| stopped while its own approach was **green** | 27 — capacity-bound |
| stopped while its own approach was **red** | 58 — priority-bound |
| moving | 53 |

Through most of that red the standing request named the *wrong phases* — the
vehicle was on the east approach (phases 2 and 3) while the request read
`[0, 1]`. Aggregate request recall is 75%, so roughly one poll in four either
misses the vehicle or names another arm.

That is the binding constraint, and it is a **sensing** problem, not a control
one. It is also consistent with this project's earlier finding that perfect
detection buys nothing over four microphones (p = 0.98) because the constraint
has moved to the intersection's capacity. Raising localisation accuracy is the
highest-value remaining work here; another control rule on top of a 75%-accurate
request will not help, and §4.1 shows it can hurt.

## 5. Defect three: the detector had never met the sound that fools it

### 5.1 The gap

Every negative in the training set was **broadband traffic noise**. That is an
easy negative for a harmonic detector: it is rejected almost regardless of
level. The hard negative -- a **road horn** -- did not exist anywhere in the
model, so the network was never asked to learn the distinction, and no
experiment in this repository could have detected that it had not.

The horn wins on every axis a level-based detector could use:

| | siren | road horn |
|---|---|---|
| source level at 1 m | 109-116 dB | 100-112 dB (UNECE R28) |
| spectrum | harmonic | harmonic |
| fundamentals | 420-1750 Hz (sweeping) | 320-620 Hz (fixed) |
| where it is | still 200 m away | **in the queue, metres from the mast** |
| **level at the mast** | 15.5 dB SNR at 200 m | **23 dB SNR at 30 m** |

**The confuser is the louder signal.** What separates them is not level but
*modulation*: a siren sweeps its fundamental, a horn does not. Over a
one-second frame a siren draws a moving ridge on the spectrogram and a horn
draws horizontal lines -- a cue invariant to distance, directivity and the
ambient floor. Horns are now synthesised physically (two diaphragms a third
apart, beating, with press-and-release envelopes), sounded by stopped vehicles
in the simulation at a rate tied to congestion, given their own detector class,
and rendered in the demo audio.

### 5.2 The bug that hid the fix

The retrained detector, evaluated through the existing readout, called a horn a
siren with probability **1.000 at every SNR above 0 dB**. That reads as a
network that cannot tell them apart. It was not.

`p_siren` was computed as `1 - P(none)`. That is correct only while every
non-silent class is an emergency class. With the confuser given its own label it
is wrong in the worst possible way: a detector that has confidently and
*correctly* identified a horn puts its mass on `horn`, so `1 - P(none)` reads
1.0. **The network had learned the distinction; the readout was discarding it.**

Summing the siren classes instead (`belief.siren_probability`, five call sites):

| SNR (dB) | true siren | horn, `1 - P(none)` | horn, siren-sum |
|---|---|---|---|
| -10..-5 | 0.989 | 0.937 | **0.126** |
| -5..0 | 0.999 | 0.997 | **0.019** |
| 0..5 | 1.000 | 1.000 | **0.001** |
| > 5 | 1.000 | 1.000 | **0.000** |

Broadband false alarms fell too (0.239 to 0.096), for the same reason: horn mass
had been counted as siren there as well. Pinned by
`test_siren_probability_excludes_the_horn_class` and
`test_fitted_horn_curve_rejects_horns_where_it_matters`; drawn in
`figures/fig_horn_discrimination`.

### 5.3 Closed loop: 8 shared seeds, 15 horn presses per queued vehicle-hour

Both horn arms are built from the *same* fitted error model and differ in
exactly one field -- the blind arm has its horn curve stripped, so horns fall
back to the siren curve. (Loading the archived pre-horn model instead was tried
first and is wrong: it differs in its siren curves, noise floor and range error
too, and measured that way the "blind" arm came out better on the first seed,
which is not a finding about horns at all.)

| | silent traffic | horns, blind | horns, aware |
|---|---|---|---|
| **spurious request rate** | 11.21% | **15.94%** | **11.32%** |
| EV stopped delay (s/veh) | 12.95 | 12.89 | 15.68 |
| background queue (veh) | 10.28 | 10.65 | 11.15 |
| teleports | 0 | 0 | 0 |

**The sensing claim holds and is significant.** Honking traffic costs a blind
detector 4.7 points of precision; horn-awareness recovers all of it, landing
within **0.11 points** of the silent-traffic reference (-4.62 pp, **p = 0.0014**,
6 of 8 seeds). The detector is genuinely rejecting the confuser in closed loop,
not merely on a held-out set.

**The control side is not established, and the point estimate is adverse.**
EV stopped delay is +2.79 s in the aware arm, but **p = 0.11** with a paired sd
of 5.37 s and only 2 of 8 seeds improving -- that is not a result, in either
direction. One hypothesis worth testing rather than asserting: a horn-triggered
request necessarily points at a *congested* approach, because that is where
honking happens, so a blind detector's false requests behave a little like a
queue-responsive controller and collect accidental benefit. If that is right,
the honest statement is that precision and delay are not the same objective, and
this repository's own §4 result -- that a guarantee amplifies whatever precision
it is given -- cuts both ways.

## 6. What is now guaranteed

Reported by `SafetyShield.guarantees()` and asserted in `tests/test_shield.py`:

| rule | bound | verified |
|---|---|---|
| S1 minimum green | 8 s (6 s under an emergency call) | unit |
| S2 maximum green | 55 s | unit |
| S3 bounded starvation | **180 s**, any movement | adversarial policy |
| S6 emergency service latency | **16 s** with 1 s polling (corridor active); **24 s** at the 5 s decision points | adversarial policy (max 21 s observed) |
| S6 composed with S3 contention | **88 s** = 16 + 4 x (8 + 5 + 3 + 2) with 1 s polling; **96 s** = 24 + 4 x 18 at decision points | derived + asserted (max 36 s observed) |

The composed bound was stated as 34 s and was **wrong**: it charged one
starvation service where the reachable worst case is one per phase, which is
the same `n_phases` factor S3's own proof carries and which S3 had already been
corrected for once. Measured against the old figure, an emergency request stood
49 s while S3 served three other phases in turn -- the shield behaving exactly
as specified, against a promise the arithmetic never supported. The number above
is the one the rule can keep. S3 now breaks its own ties toward a due emergency
phase where it can, which shortens the typical case without moving it.

S6 is ordered **below** S3 deliberately: a detector with any false-positive rate
must not be able to starve an ordinary movement, which is exactly the property
that makes classical hard preemption unsafe on a busy network. And S6 consumes a
*request*, never a posterior, so the safety argument does not depend on the
detector being correct — a wrong request costs one bounded service and cannot
violate S1, S2 or S3.

**244 tests pass.** The additions cover the mast array's arm-separation
requirement (asserted as separation rather than as a distance proxy, which is
what let the blind geometry through), the level term in the joint likelihood,
the steered pair's pattern loss, the per-engagement charging of the
false-preemption budget, and each of the S6 bugs above as a regression.

---

## 6b. The re-measurement, and two defects that had nothing to do with traffic

Everything above was measured before the headline table was re-run end to end.
It has since been re-run — 19 controllers, 12 seeds, 3600 s each, on the
corrected sensing — and the conclusions are reported in the README. Three of them changed, and they are listed here
because a document about negative results should not quietly drop the ones that
went the other way:

* **The corridor's contribution got much larger and became unambiguous.**
  Against the identical policy weights with no priority layer, median emergency
  stopped delay now falls 17.47 → 8.83 s and clean pass 0.108 → 0.305, both at
  p = 0.0001. The previous run put the same two differences at p = 0.09 and
  p = 0.07 and explained the non-significance by the *reference* being unstable:
  that checkpoint gridlocked two seeds in twelve, and the variance from them
  swamped a real median difference. This time the bare policy gridlocks none,
  the masking variance is gone, and the effect resolves. The corollary is that
  the previous claim — that the corridor's most defensible contribution was
  *preventing* those two gridlocks — has no support here, because there were
  none to prevent.

* **"Perfect detection would buy nothing" was half right.** Oracle sensing is
  indistinguishable from four microphones on emergency *delay* (p = 0.95) and
  decisively better on *clean pass* (0.491 against 0.305, p = 0.0001). The
  sensor is not the bottleneck on how long a vehicle waits and it is the
  bottleneck on whether it stops at all. The blanket version of the claim was
  wrong in a way that only splitting the two metrics reveals.

* **Calibration and the constrained controller are tail stories, not central
  ones.** Dropping the calibration map *improves* the median by 0.56 s
  (p = 0.19) and gridlocks two seeds where the calibrated scheduler gridlocks
  none. SIREN-CMDP matches SIREN-ARC on medians (8.75 against 8.83 s) and
  gridlocks one seed against none. In both cases the component earns its place
  in the tail, which is a weaker and more specific claim than "it is better".

### The two defects were in the harness, not the controller

Neither had anything to do with traffic, and between them they cost more
wall-clock than every experiment in this document.

**A path length.** Scenario files are written to
`<runs dir>/<regime>_<controller>/scenario_<controller>_<seed>.sumocfg`. Past
260 characters — Windows' `MAX_PATH` — SUMO cannot open a file Python has just
written successfully, because Python uses the long-path API and SUMO is a C++
binary using the legacy one. `Path.exists()` is `True`; SUMO reports *"Could not
access configuration"* naming the file that is provably there. TraCI then
**retries on a different port**, over and over, launching and losing a process
each time, so the client sits at 0 % CPU with no SUMO alive — a hang, not an
error. Because the overflow depends on the length of the controller's *name*, a
run directory 100 characters deep put `max_pressure` at 248 (fine) and
`max_pressure_preempt` at 264 (broken): every attempt died on the fifth
controller of nineteen and reported nothing. The length check now runs *before*
the first launch, so it is a clear error in nine seconds. A diagnostic for this
already existed in the file and was unreachable, sitting after the retry loop
that never terminated — which is worse than not having one.

**An empty argument.** `--baselines ''` is how a group that adds only learned
controllers suppresses the baselines. PowerShell drops an empty argument when it
builds a child command line, so argparse saw the option with no value and
refused. It is now an explicit `none` sentinel, which does not depend on
quoting.

### A third instance of the same mistake, in the viewer

The camera took three attempts, and the two failed ones repeat §2 and §5 exactly:
**the quantity that was measured was not the quantity that mattered.**

The report was that the view zoomed in on an arriving emergency vehicle and then
flew back out. `--follow-ev` used SUMO's `gui.trackVehicle`, so the first thing
checked was the zoom — instrumented over 3670 frames of a live run, `gui.getZoom`
returned a constant 125.23 before, during and after tracking, and
`gui.getBoundary` kept describing the full wide frame. The conclusion drawn was
that the complaint must be about *panning*, and a real panning defect was found
and fixed: following was released only when the scheduler dropped its target,
while the vehicle was matched by road id, which stops matching the moment it
enters the junction — so the camera chased already-served vehicles out of the
network and then snapped back 320 m.

That fix was correct and did not address the complaint. A screenshot of the same
behaviour showed a ten-metre scale bar and no junction on screen. `trackVehicle`
zooms the *rendered* view and reports nothing through TraCI; only the offset is
written back. The getters are what a test can see and the picture is what the
viewer sees, and for two rounds the getters were believed over a user saying
plainly that it was zooming.

Removing tracking exposed the second defect immediately. `setBoundary` *fits* the
box it is given into the window rather than filling it, so a square 520 m box in
a 2.39:1 window shows **1241.5 × 520 m** — the whole 640 m network plus margin.
The old docstring claimed a boundary in metres was "the same picture everywhere";
it is the same *box* everywhere and a different picture on every window shape.
The frame is now shaped to the window's measured aspect, so the longest visible
axis is always `2 × view_radius_m`:

| | requested | actually shown |
|---|---|---|
| before | 520 × 520 m | 1241.5 × 520 m |
| after | aspect-shaped | 380.0 × 159.2 m |

A third round followed, and it was caused by the fix for the first two. With
tracking gone the camera set the boundary on every frame, which meant the view
could not be zoomed at all — the scroll wheel moved it and the next frame put it
back. That is a worse defect than either of the originals, because it takes
control away from the person the picture is *for*, and no test would have caught
it: every frame was correct by the standard the tests applied. The camera now
detects a hand on the mouse and stands down for the rest of the run.

None of these affects a single reported number — the camera is presentational,
and the same seed produces byte-identical episode summaries before and after. It
is recorded because the failure mode is the one this whole document is about, and
because it took a user's screenshot to break it: no amount of further
instrumentation would have, since the instrument was the thing that was wrong.

**258 tests pass**, including fourteen added here. Eleven cover the camera: that
it acquires a vehicle only inside the frame, releases the moment that vehicle is
no longer upstream on the approach being served, keeps the junction on screen,
holds one single frame size across acquire and release, spans correctly at four
window aspect ratios, degrades gracefully on a TraCI without `getLaneID`, and
--- the one that matters most --- stands down permanently once the viewer has
touched the mouse. Two more pin the path-length check, and one the merge.

The lesson is the same one as §2 and §5, in a different register: **the failure
was not where the thinking was.** Three sessions were spent on the assumption
that a long evaluation kept dying because the session hosting it ended. It was
dying on its own, at the same controller every time, for a reason with no
connection to signals, sensing or reinforcement learning — and the evidence
distinguishing the two was sitting in a process list the whole time, showing
zero CPU and no simulator.

---

## 7. Reproducing this

```bash
python scripts/ablation_ev.py     --seeds 12   # the four-arm table (§4.1)
python scripts/s6_sweep.py        --seeds 12   # the deadline trade-off (§4.2)
python scripts/ev_latency.py      --seeds 12   # the per-vehicle distribution
python scripts/request_quality.py --seeds 6    # recall by distance band (§2)
python scripts/refit_error_model.py             # horn curve, no retraining (§5.2)
python scripts/horn_robustness.py --seeds 8   # silent / blind / aware (§5.3)
python scripts/make_figures.py                 # every figure, from those files
python tests/run_tests.py                      # 258 tests
```

Figures: `fig_mast_response`, `fig_ev_ablation`, `fig_request_quality`,
`fig_horn_discrimination`.

Watch it run, with the sirens audible:

```bash
python scripts/demo.py --follow-ev        # sirens play by default
```

---

## 8. If you take one thing from this

The instinct when emergency vehicles are seen waiting is to add a rule that
forces the issue. We added one, with a proof, and on the sensing that was in
place it made the intersection **worse on every axis at once** — including
producing the only gridlocked seed in the study.

What made the guarantee safe was not tuning it. It was making the requests it
acts on right more often. A guarantee converts detector precision directly into
intersection behaviour, in both directions, and it is worth measuring that
conversion before shipping the rule.

And then, if you have room for a second thing: **check the quantity the system
consumes, not the quantity you were reasoning about.** Both rounds of this
revision found a real defect, fixed it, verified the fix, and verified the wrong
thing. The first verified that the mast could *hear* a vehicle at the stop line
when what the controller needs is to know *which arm* it is on. The
`pred_spl` array was computed correctly, for every hypothesis, on every step,
and then used only to index an SNR bin — so it was live code by every test that
could be written about it, and dead by the only one that mattered.

The cheapest guard against both is the same one: write the test against the
property, not against a proxy for it. "Each mast is 20-200 m upstream and no two
are within 50 m" is a proxy. "The correct mast is louder than every wrong one,
everywhere on the arm, by more than the measurement noise" is the property, and
it fails on the geometry that shipped for two revisions.
