# Where this sits in the literature

Twenty-one papers, grouped by the assumption each one makes about how the
controller finds out that an emergency vehicle exists. That is the axis this
project is organised around, because it is the assumption that decides
everything downstream: what can be guaranteed, what has to be inferred, and
what the failure modes look like when the inference is wrong.

**A note on what is and is not compared here.** The tables below compare
*assumptions, mechanisms and guarantees*, not headline numbers. Published
travel-time reductions in this field are not commensurable across papers —
CityFlow grid networks, SUMO corridors, Hangzhou and Jinan real-world traces,
and this project's generated four-approach intersection have different
saturation levels, different demand processes and different emergency-vehicle
rates, so "20% better than IntelliLight" from one paper and "19% better than
PressLight" from another cannot be chained, and neither can be compared against
a number measured here. Where a paper's own reported figure is quoted it is
labelled as such and attributed to its own baseline and benchmark. The
quantitative claims *this* project makes are confined to
[`docs/RESULTS.md`](RESULTS.md), where every arm was run on shared seeds through
identical code.

---

## 1. The dominant assumption: the vehicle's position is given

Most emergency-vehicle work — and effectively all the reinforcement-learning
work — starts from a known EV position, supplied by GPS, V2I/DSRC, or by reading
the simulator's state directly.

| # | Work | What it assumes about detection | What it contributes |
|---|---|---|---|
| 1 | **EMVLight** (Su et al., ICML-W 2021 / AAAI 2022) [[arXiv](https://arxiv.org/abs/2109.05429)] | EV position and route known to every agent | Decentralised multi-agent RL that couples EV *routing* with signal preemption; pressure-based reward; reduces EV travel time **and** background travel time |
| 2 | **EMVLight** (extended, *Transportation Research Part C*, 2022) [[arXiv](https://arxiv.org/abs/2206.13441)] [[TR-C](https://www.sciencedirect.com/science/article/abs/pii/S0968090X22003680)] | as above | Multi-class agents, policy sharing, spatial discounting; the strongest RL-based EV-preemption baseline in the literature |
| 3 | **Informed Traffic Signal Preemption for Emergency Vehicles** (FLAIRS 2024) [[link](https://journals.flvc.org/FLAIRS/article/view/135261)] | Tracked EV location, predicted trajectory | Uses *predicted* trajectories rather than instantaneous position to time the green earlier |
| 4 | **A Collaborative Control Framework** (Long et al., *J. Adv. Transp.* 2024) [[link](https://onlinelibrary.wiley.com/doi/abs/10.1155/2024/9691419)] | Known EV state | Explicitly minimises the collateral cost to ordinary vehicles — the same trade-off this project prices, solved with a different mechanism |
| 5 | **Emergency Preemption Without Online Exploration: A Decision Transformer Approach** (2026) [[arXiv](https://arxiv.org/html/2603.22315)] | Known EV state | Offline sequence modelling, removing the need for online exploration during preemption |
| 6 | **RAG-Enhanced Distributed LLM Agents for TSC with Emergency Vehicles** (2025) [[link](https://www.researchgate.net/publication/397087983)] | Known EV state | LLM agents with retrieval for generalisation across intersections |
| 7 | **Virtual Traffic Police: LLM-Augmented TSC for Unforeseen Incidents** (2026) [[arXiv](https://arxiv.org/html/2601.15816v1)] | Incident reported, not sensed | Argues that single-incident-type methods (EV priority among them) do not generalise |
| 8 | Deployed practice: **Opticom-style EVP**, GPS/DSRC preemption [[Econolite](https://www.econolite.com/application-areas/emergency-vehicle-preemption/)], [US 5,926,113](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/5926113), [US 7,327,280](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7327280) | Line-of-sight IR emitter or differential GPS on every vehicle, receiver in every cabinet | The operational reference point: detect → enter preemption sequence → run it → dwell → return |

**What this project changes.** The only emergency sensor is a calibrated
microphone per approach. Nothing tells the controller where the vehicle is, and
a microphone cannot hear turn intent, so an entire approach must be committed to
on partial information. Removing the position assumption is what forces every
other design decision in this repository, and it is also what makes the results
comparable to a deployment with no vehicle-side equipment at all.

**What this project keeps from #8.** Deployed preemption *runs the sequence*: a
detected call is not re-priced against cross-street delay every second. The
measured failure of the previous version of this code was precisely that it
had no such property — see §4.

---

## 2. Acoustic siren detection: strong perception, no control loop

This literature detects sirens well and stops there. None of it closes the loop
onto a signal controller, and none of it reports what its false-positive rate
does to an intersection.

| # | Work | Task | Reported result (its own benchmark) |
|---|---|---|---|
| 9 | **Listening for Sirens: Locating and Classifying Acoustic Alarms in City Scenes** (2018) [[arXiv](https://arxiv.org/pdf/1810.04989)] | Stereo spectrogram → U-Net semantic segmentation → classify | Separates alarm from background before classifying |
| 10 | **Emergency Vehicles Audio Detection and Localization in Autonomous Driving** (2021) [[arXiv](https://arxiv.org/abs/2109.14797)] | Two cheap microphones, detect + bearing | Recall 99.16%; median angular error 9.64°, mean 19.18° |
| 11 | **Acoustic Features for Deep-Learning Siren Detection: An Evaluation Study** (2021) [[link](https://www.researchgate.net/publication/355112750)] | Which features matter | Systematic feature comparison for siren detection |
| 12 | **Acoustic Based EV Detection Using an Ensemble of Deep Learning Models** (2023) [[link](https://www.researchgate.net/publication/367591515)] | 3-class siren ID (ambulance / fire / police) | Ensemble improves per-class accuracy |
| 13 | **SACNN — self-attention CNN on a large siren + road-noise corpus** (*Expert Systems with Applications*, 2024) [[link](https://www.sciencedirect.com/science/article/abs/pii/S0957417424004731)] | Detection at scale | EfficientNet + 1D-CNN hybrid with self-attention |

**What this project changes.** Four things, all of which only appear once
detection is wired to a controller:

1. **Cross-talk is treated as the localisation cue, not as noise.** Four masts
   within a couple of hundred metres all hear one siren, and each correctly
   answers "is a siren audible here?" with *yes*. Four independent per-approach
   filters therefore light **2.5 arms when exactly one is occupied, at precision
   0.27** (measured against ground truth over six hours). `joint_belief.py`
   replaces them with one exact filter over the joint state — which arm, and
   where on it — over 2731 hypotheses, ~1 ms against a 5 s decision epoch.
2. **The mast geometry is part of the perception problem.** A single cardioid
   mounted upstream of the stop line and aimed away from the junction is *deaf
   over the last 70 m of every approach* — the stretch where the vehicle is
   queued at the red. See §3; this is the defect that produced the behaviour
   this revision was opened to fix.
3. **The posterior is calibrated before anything numeric is done with it.** A
   raw filter reading of 1.00 corresponds to a true probability near 0.46, so
   any rule comparing a posterior against a cost is meaningless without the
   monotone correction in `sensing/calibration.py`.
4. **The detector is tested against the sound that actually fools it.** Papers
   #9–#13 all report separation of sirens from *background noise*, which is an
   easy negative — broadband, and rejected almost regardless of level. The hard
   negative is a **road horn**: harmonic, overlapping the low siren band, and
   sounded from the queue metres from the mast while the siren that matters is
   still 200 m away. Measured on this geometry a horn arrives at **23 dB SNR
   against an approaching ambulance's 15.5 dB** — the confuser is the *louder*
   signal, so no level threshold separates them. What does separate them is
   modulation: a siren sweeps its fundamental and a horn does not. Horns are
   synthesised physically, sounded by stopped vehicles at a rate tied to
   congestion, given their own detector class, and rendered in the demo audio.
   With the readout corrected to sum the siren classes rather than take
   `1 − P(none)`, horn-as-siren confusion falls from 1.000 to 0.001 above 0 dB,
   and closed-loop precision under honking traffic returns to within 0.11 points
   of the silent-traffic reference (p = 0.0014). See
   [`RESULTS.md`](RESULTS.md) §5.

---

## 3. What the two-element mast contributes

**The defect.** Each mast sits `mic_offset_m = 70 m` upstream of its stop line
as a single cardioid aimed *up* the approach. Everything between the mast and
the line is therefore behind it, in the rear null. Evaluated on the shipped
geometry (`AcousticConfig`, ambient 55–80 dB):

| EV distance from stop line | Single cardioid | Steered two-element pair |
|---|---|---|
| 300 m | 63.5 dB | 63.5 dB |
| 71 m (at the mast) | 112.0 dB | 112.0 dB |
| 30 m | 65.7 dB | **79.7 dB** |
| **1 m (at the stop line)** | **60.8 dB** | **74.8 dB** |

The single-element response is not merely weak at the stop line, it is
**inverted**: a vehicle 1 m from the line is *quieter* than the same vehicle
300 m away, and sits at or below the ambient floor. Measured per-second against
ground truth, the consequences were the whole complaint — presence collapsing
from 1.00 to 0.00 with the vehicle 15 m short of the line, a range estimate
reading 65 m with the vehicle 1 m away, and the green handed to a conflicting
approach with the ambulance stationary at the stop line.

**The fix, and why it is not extra hardware.** A compact endfire pair is steered
by the *sign* of the delay between its two capsules; flipping that sign rotates
the cardioid through 180°. The front-back ratio the config already assumed
(14 dB, "consistent with a compact two-element endfire array") therefore implies
two capsules that were only ever being read once. Reading them twice gives:

- the **steered level** — the louder lobe, which removes the null; and
- the **front-minus-back ratio**, a *new observable*. Taken at the same instant
  through the same preamp on the same mast, source level, range and ambient
  floor all cancel, leaving geometry: the sign says which side of the mast the
  source is on. That is the one thing level alone can never say, because a mast
  mounted partway up the approach makes level **non-monotone** in distance to
  the stop line — 85 dB is equally consistent with 90 m and with 50 m.

Against papers #9–#13 this is a different kind of contribution: not a better
classifier, but a mast geometry and a second observable chosen so that the
quantity the *controller* needs is observable at all. Measured over six seeds,
the spurious-request rate falls **11.7% → 6.3%**. The direction is worth noting:
the filter does not become more sensitive but better able to *reject*, so recall
falls in most distance bands while precision roughly doubles. Precision is the
quantity that matters here, because the guarantee in §4 acts on a request
decisively — see [`RESULTS.md`](RESULTS.md) §4.1, where the same guarantee on the
*imprecise* mast makes every metric worse.

---

## 4. Safe RL and formal guarantees

| # | Work | Guarantee | Scope |
|---|---|---|---|
| 14 | **SafeLight** (2023) [[arXiv](https://arxiv.org/pdf/2211.10871)] | Collision-free signal control | Safety of the *signal*, not service of any movement |
| 15 | **Provably Safe RL: Conceptual Analysis, Survey, Benchmarking** (2022) [[arXiv](https://arxiv.org/pdf/2205.06750)] | Taxonomy of shielding / action projection | The framework this project's shield instantiates |
| 16 | **Constrained TSC under competing public-transport priority requests via safe RL** (*ESWA*, 2025) [[link](https://www.sciencedirect.com/science/article/pii/S0957417425012989)] | CMDP with a cost estimator; hard queue constraints | Closest in spirit: competing priority requests under constraints — but priority requests are *known*, not sensed |
| 17 | **LibSignal** (2022) [[arXiv](https://arxiv.org/pdf/2211.10649)] | — | Benchmark library; the reproducibility standard this field is converging on |

**What this project changes.** The shield already proved S1 (minimum green),
S2 (maximum green) and S3 (bounded starvation, `W_max = 180 s`, verified
adversarially in `tests/test_shield.py`). What it did not have — and what no
paper in this table has — is a **bounded worst-case latency from a detected
emergency request to green**:

> **S6.** A standing emergency request naming phase set `P` is granted one of
> those phases within `ev_service_deadline_s`, provided the request persists.
> With `g_ev + poll + yellow + all_red = 14 s` on the shipped timings, the
> deadline is 16 s while the corridor polls the shield every second (24 s at the
> 5 s decision points between corridor calls); the shield refuses any value
> below the floor rather than promising what the change interval forbids.

Three properties make this more than a re-labelled preemption rule:

- **It sits *below* S3, not above it.** A movement about to breach its 180 s
  starvation bound is still served first. Inverting that order is exactly what
  makes classical hard preemption (#8) unsafe on a busy network — a detector
  with any false-positive rate could otherwise starve an approach without limit.
  The composed worst case, `W_ev + (g_min + dt + T)`, is reported by
  `SafetyShield.guarantees()` and asserted in the tests.
- **It consumes a *request*, never a posterior.** The safety argument therefore
  does not depend on the detector being right. A wrong request costs cross
  traffic one bounded service; it cannot violate S1, S2 or S3.
- **It is raised on evidence, not on economics.** This is the subtle part. The
  risk-calibrated corridor (§5) is an *option* — at every poll it may decline,
  defer, or be outbid by the cross-street queue, and its correct answer is
  sometimes "no" for a vehicle that is genuinely there. Feeding S6 the same
  condition would make the guarantee vacuous, promising only what the economics
  had already chosen to do. So the request is raised on the calibrated posterior
  alone, and the vehicle is then owed a green whatever the queue is doing.

---

## 5. Conditional priority and the cost of preempting

The transit-signal-priority literature has thought hardest about *when priority
should be refused*, and this project's corridor is closer to that tradition than
to EV preemption.

| # | Work | Mechanism |
|---|---|---|
| 18 | **Conditional TSP** [[eScholarship](https://escholarship.org/uc/item/7104w1jz)] | Priority granted only when the bus is late — a *state-dependent* condition rather than a fixed rule |
| 19 | **Person-based TSP optimisation** [[eScholarship](https://escholarship.org/uc/item/0t4336f3)] | Optimise total *person* delay, so priority is priced against the people it delays |
| 20 | **TSP accommodating conflicting requests under CV** (*TR-C* 2016) [[link](https://www.sciencedirect.com/science/article/abs/pii/S0968090X16300614)] | Resolves multiple simultaneous priority requests — the problem EV preemption reviews list as open |
| 21 | **OCC-MP: max-pressure prioritising transit and HOV** (2024) [[arXiv](https://arxiv.org/pdf/2406.19269)] | Occupancy-weighted max-pressure |

**What this project changes.** Two things the EV literature generally does not do,
borrowed from this tradition and made explicit:

- **The commitment threshold is derived, not tuned.** Published systems (and this
  project's own earlier overlay) pick a number — 0.5, 0.6, 0.58 — and tune it. A
  fixed threshold answers a decision problem with a constant: a 0.55 posterior is
  worth acting on when the cross street is empty and is not when forty cars are
  queued on it. The corridor instead preempts iff `p·(1+λ)·B > C` in
  vehicle-seconds, so `θ* = C / ((1+λ)B)` is recomputed every poll from the queue
  state. Anticipation falls out of it rather than being added as a lead-time
  window.
- **Concurrent conflicting requests are scheduled exactly.** With at most four
  phases, every ordering of every prefix is at most 64 evaluations, so the
  sequencing minimising priority-weighted lateness plus background delay is
  solved optimally rather than heuristically. At the demo's emergency rate,
  concurrent committed requests occur in roughly one preemption epoch in six.

---

## Summary: the combination that is not in any single row above

| Property | #1–#8 EV preemption | #9–#13 siren detection | #14–#17 safe RL | #18–#21 conditional priority | **This work** |
|---|---|---|---|---|---|
| EV position assumed known | ✅ | — | — | ✅ | ❌ (microphone only) |
| Detection uncertainty propagated to control | ❌ | n/a | ❌ | partly | ✅ (calibrated posterior) |
| Cross-mast interference modelled as signal | ❌ | ❌ | — | — | ✅ (exact joint filter) |
| Commitment threshold derived from state | ❌ | — | — | ✅ | ✅ |
| Concurrent EV requests scheduled optimally | ❌ | — | — | ✅ (transit) | ✅ |
| Bounded starvation guarantee | ❌ | — | ✅ | ❌ | ✅ (S3, 180 s) |
| **Bounded EV service latency guarantee** | ❌ | — | ❌ | ❌ | ✅ (**S6**, 16 s / 24 s) |
| Guarantee holds under adversarial detector | ❌ | — | partly | ❌ | ✅ (S6 consumes a request, not a belief) |

The claim is the *conjunction*, not any single cell: a controller that hears its
emergency vehicles rather than being told about them, propagates the resulting
uncertainty all the way into a decision-theoretic commitment rule, and still
carries a hard worst-case bound on both ordinary-movement starvation and
emergency service latency — bounds that hold for any detector output, including
an adversarial one.

## Honest limitations

- **Single intersection.** #1–#2 solve network-level routing and coordination;
  this does not. The corridor overlay is per-intersection.
- **Simulated acoustics.** The waveform, propagation (spherical spreading +
  ISO 9613-1 absorption per harmonic) and directivity are modelled, not
  recorded. #9–#13 use real audio. The surrogate is validated against the
  end-to-end CNN (`scripts/validate_surrogate.py`), but a real deployment would
  need the detector retrained on field recordings.
- **No head-to-head against #1–#2.** EMVLight is not reimplemented here, so no
  numerical comparison against it is claimed. The baselines actually run are
  Webster fixed-time, SUMO actuated, max-pressure, greedy-queue and
  Opticom-style hard preemption, all through the identical shield and runner.
