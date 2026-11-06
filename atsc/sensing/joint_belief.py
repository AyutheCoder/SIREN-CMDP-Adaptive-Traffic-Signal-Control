"""Joint, mutually-exclusive belief over which approach carries the siren.

Why the per-approach filter cannot work
---------------------------------------
Four microphones sit within a couple of hundred metres of one another. A siren
on the eastbound arm is plainly audible at all four masts. If each approach then
runs its own independent two-state HMM -- which is what ``belief.py`` does, and
what every per-approach detector we could find in the literature does -- all
four approaches independently convince themselves that a vehicle is inbound *on
them*. That is not a subtle effect. In the shipped demo the presence vector
routinely reads ``1.00 / 1.00 / 1.00 / 0.82`` while a single ambulance crosses,
and against ground truth the raw posterior is so over-confident that a reading
of 1.0 corresponds to a true probability of 0.46 (``models/presence_calibration.json``).

Calibration repairs the *units* of that number but it cannot repair the
*information*: after calibration the maximum attainable posterior is 0.46, so any
downstream decision rule that must compare a probability against a cost is
capped at acting on evidence no stronger than a coin flip. That single fact is
what kept the risk-calibrated corridor from opening until the vehicle was
already at the stop line.

The fix: cross-talk is the signal
---------------------------------
A siren is *one object*. Its position determines what every microphone hears --
level through spherical spreading, frequency-dependent atmospheric absorption
and the cardioid's off-axis rejection; range through each mast's own regression
head. So the four readings are not four independent questions with four
independent answers; they are four measurements of one latent position, and the
pattern *across* masts is exactly the localisation cue.

This module therefore replaces four independent Bernoulli filters with one
filter over the joint state

    S subset of approaches,  and for each a in S a range r_a on that arm,

truncated at two simultaneous sources. Three concurrent emergency vehicles
inside one 300 m detection radius is far outside anything the demand process
produces, and rather than assert that, the filter watches the boundary: the
posterior mass it puts on two-source hypotheses is tracked as
:attr:`JointBeliefTracker.two_source_mass`, and if that were routinely large the
truncation would be suspect.
The hypothesis space is small enough -- 1 + n*K + C(n,2)*K^2, i.e. about 2000
for four approaches and an eighteen-point position grid -- that inference is
*exact* over it: no sampling, no gating heuristic, around a millisecond per
decision epoch against a 5 s epoch.

The measurement model is not fitted here; it is the *same* model the surrogate
sensor draws from. For a hypothesised source position the predicted level at
every mast follows from :func:`atsc.sensing.siren_synth.received_spl_db`, the
predicted SNR from the ambient floor, and the predicted distribution of the
detector's own logit from :class:`~atsc.sensing.sensor.SensorErrorModel` --
the detection curve measured on the CNN's held-out predictions. A microphone
that reports a range when the hypothesis says nothing is audible there is scored
against a uniform clutter density, in the standard multi-target way. The filter
is therefore *self-calibrating* in the sense that matters: it inherits whatever
the detector's real reliability turned out to be, per SNR bin, rather than
assuming a hand-set false-alarm rate.

Predict is factorised (independent birth/survival per approach, survival
conditioned on the sign of the tracked range rate as before), update is joint,
and the posterior is projected back onto the factorised family by marginalising
-- assumed-density filtering, the same approximation that makes multi-Bernoulli
filters tractable. The projection is where the mutual exclusion does its work:
under a hypothesis that puts one source on the east arm, the north mast's loud
reading is *explained*, so it contributes no mass to "north is occupied".

Relation to ``association.py``
------------------------------
The association test this replaces was the right idea executed as a heuristic:
it picked the nearest-reporting mast as host, asked whether each other reading
was geometrically consistent with that one source, and damped the evidence of
those that were. It is greedy (one host at a time, never revisited), it scores
consistency only in range and never in level, and its output is a temper weight
rather than a probability -- so it cannot say *how likely* it is that the east
arm is the one. The joint filter subsumes it: the same geometric consistency is
present as the likelihood of the range readings, but every hypothesis is scored
against every reading simultaneously and the answer comes out as a posterior.
``use_association`` is kept as the ablation.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import AcousticConfig, BeliefConfig
from .belief import SIREN_CLASSES, ApproachBelief, BeliefTracker
from .sensor import SensorErrorModel, cardioid_gain_db, endfire_pair_gain_db
from .siren_synth import CLASS_TIMBRE, received_spl_db

CLUTTER_RANGE_DENSITY = 1.0 / 900.0
"""Density assigned to a range reading no hypothesised source explains.

Uniform over the 1--900 m interval the detector's range head can emit. This is
the standard clutter model of multi-target tracking, and it is what lets the
filter say "that mast is hearing something none of my hypotheses account for"
instead of silently forcing the reading onto the nearest source."""

MAX_SIMULTANEOUS = 2
"""Sources considered at once. See the module docstring."""


def _log_gauss(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = np.maximum(std, 1e-3)
    z = (x - mean) / std
    return -0.5 * z * z - np.log(std) - 0.5 * math.log(2.0 * math.pi)


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


@dataclass
class MicArray:
    """Fixed geometry of the microphone masts, plus the tables it implies.

    Source position is parametrised by **signed distance to the stop line along
    the arm**, ``s``: positive is upstream (a vehicle still approaching),
    the mast sits at whatever offset the network reports, and negative is a
    vehicle that has already crossed. That sign is not a detail. The single largest error mode of
    the previous filter was the audible tail *after* a vehicle passed -- the
    detector keeps firing, the level keeps rising for a moment, and a filter
    whose state space contains only "inbound at range r" has no hypothesis that
    explains a loud on-axis reading other than "inbound and very close". Giving
    the state space somewhere for a departed vehicle to *be* is what lets the
    posterior say it has gone.

    Built once at :meth:`~atsc.env.SumoSignalEnv.reset` and thereafter read
    only, so the per-epoch cost of the joint filter is a handful of array
    lookups rather than any trigonometry.
    """

    approaches: List[str]
    stop_line: Dict[str, Tuple[float, float]]
    pos: Dict[str, Tuple[float, float]]
    """Where each mast actually is, as discovered from the network."""
    axis: Dict[str, Tuple[float, float]]
    """Unit vector pointing *up* each approach, away from the junction."""
    front_back_db: float = 14.0
    dual_element: bool = True
    """Whether the masts are steerable two-element endfire pairs.

    This has to match the sensor that is actually running. The likelihood below
    asks "what level should mast ``i`` be reporting if the source were here?",
    and answering it with a single-cardioid model while the mast is steered
    misprices every hypothesis in the last ``mic_offset_m`` of each approach by
    the full front-to-back ratio."""
    detection_radius_m: float = 300.0
    max_range_m: float = 300.0
    past_range_m: float = 300.0
    """How far beyond the stop line the state space extends.

    It has to reach the whole way down the exit, not just across the junction.
    A vehicle that has cleared the north arm travels *up the south arm's axis*,
    straight past the south mast and on-axis to it, at a range that looks
    exactly like an inbound vehicle. With nowhere in the state space for that
    vehicle to be, the only hypothesis that explains a loud on-axis reading is
    an inbound one, and the filter invents an emergency vehicle on an arm that
    has just been cleared. Measured: truncating this at 70 m cost 9 points of
    precision at the 0.5 operating point."""

    s_grid: np.ndarray = field(init=False)
    inbound: np.ndarray = field(init=False)
    """``inbound[k]``: does grid point ``k`` sit upstream of the stop line?"""
    dist: np.ndarray = field(init=False)
    """``dist[a, k, i]``: metres from a source on arm ``a`` at grid point ``k``
    to microphone ``i``."""
    spl: np.ndarray = field(init=False)
    """``spl[c, a, k, i]``: received level at mic ``i`` from a class-``c`` source
    on arm ``a`` at grid point ``k``, directivity included."""
    fb_ratio: np.ndarray = field(init=False)
    """``fb_ratio[a, k, i]``: front-lobe minus back-lobe level at mic ``i``, dB.

    Zero throughout for a single-element mast. For the two-element mast this is
    the side-of-mast cue: its sign says whether a source at grid point ``k`` is
    upstream or downstream of mast ``i``, independently of how loud the source
    is or how far away it is."""

    def __post_init__(self) -> None:
        n = len(self.approaches)
        # Upstream points are log-spaced because the detector's range error is
        # log-normal, so resolution should be proportional; a linear grid would
        # spend most of its points where the estimate is least certain. The
        # handful of downstream points exist only to absorb departed vehicles.
        upstream = np.geomspace(4.0, self.max_range_m, 14)
        downstream = -np.geomspace(6.0, self.past_range_m, 7)[::-1]
        self.s_grid = np.concatenate([downstream, upstream])
        self.inbound = self.s_grid >= 0.0
        K = len(self.s_grid)

        self.dist = np.zeros((n, K, n), dtype=np.float64)
        self.spl = np.zeros((len(SIREN_CLASSES), n, K, n), dtype=np.float64)
        self.fb_ratio = np.zeros((n, K, n), dtype=np.float64)
        for ai, a in enumerate(self.approaches):
            lx, ly = self.stop_line[a]
            ax, ay = self.axis[a]
            for k, sdist in enumerate(self.s_grid):
                px, py = lx + ax * sdist, ly + ay * sdist
                for ii, b in enumerate(self.approaches):
                    bx, by = self.pos[b]
                    ux, uy = self.axis[b]
                    dx, dy = px - bx, py - by
                    d = max(math.hypot(dx, dy), 1.0)
                    self.dist[ai, k, ii] = d
                    cos_off = max(-1.0, min(1.0, (dx * ux + dy * uy) / d))
                    theta = math.acos(cos_off)
                    if self.dual_element:
                        front, back = endfire_pair_gain_db(theta, self.front_back_db)
                        gain = max(front, back)
                        self.fb_ratio[ai, k, ii] = front - back
                    else:
                        gain = cardioid_gain_db(theta, self.front_back_db)
                    for ci, cname in enumerate(SIREN_CLASSES):
                        self.spl[ci, ai, k, ii] = received_spl_db(cname, d, gain)

    @property
    def n(self) -> int:
        return len(self.approaches)

    @property
    def n_range_bins(self) -> int:
        return len(self.s_grid)


@dataclass
class _Hypothesis:
    """One entry of the enumerated hypothesis space (built once, then reused)."""

    sources: Tuple[Tuple[int, int], ...]
    """``(approach index, range bin)`` pairs. Empty means "nothing out there"."""


class JointBeliefTracker(BeliefTracker):
    """Multi-Bernoulli presence filter with an exclusive joint measurement update.

    Drop-in for :class:`~atsc.sensing.belief.BeliefTracker`: ``predict`` and the
    read-side API (``belief``, ``beliefs``, ``feature_vector``) are unchanged, so
    the observation vector the policy consumes has the same shape and meaning.
    Only the measurement update differs -- it takes *all* the readings at once
    through :meth:`update_joint` rather than one approach at a time.
    """

    def __init__(
        self,
        approaches: Sequence[str],
        cfg: Optional[BeliefConfig] = None,
        priority_map: Optional[Dict[str, float]] = None,
        detector_prior: float = 0.5,
        mics: Optional[MicArray] = None,
        error_model: Optional[SensorErrorModel] = None,
        acoustic: Optional[AcousticConfig] = None,
        detection_gate: str = "acoustic",
    ):
        # The joint filter's state is already signed distance to the stop line,
        # so it needs no offset correction.
        super().__init__(approaches, cfg, priority_map, detector_prior,
                         stop_line_offset_m=0.0)
        self.mics = mics
        self.error_model = error_model or SensorErrorModel.default()
        self.acoustic = acoustic or AcousticConfig()
        self._ambient_db = float(np.mean(self.acoustic.ambient_noise_db_range))
        self.use_level = bool(getattr(self.acoustic, "score_level", True))
        """Whether the likelihood reads the measured level. The ablation."""
        self.level_noise_db = max(
            float(getattr(self.acoustic, "level_noise_db", 2.5)), 1e-3)
        """Std of the level measurement, dB. Taken from the same config the
        sensor draws its noise from, so the filter's likelihood and the
        sensor's sampling distribution cannot drift apart."""
        self.fb_noise_db = max(float(self.acoustic.front_back_noise_db), 1e-3)
        """Std of the front-back ratio measurement, dB. Taken from the same
        config the sensor draws its noise from, so the filter's likelihood and
        the sensor's sampling distribution cannot drift apart."""
        self.detection_gate = detection_gate
        """Which sensor's reporting rule the likelihood should assume.

        ``acoustic`` -- a mast reports whatever is audible to it, which is any
        source inside its detection radius wherever it is. ``on_arm`` -- a mast
        reports only a source inbound on its *own* approach, which is what the
        oracle sensor does and what a V2I feed would do. The two imply
        completely different likelihoods for the same reading: under the
        acoustic gate a loud reading at three masts is evidence of one source,
        under the on-arm gate it is evidence of three."""

        self.disabled_mics: set = set()
        """Masts excluded from the likelihood because they have failed.

        A dead mast is silent, and silence under this measurement model is
        strong evidence *against* every hypothesis that predicts audibility
        there -- so leaving it in makes the filter confidently conclude that
        nothing is inbound anywhere near it. Dropping the term is the correct
        marginalisation over a missing observation, and it is the reason the
        controller degrades rather than fails when a microphone is lost."""
        self._hyps: List[_Hypothesis] = []
        self._hyp_src_mask: Optional[np.ndarray] = None
        self._hyp_range_idx: Optional[np.ndarray] = None
        self._hyp_flat: Optional[np.ndarray] = None
        self._hyp_n_src: Optional[np.ndarray] = None
        self.two_source_mass = 0.0
        """Posterior mass on the largest cardinality the state space contains.

        The hypothesis set is truncated at two simultaneous sources, and the
        honest way to defend that is not to assert it is enough but to watch the
        boundary: if the filter routinely puts mass on two-source hypotheses,
        the third is probably being missed. This is that number, as a running
        mean over epochs, and it is reported alongside the results rather than
        left as an assumption."""
        self._mass_sum = 0.0
        self._mass_n = 0
        self.last_posterior: Optional[np.ndarray] = None
        if mics is not None:
            self._build_hypotheses()

    # ------------------------------------------------------------------ setup
    def predict(self, dt: float) -> None:
        """Time update, with the range state allowed to go *negative*.

        :class:`~atsc.sensing.belief.BeliefTracker` floors range at zero, which
        is correct when range means "distance from the mast" and meaningless
        here, where it means "signed distance to the stop line". A vehicle that
        crosses does not stop at the stop line; it keeps going, and the whole
        point of the signed state space is that the filter can follow it there
        and stop calling it inbound. Flooring the track at zero pins a departing
        vehicle at the stop line, where every remaining loud frame keeps being
        explained as an imminent arrival -- which is exactly the stale tail that
        dominated the false positives.
        """
        super().predict(dt)
        if self.mics is not None:
            floor = -float(self.mics.past_range_m)
            for a in self.approaches:
                self._x[a][0] = max(self._x[a][0], floor)

    def attach(self, mics: MicArray, error_model: Optional[SensorErrorModel] = None) -> None:
        self.mics = mics
        if error_model is not None:
            self.error_model = error_model
        self._build_hypotheses()

    def _build_hypotheses(self) -> None:
        assert self.mics is not None
        n, K = self.mics.n, self.mics.n_range_bins
        hyps: List[_Hypothesis] = [_Hypothesis(())]
        for a in range(n):
            for k in range(K):
                hyps.append(_Hypothesis(((a, k),)))
        if MAX_SIMULTANEOUS >= 2:
            for a, b in itertools.combinations(range(n), 2):
                for k in range(K):
                    for l in range(K):
                        hyps.append(_Hypothesis(((a, k), (b, l))))
        self._hyps = hyps
        H = len(hyps)
        # Dense index tables so the per-epoch update is pure array arithmetic:
        # 2000 hypotheses in a Python loop cost 45 ms an epoch, which is more
        # than the rest of the environment step put together.
        mask = np.zeros((H, n), dtype=bool)
        ridx = np.full((H, n), -1, dtype=np.int64)
        flat = np.full((H, MAX_SIMULTANEOUS), -1, dtype=np.int64)
        for h, hyp in enumerate(hyps):
            for j, (a, k) in enumerate(hyp.sources):
                mask[h, a] = True
                ridx[h, a] = k
                flat[h, j] = a * K + k
        self._hyp_src_mask = mask
        self._hyp_range_idx = ridx
        self._hyp_flat = flat
        self._hyp_n_src = (flat >= 0).sum(axis=1)

    # ----------------------------------------------------------- measurement
    def _predicted_pattern(
        self, class_weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Level and distance each hypothesis predicts at every microphone.

        ``class_weights`` marginalises the unknown siren class: fire engines are
        7 dB louder at source than police, which is a factor of two in range if
        it is ignored, so the level table is averaged in the *power* domain over
        the current class posterior rather than evaluated at a nominal class.

        Fully vectorised. ``spl_flat[j, i]`` is the level at mic ``i`` from
        source-slot ``j = a*K + k``; a hypothesis is then a gather of at most
        ``MAX_SIMULTANEOUS`` slots followed by an elementwise max, because the
        sensor reports the *dominant* source at each mast.
        """
        mics = self.mics
        assert mics is not None and self._hyp_flat is not None
        n, K = mics.n, mics.n_range_bins

        w = class_weights.reshape(-1, 1, 1, 1)
        spl_mix = 10.0 * np.log10(np.sum(w * 10.0 ** (mics.spl / 10.0), axis=0))
        spl_flat = spl_mix.reshape(n * K, n)
        dist_flat = mics.dist.reshape(n * K, n)

        flat = self._hyp_flat                                   # (H, S)
        gathered_spl = np.where(
            (flat >= 0)[:, :, None], spl_flat[np.maximum(flat, 0)], -np.inf)
        gathered_dist = np.where(
            (flat >= 0)[:, :, None], dist_flat[np.maximum(flat, 0)], np.inf)
        fb_flat = mics.fb_ratio.reshape(n * K, n)
        gathered_fb = np.where(
            (flat >= 0)[:, :, None], fb_flat[np.maximum(flat, 0)], 0.0)

        best = np.argmax(gathered_spl, axis=1)                  # (H, n)
        rows = np.arange(flat.shape[0])[:, None]
        cols = np.arange(n)[None, :]
        pred_spl = gathered_spl[rows, best, cols]
        pred_dist = gathered_dist[rows, best, cols]
        # The front-back ratio belongs to whichever source *dominates* that
        # mast, which is the same source the level and range are taken from --
        # so it is gathered with the same index rather than recomputed.
        pred_fb = gathered_fb[rows, best, cols]
        empty = self._hyp_n_src == 0
        pred_spl[empty] = -np.inf
        pred_dist[empty] = np.inf
        pred_fb[empty] = 0.0
        return pred_spl, pred_dist, pred_fb

    def _log_likelihood(self, logits: np.ndarray, ranges: np.ndarray,
                        pred_spl: np.ndarray, pred_dist: np.ndarray,
                        fb: Optional[np.ndarray] = None,
                        pred_fb: Optional[np.ndarray] = None,
                        spl: Optional[np.ndarray] = None) -> np.ndarray:
        """log p(z | hypothesis) for every hypothesis, summed over microphones."""
        em = self.error_model
        radius = self.mics.detection_radius_m

        if self.detection_gate == "on_arm":
            # A source is reported by mast i only if it is inbound on arm i.
            inbound = self.mics.inbound
            audible = self._hyp_src_mask & np.where(
                self._hyp_range_idx >= 0, inbound[self._hyp_range_idx], False)
            audible = audible & (pred_dist <= radius)
        else:
            audible = pred_dist <= radius                  # (H, n)
        snr = pred_spl - self._ambient_db
        bins = np.clip(np.digitize(snr, em.snr_edges), 0, em.n_bins - 1)

        siren_mu = np.asarray(em.siren_logit_mean, dtype=np.float64)
        siren_sd = np.asarray(em.siren_logit_std, dtype=np.float64)
        mu = np.where(audible, siren_mu[bins], em.noise_logit_mean)
        sd = np.where(audible, siren_sd[bins], em.noise_logit_std)

        ll = _log_gauss(logits[None, :], mu, sd)
        if self.disabled_mics:
            keep = np.array([a not in self.disabled_mics for a in self.approaches])
            ll = ll * keep[None, :]

        # Range readings. A mast that emitted one is scored against the distance
        # the hypothesis predicts for it, in log space because that is the space
        # the detector's range head is noisy in; a mast whose reading no source
        # explains is scored against the clutter density instead. Masts that
        # emitted nothing contribute only through the logit term above.
        have_r = np.isfinite(ranges)
        if have_r.any():
            logr_sd = np.asarray(em.logr_std, dtype=np.float64)
            logr_b = np.asarray(em.logr_bias, dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                log_obs = np.log(np.maximum(ranges, 1.0))
                log_pred = np.log(np.maximum(pred_dist, 1.0))
            r_ll = _log_gauss(log_obs[None, :] - log_pred,
                              logr_b[bins], logr_sd[bins])
            # -log r converts the log-space density back to a density in metres,
            # so it is commensurate with the uniform clutter density.
            r_ll = r_ll - log_obs[None, :]
            r_ll = np.where(audible, r_ll, math.log(CLUTTER_RANGE_DENSITY))
            use_r = have_r.copy()
            if self.disabled_mics:
                for i, a in enumerate(self.approaches):
                    if a in self.disabled_mics:
                        use_r[i] = False
            ll = ll + np.where(use_r[None, :], r_ll, 0.0)

        # Level. The strongest localisation cue the array produces, and until
        # now the filter did not read it: ``pred_spl`` was computed only to
        # pick an SNR bin, and everything that separated one hypothesis from
        # another came from the detector logit and the range head.
        #
        # That works only while the detector is unsaturated. It is not: at the
        # ranges that matter every mast within earshot reports p_siren = 1.00,
        # so the logit term is identical under "the source is on the east arm"
        # and "the source is on the south arm" and contributes nothing to
        # telling them apart, and the range head's error is tens of metres.
        # The levels are not identical -- traced on a stalled emergency vehicle
        # nine metres short of the east stop line, the four masts read 28, 17,
        # 12 and -6 dB SNR, which is a fifteen-decibel separation and exactly
        # the answer -- and the filter was throwing all of it away. It spent
        # fifty-seven consecutive seconds holding a green for a phantom on the
        # south arm while the vehicle stood at a red on the east one.
        #
        # Scored only where the hypothesis says the mast can hear the source:
        # under a hypothesis that predicts nothing audible there, the level is
        # the ambient floor and says nothing about where anything is. The
        # detection term above is what penalises a hypothesis that fails to
        # explain a loud mast.
        if spl is not None and self.use_level:
            have_spl = np.isfinite(spl)
            if have_spl.any():
                spl_ll = _log_gauss(spl[None, :], pred_spl, self.level_noise_db)
                spl_ll = np.where(audible, spl_ll, 0.0)
                use_spl = have_spl.copy()
                if self.disabled_mics:
                    for i, a in enumerate(self.approaches):
                        if a in self.disabled_mics:
                            use_spl[i] = False
                ll = ll + np.where(use_spl[None, :], spl_ll, 0.0)

        # Front-back ratio. The one measurement in this filter that is about
        # *geometry* rather than about level, and the only one that separates a
        # vehicle 20 m short of the stop line from one 20 m past it -- those two
        # are equidistant from a mast mounted between them and therefore
        # identical in level and in range, which is why the previous filter
        # reset its track at exactly the moment the corridor needed it most.
        #
        # Scored only where the hypothesis says the mast can hear the source at
        # all: with nothing audible the two lobes both see ambient and their
        # difference carries no information about where anything is.
        if fb is not None and pred_fb is not None:
            have_fb = np.isfinite(fb)
            if have_fb.any():
                fb_ll = _log_gauss(fb[None, :], pred_fb, self.fb_noise_db)
                fb_ll = np.where(audible, fb_ll, 0.0)
                use_fb = have_fb.copy()
                if self.disabled_mics:
                    for i, a in enumerate(self.approaches):
                        if a in self.disabled_mics:
                            use_fb[i] = False
                ll = ll + np.where(use_fb[None, :], fb_ll, 0.0)
        return ll.sum(axis=1)

    def _log_prior(self) -> np.ndarray:
        """log P(S, s) from the factorised predicted presence and range tracks."""
        mics = self.mics
        n, K = mics.n, mics.n_range_bins
        pres = np.array([self._presence[a] for a in self.approaches])
        pres = np.clip(pres, 1e-6, 1.0 - 1e-6)

        # Position prior per approach: the Kalman track when there is one, a
        # uniform arm otherwise, always floored so a lost track can be
        # reacquired without waiting for the prior to drift back.
        rprior = np.zeros((n, K), dtype=np.float64)
        for ai, a in enumerate(self.approaches):
            if self._tracked[a]:
                r_hat = float(self._x[a][0])
                sd = float(np.sqrt(max(self._P[a][0, 0], 25.0)))
                w = np.exp(-0.5 * ((mics.s_grid - r_hat) / sd) ** 2)
                w = w / max(w.sum(), 1e-12)
                rprior[ai] = 0.85 * w + 0.15 / K
            else:
                rprior[ai] = 1.0 / K

        absent = np.log(1.0 - pres)
        # log-odds of occupancy plus the position prior, per (approach, grid) slot.
        slot = (np.log(pres) - absent)[:, None] + np.log(np.maximum(rprior, 1e-12))
        slot_flat = slot.reshape(n * K)
        flat = self._hyp_flat
        contrib = np.where(flat >= 0, slot_flat[np.maximum(flat, 0)], 0.0)
        return absent.sum() + contrib.sum(axis=1)

    # ---------------------------------------------------------------- update
    def update_joint(self, readings: Dict[str, object]) -> Dict[str, float]:
        """One exact joint measurement update from all microphones at once.

        ``readings`` maps approach -> :class:`~atsc.sensing.sensor.SensorReading`.
        Returns the marginal presence per approach (also written into the
        filter state, which is what makes this assumed-density filtering).
        """
        if self.mics is None:
            raise RuntimeError("JointBeliefTracker needs a MicArray; call attach()")

        n = self.mics.n
        logits = np.array([_logit(getattr(readings[a], "p_siren", 0.0))
                           for a in self.approaches])
        ranges = np.array([
            (lambda d: float(d) if d is not None and math.isfinite(float(d)) else np.nan)
            (getattr(readings[a], "distance_m", None))
            for a in self.approaches])

        # Class marginal, pooled across approaches: the level table needs a
        # class and we do not yet know which approach hosts the source.
        cw = np.zeros(len(SIREN_CLASSES))
        for a in self.approaches:
            for ci, c in enumerate(SIREN_CLASSES):
                cw[ci] += self._class[a][c]
        cw = cw / max(cw.sum(), 1e-12)

        fb = np.array([
            (lambda v: float(v) if v is not None and math.isfinite(float(v)) else np.nan)
            (getattr(readings[a], "front_back_db", None))
            for a in self.approaches])

        spl = np.array([
            (lambda v: float(v) if v is not None and math.isfinite(float(v)) else np.nan)
            (getattr(readings[a], "spl_db", None))
            for a in self.approaches])

        pred_spl, pred_dist, pred_fb = self._predicted_pattern(cw)
        # Scaling the log-likelihood is the joint filter's form of the same
        # tempering the per-approach filter applies to its likelihood ratio:
        # a reference interval's worth of evidence however often it is sampled.
        # It is the likelihood alone -- the prior already carries the elapsed
        # time through the transition model.
        log_post = self._log_prior() + self.evidence_weight * self._log_likelihood(
            logits, ranges, pred_spl, pred_dist, fb, pred_fb, spl)
        log_post -= log_post.max()
        post = np.exp(log_post)
        post /= max(post.sum(), 1e-300)
        self.last_posterior = post
        two = float(post[self._hyp_n_src >= 2].sum()) if self._hyp_n_src is not None             else 0.0
        self._mass_sum += two
        self._mass_n += 1
        self.two_source_mass = self._mass_sum / max(self._mass_n, 1)

        # --- project back onto the factorised family ---
        # "Presence" is the event the controller can act on: a vehicle still
        # upstream of the stop line. Mass that the filter has placed *past* the
        # stop line is a vehicle it believes has already crossed, and crediting
        # that to presence is precisely the stale-tail failure the signed grid
        # exists to remove.
        inbound = self.mics.inbound
        occ = self._hyp_src_mask & np.where(
            self._hyp_range_idx >= 0, inbound[self._hyp_range_idx], False)
        marg = (post[:, None] * occ).sum(axis=0)                 # (n,)
        any_marg = (post[:, None] * self._hyp_src_mask).sum(axis=0)
        for ai, a in enumerate(self.approaches):
            self._presence[a] = float(np.clip(marg[ai], 1e-6, 1.0 - 1e-6))

        # --- range: posterior mean and variance on the arm, as a pseudo-measurement ---
        grid = self.mics.s_grid
        for ai, a in enumerate(self.approaches):
            sel = self._hyp_src_mask[:, ai]
            if not sel.any() or any_marg[ai] < 1e-6:
                continue
            w = post[sel]
            k = self._hyp_range_idx[sel, ai]
            wsum = w.sum()
            if wsum <= 1e-12:
                continue
            w = w / wsum
            r_mean = float(np.dot(w, grid[k]))
            r_var = float(np.dot(w, (grid[k] - r_mean) ** 2))
            # The pseudo-measurement is only as good as our confidence that the
            # arm is occupied at all: an approach at presence 0.2 must not drag
            # its track as hard as one at 0.95. Inflating R by 1/presence is the
            # standard probabilistic-data-association weighting.
            r_var = max(r_var, 25.0) / max(any_marg[ai], 1e-3)
            self._kalman_range_update(a, r_mean, r_var,
                                      rate_hint(readings[a]))

        # --- class posterior, gated by presence, per approach as before ---
        for a in self.approaches:
            cp = getattr(readings[a], "class_probs", None)
            if cp:
                self._class_update(a, cp)

        return {a: self._presence[a] for a in self.approaches}

    # -------------------------------------------------------------- internals
    def _kalman_range_update(self, approach: str, z: float, R: float,
                             rate_hint_ms: Optional[float]) -> None:
        floor = -float(self.mics.past_range_m) if self.mics is not None else 1.0
        z = float(np.clip(z, floor, 2.0 * self.DEFAULT_RANGE_M))
        if not self._tracked[approach]:
            self._x[approach] = np.array(
                [z, rate_hint_ms if rate_hint_ms is not None else self.DEFAULT_RATE_MS])
            self._P[approach] = np.diag([R, 144.0])
            self._tracked[approach] = True
        else:
            H = np.array([[1.0, 0.0]])
            x, P = self._x[approach], self._P[approach]
            y = z - float(H @ x)
            S = float(H @ P @ H.T) + R
            K = (P @ H.T / S).reshape(2)
            self._x[approach] = x + K * y
            self._P[approach] = (np.eye(2) - np.outer(K, H)) @ P
        if rate_hint_ms is not None and self._x[approach][1] * rate_hint_ms < 0:
            self._x[approach][1] *= 0.5
        self._x[approach][0] = max(self._x[approach][0], floor)
        if self._presence[approach] < 0.05:
            self._x[approach] = np.array([self.DEFAULT_RANGE_M, self.DEFAULT_RATE_MS])
            self._P[approach] = np.diag([self.DEFAULT_RANGE_M ** 2, 100.0])
            self._tracked[approach] = False

    def _class_update(self, approach: str, class_probs: Dict[str, float]) -> None:
        current = self._class[approach]
        weight = self._presence[approach]
        updated, total = {}, 0.0
        for c in SIREN_CLASSES:
            lik = max(class_probs.get(c, 0.0), 1e-6)
            blended = (1.0 - weight) * current[c] + weight * current[c] * lik
            updated[c] = blended
            total += blended
        self._class[approach] = {c: v / max(total, 1e-12) for c, v in updated.items()}

    def belief(self, approach: str) -> ApproachBelief:
        b = super().belief(approach)
        b.signed_range = True
        return b

    # ------------------------------------------------------------------ stats
    def hypothesis_report(self, top: int = 3) -> List[Tuple[str, float]]:
        """Human-readable top hypotheses, for the demo HUD and for debugging."""
        if self.last_posterior is None:
            return []
        order = np.argsort(-self.last_posterior)[:top]
        out = []
        for h in order:
            hyp = self._hyps[int(h)]
            if not hyp.sources:
                label = "nothing"
            else:
                label = " + ".join(
                    f"{self.approaches[a]}@{self.mics.s_grid[k]:+.0f}m"
                    for a, k in hyp.sources)
            out.append((label, float(self.last_posterior[int(h)])))
        return out


def rate_hint(reading: object) -> Optional[float]:
    v = getattr(reading, "radial_velocity_ms", None)
    return None if v is None else float(v)
