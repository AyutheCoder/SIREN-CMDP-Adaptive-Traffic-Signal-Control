"""Recursive Bayesian belief over emergency-vehicle presence, per approach.

The detector is a *noisy* sensor: it hallucinates sirens in bus noise, misses
distant ones, confuses an ambulance for a fire engine, and its range estimate is
only good to tens of metres. Feeding its raw per-frame output straight to the
policy would make the controller twitch on every false positive. Instead each
approach carries a small filter:

* **Presence** - a two-state hidden Markov model (present / absent) with birth
  and survival probabilities, updated by the detector's *likelihood ratio*
  rather than its hard decision. Because the classifier is trained with a known
  siren prior ``pi``, its posterior odds can be converted back to a likelihood
  ratio exactly:

      L = [P(siren|z) / P(none|z)] * [(1 - pi) / pi]

  so a single confident frame moves the belief a lot and a stream of marginal
  frames moves it a little, which is the behaviour we want.

* **Time to arrival** - a two-state constant-velocity Kalman filter on
  ``[range, range_rate]``, driven by the detector's per-frame range estimate.
  Arrival time is then ``range / -range_rate``.

  It is worth being explicit about why range-rate is *tracked* rather than
  measured. Doppler shift at road speeds is about 6 % - some tens of hertz on a
  700 Hz fundamental - which is well below one mel band, and the siren's own
  sweep moves the fundamental far more than that within a frame. Single-frame
  range-rate regression is therefore close to ill-posed, and measurably so: the
  detector's velocity head reaches only ~12 m/s MAE on a +/-30 m/s range. What
  *is* observable is that the received level climbs as the vehicle closes, and
  that a vehicle on a given approach stays near a given microphone's axis
  throughout its run, so the unknown directivity offset is roughly constant and
  cancels out of the *trend*. Differencing successive range estimates therefore
  recovers range-rate even though no single estimate is trustworthy - which is
  exactly what a constant-velocity filter does optimally. The velocity head is
  retained only as a weak prior on the sign of closure, so a departing vehicle
  is not mistaken for an arriving one.

* **Class** - a categorical posterior over {ambulance, fire, police} that yields
  an expected triage priority, which scales the constraint budget.

Both the presence prior and the observation likelihood are deliberately
conservative, so the cost of a false positive to ordinary traffic stays small.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import BeliefConfig

SIREN_CLASSES = ("ambulance", "fire", "police")


def siren_probability(class_probs, class_map) -> "np.ndarray":
    """P(an emergency siren is present), from the detector's class posterior.

    This is a **sum over the siren classes**, not ``1 - P(none)``, and the
    difference is not cosmetic. Once the confuser has its own label, the two
    stop being equal: a detector that has confidently and correctly identified a
    *road horn* puts its mass on ``horn``, so ``1 - P(none)`` reads 1.0 and the
    controller is told a siren is present.

    That was a live bug, and it hid the very capability the horn class was added
    to provide. Measured on the retrained detector's held-out set, ``1 - P(none)``
    called a horn a siren with probability **1.000** at every SNR above 0 dB --
    which reads as "the network cannot tell them apart" and is not what the
    network was doing at all. The network had learned the distinction; the
    readout was discarding it.

    Accepts a single posterior vector or a batch, and returns the matching
    shape.
    """
    import numpy as _np

    probs = _np.asarray(class_probs, dtype=_np.float64)
    idx = [class_map[c] for c in SIREN_CLASSES if c in class_map]
    if probs.ndim == 1:
        return _np.clip(probs[idx].sum(), 0.0, 1.0)
    return _np.clip(probs[:, idx].sum(axis=1), 0.0, 1.0)


@dataclass
class ApproachBelief:
    """Posterior state for a single approach."""

    presence: float
    eta_s: float
    eta_var: float
    class_probs: Dict[str, float]
    expected_priority: float
    range_m: float = float("nan")
    range_rate_ms: float = float("nan")
    signed_range: bool = False
    """Is ``range_m`` signed about the stop line?

    The joint filter tracks distance to the stop line and lets it go negative
    once the vehicle has crossed, so ``range_m > 0`` *is* the inbound test. The
    per-approach filter tracks distance to the mast, which is unsigned, so
    inbound has to be inferred from the sign of the range rate instead. The two
    are not interchangeable and a consumer must know which it has: a vehicle
    stopped in a queue has a range rate of zero, and a rule that reads that as
    "not approaching" will drop the corridor for exactly the vehicle it exists
    to serve."""

    @property
    def committed(self) -> bool:
        return self.presence >= self._commit

    _commit: float = 0.6

    def imminence(self, max_eta_s: float, max_range_m: float = 300.0) -> float:
        """How close to arriving this approach's vehicle is, in ``[0, 1]``.

        Arrival time alone cannot express this, and the case it fails on is the
        one that matters most. ``eta`` is ``range / closing speed`` clamped at
        a ceiling, so a vehicle that has *stopped* -- closing speed zero,
        because it is held at the red the controller is being asked to lift --
        reports the ceiling and scores an imminence of exactly zero. Measured
        against ground truth, the estimate sat at its ceiling in every epoch
        where an emergency vehicle was queued on a red approach: the feature
        the policy is supposed to learn priority from was identically zero for
        the vehicle most in need of it.

        Range does not have that failure mode -- a stopped vehicle 8 m from the
        stop line is 8 m from the stop line regardless of its speed -- so the
        range is used whenever the arrival time is uninformative. The two agree
        where both are valid, and this is the same fallback the corridor
        scheduler already makes for the same reason.
        """
        if np.isfinite(self.eta_s) and self.eta_s < max_eta_s - 1e-3:
            return float(np.clip(1.0 - self.eta_s / max(max_eta_s, 1e-6), 0.0, 1.0))
        if not np.isfinite(self.range_m):
            return 0.0
        if self.range_m <= 0.0:
            # At or past the stop line: as imminent as it gets, whether that is
            # a vehicle in the junction or one waiting to enter it.
            return 1.0 if self.signed_range else 0.0
        return float(np.clip(1.0 - self.range_m / max(max_range_m, 1e-6), 0.0, 1.0))

    def as_features(self, max_eta_s: float, max_range_m: float = 300.0) -> List[float]:
        """The three numbers the policy actually consumes for this approach."""
        imminence = self.imminence(max_eta_s, max_range_m)
        # Reported as the complement so the feature keeps its previous sense --
        # small means "about to arrive" -- and the observation vector keeps its
        # shape and its scale.
        eta_norm = 1.0 - imminence
        # Urgency decays with time-to-arrival: an EV 5 s away matters far more
        # than one 40 s away, and one that is absent does not matter at all.
        urgency = self.presence * imminence
        return [self.presence, eta_norm, urgency * self.expected_priority]


def _scale_transition(survival: float, birth: float, dt: float,
                      reference_s: float) -> Tuple[float, float]:
    """Re-express a per-epoch two-state chain for a step of ``dt`` seconds.

    Survival compounds, so it is raised to ``dt / reference``; birth is the
    complement of "no birth in any sub-interval", so ``1 - (1 - b) ** k``. At
    ``dt == reference`` this is the identity, and the stationary occupancy
    ``b / (b + 1 - s)`` -- which is what the filter's prior was pinned to -- is
    preserved to first order at every other ``dt``.
    """
    if reference_s <= 0.0:
        return survival, birth
    k = float(dt) / float(reference_s)
    if abs(k - 1.0) < 1e-9:
        return survival, birth
    k = max(k, 0.0)
    return float(survival ** k), float(1.0 - (1.0 - birth) ** k)


class BeliefTracker:
    """Maintains one :class:`ApproachBelief` per approach."""

    N_FEATURES_PER_APPROACH = 3

    def __init__(
        self,
        approaches: Sequence[str],
        cfg: Optional[BeliefConfig] = None,
        priority_map: Optional[Dict[str, float]] = None,
        detector_prior: float = 0.5,
        stop_line_offset_m: float = 0.0,
    ):
        self.approaches = list(approaches)
        self.stop_line_offset_m = float(stop_line_offset_m)
        """Metres from the microphone mast to the stop line.

        The detector measures range to the *mast*, which sits upstream of the
        stop line so that the four approaches are acoustically separable at all.
        Everything downstream -- arrival time, the corridor's clearance
        arithmetic -- is about the stop line. Reporting the raw mast range makes
        every arrival time short by this offset divided by the approach speed,
        about five seconds here, and five seconds is most of a changeover."""
        self.cfg = cfg or BeliefConfig()
        self.priority_map = priority_map or {"ambulance": 1.0, "fire": 0.9, "police": 0.7}
        self.detector_prior = float(np.clip(detector_prior, 1e-3, 1 - 1e-3))
        self._prior_odds_correction = (1.0 - self.detector_prior) / self.detector_prior
        self.reset()

    # ------------------------------------------------------------------- state
    @property
    def evidence_weight(self) -> float:
        """Power to raise this cycle's likelihood to, in [0, 1].

        One at the reference interval, and proportionally less when the filter
        is sampled faster: five one-second looks then carry the evidence of one
        five-second look rather than five times it. See
        BeliefConfig.evidence_per_reference_interval for why that is the
        correct accounting and what over-counting cost when it was absent.
        """
        cfg = self.cfg
        ref = float(cfg.transition_reference_s)
        if ref <= 0.0:
            return float(cfg.evidence_per_reference_interval)
        w = cfg.evidence_per_reference_interval * float(self._last_dt) / ref
        return float(np.clip(w, 0.0, 1.0))

    DEFAULT_RANGE_M = 400.0
    DEFAULT_RATE_MS = -8.0
    """Prior closing speed used before any range trend has been observed."""

    def reset(self) -> None:
        cfg = self.cfg
        self._presence = {a: cfg.prior_presence for a in self.approaches}
        self._class = {
            a: {c: 1.0 / len(SIREN_CLASSES) for c in SIREN_CLASSES} for a in self.approaches
        }
        # Kalman state x = [range_m, range_rate_ms] with covariance P.
        self._x = {
            a: np.array([self.DEFAULT_RANGE_M, self.DEFAULT_RATE_MS], dtype=np.float64)
            for a in self.approaches
        }
        self._P = {
            a: np.diag([self.DEFAULT_RANGE_M ** 2, 100.0]) for a in self.approaches
        }
        self._tracked = {a: False for a in self.approaches}
        self._last_dt = float(self.cfg.transition_reference_s)

    # ------------------------------------------------------------------ update
    def predict(self, dt: float) -> None:
        """Time update: presence relaxes toward the birth rate, the target moves."""
        cfg = self.cfg
        q = cfg.eta_process_noise_s ** 2
        F = np.array([[1.0, dt], [0.0, 1.0]])
        # Continuous white-noise-acceleration process noise.
        Q = q * np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0], [dt ** 2 / 2.0, dt]])

        for a in self.approaches:
            p = self._presence[a]
            # Survival is conditioned on direction of travel. "Present" means
            # *inbound*, so a source the tracker says is receding has most
            # likely left the state, however loud it still is. Without this the
            # posterior stays pinned near 1 for the whole audible tail after a
            # vehicle crosses, which was the single largest source of apparent
            # false alarms in this filter.
            receding = float(self._x[a][1]) > 0.0
            survival = (cfg.survival_prob_receding if receding
                        else cfg.survival_prob)
            # Both probabilities are quoted per ``transition_reference_s`` of
            # elapsed time, not per call. The environment now advances the
            # filter once per detector frame rather than once per decision
            # epoch, so applying them raw would run the same chain five times
            # as fast -- see BeliefConfig.transition_reference_s.
            survival, birth = _scale_transition(
                survival, cfg.birth_prob, dt, cfg.transition_reference_s)
            self._presence[a] = p * survival + (1.0 - p) * birth
            # How much elapsed time this cycle's measurement stands for. The
            # update that follows weights its evidence by it -- see
            # BeliefConfig.evidence_per_reference_interval.
            self._last_dt = float(dt)

            self._x[a] = F @ self._x[a]
            self._x[a][0] = max(self._x[a][0], 0.0)
            self._P[a] = F @ self._P[a] @ F.T + Q

            probs = self._class[a]
            uniform = 1.0 / len(SIREN_CLASSES)
            decay = 0.9 ** dt
            self._class[a] = {c: decay * v + (1 - decay) * uniform for c, v in probs.items()}

    def update(
        self,
        approach: str,
        p_siren: float,
        class_probs: Optional[Dict[str, float]],
        range_measurement_m: Optional[float],
        range_measurement_var: Optional[float] = None,
        rate_hint_ms: Optional[float] = None,
    ) -> None:
        """Measurement update from one detector output on ``approach``."""
        p_siren = float(np.clip(p_siren, 1e-6, 1.0 - 1e-6))

        # --- presence: likelihood-ratio Bayes update ---
        # Raised to the evidence weight: L**w is the evidence of w independent
        # looks, so sampling the microphone faster than the scene changes no
        # longer inflates the posterior. w = 1 leaves this exactly as it was.
        likelihood_ratio = ((p_siren / (1.0 - p_siren))
                            * self._prior_odds_correction) ** self.evidence_weight
        prior = self._presence[approach]
        posterior = prior * likelihood_ratio / (prior * likelihood_ratio + (1.0 - prior))
        self._presence[approach] = float(np.clip(posterior, 1e-6, 1.0 - 1e-6))

        # --- class: categorical Bayes, weighted by how sure we are it is a siren ---
        if class_probs:
            current = self._class[approach]
            weight = self._presence[approach]
            updated = {}
            total = 0.0
            for c in SIREN_CLASSES:
                lik = max(class_probs.get(c, 0.0), 1e-6)
                blended = (1.0 - weight) * current[c] + weight * current[c] * lik
                updated[c] = blended
                total += blended
            self._class[approach] = {c: v / max(total, 1e-12) for c, v in updated.items()}

        if range_measurement_m is not None and math.isfinite(range_measurement_m):
            z = float(np.clip(range_measurement_m, 1.0, 2.0 * self.DEFAULT_RANGE_M))
            R = float(range_measurement_var if range_measurement_var is not None else 60.0 ** 2)

            if not self._tracked[approach]:
                # First sighting: seed the track rather than dragging the prior
                # range down one Kalman gain at a time.
                self._x[approach] = np.array(
                    [z, rate_hint_ms if rate_hint_ms is not None else self.DEFAULT_RATE_MS]
                )
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

            # The velocity head is a weak prior on the *sign* of closure only.
            if rate_hint_ms is not None and self._x[approach][1] * rate_hint_ms < 0:
                self._x[approach][1] *= 0.5
            self._x[approach][0] = max(self._x[approach][0], 0.0)

        elif self._presence[approach] < 0.1:
            # Nothing there: relax back to "far away" and drop the track.
            self._x[approach] = np.array([self.DEFAULT_RANGE_M, self.DEFAULT_RATE_MS])
            self._P[approach] = np.diag([self.DEFAULT_RANGE_M ** 2, 100.0])
            self._tracked[approach] = False

    # -------------------------------------------------------------------- read
    def _eta_of(self, approach: str) -> Tuple[float, float]:
        """Arrival time and its variance, propagated from the range track."""
        cfg = self.cfg
        r, rdot = self._x[approach]
        r = r + self.stop_line_offset_m
        closing = -rdot
        # A track whose range has gone negative is a vehicle that has crossed
        # the stop line. It has no arrival time, and reporting one -- which
        # r/closing cheerfully does, as a small positive number -- makes a
        # departed vehicle look like the most urgent thing on the network.
        if closing < 0.5 or r <= 0.0:
            return cfg.max_eta_s, cfg.max_eta_s ** 2
        eta = float(np.clip(r / closing, 0.0, cfg.max_eta_s))
        P = self._P[approach]
        # First-order propagation of eta = r / (-rdot).
        J = np.array([1.0 / closing, r / (closing ** 2)])
        var = float(np.clip(J @ P @ J.T, 1.0, cfg.max_eta_s ** 2))
        return eta, var

    def belief(self, approach: str) -> ApproachBelief:
        probs = self._class[approach]
        priority = sum(probs[c] * self.priority_map.get(c, 1.0) for c in SIREN_CLASSES)
        eta, eta_var = self._eta_of(approach)
        return ApproachBelief(
            presence=self._presence[approach],
            eta_s=eta,
            eta_var=eta_var,
            class_probs=dict(probs),
            expected_priority=priority,
            range_m=float(self._x[approach][0]) + self.stop_line_offset_m,
            range_rate_ms=float(self._x[approach][1]),
            _commit=self.cfg.commit_threshold,
        )

    def beliefs(self) -> Dict[str, ApproachBelief]:
        return {a: self.belief(a) for a in self.approaches}

    def feature_vector(self, max_range_m: float = 300.0) -> np.ndarray:
        max_eta = self.cfg.max_eta_s
        out: List[float] = []
        for a in self.approaches:
            out.extend(self.belief(a).as_features(max_eta, max_range_m))
        return np.asarray(out, dtype=np.float32)

    @property
    def n_features(self) -> int:
        return len(self.approaches) * self.N_FEATURES_PER_APPROACH
