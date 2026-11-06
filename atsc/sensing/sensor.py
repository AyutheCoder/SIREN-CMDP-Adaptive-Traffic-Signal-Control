"""The runtime acoustic sensor: four calibrated microphones, one per approach.

Sensor placement
----------------
A microphone sits on each approach mast at the stop line. This mirrors how
roadside detection is actually deployed (per-approach cameras and radar already
live there) and, importantly, it means localisation falls out of geometry rather
than requiring a beamforming array. It is not clean, though: a siren on the
northbound approach is still audible at the eastbound microphone, just quieter.
That cross-talk is a genuine source of ambiguity and the belief filter has to
survive it.

What the sensor cannot tell you
-------------------------------
A microphone hears *that* an emergency vehicle is approaching and roughly how
fast and how far. It cannot hear which way the vehicle intends to turn. The
controller therefore has to serve an entire approach on partial information -
a constraint most emergency-preemption work sidesteps by assuming V2I gives an
exact route. Keeping it is a large part of why this problem is interesting.

Modes
-----
``end_to_end``  synthesise the mixed waveform and run the CNN. Physically
                faithful; roughly 30x slower than the alternatives, so it is
                used to *validate* rather than to train.
``surrogate``   sample the detector's output from an error model fitted to its
                own held-out predictions. Statistically matched to
                ``end_to_end`` (see ``scripts/validate_surrogate.py``) and fast
                enough for millions of RL steps.
``oracle``      perfect detection. The ablation upper bound: how much is lost to
                imperfect sensing rather than to imperfect control.
``none``        no emergency awareness whatsoever.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import AcousticConfig
from .belief import SIREN_CLASSES, siren_probability
from .siren_synth import CLASS_TIMBRE, SirenSynthesizer
from .siren_synth import received_spl_db as _propagate_spl

MODES = ("end_to_end", "surrogate", "oracle", "none")


def cardioid_gain_db(off_axis_rad: float, front_back_db: float) -> float:
    """Directional response of a cardioid microphone aimed along its approach.

    ``g(theta) = (1 + cos theta) / 2``, floored so the rear null attenuates by
    ``front_back_db`` rather than by minus infinity. Combined with mounting the
    microphone upstream of the stop line, this is what makes the four approaches
    acoustically separable at all.
    """
    g = 0.5 * (1.0 + math.cos(off_axis_rad))
    floor = 10.0 ** (-front_back_db / 20.0)
    return 20.0 * math.log10(max(g, floor))


def endfire_pair_gain_db(off_axis_rad: float,
                         front_back_db: float) -> Tuple[float, float]:
    """The two cardioids a two-element endfire mast delivers, ``(front, back)``.

    A compact endfire pair is steered by the *sign* of the delay applied between
    its two capsules; flipping that sign rotates the cardioid through 180
    degrees. Both lobes are therefore available from the same two capsules at
    the same instant, at the cost of one extra delay-and-sum per frame.

    ``front`` looks up the approach, away from the junction -- the lobe the
    single-element mast had. ``back`` looks at the junction, and is what makes
    the last ``mic_offset_m`` of the approach audible instead of nulled.
    """
    front = cardioid_gain_db(off_axis_rad, front_back_db)
    back = cardioid_gain_db(math.pi - off_axis_rad, front_back_db)
    return front, back


@dataclass
class SourceState:
    """Ground-truth kinematics of one emergency vehicle relative to one mic."""

    veh_class: str
    distance_m: float
    radial_velocity_ms: float
    """Positive means receding."""
    off_axis_rad: float = 0.0
    """Angle between the microphone's look direction and the source bearing."""
    front_back_db: float = 14.0
    dual_element: bool = True
    """Read the mast as a steerable two-element endfire pair. See
    :attr:`~atsc.config.AcousticConfig.dual_element_mast`."""

    @property
    def directivity_db(self) -> float:
        """Gain of the lobe the mast is actually listening on.

        For a single-element mast that is the upstream cardioid. For the
        two-element mast it is the louder of the two lobes, which is what
        steering the pair delivers -- and what removes the rear null over the
        last ``mic_offset_m`` of the approach.
        """
        if not self.dual_element:
            return cardioid_gain_db(self.off_axis_rad, self.front_back_db)
        front, back = endfire_pair_gain_db(self.off_axis_rad, self.front_back_db)
        return max(front, back)

    @property
    def front_back_ratio_db(self) -> float:
        """Front lobe minus back lobe, dB. Zero for a single-element mast.

        The new observable the second element buys. Source level and range
        cancel out of a ratio taken at the same instant on the same mast, so
        this is a near-pure statement of *which side of the mast* the source is
        on: strongly positive upstream (still approaching), strongly negative
        downstream (across the line and leaving). It is the cue the
        single-element geometry could not provide, and the one the filter needs
        to tell an inbound vehicle from a departed one.
        """
        if not self.dual_element:
            return 0.0
        front, back = endfire_pair_gain_db(self.off_axis_rad, self.front_back_db)
        return front - back

    def received_spl_db(self) -> float:
        """Closed-form level, matching what the synthesiser would render."""
        return _propagate_spl(self.veh_class, self.distance_m, self.directivity_db)


@dataclass
class SensorReading:
    """One microphone's output at one decision epoch."""

    approach: str
    p_siren: float
    class_probs: Dict[str, float]
    radial_velocity_ms: Optional[float] = None
    distance_m: Optional[float] = None
    spl_db: Optional[float] = None
    """Sound pressure level this microphone actually measured, dB re 20 uPa.

    The classifier posterior is a decision about the level; the level itself is
    a measurement, and it is monotone in closeness in a way the posterior is
    not -- a saturated detector reads 1.00 at 200 m and at 20 m alike, while
    the level between those two differs by twenty decibels. It is carried
    through so the controller can use loudness and its trend directly, so the
    demo can render the siren at the level the intersection is hearing, and so
    the HUD can show it."""

    front_back_db: Optional[float] = None
    """Front lobe minus back lobe on this mast, dB. ``None`` for a
    single-element mast, which cannot measure it.

    A ratio of two levels taken at the same instant through the same preamp on
    the same mast, so the source's own level, its distance and the ambient
    floor all cancel to first order and what is left is geometry: the sign says
    which side of the mast the source is on. That is the one thing the level
    alone can never say, because the mast sits partway up the approach and the
    level is therefore *non-monotone* in distance to the stop line -- it peaks
    at the mast and falls away on both sides, so a reading of 85 dB is equally
    consistent with a vehicle 90 m from the line and one 50 m from it. The
    ratio breaks that tie, and it is what lets the filter keep a track through
    the last stretch instead of resetting it at the moment the vehicle arrives.

    Noisier than the geometry alone implies, because the two lobes are formed
    from the same two capsules and their mismatch, the wind and the finite
    element spacing all land here; see
    ``AcousticConfig.front_back_noise_db``."""

    ambient_db: Optional[float] = None
    """Noise floor the level was measured against, for the SNR."""

    @property
    def snr_db(self) -> Optional[float]:
        """Level above the ambient floor. ``None`` when nothing was audible."""
        if self.spl_db is None or self.ambient_db is None:
            return None
        return float(self.spl_db - self.ambient_db)

    @property
    def eta_s(self) -> Optional[float]:
        if self.radial_velocity_ms is None or self.distance_m is None:
            return None
        if self.radial_velocity_ms >= -0.5:
            return None  # stationary or receding: no meaningful arrival time
        return self.distance_m / abs(self.radial_velocity_ms)

    @property
    def range_var(self) -> Optional[float]:
        """Variance of the range estimate, growing with the estimate itself.

        Range is regressed in log space, so a fixed log-error corresponds to a
        proportional error in metres: a 200 m estimate is far less certain in
        absolute terms than a 30 m one, and the tracker must be told so or it
        will over-trust distant measurements.
        """
        if self.distance_m is None:
            return None
        sigma_log = 0.45
        sigma_m = self.distance_m * sigma_log
        return float(np.clip(sigma_m ** 2, 25.0, 4.0e4))


def _logit(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


@dataclass
class SensorErrorModel:
    """Empirical model of the detector's behaviour, fitted on held-out data.

    Everything is conditioned on signal-to-noise ratio, because SNR (not raw
    distance) is what actually governs whether the detector fires: a siren at
    200 m on a quiet night is easier than one at 60 m next to a bus.
    """

    snr_edges: List[float] = field(default_factory=lambda: [-20.0, -10.0, -5.0, 0.0, 5.0, 15.0])
    siren_logit_mean: List[float] = field(default_factory=list)
    siren_logit_std: List[float] = field(default_factory=list)
    noise_logit_mean: float = -4.0
    noise_logit_std: float = 1.5
    class_posterior: List[List[List[float]]] = field(default_factory=list)
    """``[snr_bin][true_class][predicted_class]`` mean posterior over siren classes."""
    vr_bias: List[float] = field(default_factory=list)
    vr_std: List[float] = field(default_factory=list)
    logr_bias: List[float] = field(default_factory=list)
    logr_std: List[float] = field(default_factory=list)
    horn_logit_mean: List[float] = field(default_factory=list)
    horn_logit_std: List[float] = field(default_factory=list)
    """Detector siren-logit when the source is a **horn**, per SNR bin.

    The confuser's own detection curve, and the reason it is kept separate from
    ``noise_logit_*`` rather than folded into it. Bare traffic noise is an easy
    negative: it is broadband, and the detector rejects it almost regardless of
    level. A horn is a hard negative -- loud, harmonic, and occupying the low
    end of the siren band -- so its false-alarm rate rises with SNR where the
    noise floor's does not. Scoring both against one flat ``noise_logit_mean``
    asserts that a horn at 3 m is no more confusable than distant tyre roar,
    which is exactly the assumption that made the previous detector untested
    against the confuser that matters.

    Empty when the model was fitted before horns existed, in which case the
    horn falls back to the siren curve -- the pessimistic reading, and the right
    default for a model that cannot say otherwise."""

    detector_prior: float = 0.5
    n_bins: int = 6

    def horn_logit(self, snr_bin: int) -> Tuple[float, float]:
        """``(mean, std)`` siren-logit for a horn in this SNR bin.

        Falls back to the *siren* curve when no horn data was fitted, so an
        un-updated error model treats every horn as maximally confusable rather
        than silently as free.
        """
        if self.horn_logit_mean and snr_bin < len(self.horn_logit_mean):
            return (float(self.horn_logit_mean[snr_bin]),
                    float(max(self.horn_logit_std[snr_bin], 1e-3)))
        if self.siren_logit_mean and snr_bin < len(self.siren_logit_mean):
            return (float(self.siren_logit_mean[snr_bin]),
                    float(max(self.siren_logit_std[snr_bin], 1e-3)))
        return (float(self.noise_logit_mean), float(self.noise_logit_std))

    # ---------------------------------------------------------------- fitting
    @staticmethod
    def _bin_index(snr: np.ndarray, edges: Sequence[float]) -> np.ndarray:
        return np.digitize(snr, edges)

    @classmethod
    def fit(
        cls,
        y_class: np.ndarray,
        snr_db: np.ndarray,
        pred_class_probs: np.ndarray,
        pred_vr: np.ndarray,
        pred_logr: np.ndarray,
        true_vr: np.ndarray,
        true_logr: np.ndarray,
        cfg: AcousticConfig,
        detector_prior: float = 0.5,
        snr_edges: Optional[Sequence[float]] = None,
    ) -> "SensorErrorModel":
        edges = list(snr_edges) if snr_edges is not None else [-20.0, -10.0, -5.0, 0.0, 5.0, 15.0]
        n_bins = len(edges) + 1
        model = cls(snr_edges=edges, n_bins=n_bins, detector_prior=detector_prior)

        # Sum over the siren classes rather than 1 - P(none): with the horn
        # class present the two differ, and the difference is precisely the
        # confuser. See belief.siren_probability.
        p_siren = siren_probability(pred_class_probs, cfg.class_map)
        logits = _logit(p_siren)

        is_noise = y_class == cfg.class_map["none"]
        if is_noise.any():
            model.noise_logit_mean = float(np.mean(logits[is_noise]))
            model.noise_logit_std = float(max(np.std(logits[is_noise]), 1e-3))

        # Horns are neither noise nor siren. Pooling them with either would
        # corrupt that curve: with noise, because a horn is far more confusable
        # than tyre roar and would inflate the apparent false-alarm rate of
        # silence; with siren, because a correctly rejected horn would read as a
        # missed detection and understate the detector's real sensitivity.
        horn_idx = cfg.class_map.get("horn")
        is_horn = (y_class == horn_idx) if horn_idx is not None else np.zeros_like(is_noise)
        is_siren_row = ~is_noise & ~is_horn

        siren_idx = [cfg.class_map[c] for c in SIREN_CLASSES]
        bins = cls._bin_index(snr_db, edges)

        model.siren_logit_mean, model.siren_logit_std = [], []
        model.vr_bias, model.vr_std = [], []
        model.logr_bias, model.logr_std = [], []
        model.class_posterior = []

        model.horn_logit_mean, model.horn_logit_std = [], []

        for b in range(n_bins):
            sel = is_siren_row & (bins == b)
            if sel.sum() < 10:
                # Too few samples: fall back to the pooled siren statistics.
                sel = is_siren_row

            hsel = is_horn & (bins == b)
            if hsel.sum() < 10:
                hsel = is_horn
            if hsel.any():
                model.horn_logit_mean.append(float(np.mean(logits[hsel])))
                model.horn_logit_std.append(float(max(np.std(logits[hsel]), 1e-3)))
            model.siren_logit_mean.append(float(np.mean(logits[sel])))
            model.siren_logit_std.append(float(max(np.std(logits[sel]), 1e-3)))

            vr_err = (pred_vr[sel] - true_vr[sel]) * cfg.max_radial_speed_ms
            model.vr_bias.append(float(np.mean(vr_err)))
            model.vr_std.append(float(max(np.std(vr_err), 1e-3)))

            logr_err = pred_logr[sel] - true_logr[sel]
            model.logr_bias.append(float(np.mean(logr_err)))
            model.logr_std.append(float(max(np.std(logr_err), 1e-3)))

            rows = []
            for ci, cname in enumerate(SIREN_CLASSES):
                true_c = cfg.class_map[cname]
                m = (y_class == true_c) & (bins == b)
                if m.sum() < 5:
                    m = y_class == true_c
                if m.sum() == 0:
                    rows.append([1.0 / len(SIREN_CLASSES)] * len(SIREN_CLASSES))
                    continue
                sub = pred_class_probs[np.ix_(m, siren_idx)]
                sub = sub / np.maximum(sub.sum(axis=1, keepdims=True), 1e-12)
                rows.append([float(v) for v in sub.mean(axis=0)])
            model.class_posterior.append(rows)

        return model

    # --------------------------------------------------------------- sampling
    def sample(
        self, rng: np.random.Generator, snr_db: Optional[float], veh_class: Optional[str]
    ) -> Tuple[float, Dict[str, float], Optional[float], Optional[float]]:
        """Draw one synthetic detector output.

        Returns ``(p_siren, class_probs, vr_error_ms, logr_error)``. The errors
        are *additive corrections* the caller applies to ground truth.
        """
        if veh_class is None or snr_db is None:
            logit = rng.normal(self.noise_logit_mean, self.noise_logit_std)
            probs = {c: 1.0 / len(SIREN_CLASSES) for c in SIREN_CLASSES}
            return float(_sigmoid(logit)), probs, None, None

        b = int(np.digitize([snr_db], self.snr_edges)[0])
        b = int(np.clip(b, 0, self.n_bins - 1))
        logit = rng.normal(self.siren_logit_mean[b], self.siren_logit_std[b])
        ci = SIREN_CLASSES.index(veh_class)
        mean_post = np.asarray(self.class_posterior[b][ci], dtype=np.float64)
        # Dirichlet around the mean posterior keeps per-frame outputs varied
        # rather than emitting the same vector every epoch.
        conc = np.maximum(mean_post * 25.0, 0.1)
        probs_arr = rng.dirichlet(conc)
        probs = {c: float(probs_arr[i]) for i, c in enumerate(SIREN_CLASSES)}
        vr_err = float(rng.normal(self.vr_bias[b], self.vr_std[b]))
        logr_err = float(rng.normal(self.logr_bias[b], self.logr_std[b]))
        return float(_sigmoid(logit)), probs, vr_err, logr_err

    # ------------------------------------------------------------------- i/o
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SensorErrorModel":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def oracle(cls) -> "SensorErrorModel":
        """The error model of a *perfect* detector.

        The joint belief filter scores each microphone's logit against the
        distribution the detector is known to produce, so the model has to
        describe whichever sensor is actually running. Handing it the CNN's
        fitted curve while the oracle sensor is emitting 0.999 and 0.001 puts
        every observation many standard deviations into the tail of every
        hypothesis, and the posterior degenerates -- which is not "perfect
        sensing performs like imperfect sensing", it is the filter being lied
        to about its own sensor. Measured, that made the oracle ablation
        indistinguishable from having no emergency-vehicle awareness at all.
        """
        n_bins = 6
        return cls(
            snr_edges=[-20.0, -10.0, -5.0, 0.0, 5.0, 15.0],
            siren_logit_mean=[6.9] * n_bins,
            siren_logit_std=[0.25] * n_bins,
            noise_logit_mean=-6.9,
            noise_logit_std=0.25,
            class_posterior=[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                             for _ in range(n_bins)],
            vr_bias=[0.0] * n_bins,
            vr_std=[0.05] * n_bins,
            logr_bias=[0.0] * n_bins,
            logr_std=[0.02] * n_bins,
            n_bins=n_bins,
        )

    @classmethod
    def default(cls) -> "SensorErrorModel":
        """A hand-specified fallback used when no detector has been trained yet.

        Pessimistic on purpose: near-chance detection at very low SNR, high
        confidence only when the siren clearly dominates the noise floor.
        """
        n_bins = 6
        return cls(
            snr_edges=[-20.0, -10.0, -5.0, 0.0, 5.0, 15.0],
            siren_logit_mean=[-2.5, -0.8, 0.4, 1.6, 3.0, 4.2][:n_bins],
            siren_logit_std=[1.6, 1.5, 1.3, 1.1, 0.9, 0.7][:n_bins],
            noise_logit_mean=-3.6,
            noise_logit_std=1.4,
            class_posterior=[
                [[0.45, 0.30, 0.25], [0.30, 0.45, 0.25], [0.25, 0.25, 0.50]]
                for _ in range(n_bins)
            ],
            vr_bias=[0.0] * n_bins,
            vr_std=[9.0, 7.0, 5.0, 3.5, 2.2, 1.5][:n_bins],
            logr_bias=[0.0] * n_bins,
            logr_std=[0.9, 0.7, 0.5, 0.35, 0.22, 0.15][:n_bins],
            n_bins=n_bins,
        )


class AcousticSensor:
    """Turns ground-truth emergency-vehicle kinematics into noisy readings."""

    ORACLE_ON_AXIS_RAD = math.radians(40.0)
    """Half-angle within which an oracle counts a source as being *on* this
    approach. Wide enough to include a vehicle part-way through the junction,
    narrow enough to exclude the other three arms."""

    def __init__(
        self,
        approaches: Sequence[str],
        cfg: Optional[AcousticConfig] = None,
        mode: str = "surrogate",
        error_model: Optional[SensorErrorModel] = None,
        detector=None,
        mel=None,
        seed: int = 0,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.approaches = list(approaches)
        self.cfg = cfg or AcousticConfig()
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.error_model = error_model or SensorErrorModel.default()
        self.detector = detector
        self.mel = mel
        self._synth: Optional[SirenSynthesizer] = None
        if mode == "end_to_end":
            if detector is None or mel is None:
                raise ValueError("end_to_end mode requires both a detector and a mel front-end")
            self._synth = SirenSynthesizer(self.cfg, seed=seed + 1)

        self.dead_approaches: set = set()
        """Masts that have failed. A failed microphone is not a noisy one: it
        reports nothing at all, and the distinction matters because a filter
        that is told which masts are dead can still localise from the rest,
        whereas one that treats silence as evidence of absence will conclude
        that nothing is ever inbound on that arm. Graceful degradation under
        partial sensor loss is the property a passive-acoustic system has and a
        transponder-based one does not, so it is worth measuring."""

        self._ambient: Dict[str, float] = {}
        """Per-mast noise floor, carried across calls; see :meth:`_ambient_db`."""

        self.n_calls = 0
        self.n_true_positive = 0
        self.n_false_positive = 0
        self.n_false_negative = 0

    # ------------------------------------------------------------------ helpers
    def _level_noise(self) -> float:
        """Measurement error on one level reading, dB. See
        :attr:`~atsc.config.AcousticConfig.level_noise_db`."""
        sd = float(getattr(self.cfg, "level_noise_db", 0.0))
        return 0.0 if sd <= 0.0 else float(self.rng.normal(0.0, sd))

    def _ambient_db(self, approach: str, dt: float) -> float:
        """Noise floor at one mast, as a slowly drifting process.

        It used to be drawn independently at every call, uniformly over the
        25 dB the configuration allows. As long as the filter was only asked
        once per five-second epoch that was merely crude; once the level itself
        is used -- and it must be, because loudness is the part of the evidence
        that keeps rising all the way in while the posterior saturates -- it is
        wrong in a way that matters. Twenty-five decibels of white noise on the
        floor is far larger than the level change a vehicle produces between
        frames, so the trend of the signal-to-noise ratio was pure noise and no
        closing test could be built on it.

        Traffic noise does not do that. It drifts. This is a random walk
        reflected in the configured interval, whose stationary distribution is
        exactly the uniform the configuration asks for -- so the *marginal* the
        detector's error model was fitted against is unchanged -- but whose
        increments are small over a second. Each mast drifts independently,
        which is also true: they are a couple of hundred metres apart.
        """
        lo, hi = self.cfg.ambient_noise_db_range
        span = max(hi - lo, 1e-6)
        x = self._ambient.get(approach)
        if x is None:
            x = float(self.rng.uniform(lo, hi))
        else:
            step = self.cfg.ambient_drift_db_per_sqrt_s * math.sqrt(max(dt, 0.0))
            x = x + float(self.rng.normal(0.0, step))
            # Reflect at the bounds. A reflected random walk on an interval has
            # a uniform stationary law, which is the point.
            period = 2.0 * span
            y = (x - lo) % period
            x = lo + (y if y <= span else period - y)
        self._ambient[approach] = float(x)
        return float(x)

    @staticmethod
    def _dominant(sources: Sequence[SourceState]) -> Optional[SourceState]:
        audible = [s for s in sources if s.veh_class in CLASS_TIMBRE]
        if not audible:
            return None
        return max(audible, key=lambda s: s.received_spl_db())

    def _front_back(self, dominant, noiseless: bool = False) -> Optional[float]:
        """Measured front-minus-back ratio for the dominant source, dB.

        ``None`` when the mast has only one element, so a filter reading this
        can tell "no second element" apart from "a ratio that happens to be
        zero" -- which is a real reading, meaning the source is broadside.
        """
        if dominant is None or not self.cfg.dual_element_mast:
            return None
        true = float(dominant.front_back_ratio_db)
        if noiseless or self.cfg.front_back_noise_db <= 0.0:
            return true
        return float(true + self.rng.normal(0.0, self.cfg.front_back_noise_db))

    # -------------------------------------------------------------------- sense
    def sense(self, scene: Dict[str, List[SourceState]],
              dt: float = 1.0) -> Dict[str, SensorReading]:
        """One reading per approach given the true sources heard by each mic.

        ``dt`` is the simulated time since the previous call, used only to
        advance the ambient-noise walk at the right rate.
        """
        readings: Dict[str, SensorReading] = {}
        for approach in self.approaches:
            sources = scene.get(approach, [])
            readings[approach] = self._sense_one(approach, sources, dt)
        return readings

    def _sense_one(self, approach: str, sources: Sequence[SourceState],
                   dt: float = 1.0) -> SensorReading:
        self.n_calls += 1
        if approach in self.dead_approaches:
            return SensorReading(approach, 0.0, {c: 1 / 3 for c in SIREN_CLASSES})
        dominant = self._dominant(sources)
        in_range = dominant is not None and dominant.distance_m <= self.cfg.detection_radius_m

        if self.mode == "none":
            return SensorReading(approach, 0.0, {c: 1 / 3 for c in SIREN_CLASSES})

        if self.mode == "oracle":
            # "Perfect detection" has to mean perfect detection *of this
            # approach*, not "some emergency vehicle exists nearby". Gating on
            # range alone reports every emergency vehicle at every microphone,
            # because all four microphones sit within a couple of hundred metres
            # of each other. That is not an upper bound on sensing, it is a
            # different and much worse sensor: it drove a hard-preemption
            # controller to seize all four phases at once and gridlock the
            # intersection. An oracle therefore reports only vehicles that are
            # inbound (closing) and genuinely on this approach (near the
            # microphone's axis).
            inbound = (
                in_range
                and dominant.radial_velocity_ms < -0.5
                and dominant.off_axis_rad <= self.ORACLE_ON_AXIS_RAD
            )
            if not inbound:
                return SensorReading(approach, 0.001, {c: 1 / 3 for c in SIREN_CLASSES})
            probs = {c: 0.0 for c in SIREN_CLASSES}
            probs[dominant.veh_class] = 1.0
            return SensorReading(
                approach, 0.999, probs, dominant.radial_velocity_ms, dominant.distance_m,
                spl_db=float(dominant.received_spl_db()),
                ambient_db=float(np.mean(self.cfg.ambient_noise_db_range)),
                front_back_db=self._front_back(dominant, noiseless=True),
            )

        ambient = self._ambient_db(approach, dt)

        if self.mode == "end_to_end":
            return self._sense_end_to_end(approach, dominant, ambient)

        # --- surrogate ---
        if not in_range:
            p, probs, _, _ = self.error_model.sample(self.rng, None, None)
            if p >= 0.5:
                self.n_false_positive += 1
            return SensorReading(approach, p, probs, ambient_db=float(ambient))

        # The mast measures a level; it does not evaluate a propagation model.
        # The noise is applied once here and everything downstream -- the SNR
        # the detector is conditioned on, the level the filter scores, the
        # trend the corridor differences -- sees the same measured number, so
        # they cannot disagree about what the microphone heard.
        spl = dominant.received_spl_db() + self._level_noise()
        snr = spl - ambient

        if dominant.veh_class == "horn":
            # The loudest thing this mast can hear is a car horn. It is not an
            # emergency vehicle, and the detector's job is to say so -- but it
            # does not always, and how often it fails is the horn confusion
            # curve measured on the CNN's held-out predictions rather than
            # assumed here.
            #
            # When it does fail, what it emits is a *detection with no vehicle
            # behind it*: the range head was never trained on horns and the
            # class head is guessing, so the reading carries a range drawn from
            # the head's own error rather than from any true kinematics. That is
            # exactly the clutter the joint filter's uniform density exists to
            # absorb, and feeding it a confident wrong range is the honest
            # version of the failure.
            return self._sense_horn(approach, dominant, snr, ambient)

        p, probs, vr_err, logr_err = self.error_model.sample(self.rng, snr, dominant.veh_class)
        if p >= 0.5:
            self.n_true_positive += 1
        else:
            self.n_false_negative += 1
        vr_hat = dominant.radial_velocity_ms + (vr_err or 0.0)
        r_hat = float(np.exp(np.log(max(dominant.distance_m, 1.0)) + (logr_err or 0.0)))
        r_hat = float(np.clip(r_hat, 1.0, 3.0 * self.cfg.detection_radius_m))
        return SensorReading(approach, p, probs, vr_hat, r_hat,
                             spl_db=float(spl), ambient_db=float(ambient),
                             front_back_db=self._front_back(dominant))

    def _sense_horn(self, approach: str, dominant, snr: float,
                    ambient: float) -> SensorReading:
        """One reading when the dominant source at this mast is a road horn."""
        em = self.error_model
        spl_measured = snr + ambient
        b = int(np.clip(np.digitize([snr], em.snr_edges)[0], 0, em.n_bins - 1))
        mu, sd = em.horn_logit(b)
        logit = float(self.rng.normal(mu, sd))
        p = float(1.0 / (1.0 + np.exp(-logit)))
        if p >= 0.5:
            self.n_false_positive += 1

        # The class posterior is near-uniform: a detector fooled by a horn has
        # no information about *which* siren it thinks it heard.
        probs = {c: 1.0 / len(SIREN_CLASSES) for c in SIREN_CLASSES}
        if p < 0.5:
            return SensorReading(approach, p, probs, spl_db=float(spl_measured),
                                 ambient_db=float(ambient),
                                 front_back_db=self._front_back(dominant))

        # Fooled: emit a range from the head's own error about the horn's true
        # distance, and a range rate near zero, because a honking vehicle is
        # stationary. Both are wrong in the way that matters -- they describe a
        # vehicle that is not coming.
        logr_err = float(self.rng.normal(
            em.logr_bias[b] if em.logr_bias else 0.0,
            em.logr_std[b] if em.logr_std else 0.5))
        r_hat = float(np.clip(np.exp(np.log(max(dominant.distance_m, 1.0)) + logr_err),
                              1.0, 3.0 * self.cfg.detection_radius_m))
        vr_hat = float(self.rng.normal(0.0, em.vr_std[b] if em.vr_std else 5.0))
        return SensorReading(approach, p, probs, vr_hat, r_hat,
                             spl_db=float(spl_measured),
                             ambient_db=float(ambient),
                             front_back_db=self._front_back(dominant))

    def _sense_end_to_end(
        self,
        approach: str,
        dominant: Optional[SourceState],
        ambient: float,
    ) -> SensorReading:
        assert self._synth is not None
        if dominant is None or dominant.distance_m > self.cfg.detection_radius_m:
            obs = self._synth.synthesize(None, ambient_db=ambient)
        else:
            obs = self._synth.synthesize(
                dominant.veh_class,
                dominant.distance_m,
                dominant.radial_velocity_ms,
                ambient_db=ambient,
                gain_db=dominant.directivity_db,
            )
        feats = self.mel(obs.waveform)[None, :, :, None]
        class_probs, vr_norm, logr = self.detector.predict_frame(feats)
        p_siren = float(siren_probability(class_probs, self.cfg.class_map))
        siren_probs = np.array([class_probs[self.cfg.class_map[c]] for c in SIREN_CLASSES])
        siren_probs = siren_probs / max(siren_probs.sum(), 1e-12)
        probs = {c: float(siren_probs[i]) for i, c in enumerate(SIREN_CLASSES)}

        spl = float(obs.received_spl_db)
        if p_siren < 0.5:
            return SensorReading(approach, p_siren, probs,
                                 spl_db=spl, ambient_db=float(ambient))
        vr_hat = float(vr_norm * self.cfg.max_radial_speed_ms)
        r_hat = float(np.clip(np.exp(logr), 1.0, 3.0 * self.cfg.detection_radius_m))
        return SensorReading(approach, p_siren, probs, vr_hat, r_hat,
                             spl_db=spl, ambient_db=float(ambient))

    # ------------------------------------------------------------------ stats
    def stats(self) -> Dict[str, float]:
        return {
            "sensor_calls": self.n_calls,
            "sensor_true_positive": self.n_true_positive,
            "sensor_false_positive": self.n_false_positive,
            "sensor_false_negative": self.n_false_negative,
        }

    def reset_stats(self) -> None:
        self.n_calls = self.n_true_positive = self.n_false_positive = self.n_false_negative = 0
        self._ambient = {}
