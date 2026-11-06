"""Physically-grounded siren synthesis with Doppler, absorption and scene noise.

Why synthesise rather than use a corpus
---------------------------------------
The detector must regress *range*, because range is what turns a detection into
an arrival-time estimate the controller can act on. No public siren corpus
carries range or range-rate labels; they simply are not recoverable from a
downloaded clip. Synthesising from the physics gives exact labels tied to the
emitting vehicle's true kinematics in SUMO.

Signal chain
------------
1. **Source.** Electronic sirens sweep a fundamental between roughly 500 and
   1800 Hz through a horn that adds strong harmonics. Three standard patterns
   are modelled: ``wail`` (slow ~4-6 s sweep), ``yelp`` (fast ~0.2-0.4 s sweep),
   ``hilo`` (European two-tone).
2. **Doppler.** ``f_obs = f_src * c / (c + v_r)``, ``v_r`` the range rate
   (positive receding), applied to the instantaneous frequency before phase
   integration.
3. **Propagation, per harmonic.** Spherical spreading ``-20 log10 r`` plus
   *frequency-dependent* atmospheric absorption from ISO 9613-1. The frequency
   dependence is essential rather than cosmetic: absorption at 4 kHz is roughly
   17x that at 500 Hz, so distance imposes a spectral **tilt** on top of a level
   drop. Level alone cannot be inverted for range - a near source off the
   microphone's axis and a far source on-axis produce the same level - but the
   tilt is unaffected by the microphone's (frequency-flat) directivity, so it
   supplies the range cue that breaks that ambiguity. An earlier version applied
   a flat absorption coefficient and the range head could not do better than 57 %
   error as a result.
4. **Scene.** Pink traffic rumble plus broadband tyre noise at a sampled ambient
   level, then a short exponential reverb tail for the urban canyon.

Everything is carried in pascals so absolute level remains a usable cue (see
``dsp.MelSpectrogram``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple  # noqa: F401  (used in annotations)

import numpy as np

from ..config import AcousticConfig
from .dsp import db_to_pa, exponential_reverb_ir, fft_convolve, pink_noise

# ISO 9613-1 atmospheric absorption at 20 C, 50 % relative humidity, dB/km.
_ABSORPTION_FREQ_HZ = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
_ABSORPTION_DB_PER_KM = np.array([0.1, 0.4, 1.0, 1.9, 3.7, 9.7, 32.8, 117.0])


def absorption_db_per_m(freq_hz: np.ndarray | float) -> np.ndarray:
    """Atmospheric absorption coefficient, interpolated log-linearly in frequency."""
    f = np.clip(np.asarray(freq_hz, dtype=np.float64), 20.0, 16000.0)
    return np.interp(
        np.log10(f), np.log10(_ABSORPTION_FREQ_HZ), _ABSORPTION_DB_PER_KM
    ) / 1000.0


@dataclass
class SirenPattern:
    """Parameter ranges for one siren waveform family."""

    name: str
    f_lo_range: Tuple[float, float]
    f_hi_range: Tuple[float, float]
    period_range: Tuple[float, float]


PATTERNS: Dict[str, SirenPattern] = {
    "wail": SirenPattern("wail", (560.0, 700.0), (1300.0, 1600.0), (3.8, 6.0)),
    "yelp": SirenPattern("yelp", (620.0, 780.0), (1400.0, 1750.0), (0.20, 0.40)),
    "hilo": SirenPattern("hilo", (420.0, 520.0), (620.0, 760.0), (0.85, 1.15)),
}

# Which patterns each vehicle class uses, and how often. The overlap is
# deliberate and true to practice: an ambulance and a fire engine may both wail,
# which is exactly why class discrimination from a short frame is hard.
CLASS_PATTERNS: Dict[str, List[Tuple[str, float]]] = {
    "ambulance": [("wail", 0.60), ("yelp", 0.30), ("hilo", 0.10)],
    "fire": [("wail", 0.45), ("hilo", 0.45), ("yelp", 0.10)],
    "police": [("yelp", 0.65), ("wail", 0.20), ("hilo", 0.15)],
}

# Class-characteristic timbre: fire appliances use larger, lower horns; police
# units use compact high-output speakers. This is the cue the CNN can exploit.
CLASS_TIMBRE: Dict[str, Dict[str, float]] = {
    "ambulance": {"freq_scale": 1.00, "rolloff": 1.35, "source_spl": 112.0},
    "fire": {"freq_scale": 0.84, "rolloff": 1.05, "source_spl": 116.0},
    "police": {"freq_scale": 1.12, "rolloff": 1.55, "source_spl": 109.0},
    # An ordinary vehicle horn. Not an emergency class -- it is the *confuser*,
    # and it is the one this detector previously had no concept of.
    "horn": {"freq_scale": 1.00, "rolloff": 1.15, "source_spl": 107.0},
}

HORN_TONES_HZ: Tuple[Tuple[float, float], ...] = (
    (370.0, 440.0),   # F#4 + A4, a minor third -- the common European pairing
    (400.0, 500.0),   # G4 + B4, a major third
    (440.0, 550.0),
    (320.0, 400.0),   # larger vehicles sit lower
    (500.0, 620.0),   # compact cars sit higher
)
"""Dual-tone pairs a road horn emits. Almost every automotive horn is two
diaphragms sounded together a third apart, which is what gives the sound its
characteristic beating roughness.

The frequencies overlap the low end of the siren band on purpose, because they
really do: a 500 Hz horn tone and the bottom of a ``hilo`` siren sweep sit on
top of each other. Level and band occupancy therefore cannot separate them, and
a detector trained only against broadband traffic noise -- which is what this
one was, before horns existed in the model at all -- has been asked to reject a
confuser it has never seen."""

HORN_SOURCE_SPL_RANGE_DB: Tuple[float, float] = (100.0, 112.0)
"""Horn level at 1 m. UNECE R28 requires 105-118 dB(A) at 2 m for the type
approval test; measured at 1 m and allowing for wear and for the cheap
aftermarket end of the range, this is the operating envelope. It overlaps the
siren range (109-116 dB) almost completely, which is the point: **a horn close
by is louder at the mast than a siren far away**, so no level threshold
separates them."""


def horn_instantaneous_frequency(t: np.ndarray, f0: float) -> np.ndarray:
    """A horn's fundamental over time: flat, to within a little wobble.

    This is the whole discriminator, and it is worth stating as a physical claim
    rather than as a feature choice. A siren *sweeps* -- ``wail`` glides over
    roughly an octave in 4-6 s, ``yelp`` in 0.2-0.4 s, ``hilo`` alternates
    between two tones about once a second. A horn is a pair of fixed-pitch
    diaphragms and does not sweep at all.

    Over a one-second analysis frame that difference is stark and it survives
    low SNR: on a log-mel spectrogram a siren draws a moving ridge and a horn
    draws horizontal lines. It is also invariant to everything that makes level
    unusable -- distance, directivity, the ambient floor -- because it is a
    property of the source's *modulation*, not of its amplitude.

    The small drift modelled here is real: horn pitch falls slightly as
    the diaphragm heats and as battery voltage sags under load. It is
    deliberately included so the detector cannot key on "exactly constant" as a
    give-away that a real recording would not offer.
    """
    drift = 1.0 - 0.012 * (t / max(float(t[-1]) if t.size else 1.0, 1e-6))
    wobble = 1.0 + 0.004 * np.sin(2.0 * np.pi * 7.0 * t)
    return f0 * drift * wobble

N_HARMONICS = 8
NOMINAL_FUNDAMENTAL_HZ = 950.0
"""Mid-sweep fundamental, used for the closed-form level estimate the surrogate
sensor needs (it must agree with the rendered waveform's level)."""


def harmonic_amplitudes(rolloff: float, n: int = N_HARMONICS) -> np.ndarray:
    """Unit-power harmonic amplitude envelope ``a_k ~ k^-rolloff``."""
    a = np.arange(1, n + 1, dtype=np.float64) ** (-rolloff)
    return a / np.sqrt(np.sum(a ** 2))


def received_spl_db(
    veh_class: str,
    distance_m: float,
    directivity_db: float = 0.0,
    fundamental_hz: float = NOMINAL_FUNDAMENTAL_HZ,
) -> float:
    """Total received level, summing harmonics under spreading and absorption.

    Shared by the synthesiser and the surrogate sensor so that the SNR the
    surrogate conditions on is the same quantity the CNN actually saw.
    """
    r = max(float(distance_m), 1.0)
    timbre = CLASS_TIMBRE[veh_class]
    amps = harmonic_amplitudes(timbre["rolloff"])
    freqs = fundamental_hz * np.arange(1, N_HARMONICS + 1)
    level_k = (
        timbre["source_spl"]
        + 20.0 * np.log10(amps)
        - 20.0 * np.log10(r)
        - absorption_db_per_m(freqs) * r
        + directivity_db
    )
    return float(10.0 * np.log10(np.sum(10.0 ** (level_k / 10.0))))


@dataclass
class SirenObservation:
    """One synthesised microphone frame plus its ground-truth labels."""

    waveform: np.ndarray
    label: str
    """``none``, ``ambulance``, ``fire`` or ``police``."""
    distance_m: float
    radial_velocity_ms: float
    """Positive means receding."""
    ambient_db: float
    received_spl_db: float
    pattern: Optional[str] = None
    off_axis_rad: float = 0.0

    @property
    def snr_db(self) -> float:
        return self.received_spl_db - self.ambient_db

    @property
    def eta_s(self) -> float:
        """Time until the source reaches the microphone; ``inf`` if receding."""
        if self.radial_velocity_ms >= -0.1:
            return float("inf")
        return self.distance_m / abs(self.radial_velocity_ms)


class SirenSynthesizer:
    def __init__(self, cfg: Optional[AcousticConfig] = None, seed: int = 0):
        self.cfg = cfg or AcousticConfig()
        self.rng = np.random.default_rng(seed)
        self.n_samples = int(round(self.cfg.sample_rate * self.cfg.frame_s))
        self._ir_bank = [
            exponential_reverb_ir(self.cfg.sample_rate, self.cfg.reverb_rt60_s, self.rng)
            for _ in range(8)
        ]

    # -------------------------------------------------------------- source model
    def _sample_pattern(self, veh_class: str) -> str:
        if veh_class == "horn":
            return "horn"
        options = CLASS_PATTERNS[veh_class]
        names = [n for n, _ in options]
        probs = np.array([p for _, p in options], dtype=np.float64)
        return str(self.rng.choice(names, p=probs / probs.sum()))

    def _instantaneous_frequency(
        self, pattern: str, t: np.ndarray, veh_class: str
    ) -> np.ndarray:
        if pattern == "horn":
            # Two diaphragms sounded together. The pair is returned as a single
            # fundamental and the second tone is added as its own partial series
            # in `synthesize`, so the beating between them survives.
            lo, _hi = HORN_TONES_HZ[int(self.rng.integers(len(HORN_TONES_HZ)))]
            self._horn_pair = (lo, _hi)
            return horn_instantaneous_frequency(t, lo)
        spec = PATTERNS[pattern]
        scale = CLASS_TIMBRE[veh_class]["freq_scale"]
        f_lo = self.rng.uniform(*spec.f_lo_range) * scale
        f_hi = self.rng.uniform(*spec.f_hi_range) * scale
        period = self.rng.uniform(*spec.period_range)
        phase0 = self.rng.uniform(0.0, period)
        u = ((t + phase0) % period) / period

        if pattern == "hilo":
            sq = np.tanh(np.sin(2 * np.pi * u) * 12.0)
            return f_lo + (f_hi - f_lo) * (0.5 * (sq + 1.0))
        if pattern == "yelp":
            return f_lo + (f_hi - f_lo) * u
        return f_lo + (f_hi - f_lo) * 0.5 * (1.0 - np.cos(2 * np.pi * u))

    # ------------------------------------------------------------------ scenes
    def synthesize(
        self,
        veh_class: Optional[str],
        distance_m: float = 0.0,
        radial_velocity_ms: float = 0.0,
        ambient_db: Optional[float] = None,
        reverb: bool = True,
        gain_db: float = 0.0,
        off_axis_rad: float = 0.0,
    ) -> SirenObservation:
        """Render one frame of the microphone signal, in pascals.

        ``veh_class=None`` renders a noise-only frame. ``gain_db`` is a level
        offset applied to the source only, carrying the microphone's directional
        response (which is frequency-flat, and so leaves the spectral tilt that
        encodes range intact).
        """
        cfg = self.cfg
        n = self.n_samples
        t = np.arange(n) / cfg.sample_rate

        if ambient_db is None:
            ambient_db = float(self.rng.uniform(*cfg.ambient_noise_db_range))

        noise = 0.85 * pink_noise(n, self.rng) + 0.15 * self.rng.standard_normal(n)
        noise /= max(np.sqrt(np.mean(noise ** 2)), 1e-12)
        signal = noise * db_to_pa(ambient_db)

        total_spl = -np.inf
        pattern = None

        if veh_class is not None:
            timbre = CLASS_TIMBRE[veh_class]
            # Range varies across the frame because the source is moving.
            r_t = np.maximum(distance_m + radial_velocity_ms * t, 1.0)
            doppler = cfg.speed_of_sound / (cfg.speed_of_sound + radial_velocity_ms)

            pattern = self._sample_pattern(veh_class)
            f_src = self._instantaneous_frequency(pattern, t, veh_class)
            f_obs = f_src * doppler
            dt = 1.0 / cfg.sample_rate
            phase = 2.0 * np.pi * np.cumsum(f_obs) * dt

            if veh_class == "horn":
                # Drawn per press rather than fixed per class: horns vary far
                # more between vehicles than sirens do between services, and the
                # range overlaps the siren range almost completely.
                timbre = dict(timbre)
                timbre["source_spl"] = float(self.rng.uniform(*HORN_SOURCE_SPL_RANGE_DB))
            rolloff = timbre["rolloff"] * self.rng.uniform(0.92, 1.08)
            amps = harmonic_amplitudes(rolloff)
            spreading = -20.0 * np.log10(r_t)
            nyquist = cfg.sample_rate / 2.0

            wave = np.zeros(n, dtype=np.float64)
            power_sum = 0.0
            for k in range(1, N_HARMONICS + 1):
                f_k = f_obs * k
                if np.min(f_k) >= nyquist:
                    break
                alive = f_k < nyquist
                # Each harmonic is attenuated by its own absorption coefficient;
                # this is what imprints the range-dependent spectral tilt.
                alpha_k = absorption_db_per_m(f_k)
                level_k = (
                    timbre["source_spl"]
                    + 20.0 * np.log10(amps[k - 1])
                    + spreading
                    - alpha_k * r_t
                    + gain_db
                )
                amp_pa = db_to_pa(level_k) * np.sqrt(2.0) * alive
                wave += amp_pa * np.sin(k * phase + self.rng.uniform(0, 2 * np.pi))
                power_sum += float(np.mean((amp_pa / np.sqrt(2.0)) ** 2))

            if veh_class == "horn":
                # The second diaphragm. Rendered as its own harmonic stack at
                # the same instant rather than mixed in afterwards, so the two
                # tones beat against each other the way a real pair does -- the
                # roughness is part of what the sound *is*.
                lo, hi = getattr(self, "_horn_pair", HORN_TONES_HZ[0])
                f2 = f_obs * (hi / max(lo, 1e-6))
                phase2 = 2.0 * np.pi * np.cumsum(f2) * dt
                for k in range(1, N_HARMONICS + 1):
                    f_k = f2 * k
                    if np.min(f_k) >= nyquist:
                        break
                    alive = f_k < nyquist
                    alpha_k = absorption_db_per_m(f_k)
                    level_k = (
                        timbre["source_spl"]
                        + 20.0 * np.log10(amps[k - 1])
                        + spreading - alpha_k * r_t + gain_db
                        - 1.5           # the upper diaphragm is the quieter one
                    )
                    amp_pa = db_to_pa(level_k) * np.sqrt(2.0) * alive
                    wave += amp_pa * np.sin(k * phase2 + self.rng.uniform(0, 2 * np.pi))
                    power_sum += float(np.mean((amp_pa / np.sqrt(2.0)) ** 2))

                # A horn is pressed, not run continuously. The burst envelope
                # matters to the classifier: a frame may catch a horn's onset,
                # its tail, or a gap between two presses, and a detector that
                # only ever saw full-frame horns would not reject a partial one.
                wave = wave * self._horn_envelope(t)

            signal = signal + wave
            total_spl = float(20.0 * np.log10(max(np.sqrt(power_sum), 1e-12) / 2e-5))

        if reverb:
            ir = self._ir_bank[int(self.rng.integers(len(self._ir_bank)))]
            signal = fft_convolve(signal, ir)

        return SirenObservation(
            waveform=signal,
            label=veh_class if veh_class is not None else "none",
            distance_m=float(distance_m),
            radial_velocity_ms=float(radial_velocity_ms),
            ambient_db=float(ambient_db),
            received_spl_db=total_spl,
            pattern=pattern,
            off_axis_rad=float(off_axis_rad),
        )

    def _horn_envelope(self, t: np.ndarray) -> np.ndarray:
        """Press-and-release gating for a horn burst.

        Bursts of 0.25-1.6 s with 10 ms attack and 40 ms release, and a random
        offset so a one-second frame can land anywhere in the press. Roughly one
        frame in six catches only silence or a tail, which is exactly the
        partial evidence the classifier has to handle without either firing or
        going deaf.
        """
        dur = float(self.rng.uniform(0.25, 1.6))
        start = float(self.rng.uniform(-0.6, max(float(t[-1]) if t.size else 1.0, 1e-6)))
        env = np.clip((t - start) / 0.010, 0.0, 1.0)
        env *= np.clip((start + dur - t) / 0.040, 0.0, 1.0)
        return env

    # ------------------------------------------------------------------ dataset
    def _sample_off_axis(self) -> float:
        """Bearing distribution matching the deployed microphone geometry.

        A microphone mounted up its own approach sees vehicles on that approach
        almost head-on, and vehicles on every other approach from well off-axis.
        The angle distribution is therefore bimodal, not uniform - training on a
        uniform bearing would over-represent bearings that never occur and
        weaken the range head where it is actually used.
        """
        if self.rng.random() < 0.45:
            return float(self.rng.uniform(0.0, np.deg2rad(20.0)))
        return float(self.rng.uniform(np.deg2rad(55.0), np.pi))

    def sample_scene(self, p_siren: float = 0.5,
                     p_horn: float = 0.25) -> SirenObservation:
        """Draw a random training scene from the operating envelope.

        ``p_horn`` is the share of the *non-siren* half given to horn presses
        rather than to bare traffic noise. A quarter is far above the duty cycle
        any real junction sustains, and deliberately so: the negative class only
        teaches rejection in proportion to how often the hard negative appears,
        and bare pink noise is not a hard negative for a siren detector at all.
        """
        cfg = self.cfg
        if self.rng.random() >= p_siren:
            if self.rng.random() < p_horn:
                # A horn press. Close, because that is when it is confusable:
                # honking happens in the queue, which is at the stop line, which
                # is where the mast is listening hardest.
                dist = float(self.rng.uniform(5.0, 120.0))
                theta = self._sample_off_axis()
                g = 0.5 * (1.0 + np.cos(theta))
                floor = 10.0 ** (-cfg.directivity_front_back_db / 20.0)
                return self.synthesize(
                    "horn", dist, float(self.rng.uniform(-8.0, 8.0)),
                    gain_db=20.0 * np.log10(max(g, floor)), off_axis_rad=theta)
            return self.synthesize(None)

        veh_class = str(self.rng.choice(["ambulance", "fire", "police"]))
        distance = float(self.rng.uniform(5.0, cfg.detection_radius_m))
        # Mostly approaching (the operationally relevant case), but receding and
        # near-tangential cases must appear or the model will call every
        # detection inbound.
        u = self.rng.random()
        if u < 0.70:
            v_r = -float(self.rng.uniform(3.0, cfg.max_radial_speed_ms))
        elif u < 0.90:
            v_r = float(self.rng.uniform(3.0, cfg.max_radial_speed_ms))
        else:
            v_r = float(self.rng.uniform(-3.0, 3.0))

        theta = self._sample_off_axis()
        g = 0.5 * (1.0 + np.cos(theta))
        floor = 10.0 ** (-cfg.directivity_front_back_db / 20.0)
        gain_db = 20.0 * np.log10(max(g, floor))
        return self.synthesize(veh_class, distance, v_r, gain_db=gain_db, off_axis_rad=theta)

    def build_dataset(
        self,
        n_clips: int,
        mel,
        p_siren: float = 0.5,
        progress_every: int = 0,
    ) -> Dict[str, np.ndarray]:
        """Generate ``n_clips`` labelled log-mel frames.

        Returns ``x`` (N, n_mels, n_frames, 1), ``y_class``, ``y_vr``
        (normalised range rate), ``y_logr`` (natural log of range in metres),
        plus ``snr_db``, ``distance_m`` and ``off_axis_rad`` for stratified
        evaluation.
        """
        cfg = self.cfg
        class_map = cfg.class_map
        probe = mel(np.zeros(self.n_samples))
        x = np.zeros((n_clips, probe.shape[0], probe.shape[1], 1), dtype=np.float32)
        y_class = np.zeros(n_clips, dtype=np.int64)
        y_vr = np.zeros(n_clips, dtype=np.float32)
        y_logr = np.zeros(n_clips, dtype=np.float32)
        snr = np.zeros(n_clips, dtype=np.float32)
        dist = np.zeros(n_clips, dtype=np.float32)
        theta = np.zeros(n_clips, dtype=np.float32)

        for i in range(n_clips):
            obs = self.sample_scene(p_siren=p_siren)
            x[i, :, :, 0] = mel(obs.waveform)
            y_class[i] = class_map[obs.label]
            y_vr[i] = obs.radial_velocity_ms / cfg.max_radial_speed_ms
            # A horn has a range, but the range head is only ever read for a
            # siren, so training it on horns would spend capacity regressing a
            # quantity nothing consumes.
            is_siren = obs.label not in ("none", "horn")
            y_logr[i] = np.log(max(obs.distance_m, 1.0)) if is_siren else 0.0
            snr[i] = obs.snr_db if obs.label != "none" else -np.inf
            dist[i] = obs.distance_m
            theta[i] = obs.off_axis_rad
            if progress_every and (i + 1) % progress_every == 0:
                print(f"    synthesised {i + 1}/{n_clips} clips", flush=True)

        return {
            "x": x,
            "y_class": y_class,
            "y_vr": y_vr,
            "y_logr": y_logr,
            "snr_db": snr,
            "distance_m": dist,
            "off_axis_rad": theta,
        }
