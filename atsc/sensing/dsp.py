"""Minimal, dependency-free signal processing.

Implemented in pure NumPy rather than pulling in librosa/scipy so that the
feature extraction a reviewer runs is byte-identical to the one used here, with
no transitive-dependency drift. The mel filterbank follows the Slaney-style
triangular construction used by HTK/librosa.

Frames are **not** per-frame energy-normalised. Absolute level is a physical cue
for source distance, and the detector's distance head depends on it; normalising
it away would silently destroy the range estimate. This is why the model assumes
a *calibrated* microphone, as fixed roadside acoustic sensors are.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Reference sound pressure (20 uPa) used to convert Pa <-> dB SPL.
P_REF = 2e-5


def hz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float
) -> np.ndarray:
    """Triangular mel filterbank, shape ``(n_mels, n_fft // 2 + 1)``."""
    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_bins)

    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for i in range(n_mels):
        lo, ctr, hi = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        if hi <= lo:
            continue
        rising = (fft_freqs - lo) / max(ctr - lo, 1e-9)
        falling = (hi - fft_freqs) / max(hi - ctr, 1e-9)
        fb[i] = np.clip(np.minimum(rising, falling), 0.0, None)
    # Unit-*peak* triangles, deliberately not Slaney (unit-area) normalised.
    # Slaney scaling divides each filter by its bandwidth in hertz, which at
    # 1 kHz is a factor of ~150 and therefore shifts the output about 15 dB off
    # the true sound pressure level. That is harmless for speech recognition,
    # where only relative shape matters, but fatal here: the detector's range
    # head reads absolute level, so the filterbank has to behave like an
    # acoustic band analyser and report true band levels.
    return fb


def frame_signal(x: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Split ``x`` into overlapping frames, shape ``(n_frames, frame_length)``."""
    if x.shape[0] < frame_length:
        x = np.pad(x, (0, frame_length - x.shape[0]))
    n_frames = 1 + (x.shape[0] - frame_length) // hop_length
    idx = np.arange(frame_length)[None, :] + hop_length * np.arange(n_frames)[:, None]
    return x[idx]


@dataclass
class MelSpectrogram:
    """Log-mel front-end with an absolute (dB SPL referenced) magnitude scale."""

    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 160
    n_mels: int = 40
    fmin: float = 50.0
    fmax: float = 7500.0
    floor_db: float = -20.0
    """Clamp on the output in dB; roughly the quietest urban night-time level."""

    def __post_init__(self) -> None:
        self._fb = mel_filterbank(
            self.sample_rate, self.n_fft, self.n_mels, self.fmin, self.fmax
        )
        self._window = np.hanning(self.n_fft).astype(np.float64)
        # Power-preserving (Parseval) normalisation: with this constant, summing
        # the one-sided bin powers over a band recovers the mean-square pressure
        # in that band, independently of the window shape. Normalising by the
        # window's coherent gain instead would be correct for pure tones and
        # wrong by the window's noise bandwidth for the broadband part of the
        # signal - and a siren in traffic noise is both.
        self._power_norm = self.n_fft * float(np.sum(self._window ** 2))

    def n_frames(self, n_samples: int) -> int:
        return 1 + max(n_samples - self.n_fft, 0) // self.hop_length

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """Waveform in pascals -> log-mel band levels in dB SPL.

        Output shape ``(n_mels, n_frames)``. Values are true band sound pressure
        levels, so a source 6 dB louder produces an output 6 dB higher.
        """
        x = np.asarray(waveform, dtype=np.float64)
        frames = frame_signal(x, self.n_fft, self.hop_length) * self._window
        spec = np.abs(np.fft.rfft(frames, n=self.n_fft, axis=-1)) ** 2
        power = 2.0 * spec / self._power_norm  # one-sided mean-square, Pa^2
        power[..., 0] *= 0.5
        if self.n_fft % 2 == 0:
            power[..., -1] *= 0.5
        mel = power @ self._fb.T  # (n_frames, n_mels)
        db = 10.0 * np.log10(np.maximum(mel, 1e-20) / (P_REF ** 2))
        return np.maximum(db, self.floor_db).T.astype(np.float32)

    @property
    def output_shape_for(self):
        def _shape(n_samples: int):
            return (self.n_mels, self.n_frames(n_samples))
        return _shape


def db_to_pa(spl_db: float | np.ndarray) -> np.ndarray:
    """Convert a sound pressure level in dB SPL to an RMS pressure in pascals."""
    return P_REF * 10.0 ** (np.asarray(spl_db, dtype=np.float64) / 20.0)


def pa_to_db(pressure: float | np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(np.asarray(pressure)), 1e-12) / P_REF)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-RMS pink (1/f) noise via spectral shaping of white noise."""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    shaped = np.fft.irfft(spec * scale, n=n)
    rms = np.sqrt(np.mean(shaped ** 2))
    return shaped / max(rms, 1e-12)


def exponential_reverb_ir(
    sample_rate: int, rt60_s: float, rng: np.random.Generator, length_s: float = 0.25
) -> np.ndarray:
    """Exponentially decaying noise burst approximating an urban canyon tail."""
    n = max(int(sample_rate * length_s), 1)
    t = np.arange(n) / sample_rate
    decay = 10.0 ** (-3.0 * t / max(rt60_s, 1e-3))
    ir = rng.standard_normal(n) * decay
    ir[0] += 1.0  # direct path
    ir /= np.sqrt(np.sum(ir ** 2))
    return ir


def fft_convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Linear convolution keeping the first ``len(x)`` samples."""
    n = 1 << int(np.ceil(np.log2(len(x) + len(h) - 1)))
    y = np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(h, n), n)
    return y[: len(x)]
