"""Audible sirens: what the intersection is hearing, played out loud.

Why this is in the repository rather than in a demo script
----------------------------------------------------------
The controller's only emergency-vehicle sensor is a microphone, and every
decision it takes is downstream of a level and a spectrum. That makes the
system unusually hard to read from a map alone: the picture shows a vehicle
approaching, but not the quantity the controller is actually reacting to. The
whole argument -- *the siren gets louder, so the vehicle is getting closer, so
the corridor opens* -- is an argument about a sound, and it is far more legible
when you can hear it.

So this renders the scene the masts are hearing and plays it. It is not a sound
effect bolted on for atmosphere: the waveform comes from the same physical
model the detector is trained on and the joint filter's likelihood is built
from, so what you hear and what the controller decides on are the same signal.

* **Pattern and timbre** come from :mod:`atsc.sensing.siren_synth` -- the wail,
  yelp and hi-lo sweeps, and the per-class horn characteristics (a fire
  appliance's lower, louder horn; a police unit's compact high-output speaker).
* **Level** is :func:`~atsc.sensing.siren_synth.received_spl_db` evaluated at
  the vehicle's true distance from the junction: spherical spreading plus
  ISO 9613-1 atmospheric absorption, summed over harmonics. So the loudness
  really is the loudness, and it really does rise as the vehicle approaches --
  about 6 dB per halving of distance.
* **Pitch** carries the Doppler shift ``f * c / (c + v_r)``, which is why a
  vehicle that has crossed sounds different from one still inbound. It is a
  small effect at road speeds -- about 6 % -- and hearing how small it is
  explains, better than the ablation table does, why single-frame Doppler
  regression did not work.
* **Pan** places each vehicle left or right by its true bearing from the
  junction, so which arm a siren is on is audible as well as visible.

What it deliberately does *not* do is use the belief. This is the ground truth
of the scene, not the controller's estimate of it -- so when the filter is
wrong, you can hear that it is wrong.

Backends
--------
Real-time playback needs an output device, and nothing in this project's
dependency list provides one. Rather than add a dependency for a demo, the
renderer is separated from the sink and there are three sinks:

``winsound``   the Windows standard library. Each chunk is handed over as an
               in-memory WAV; phase is carried across chunks so the seam is
               inaudible.
``command``    ``aplay`` / ``paplay`` / ``afplay``, whichever exists.
``null``       no device: the mix is still rendered, and still recorded if a
               recording path was given.

``record_path`` writes the whole soundtrack to a WAV file whatever the sink is,
so the output is inspectable on a machine with no audio at all -- which is the
only way to check the thing in a test.
"""

from __future__ import annotations

import ctypes
import io
import math
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import AcousticConfig
from ..sensing.siren_synth import (
    CLASS_PATTERNS, CLASS_TIMBRE, N_HARMONICS, PATTERNS, harmonic_amplitudes,
    received_spl_db,
)

SAMPLE_RATE = 22050
"""Enough for the eighth harmonic of a 1.6 kHz sweep, small enough to render in
a fraction of real time on one core."""

CHUNK_S = 0.25
"""Render granularity. Short enough that the level tracks a vehicle moving at
14 m/s, long enough that the render loop is not the bottleneck."""

FLOOR_DB = 58.0
"""Level mapped to silence. Below the urban noise floor, so a siren that has
faded into the background is inaudible rather than quietly present."""

CEIL_DB = 104.0
"""Level mapped to full scale, i.e. a vehicle right at the junction."""

LEVEL_CURVE = 1.2
"""Shape of the level -> amplitude ramp.

The compression is already doing the work (see :func:`level_to_gain`); an
exponent much above 1 compresses it a second time and buries the part of the
approach the demonstration is about. At 1.5, where this started, a siren 300 m
out rendered at 0.016 of full scale -- inaudible on laptop speakers -- so the
corridor appeared to open in silence and only became audible once the vehicle
was at the line, which is the opposite of the thing being shown."""

MASTER_GAIN = 2.5
"""Pre-limiter gain on the summed mix.

One siren at 50 m peaked at 0.16 of full scale, about -16 dBFS, which is quiet
enough on built-in speakers to be mistaken for no sound at all -- and "there is
no siren" was the first thing anyone said about this demo. The limiter below is
what makes raising it safe: several vehicles at once compress rather than
clip."""


@dataclass
class AudioSource:
    """One emergency vehicle as the junction hears it."""

    key: str
    veh_class: str
    distance_m: float
    radial_velocity_ms: float
    """Positive means receding, matching the convention everywhere else here."""
    pan: float = 0.0
    """-1 hard left, +1 hard right, from the vehicle's true bearing."""


class _Voice:
    """Per-vehicle oscillator state, carried across chunks.

    Phase continuity is the whole reason this is a class. Re-rendering an
    independent frame each chunk -- which is what the training-data synthesiser
    does, correctly, because frames there are independent samples -- puts a
    click at every seam, four times a second.
    """

    def __init__(self, src: "AudioSource", rng: np.random.Generator):
        self.veh_class = src.veh_class
        if src.veh_class == "horn":
            # A horn is not a siren pattern: two fixed tones, no sweep. Given
            # its own branch rather than a degenerate PATTERNS entry, because
            # "a sweep with zero span" would still be re-randomised per chunk
            # and would wander audibly.
            from ..sensing.siren_synth import HORN_TONES_HZ
            lo, hi = HORN_TONES_HZ[int(rng.integers(len(HORN_TONES_HZ)))]
            self.pattern = "horn"
            self.f_lo = self.f_hi = float(lo)
            self.horn_hi = float(hi)
            self.period = 1.0
            self.sweep_t = 0.0
            self.phase = 0.0
            self.phase_hi = 0.0
            self.gain = 0.0
            self.pan = float(src.pan)
            return
        options = CLASS_PATTERNS[src.veh_class]
        names = [n for n, _ in options]
        probs = np.array([p for _, p in options], dtype=np.float64)
        self.pattern = str(rng.choice(names, p=probs / probs.sum()))
        spec = PATTERNS[self.pattern]
        scale = CLASS_TIMBRE[src.veh_class]["freq_scale"]
        self.f_lo = float(rng.uniform(*spec.f_lo_range)) * scale
        self.f_hi = float(rng.uniform(*spec.f_hi_range)) * scale
        self.period = float(rng.uniform(*spec.period_range))
        self.sweep_t = float(rng.uniform(0.0, self.period))
        self.phase = 0.0
        self.gain = 0.0
        self.pan = float(src.pan)

    def sweep(self, t: np.ndarray) -> np.ndarray:
        """Instantaneous source frequency over a chunk, as in the training synth."""
        if self.pattern == "horn":
            return np.full_like(t, self.f_lo)
        u = ((self.sweep_t + t) % self.period) / self.period
        if self.pattern == "hilo":
            sq = np.tanh(np.sin(2 * np.pi * u) * 12.0)
            return self.f_lo + (self.f_hi - self.f_lo) * (0.5 * (sq + 1.0))
        if self.pattern == "yelp":
            return self.f_lo + (self.f_hi - self.f_lo) * u
        return self.f_lo + (self.f_hi - self.f_lo) * 0.5 * (1.0 - np.cos(2 * np.pi * u))


def level_to_gain(spl_db: float) -> float:
    """Received level -> linear amplitude, on a perceptual rather than physical scale.

    A true pressure mapping spans 45 dB between a siren at the detection radius
    and one at the stop line, which is a factor of 180 in amplitude: the
    approach would be silent until the last few seconds and then deafening.
    Compressing it to a 0..1 ramp in decibels keeps what the demonstration is
    for -- loudness rising steadily as the vehicle closes -- audible over the
    whole approach, which is the same reason volume controls are logarithmic.
    """
    x = (float(spl_db) - FLOOR_DB) / max(CEIL_DB - FLOOR_DB, 1e-6)
    return float(np.clip(x, 0.0, 1.0)) ** LEVEL_CURVE


class SirenRenderer:
    """Phase-continuous mixer for a set of moving sirens."""

    def __init__(self, cfg: Optional[AcousticConfig] = None,
                 sample_rate: int = SAMPLE_RATE, seed: int = 0):
        self.cfg = cfg or AcousticConfig()
        self.sample_rate = int(sample_rate)
        self.rng = np.random.default_rng(seed)
        self._voices: Dict[str, _Voice] = {}
        self._sources: Dict[str, AudioSource] = {}

    def set_sources(self, sources: Sequence[AudioSource]) -> None:
        """Replace the scene. Voices persist for vehicles that are still here."""
        self._sources = {s.key: s for s in sources}
        for key in list(self._voices):
            if key not in self._sources:
                del self._voices[key]
        for key, src in self._sources.items():
            if key not in self._voices:
                self._voices[key] = _Voice(src, self.rng)
            self._voices[key].pan = src.pan

    @property
    def active(self) -> bool:
        return bool(self._sources)

    def render(self, duration_s: float) -> np.ndarray:
        """Render ``duration_s`` of the current scene as float32 stereo in [-1, 1]."""
        n = max(int(round(duration_s * self.sample_rate)), 1)
        out = np.zeros((n, 2), dtype=np.float64)
        if not self._sources:
            return out.astype(np.float32)

        t = np.arange(n) / self.sample_rate
        c = self.cfg.speed_of_sound
        nyquist = self.sample_rate / 2.0

        for key, src in self._sources.items():
            voice = self._voices[key]
            timbre = CLASS_TIMBRE[src.veh_class]
            amps = harmonic_amplitudes(timbre["rolloff"])

            # Distance over the chunk, so the level ramps within it rather than
            # stepping at the seam.
            r_t = np.maximum(src.distance_m + src.radial_velocity_ms * t, 1.0)
            doppler = c / (c + float(src.radial_velocity_ms))
            f_obs = voice.sweep(t) * doppler
            phase = voice.phase + 2.0 * np.pi * np.cumsum(f_obs) / self.sample_rate

            wave_out = np.zeros(n, dtype=np.float64)
            for k in range(1, N_HARMONICS + 1):
                if float(np.min(f_obs)) * k >= nyquist:
                    break
                wave_out += amps[k - 1] * np.sin(k * phase) * (f_obs * k < nyquist)

            # Level at the junction, ramped across the chunk and crossfaded from
            # the previous chunk's gain so a jump in the reported distance does
            # not click.
            spl_start = received_spl_db(src.veh_class, float(r_t[0]))
            spl_end = received_spl_db(src.veh_class, float(r_t[-1]))
            g_end = level_to_gain(spl_end)
            g_start = voice.gain if voice.gain > 0.0 else level_to_gain(spl_start)
            g = np.linspace(g_start, g_end, n)
            voice.gain = g_end

            # Constant-power pan: the total stays put as a vehicle crosses.
            angle = (float(np.clip(voice.pan, -1.0, 1.0)) + 1.0) * (math.pi / 4.0)
            body = wave_out * g
            out[:, 0] += body * math.cos(angle)
            out[:, 1] += body * math.sin(angle)

            voice.phase = float(phase[-1] % (2.0 * math.pi))
            voice.sweep_t = float((voice.sweep_t + duration_s) % voice.period)

        # Soft clip rather than hard: several vehicles at once should sound
        # crowded, not distorted.
        return np.tanh(out * MASTER_GAIN).astype(np.float32)


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------
def to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Float stereo in [-1, 1] -> a complete 16-bit PCM WAV file."""
    pcm = np.clip(np.asarray(samples), -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(pcm.shape[1] if pcm.ndim > 1 else 1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class _NullSink:
    """No device. The mix is still rendered, and still recorded."""

    name = "null"
    blocking = False

    def play(self, samples: np.ndarray, sample_rate: int) -> None:
        pass

    def close(self) -> None:
        pass


class _WaveOutSink:
    """Gapless streaming through the Windows ``winmm`` waveOut API.

    This exists because the obvious standard-library route does not work.
    ``winsound.PlaySound`` is fire-and-forget: asynchronously it *replaces* the
    sound currently playing, so a chunk stream cancels itself before anything
    reaches the device, and synchronously it costs a fixed ~150 ms of setup per
    call whatever the buffer length -- measured at 56 % duty for 250 ms chunks
    and still only 88 % at one second. There is no chunk size at which it
    streams; the gap is per call, not per sample.

    waveOut is the API the problem actually wants. Buffers are *queued* on the
    device and play back to back, so the only requirement is to keep the queue
    non-empty, and the audio hardware provides the timing rather than a timer
    in Python racing it. Three buffers of a quarter-second each give ~750 ms of
    slack, which is far more than a render takes and enough to ride out an
    interpreter pause.

    Written against ctypes rather than taking a dependency: the project ships
    with NumPy, TensorFlow and matplotlib, and adding an audio library so a
    demonstration can make a noise is a poor trade for anyone installing this.
    """

    name = "waveout"
    blocking = False

    _WHDR_DONE = 0x00000001
    _WAVE_MAPPER = 0xFFFFFFFF
    _WAVE_FORMAT_PCM = 1
    _N_BUFFERS = 3

    class _WAVEFORMATEX(ctypes.Structure):
        _fields_ = [("wFormatTag", ctypes.c_ushort),
                    ("nChannels", ctypes.c_ushort),
                    ("nSamplesPerSec", ctypes.c_uint),
                    ("nAvgBytesPerSec", ctypes.c_uint),
                    ("nBlockAlign", ctypes.c_ushort),
                    ("wBitsPerSample", ctypes.c_ushort),
                    ("cbSize", ctypes.c_ushort)]

    class _WAVEHDR(ctypes.Structure):
        pass

    def __init__(self, sample_rate: int = SAMPLE_RATE, channels: int = 2):
        if sys.platform != "win32":
            raise RuntimeError("waveOut is Windows only")
        _WAVEHDR = self._WAVEHDR
        _WAVEHDR._fields_ = [("lpData", ctypes.c_void_p),
                             ("dwBufferLength", ctypes.c_uint),
                             ("dwBytesRecorded", ctypes.c_uint),
                             ("dwUser", ctypes.c_void_p),
                             ("dwFlags", ctypes.c_uint),
                             ("dwLoops", ctypes.c_uint),
                             ("lpNext", ctypes.c_void_p),
                             ("reserved", ctypes.c_void_p)]
        self._winmm = ctypes.WinDLL("winmm")
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)

        fmt = self._WAVEFORMATEX(
            wFormatTag=self._WAVE_FORMAT_PCM, nChannels=self.channels,
            nSamplesPerSec=self.sample_rate,
            nAvgBytesPerSec=self.sample_rate * self.channels * 2,
            nBlockAlign=self.channels * 2, wBitsPerSample=16, cbSize=0)
        self._h = ctypes.c_void_p()
        rc = self._winmm.waveOutOpen(ctypes.byref(self._h), self._WAVE_MAPPER,
                                     ctypes.byref(fmt), 0, 0, 0)
        if rc != 0:
            raise RuntimeError(f"waveOutOpen failed ({rc})")
        # (header, backing buffer) pairs, recycled once the device is done.
        self._slots: List = []
        self._closed = False

    def _reap(self) -> None:
        """Unprepare and drop every buffer the device has finished with."""
        alive = []
        for hdr, buf in self._slots:
            if hdr.dwFlags & self._WHDR_DONE:
                self._winmm.waveOutUnprepareHeader(
                    self._h, ctypes.byref(hdr), ctypes.sizeof(hdr))
            else:
                alive.append((hdr, buf))
        self._slots = alive

    def play(self, samples: np.ndarray, sample_rate: int) -> None:
        if self._closed:
            return
        self._reap()
        # Keep the queue shallow. Blocking here is what paces the render loop:
        # the device is the clock, and waiting for a free slot is waiting for
        # audio to actually have been played.
        waited = 0.0
        while len(self._slots) >= self._N_BUFFERS and waited < 2.0:
            time.sleep(0.01)
            waited += 0.01
            self._reap()

        pcm = np.clip(np.asarray(samples), -1.0, 1.0)
        raw = (pcm * 32767.0).astype("<i2").tobytes()
        buf = ctypes.create_string_buffer(raw, len(raw))
        hdr = self._WAVEHDR(lpData=ctypes.cast(buf, ctypes.c_void_p),
                            dwBufferLength=len(raw), dwBytesRecorded=0,
                            dwUser=None, dwFlags=0, dwLoops=0,
                            lpNext=None, reserved=None)
        rc = self._winmm.waveOutPrepareHeader(self._h, ctypes.byref(hdr),
                                              ctypes.sizeof(hdr))
        if rc != 0:
            raise RuntimeError(f"waveOutPrepareHeader failed ({rc})")
        rc = self._winmm.waveOutWrite(self._h, ctypes.byref(hdr),
                                      ctypes.sizeof(hdr))
        if rc != 0:
            self._winmm.waveOutUnprepareHeader(self._h, ctypes.byref(hdr),
                                               ctypes.sizeof(hdr))
            raise RuntimeError(f"waveOutWrite failed ({rc})")
        # Both must outlive the call: the device reads the buffer after we
        # return, and dropping the Python reference would free it underneath.
        self._slots.append((hdr, buf))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._winmm.waveOutReset(self._h)
            self._reap()
            self._winmm.waveOutClose(self._h)
        except Exception:
            pass


class _WinsoundSink:
    """Windows standard library, playing each chunk synchronously from memory.

    Synchronous is not a compromise here, it is the only thing that works.
    ``PlaySound`` with ``SND_ASYNC`` *replaces* whatever is currently playing,
    so an asynchronous chunk stream relies on each chunk starting before the
    next arrives. It does not: the call has to hand the buffer to the audio
    device first, and when that setup time approaches the chunk length every
    chunk is cancelled before it produces a sample. The failure mode is total
    silence in which every call succeeds -- measured here as 24 chunks
    submitted, peaks up to 0.49, and nothing audible.

    Playing synchronously inverts the relationship: the call returns when the
    chunk has finished, so the audio device paces the render loop instead of a
    timer racing it. The only gap is the few milliseconds spent rendering the
    next chunk, and carrying oscillator phase across chunks keeps that seam
    inaudible.

    ``SND_MEMORY`` is what makes this cheap -- no file per chunk, no temporary
    directory. It is rejected in combination with ``SND_ASYNC`` (the caller
    could free the buffer while the device is still reading it), which is what
    pushed the first version onto files; synchronously there is no such
    problem, because the buffer outlives the call by construction.
    """

    name = "winsound"
    blocking = True

    def __init__(self):
        import winsound  # noqa: F401  (the import is the availability test)
        self._winsound = winsound
        # Fail at construction rather than inside the render thread, where the
        # exception would be swallowed and the run would simply go quiet.
        self._winsound.PlaySound(
            to_wav_bytes(np.zeros((8, 2), dtype=np.float32), 22050),
            winsound.SND_MEMORY | winsound.SND_NODEFAULT)

    def play(self, samples: np.ndarray, sample_rate: int) -> None:
        self._winsound.PlaySound(
            to_wav_bytes(samples, sample_rate),
            self._winsound.SND_MEMORY | self._winsound.SND_NODEFAULT)

    def close(self) -> None:
        try:
            self._winsound.PlaySound(None, 0)
        except Exception:
            pass


class _CommandSink:
    """POSIX fallback: hand a WAV file to whichever player is installed."""

    name = "command"
    blocking = False

    _CANDIDATES = (("paplay", ()), ("aplay", ("-q",)), ("afplay", ()))

    def __init__(self):
        for exe, args in self._CANDIDATES:
            if shutil.which(exe):
                self._cmd = [exe, *args]
                break
        else:
            raise RuntimeError("no command-line audio player found")
        self._proc: Optional[subprocess.Popen] = None
        self._tmp = Path(tempfile.mkdtemp(prefix="atsc-siren-"))
        self._n = 0

    def play(self, samples: np.ndarray, sample_rate: int) -> None:
        self._n += 1
        path = self._tmp / f"chunk{self._n % 4}.wav"
        path.write_bytes(to_wav_bytes(samples, sample_rate))
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = subprocess.Popen([*self._cmd, str(path)],
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
        shutil.rmtree(self._tmp, ignore_errors=True)


def open_sink(prefer: str = "auto"):
    """Best available output device, or the null sink."""
    if prefer == "null":
        return _NullSink()
    if prefer == "waveout":
        order = [_WaveOutSink]
    elif prefer == "winsound":
        order = [_WinsoundSink]
    elif prefer == "command":
        order = [_CommandSink]
    elif sys.platform == "win32":
        # waveOut first: it is the only one of the three that streams without
        # gaps. winsound remains as the fallback because it needs nothing.
        order = [_WaveOutSink, _WinsoundSink, _CommandSink]
    else:
        order = [_CommandSink, _WaveOutSink]
    for cls in order:
        try:
            return cls()
        except Exception:
            continue
    return _NullSink()


# ---------------------------------------------------------------------------
# The player
# ---------------------------------------------------------------------------
class SirenPlayer:
    """Renders the live acoustic scene on a background thread and plays it.

    The simulation calls :meth:`update` as often as it likes -- the demo does so
    every frame -- and the render thread runs on its own clock, so audio never
    stalls the simulation and a slow frame does not open a gap in the sound.

    Two clocks, and which one you want depends on what the audio is for:

    ``live``      render on the wall clock, on a background thread. Right for
                  watching the demo, where the simulation is paced to real time
                  and the sound has to keep up with the window.
    ``offline``   render exactly the simulated time the caller reports through
                  :meth:`advance`, synchronously and with no thread. Right for
                  recording, because a headless run goes at forty times real
                  time and a wall-clock recording of it is thirteen seconds of
                  audio for an eight-minute episode -- the file has to be in
                  the simulation's time base to line up with anything.
    """

    def __init__(
        self,
        cfg: Optional[AcousticConfig] = None,
        enabled: bool = True,
        backend: str = "auto",
        record_path: Optional[Path] = None,
        chunk_s: float = CHUNK_S,
        sample_rate: int = SAMPLE_RATE,
        seed: int = 0,
        mode: str = "live",
    ):
        if mode not in ("live", "offline"):
            raise ValueError(f"mode must be 'live' or 'offline', got {mode!r}")
        self.mode = mode
        self._pending_s = 0.0
        self.enabled = bool(enabled)
        self.chunk_s = float(chunk_s)
        self.sample_rate = int(sample_rate)
        self.renderer = SirenRenderer(cfg, sample_rate=sample_rate, seed=seed)
        self.record_path = Path(record_path) if record_path else None
        self._sink = open_sink(backend) if self.enabled else _NullSink()
        self._lock = threading.Lock()
        self._pending: Optional[List[AudioSource]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._recorded: List[np.ndarray] = []
        self.n_chunks = 0

    # ------------------------------------------------------------------- api
    @property
    def backend(self) -> str:
        return self._sink.name

    def start(self) -> "SirenPlayer":
        if self._thread is not None or self.mode == "offline":
            return self
        if not self.enabled and self.record_path is None:
            return self
        self._thread = threading.Thread(target=self._run, name="siren-audio",
                                        daemon=True)
        self._thread.start()
        return self

    def advance(self, dt: float) -> None:
        """Render ``dt`` seconds of *simulated* time. Offline mode only.

        A no-op in live mode, where the render thread owns the clock, so a
        caller can drive both the same way.
        """
        if self.mode != "offline":
            return
        if not self.enabled and self.record_path is None:
            return
        self._pending_s += float(dt)
        while self._pending_s + 1e-9 >= self.chunk_s:
            self._pending_s -= self.chunk_s
            chunk = self.renderer.render(self.chunk_s)
            self.n_chunks += 1
            if self.record_path is not None:
                self._recorded.append(chunk)
            if self.enabled and self.renderer.active:
                try:
                    self._sink.play(chunk, self.sample_rate)
                except Exception:
                    self._sink = _NullSink()

    def update(self, sources: Sequence[AudioSource]) -> None:
        """Hand the render thread the current scene. Cheap and non-blocking."""
        if self.mode == "offline":
            self.renderer.set_sources(sources)
            return
        with self._lock:
            self._pending = list(sources)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._sink.close()
        self._write_recording()

    def __enter__(self) -> "SirenPlayer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    # -------------------------------------------------------------- internals
    def _run(self) -> None:
        deadline = time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                pending, self._pending = self._pending, None
            if pending is not None:
                self.renderer.set_sources(pending)
            chunk = self.renderer.render(self.chunk_s)
            self.n_chunks += 1
            if self.record_path is not None:
                self._recorded.append(chunk)
            if self.renderer.active:
                try:
                    self._sink.play(chunk, self.sample_rate)
                except Exception:
                    # A device that disappears mid-run must not take the
                    # simulation down with it.
                    self._sink = _NullSink()
            if getattr(self._sink, "blocking", False) and self.renderer.active:
                # The sink played the chunk to completion, so it has already
                # consumed the wall time this iteration is worth. Sleeping
                # again on top of that would insert a silent chunk-length gap
                # between every chunk.
                deadline = time.monotonic()
                continue
            deadline += self.chunk_s
            wait = deadline - time.monotonic()
            if wait > 0.0:
                self._stop.wait(wait)
            else:
                deadline = time.monotonic()

    def _write_recording(self) -> None:
        if self.record_path is None or not self._recorded:
            return
        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        samples = np.concatenate(self._recorded, axis=0)
        self.record_path.write_bytes(to_wav_bytes(samples, self.sample_rate))


# ---------------------------------------------------------------------------
# Wiring to the environment
# ---------------------------------------------------------------------------
def scene_from_env(env) -> List[AudioSource]:
    """Everything currently audible at the junction: sirens *and* horns.

    Ground truth, taken from the simulator: the point of the audio is to let a
    listener check the controller against what is really out there, so it must
    not be filtered through the controller's own belief.

    The horns are here for the same reason they are in the acoustic scene the
    detector reads. They are the confuser, and a listener judging whether the
    controller is being fooled needs to hear what it is being fooled *by* --
    a demo in which the only sound is the siren makes the detection problem
    sound far easier than it is.
    """
    import traci

    scenario = getattr(env, "scenario", None)
    conn = getattr(env, "conn", None)
    if scenario is None or conn is None:
        return []
    try:
        present = set(conn.vehicle.getIDList())
    except Exception:
        return []
    active = present & set(scenario.emergency_vehicles or ())

    # The junction is the mean of the stop lines: where a listener standing at
    # the intersection would be.
    stops = list(getattr(env, "stop_line", {}).values())
    if stops:
        jx = float(np.mean([p[0] for p in stops]))
        jy = float(np.mean([p[1] for p in stops]))
    else:
        jx = jy = 0.0

    radius = float(env.cfg.acoustic.detection_radius_m)
    out: List[AudioSource] = []
    for vid in sorted(active):
        ev = scenario.emergency_vehicles[vid]
        try:
            px, py = conn.vehicle.getPosition(vid)
            speed = conn.vehicle.getSpeed(vid)
            angle = conn.vehicle.getAngle(vid)
        except traci.TraCIException:
            continue
        dx, dy = px - jx, py - jy
        dist = max(math.hypot(dx, dy), 1.0)
        if dist > radius:
            continue
        rad = math.radians(angle)          # SUMO: degrees clockwise from north
        vx, vy = speed * math.sin(rad), speed * math.cos(rad)
        v_r = (dx * vx + dy * vy) / dist   # positive = receding
        # Pan by true bearing: east of the junction is to the right.
        pan = float(np.clip(dx / max(dist, 1e-6), -1.0, 1.0))
        out.append(AudioSource(key=vid, veh_class=ev.veh_class,
                               distance_m=dist, radial_velocity_ms=v_r, pan=pan))

    # Horns, from the same press bookkeeping the microphones are reading.
    for vid in list(getattr(env, "_horn_until", {}) or {}):
        try:
            px, py = conn.vehicle.getPosition(vid)
        except Exception:
            continue
        dx, dy = px - jx, py - jy
        dist = max(math.hypot(dx, dy), 1.0)
        if dist > radius:
            continue
        pan = float(np.clip(dx / max(dist, 1e-6), -1.0, 1.0))
        out.append(AudioSource(key=f"horn_{vid}", veh_class="horn",
                               distance_m=dist, radial_velocity_ms=0.0, pan=pan))
    return out
