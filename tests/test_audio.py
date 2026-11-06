"""The audible siren.

The claim the audio exists to make legible is *the siren gets louder, so the
vehicle is getting closer, so the corridor opens*. That is only worth playing
if the first half of it is true of the waveform, so these tests check the
rendered sound against the same physics the detector is trained on rather than
merely checking that a file appears.

No audio device is required: the renderer and the sink are separate, and the
null sink is what runs here.
"""

from __future__ import annotations

import math
import wave
import io
import tempfile
from pathlib import Path

import numpy as np

from atsc.config import AcousticConfig
from atsc.evaluation.audio import (
    AudioSource, SirenPlayer, SirenRenderer, level_to_gain, open_sink,
    to_wav_bytes, CEIL_DB, FLOOR_DB,
)
from atsc.sensing.siren_synth import received_spl_db


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def test_loudness_rises_monotonically_as_the_vehicle_closes():
    """The whole point: closeness is audible.

    The posterior saturates hundreds of metres out and stays saturated, so it
    is the level -- not the detection -- that distinguishes a vehicle about to
    arrive from one that has only just been heard.
    """
    r = SirenRenderer(seed=1)
    levels = []
    for d in (280.0, 200.0, 140.0, 90.0, 50.0, 20.0, 5.0):
        r.set_sources([AudioSource("a", "ambulance", d, -14.0)])
        levels.append(_rms(r.render(0.25)))
    assert all(b > a for a, b in zip(levels, levels[1:])), levels
    # Two things are asserted, and they pull against each other. The near end
    # must be clearly louder than the far end -- that is the demonstration --
    # and the far end must be *audible at all*, which is the half that was
    # missing. At the 20x this used to require, a siren 280 m out rendered at
    # 0.016 of full scale: silent on laptop speakers, so the corridor appeared
    # to open for nothing and only became audible once the vehicle was at the
    # line. Twenty-one decibels of range with the far end at -25 dBFS is the
    # trade that makes both ends do their job.
    assert levels[-1] > 6.0 * levels[0], levels
    assert levels[0] > 0.03, "a distant siren must still be audible"
    assert levels[-1] < 0.95, "a near siren must not clip"


def test_gain_map_is_anchored_to_the_physical_level():
    """Silence at the noise floor, full scale at the junction, monotone between."""
    assert level_to_gain(FLOOR_DB - 10.0) == 0.0
    assert level_to_gain(CEIL_DB + 10.0) == 1.0
    xs = np.linspace(FLOOR_DB, CEIL_DB, 40)
    gains = [level_to_gain(float(x)) for x in xs]
    assert all(b >= a for a, b in zip(gains, gains[1:]))
    # The anchor points are the model's own: a vehicle at the stop line really
    # is near the ceiling, one at the detection radius really is near the floor.
    assert received_spl_db("ambulance", 5.0) > CEIL_DB - 15.0
    assert received_spl_db("ambulance", 300.0) < FLOOR_DB + 15.0


def test_doppler_shifts_the_pitch_by_the_closing_speed():
    """Approaching is higher than receding, by c/(c+v) either way.

    Rendered rather than asserted: the shift at road speeds is about 6 %, which
    is the measurement the detector's velocity head could not make, and hearing
    how small it is explains that result better than the ablation row does.
    """
    cfg = AcousticConfig()
    approaching = SirenRenderer(cfg, seed=4)
    receding = SirenRenderer(cfg, seed=4)
    approaching.set_sources([AudioSource("a", "police", 100.0, -25.0)])
    receding.set_sources([AudioSource("a", "police", 100.0, +25.0)])

    def centroid(x: np.ndarray) -> float:
        mono = x.mean(axis=1)
        spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
        freqs = np.fft.rfftfreq(len(mono), 1.0 / approaching.sample_rate)
        return float(np.sum(freqs * spec) / max(np.sum(spec), 1e-12))

    hi = centroid(approaching.render(1.0))
    lo = centroid(receding.render(1.0))
    assert hi > lo, (hi, lo)
    expected = (cfg.speed_of_sound + 25.0) / (cfg.speed_of_sound - 25.0)
    assert 0.5 * expected < hi / lo < 2.0 * expected, (hi / lo, expected)


def test_pan_follows_the_bearing():
    """A vehicle east of the junction is heard on the right."""
    r = SirenRenderer(seed=2)
    r.set_sources([AudioSource("a", "fire", 80.0, -12.0, pan=+1.0)])
    right = r.render(0.5)
    r.set_sources([AudioSource("b", "fire", 80.0, -12.0, pan=-1.0)])
    left = r.render(0.5)
    assert _rms(right[:, 1]) > 10.0 * _rms(right[:, 0])
    assert _rms(left[:, 0]) > 10.0 * _rms(left[:, 1])


def test_phase_is_continuous_across_chunks():
    """No click at the seam.

    The training synthesiser draws an independent phase per frame, which is
    correct for independent samples and wrong for playback: at a quarter-second
    chunk it would put a click in the output four times a second.
    """
    r = SirenRenderer(seed=5)
    r.set_sources([AudioSource("a", "ambulance", 60.0, 0.0)])
    a = r.render(0.25)
    b = r.render(0.25)
    joined = np.concatenate([a, b])
    step = np.abs(np.diff(joined[:, 0]))
    seam = float(step[len(a) - 1])
    assert seam <= 4.0 * float(np.percentile(step, 99.9)), (
        seam, float(np.percentile(step, 99.9)))


def test_silence_when_nothing_is_out_there():
    r = SirenRenderer(seed=6)
    assert not r.active
    assert _rms(r.render(0.5)) == 0.0


def test_two_vehicles_mix_without_clipping():
    r = SirenRenderer(seed=8)
    r.set_sources([AudioSource("a", "fire", 8.0, -14.0, pan=-0.5),
                   AudioSource("b", "police", 8.0, -14.0, pan=+0.5)])
    x = r.render(0.5)
    assert float(np.abs(x).max()) <= 1.0


def test_offline_render_is_in_the_simulations_time_base():
    """A recorded episode must be as long as the episode, not as long as the run.

    Headless runs go at tens of times real time. A wall-clock recording of one
    is a few seconds of audio for several minutes of simulation, which lines up
    with nothing.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "siren.wav"
        player = SirenPlayer(enabled=False, record_path=out, mode="offline",
                             backend="null", seed=3).start()
        player.update([AudioSource("a", "ambulance", 120.0, -14.0)])
        for _ in range(60):
            player.advance(0.1)
        player.close()
        with wave.open(str(out)) as w:
            seconds = w.getnframes() / w.getframerate()
        assert abs(seconds - 6.0) < 0.5, seconds


def test_recording_works_with_no_audio_device():
    """The null sink still renders and still records."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "siren.wav"
        player = SirenPlayer(enabled=True, backend="null", record_path=out,
                             mode="offline", seed=1).start()
        assert player.backend == "null"
        player.update([AudioSource("a", "police", 40.0, -12.0)])
        player.advance(2.0)
        player.close()
        assert out.exists() and out.stat().st_size > 1000


def test_a_full_pass_rises_then_falls():
    """The shape of one vehicle going by, which is the demonstration."""
    player = SirenPlayer(enabled=False, backend="null", mode="offline", seed=2)
    r = player.renderer
    peaks = []
    for d in (260.0, 180.0, 110.0, 50.0, 12.0):
        r.set_sources([AudioSource("a", "ambulance", d, -14.0)])
        peaks.append(_rms(r.render(0.25)))
    for d in (12.0, 50.0, 110.0, 180.0, 260.0):
        r.set_sources([AudioSource("a", "ambulance", d, +14.0)])
        peaks.append(_rms(r.render(0.25)))
    top = int(np.argmax(peaks))
    assert 3 <= top <= 5, peaks
    assert peaks[0] < peaks[top] and peaks[-1] < peaks[top]


def test_wav_bytes_round_trip():
    x = np.zeros((100, 2), dtype=np.float32)
    x[:, 0] = np.sin(np.linspace(0, 6.0, 100))
    data = to_wav_bytes(x, 22050)
    with wave.open(io.BytesIO(data)) as w:
        assert w.getnchannels() == 2
        assert w.getframerate() == 22050
        assert w.getnframes() == 100
        back = np.frombuffer(w.readframes(100), dtype="<i2").reshape(-1, 2) / 32767.0
    assert float(np.abs(back - x).max()) < 1e-3


def test_open_sink_never_raises():
    """A machine with no audio must still run the demo."""
    for prefer in ("auto", "null", "winsound", "command", "waveout"):
        sink = open_sink(prefer)
        sink.play(np.zeros((10, 2), dtype=np.float32), 22050)
        sink.close()


def test_streaming_sink_is_preferred_where_it_exists():
    """The fallback cannot stream, so it must not be the first choice.

    winsound.PlaySound is fire-and-forget: asynchronously each chunk cancels
    the one before it, and synchronously it costs a fixed ~150 ms per call
    whatever the buffer length -- measured at 56 % duty for quarter-second
    chunks and 88 % at a full second. There is no chunk size at which it
    streams, so a sink that queues buffers has to win the selection.
    """
    import sys as _sys
    from atsc.evaluation.audio import _WaveOutSink, _WinsoundSink

    assert getattr(_WinsoundSink, "blocking", False) is True, (
        "the fallback plays synchronously; the render loop must not sleep on "
        "top of a call that already consumed the wall time")
    if _sys.platform != "win32":
        return
    try:
        sink = _WaveOutSink()
    except Exception:
        return  # no device on this machine; nothing to assert
    try:
        assert open_sink("auto").name == "waveout"
    finally:
        sink.close()


def test_queued_buffers_outlive_the_call_that_wrote_them():
    """The device reads each buffer after play() returns.

    Dropping the Python reference would free the memory underneath a device
    that is still reading it -- which does not raise, it produces noise or a
    crash somewhere unrelated. So the sink has to hold both the header and its
    backing store until the hardware sets WHDR_DONE.
    """
    import sys as _sys
    if _sys.platform != "win32":
        return
    from atsc.evaluation.audio import _WaveOutSink
    try:
        sink = _WaveOutSink()
    except Exception:
        return
    try:
        x = np.zeros((2205, 2), dtype=np.float32)
        sink.play(x, 22050)
        assert sink._slots, "the queued buffer must be retained"
        hdr, buf = sink._slots[0]
        assert len(buf) >= 2205 * 2 * 2, "backing store must hold the samples"
    finally:
        sink.close()
