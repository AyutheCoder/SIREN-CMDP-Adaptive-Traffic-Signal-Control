"""Acoustic front end: DSP correctness, propagation physics, belief tracking."""

from __future__ import annotations

import math

import numpy as np

from atsc.config import AcousticConfig, BeliefConfig
from atsc.sensing.belief import SIREN_CLASSES, ApproachBelief, BeliefTracker
from atsc.sensing.dsp import (
    MelSpectrogram, db_to_pa, hz_to_mel, mel_to_hz, mel_filterbank, pa_to_db, pink_noise,
)
from atsc.sensing.sensor import (
    AcousticSensor, SensorErrorModel, SensorReading, SourceState, cardioid_gain_db)
from atsc.sensing.siren_synth import (
    SirenSynthesizer, absorption_db_per_m, harmonic_amplitudes, received_spl_db,
)

CFG = AcousticConfig()


def _mel() -> MelSpectrogram:
    return MelSpectrogram(CFG.sample_rate, CFG.n_fft, CFG.hop_length,
                          CFG.n_mels, CFG.fmin, CFG.fmax)


# ------------------------------------------------------------------------ dsp
def test_mel_scale_roundtrip():
    f = np.array([50.0, 440.0, 1000.0, 4000.0, 7500.0])
    assert np.allclose(mel_to_hz(hz_to_mel(f)), f, rtol=1e-9)


def test_mel_filterbank_is_nonnegative_and_covers_band():
    fb = mel_filterbank(CFG.sample_rate, CFG.n_fft, CFG.n_mels, CFG.fmin, CFG.fmax)
    assert fb.shape == (CFG.n_mels, CFG.n_fft // 2 + 1)
    assert (fb >= 0).all()
    # Every filter must have some support, or a band is silently dead.
    assert (fb.sum(axis=1) > 0).all()


def test_db_pa_roundtrip():
    for db in (40.0, 60.0, 94.0, 120.0):
        assert abs(float(pa_to_db(db_to_pa(db))) - db) < 1e-6


def test_mel_recovers_absolute_level_of_a_tone():
    """A calibrated tone at a known SPL must read back close to that SPL.

    Absolute accuracy is what makes level usable as a range cue. A tone falling
    between two mel band centres is split across them, costing up to ~6 dB in
    the peak band, so the tolerance is set accordingly - the point is that the
    reading tracks true SPL rather than sitting an arbitrary constant away from
    it, which is what unit-area (Slaney) filters would give.
    """
    mel = _mel()
    t = np.arange(CFG.sample_rate) / CFG.sample_rate
    for target_db in (60.0, 80.0, 100.0):
        amp = db_to_pa(target_db) * np.sqrt(2.0)
        wave = amp * np.sin(2 * np.pi * 1000.0 * t)
        peak = float(mel(wave).max())
        assert abs(peak - target_db) < 6.0, (target_db, peak)


def test_mel_output_is_linear_in_decibels():
    """Raising the source by N dB must raise the feature by exactly N dB."""
    mel = _mel()
    t = np.arange(CFG.sample_rate) / CFG.sample_rate
    readings = []
    for target_db in (50.0, 70.0, 90.0):
        wave = db_to_pa(target_db) * np.sqrt(2.0) * np.sin(2 * np.pi * 1000.0 * t)
        readings.append(float(mel(wave).max()))
    assert abs((readings[1] - readings[0]) - 20.0) < 0.2, readings
    assert abs((readings[2] - readings[1]) - 20.0) < 0.2, readings


def test_mel_recovers_broadband_noise_level():
    """Band levels of shaped noise must sum back to the true overall level."""
    mel = _mel()
    rng = np.random.default_rng(0)
    target_db = 70.0
    x = rng.standard_normal(CFG.sample_rate)
    x = x / np.sqrt(np.mean(x ** 2)) * db_to_pa(target_db)
    bands = mel(x).mean(axis=1)
    total = 10.0 * np.log10(np.sum(10.0 ** (bands / 10.0)))
    # Mel triangles overlap and stop at fmax, so the recovered total is close to
    # but not exactly the full-band level.
    assert abs(total - target_db) < 6.0, (total, target_db)


def test_pink_noise_is_unit_rms_and_low_biased():
    rng = np.random.default_rng(0)
    x = pink_noise(16000, rng)
    assert abs(float(np.sqrt(np.mean(x ** 2))) - 1.0) < 1e-6
    spec = np.abs(np.fft.rfft(x))
    lo = spec[10:100].mean()
    hi = spec[2000:4000].mean()
    assert lo > 3 * hi, "pink noise should fall off with frequency"


# ---------------------------------------------------------------- propagation
def test_absorption_increases_with_frequency():
    a = absorption_db_per_m([250.0, 1000.0, 4000.0, 8000.0])
    assert np.all(np.diff(a) > 0)
    # 4 kHz should be roughly an order of magnitude worse than 500 Hz.
    assert absorption_db_per_m(4000.0) / absorption_db_per_m(500.0) > 10.0


def test_harmonic_amplitudes_are_unit_power():
    for rolloff in (1.0, 1.35, 1.6):
        a = harmonic_amplitudes(rolloff)
        assert abs(float(np.sum(a ** 2)) - 1.0) < 1e-9
        assert np.all(np.diff(a) < 0), "harmonics must decay"


def test_received_level_falls_with_range():
    levels = [received_spl_db("ambulance", r) for r in (10, 50, 100, 200, 300)]
    assert all(b < a for a, b in zip(levels, levels[1:]))
    # Doubling range must cost at least the 6 dB of spherical spreading.
    assert received_spl_db("ambulance", 50) - received_spl_db("ambulance", 100) >= 6.0


def test_directivity_is_a_cardioid():
    assert abs(cardioid_gain_db(0.0, 14.0)) < 1e-9              # on axis: 0 dB
    assert cardioid_gain_db(np.pi / 2, 14.0) < -5.0             # side: attenuated
    assert abs(cardioid_gain_db(np.pi, 14.0) + 14.0) < 1e-6     # rear: floored


def test_doppler_raises_pitch_when_approaching():
    """An approaching source must show a higher spectral centroid than a receding one."""
    mel = _mel()
    syn = SirenSynthesizer(CFG, seed=7)
    freqs = np.linspace(CFG.fmin, CFG.fmax, CFG.n_mels)

    def centroid(v_r: float) -> float:
        acc = []
        for _ in range(6):
            obs = syn.synthesize("ambulance", 40.0, v_r, ambient_db=45.0, reverb=False)
            power = 10 ** (mel(obs.waveform) / 10.0)
            acc.append(float((power.mean(axis=1) * freqs).sum() / power.mean(axis=1).sum()))
        return float(np.mean(acc))

    assert centroid(-28.0) > centroid(28.0)


def test_spectral_tilt_encodes_range_at_short_range():
    """Absorption must make far sources measurably duller than near ones."""
    mel = _mel()
    syn = SirenSynthesizer(CFG, seed=3)

    def tilt(r: float) -> float:
        acc = []
        for _ in range(6):
            obs = syn.synthesize("ambulance", r, -14.0, ambient_db=55.0)
            m = mel(obs.waveform)
            acc.append(float(m[:16].mean() - m[45:].mean()))
        return float(np.mean(acc))

    assert tilt(150.0) > tilt(25.0) + 5.0, "far sources must be relatively duller"


def test_noise_only_frames_carry_no_siren_label():
    syn = SirenSynthesizer(CFG, seed=1)
    obs = syn.synthesize(None)
    assert obs.label == "none"
    assert not np.isfinite(obs.snr_db)


def test_dataset_shapes_and_balance():
    mel = _mel()
    syn = SirenSynthesizer(CFG, seed=11)
    ds = syn.build_dataset(200, mel, p_siren=0.5)
    n_frames = mel.n_frames(syn.n_samples)
    assert ds["x"].shape == (200, CFG.n_mels, n_frames, 1)
    counts = np.bincount(ds["y_class"], minlength=4)
    assert counts[0] > 50 and counts[1:].sum() > 50
    assert np.isfinite(ds["x"]).all()


# --------------------------------------------------------------------- belief
def test_belief_rises_on_detection_and_decays_without():
    cfg = BeliefConfig()
    bt = BeliefTracker(["N", "E"], cfg)
    # With no evidence the filter must sit at the chain's stationary occupancy,
    # which is pinned to the deployment's real emergency-vehicle rate rather
    # than chosen: b / (b + 1 - s).
    stationary = cfg.birth_prob / (cfg.birth_prob + 1.0 - cfg.survival_prob)
    assert abs(bt.belief("N").presence - stationary) < 0.02, bt.belief("N").presence
    assert stationary < 0.10

    for _ in range(4):
        bt.predict(5.0)
        bt.update("N", 0.95, {"ambulance": 0.8, "fire": 0.1, "police": 0.1}, 120.0, 900.0, -14.0)
    assert bt.belief("N").presence > 0.8

    for _ in range(12):
        bt.predict(5.0)
        bt.update("N", 0.02, None, None)
    assert bt.belief("N").presence < 0.2


def test_belief_ignores_a_single_false_positive():
    """One confident spurious frame must not commit the controller."""
    bt = BeliefTracker(["N"], BeliefConfig())
    bt.predict(5.0)
    bt.update("N", 0.90, {"ambulance": 0.5, "fire": 0.3, "police": 0.2}, 200.0, 8000.0, -10.0)
    assert not bt.belief("N").committed, bt.belief("N").presence


def test_range_tracker_converges_to_truth():
    """Noisy range measurements must still yield a usable closing speed."""
    rng = np.random.default_rng(0)
    bt = BeliefTracker(["N"], BeliefConfig())
    r, v, dt = 260.0, -13.0, 5.0
    for _ in range(9):
        z = float(r * np.exp(rng.normal(0.0, 0.35)))
        bt.predict(dt)
        bt.update("N", 0.97, {"ambulance": 0.9, "fire": 0.05, "police": 0.05},
                  z, (0.45 * r) ** 2, -11.0)
        r = max(r + v * dt, 2.0)

    b = bt.belief("N")
    assert b.range_rate_ms < -4.0, f"tracker lost closure: {b.range_rate_ms}"
    assert abs(b.range_m - r) < 90.0, f"range {b.range_m} vs truth {r}"


def test_priority_reflects_class_posterior():
    bt = BeliefTracker(["N"], BeliefConfig(),
                       priority_map={"ambulance": 1.0, "fire": 0.9, "police": 0.7})
    for _ in range(8):
        bt.predict(5.0)
        bt.update("N", 0.97, {"ambulance": 0.9, "fire": 0.05, "police": 0.05}, 80.0, 400.0, -12.0)
    assert bt.belief("N").expected_priority > 0.9


def test_feature_vector_is_finite_and_shaped():
    bt = BeliefTracker(["N", "E", "S", "W"], BeliefConfig())
    v = bt.feature_vector()
    assert v.shape == (12,) and np.isfinite(v).all()
    assert bt.n_features == 12


# --------------------------------------------------------------------- sensor
def test_oracle_sensor_is_exact():
    sen = AcousticSensor(["N", "E"], CFG, mode="oracle", seed=0)
    scene = {"N": [SourceState("fire", 80.0, -12.0, 0.1, 14.0)], "E": []}
    r = sen.sense(scene)
    assert r["N"].p_siren > 0.99 and r["N"].class_probs["fire"] == 1.0
    assert abs(r["N"].distance_m - 80.0) < 1e-9
    assert r["E"].p_siren < 0.01


def test_none_sensor_reports_nothing():
    sen = AcousticSensor(["N"], CFG, mode="none", seed=0)
    r = sen.sense({"N": [SourceState("police", 20.0, -20.0)]})
    assert r["N"].p_siren == 0.0 and r["N"].distance_m is None


def test_surrogate_detects_more_often_when_closer():
    sen = AcousticSensor(["N"], CFG, mode="surrogate",
                         error_model=SensorErrorModel.default(), seed=5)
    def rate(distance: float) -> float:
        hits = 0
        for _ in range(400):
            r = sen.sense({"N": [SourceState("ambulance", distance, -14.0, 0.0, 14.0)]})
            hits += r["N"].p_siren >= 0.5
        return hits / 400

    near, far = rate(40.0), rate(280.0)
    assert near > far, f"near={near}, far={far}"


def test_single_element_directivity_suppresses_sources_behind_it():
    """The single-cardioid mast attenuates what is behind it.

    This is the shipped mast, and with it at the signal pole the rejection is
    aimed where it helps: a vehicle that has crossed the stop line is behind it
    and goes quiet, which is the inbound-versus-departed cue the signed state
    space consumes. It is only harmful when the mast is far upstream, where the
    whole last stretch of the approach falls behind it -- see the offset table
    in ``AcousticConfig.mic_offset_m``.
    """
    on = SourceState("ambulance", 120.0, -14.0, 0.0, 14.0, dual_element=False)
    off = SourceState("ambulance", 120.0, -14.0, np.pi * 0.85, 14.0,
                      dual_element=False)
    assert on.received_spl_db() - off.received_spl_db() > 8.0


def test_two_element_mast_hears_behind_it():
    """The steered pair does not suppress what is behind it. That is its trade.

    Reading the louder of two opposed lobes makes a source behind the mast
    audible, which is what rescued the 70 m mast. It also flattens the pattern
    to very nearly omnidirectional, which is why the shipped mast is a single
    cardioid at the signal pole instead. Both halves are asserted: here that
    the pair hears behind it, and in
    ``test_the_steered_pair_trades_its_pattern_away`` that it pays for it.
    """
    on = SourceState("ambulance", 120.0, -14.0, 0.0, 14.0)
    off = SourceState("ambulance", 120.0, -14.0, np.pi, 14.0)
    assert abs(on.received_spl_db() - off.received_spl_db()) < 0.5


def test_the_steered_pair_trades_its_pattern_away():
    """Reading ``max(front, back)`` costs the directivity that separates the arms.

    Two opposed cardioids read at their louder lobe give 0 dB on both axes and
    -6 dB broadside, so a steered mast is very nearly omnidirectional. That is
    the half of the trade that went unnoticed, because nothing was reading the
    level: the front-to-back ratio the four approaches are separated by at long
    range collapses from 14 dB to 6, and with it the array's ability to say
    which arm a distant siren is on.

    True broadside is the one angle where the two agree, at -6 dB each. The
    geometry that matters is the rear quadrant: a source on a neighbouring arm,
    seen from a mast near the junction, sits well behind that mast's axis --
    around 135 degrees off it. There the single cardioid is into its rear
    rejection and the steered pair is nearly on the axis of its *other* lobe.
    """
    off_axis = 3.0 * np.pi / 4.0
    single = SourceState("ambulance", 120.0, -14.0, off_axis, 14.0,
                         dual_element=False)
    pair = SourceState("ambulance", 120.0, -14.0, off_axis, 14.0)
    on_axis = SourceState("ambulance", 120.0, -14.0, 0.0, 14.0)

    single_rejection = on_axis.received_spl_db() - single.received_spl_db()
    pair_rejection = on_axis.received_spl_db() - pair.received_spl_db()
    assert single_rejection > 12.0, single_rejection
    assert pair_rejection < 3.0, pair_rejection


def test_the_filter_scores_the_measured_level():
    """The level is a measurement, not just a bin index.

    Every mast within earshot saturates at ``p_siren = 1.00``, so the detection
    term is identical under "the source is on this arm" and "on that one" and
    cannot separate them. The levels are not identical, and the likelihood has
    to read them: a hypothesis placing the source on the mast that is actually
    loudest must score better than one placing it elsewhere.
    """
    from atsc.sensing.joint_belief import JointBeliefTracker, MicArray
    from atsc.config import BeliefConfig

    approaches = ["N", "E", "S", "W"]
    axis = {"N": (0.0, 1.0), "E": (1.0, 0.0), "S": (0.0, -1.0), "W": (-1.0, 0.0)}
    stop = {a: (axis[a][0] * 10.0, axis[a][1] * 10.0) for a in approaches}
    off = CFG.mic_offset_m
    pos = {a: (stop[a][0] + axis[a][0] * off, stop[a][1] + axis[a][1] * off)
           for a in approaches}
    mics = MicArray(approaches=approaches, stop_line=stop, pos=pos, axis=axis,
                    front_back_db=CFG.directivity_front_back_db,
                    dual_element=CFG.dual_element_mast,
                    detection_radius_m=CFG.detection_radius_m)
    tracker = JointBeliefTracker(approaches, BeliefConfig(),
                                 priority_map={a: 1.0 for a in approaches},
                                 mics=mics, acoustic=CFG)
    tracker.reset()

    # Same saturated detection everywhere; only the levels differ, and they say
    # the source is close to the east mast.
    def reading(spl):
        return SensorReading("x", 0.999, {c: 1 / 3 for c in SIREN_CLASSES},
                             radial_velocity_ms=-12.0, distance_m=None,
                             spl_db=spl, ambient_db=65.0)

    truth = {a: received_spl_db(
        "ambulance",
        max(math.hypot(*(p - q for p, q in zip(
            (stop["E"][0] + axis["E"][0] * 12.0, stop["E"][1] + axis["E"][1] * 12.0),
            pos[a]))), 1.0), 0.0) for a in approaches}
    readings = {a: reading(truth[a]) for a in approaches}
    for _ in range(4):
        tracker.predict(1.0)
        tracker.update_joint(readings)

    marg = tracker.beliefs()
    assert marg["E"].presence > 0.5, {a: marg[a].presence for a in approaches}
    assert all(marg["E"].presence > marg[a].presence + 0.2
               for a in approaches if a != "E"),         {a: marg[a].presence for a in approaches}


def test_front_back_ratio_carries_the_side_of_mast_cue():
    """Separability moves from the level to the ratio, and gets stronger.

    The single-element mast separated front from back by ``front_back_db``
    (14 dB). The pair separates them by twice that in the ratio -- +14 for a
    source ahead, -14 for one behind -- while leaving both audible, so the cue
    is 28 dB peak-to-peak against a 3 dB measurement noise.
    """
    ahead = SourceState("ambulance", 120.0, -14.0, 0.0, 14.0)
    behind = SourceState("ambulance", 120.0, -14.0, np.pi, 14.0)
    assert ahead.front_back_ratio_db > 13.0
    assert behind.front_back_ratio_db < -13.0
    assert SourceState("ambulance", 120.0, -14.0, 0.0, 14.0,
                       dual_element=False).front_back_ratio_db == 0.0


def test_two_element_mast_hears_a_vehicle_at_the_stop_line():
    """Why the far-upstream mast needed the steered pair. Kept as the ablation.

    Geometry: mast 70 m upstream of the line, vehicle 1 m from the line, so the
    vehicle is 69 m behind the mast. Under the single-element mast the level
    there falls to 60.8 dB -- below the middle of the 55-80 dB ambient band, and
    quieter than the same vehicle 300 m away. The pair recovers the full
    ``front_back_db``, which is what made that siting usable at all.

    The shipped mast solves the same problem by not creating it: at the signal
    pole the vehicle is in front of the mast, not behind it.
    """
    ambient = float(np.mean(CFG.ambient_noise_db_range))
    at_line_single = SourceState("ambulance", 69.0, -1.0, np.pi, 14.0,
                                 dual_element=False).received_spl_db()
    at_line_dual = SourceState("ambulance", 69.0, -1.0, np.pi, 14.0).received_spl_db()
    far = SourceState("ambulance", 230.0, -14.0, 0.0, 14.0).received_spl_db()

    assert at_line_single < ambient, "the bug: inaudible at the stop line"
    assert at_line_single < far, "the bug: quieter at the line than 300 m out"
    assert at_line_dual - at_line_single > 13.0
    assert at_line_dual > ambient, "the fix: audible at the stop line"


def test_error_model_json_roundtrip(tmp_path=None):
    import tempfile
    from pathlib import Path
    em = SensorErrorModel.default()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "em.json"
        em.save(p)
        back = SensorErrorModel.load(p)
    assert back.snr_edges == em.snr_edges
    assert back.siren_logit_mean == em.siren_logit_mean


def test_oracle_only_reports_vehicles_on_its_own_approach():
    """Perfect detection must mean perfect detection *of this approach*.

    All four microphones sit within a couple of hundred metres of each other, so
    gating an oracle on range alone reports every emergency vehicle at every
    microphone. That is not an upper bound on sensing -- it is a different and
    much worse sensor, and it drove hard preemption to seize all four phases at
    once and gridlock the intersection.
    """
    sen = AcousticSensor(["N"], CFG, mode="oracle", seed=0)

    on_approach = SourceState("ambulance", 120.0, -14.0, 0.05, 14.0)
    assert sen.sense({"N": [on_approach]})["N"].p_siren > 0.9

    # Same range, but off to the side: another arm of the junction.
    cross = SourceState("ambulance", 120.0, -14.0, np.deg2rad(85.0), 14.0)
    assert sen.sense({"N": [cross]})["N"].p_siren < 0.1

    # On-axis but departing: must not trigger preemption.
    leaving = SourceState("ambulance", 120.0, +14.0, 0.05, 14.0)
    assert sen.sense({"N": [leaving]})["N"].p_siren < 0.1


def test_oracle_does_not_fire_on_all_approaches_at_once():
    """One emergency vehicle must not light up every approach."""
    approaches = ["N", "E", "S", "W"]
    sen = AcousticSensor(approaches, CFG, mode="oracle", seed=0)
    scene = {
        "N": [SourceState("fire", 100.0, -13.0, 0.02, 14.0)],
        "E": [SourceState("fire", 150.0, -3.0, np.deg2rad(80.0), 14.0)],
        "S": [SourceState("fire", 200.0, +5.0, np.deg2rad(120.0), 14.0)],
        "W": [SourceState("fire", 170.0, -2.0, np.deg2rad(95.0), 14.0)],
    }
    fired = [a for a, r in sen.sense(scene).items() if r.p_siren > 0.5]
    assert fired == ["N"], fired


# ---------------------------------------------------------------------------
# The level, and the rate the microphones are read at
# ---------------------------------------------------------------------------
def test_ambient_floor_drifts_rather_than_resampling():
    """The noise floor is a process, not an independent draw per frame.

    It used to be redrawn uniformly over its whole 25 dB range at every call.
    That is harmless while nothing reads the level, and fatal once something
    does: an approaching siren gains one or two decibels a second, which is
    invisible under 25 dB of per-frame noise. The floor must move slowly, and
    its stationary distribution must still be the uniform the detector's error
    model was fitted against.
    """
    cfg = AcousticConfig()
    sen = AcousticSensor(["N"], cfg, mode="surrogate", seed=3)
    lo, hi = cfg.ambient_noise_db_range
    series = [sen._ambient_db("N", 1.0) for _ in range(4000)]
    steps = np.abs(np.diff(series))
    assert float(np.mean(steps)) < 0.2 * (hi - lo), "the floor is still resampling"
    assert min(series) >= lo - 1e-9 and max(series) <= hi + 1e-9, "floor left its range"
    # Reflected on an interval, the walk is uniform in the long run, so it must
    # visit both halves rather than sitting where it started.
    assert min(series) < lo + 0.25 * (hi - lo)
    assert max(series) > hi - 0.25 * (hi - lo)


def test_reading_carries_the_measured_level():
    """Level and floor come back with the reading, and the SNR follows."""
    sen = AcousticSensor(["N"], AcousticConfig(), mode="surrogate", seed=1)
    r = sen.sense({"N": [SourceState("ambulance", 40.0, -12.0, 0.0, 14.0)]})["N"]
    assert r.spl_db is not None and r.ambient_db is not None
    assert abs(r.snr_db - (r.spl_db - r.ambient_db)) < 1e-9


def test_level_rises_as_the_source_closes():
    """The quantity the controller now uses: loudness is monotone in closeness.

    The posterior is not -- it saturates at 1.0 hundreds of metres out and
    stays there -- so the level is what distinguishes a vehicle about to arrive
    from one that has only just been heard.
    """
    sen = AcousticSensor(["N"], AcousticConfig(), mode="surrogate", seed=7)
    levels = [sen.sense({"N": [SourceState("ambulance", d, -14.0, 0.0, 14.0)]})["N"].spl_db
              for d in (250.0, 150.0, 80.0, 30.0)]
    assert all(b > a for a, b in zip(levels, levels[1:])), levels


def test_belief_transition_is_a_function_of_elapsed_time():
    """One 5 s predict and five 1 s predicts must leave the same posterior.

    The survival and birth probabilities are quoted per epoch. The environment
    now advances the filter once per detector frame instead of once per
    decision epoch, so applying them per *call* would run the chain five times
    as fast -- the same stationary occupancy, but a presence that decays five
    times quicker in wall-clock time, which drops a corridor through any
    two-frame detection dropout.
    """
    cfg = BeliefConfig()
    coarse = BeliefTracker(["N"], cfg)
    fine = BeliefTracker(["N"], cfg)
    coarse._presence["N"] = fine._presence["N"] = 0.8
    coarse.predict(cfg.transition_reference_s)
    for _ in range(5):
        fine.predict(cfg.transition_reference_s / 5.0)
    assert abs(coarse._presence["N"] - fine._presence["N"]) < 5e-3, (
        coarse._presence["N"], fine._presence["N"])


def test_imminence_survives_a_stopped_vehicle():
    """The feature the policy learns priority from must not vanish at the red.

    Arrival time is range over closing speed, so a vehicle held at a red -- the
    exact case a priority signal exists for -- has a closing speed of zero, an
    arrival time pinned at its ceiling, and under the previous definition an
    imminence of exactly zero. Range has no such failure mode.
    """
    cfg = BeliefConfig()
    stopped = ApproachBelief(presence=0.9, eta_s=cfg.max_eta_s, eta_var=1e4,
                             class_probs={}, expected_priority=1.0,
                             range_m=8.0, range_rate_ms=0.0, signed_range=True)
    assert stopped.imminence(cfg.max_eta_s, 300.0) > 0.9

    far = ApproachBelief(presence=0.9, eta_s=cfg.max_eta_s, eta_var=1e4,
                         class_probs={}, expected_priority=1.0,
                         range_m=280.0, range_rate_ms=0.0, signed_range=True)
    assert far.imminence(cfg.max_eta_s, 300.0) < 0.1
    assert stopped.imminence(cfg.max_eta_s, 300.0) > far.imminence(cfg.max_eta_s, 300.0)


def test_imminence_still_prefers_a_usable_arrival_time():
    """Where the arrival estimate works, it is what is used."""
    cfg = BeliefConfig()
    b = ApproachBelief(presence=0.9, eta_s=9.0, eta_var=4.0, class_probs={},
                       expected_priority=1.0, range_m=120.0, range_rate_ms=-13.0,
                       signed_range=True)
    assert abs(b.imminence(cfg.max_eta_s, 300.0) - (1.0 - 9.0 / cfg.max_eta_s)) < 1e-9


def test_evidence_is_bounded_by_elapsed_time_not_sampling_rate():
    """Sampling the microphone faster must not inflate the posterior.

    The filter multiplies likelihood ratios, which is right for independent
    measurements and wrong for a stream sampled faster than the scene changes.
    Reading a detector that says the same thing five times a second is not five
    times the evidence: the source has barely moved and the likelihood is a
    fitted model, so its misspecification compounds rather than averaging out.
    Measured, over-counting took false detections from 122/h to 1020/h.
    """
    cfg = BeliefConfig()
    coarse = BeliefTracker(["N"], cfg)
    fine = BeliefTracker(["N"], cfg)
    coarse._presence["N"] = fine._presence["N"] = cfg.prior_presence

    coarse.predict(cfg.transition_reference_s)
    coarse.update("N", 0.95, None, None)

    step = cfg.transition_reference_s / 5.0
    for _ in range(5):
        fine.predict(step)
        fine.update("N", 0.95, None, None)

    assert abs(coarse._presence["N"] - fine._presence["N"]) < 0.06, (
        coarse._presence["N"], fine._presence["N"])


def test_untempered_accumulation_would_have_been_far_more_confident():
    """The bug this guards against, stated as the difference it made."""
    cfg = BeliefConfig()
    tempered = BeliefTracker(["N"], cfg)
    raw = BeliefTracker(["N"], BeliefConfig(evidence_per_reference_interval=5.0))
    for f in (tempered, raw):
        f._presence["N"] = cfg.prior_presence
        for _ in range(5):
            f.predict(cfg.transition_reference_s / 5.0)
            f.update("N", 0.9, None, None)
    assert raw._presence["N"] > tempered._presence["N"] + 0.2, (
        raw._presence["N"], tempered._presence["N"])


# ---------------------------------------------------------------------------
# The horn confuser
# ---------------------------------------------------------------------------
def test_a_horn_is_louder_than_a_siren_at_the_same_range():
    """No level threshold can separate a horn from a siren, and this is why.

    A road horn puts out 100-112 dB at 1 m against a siren's 109-116, but it
    honks from the queue -- metres from the mast -- while the siren that matters
    is still hundreds of metres away. The confuser is routinely the *louder*
    signal at the microphone.
    """
    horn = SourceState("horn", 30.0, 0.0, 0.0, 14.0)
    siren = SourceState("ambulance", 200.0, -14.0, 0.0, 14.0)
    assert horn.received_spl_db() > siren.received_spl_db()


def test_horn_holds_its_pitch_while_a_siren_sweeps():
    """The discriminator, stated as a property of the source rather than a feature.

    A siren's fundamental glides; a horn's does not. Everything else about the
    two -- level, band, harmonic richness -- overlaps, so this modulation
    difference is what the classifier has to key on, and it is invariant to
    distance, directivity and the ambient floor.
    """
    from atsc.sensing.siren_synth import horn_instantaneous_frequency

    t = np.linspace(0.0, 1.0, 16000, endpoint=False)
    synth = SirenSynthesizer(CFG, seed=3)

    horn_f = horn_instantaneous_frequency(t, 440.0)
    horn_span = float(horn_f.max() - horn_f.min())

    for pattern in ("wail", "yelp", "hilo"):
        f = synth._instantaneous_frequency(pattern, t, "ambulance")
        span = float(f.max() - f.min())
        assert span > 8.0 * horn_span, (
            f"{pattern} sweeps only {span:.0f} Hz against the horn's {horn_span:.0f} Hz"
        )
    assert horn_span < 20.0, f"horn drifted {horn_span:.0f} Hz, which is a sweep"


def test_horn_scenes_are_generated_and_labelled():
    """Horns must actually reach the training set, or the class teaches nothing."""
    synth = SirenSynthesizer(CFG, seed=5)
    labels = [synth.sample_scene(p_siren=0.5).label for _ in range(400)]
    assert "horn" in labels, "no horn was ever drawn"
    assert 0.05 < labels.count("horn") / len(labels) < 0.30


def test_horn_is_a_distinct_detector_class():
    """Folded into `none`, a correctly rejected horn is indistinguishable in the
    confusion matrix from silence, and the rejection cannot be reported."""
    assert "horn" in CFG.class_map
    assert CFG.class_map["horn"] != CFG.class_map["none"]


def test_error_model_falls_back_pessimistically_without_horn_data():
    """An error model fitted before horns existed must not treat them as free."""
    em = SensorErrorModel.default()
    em.horn_logit_mean, em.horn_logit_std = [], []
    mu, _sd = em.horn_logit(3)
    assert mu == em.siren_logit_mean[3], "a horn must fall back to the siren curve"


def test_siren_probability_excludes_the_horn_class():
    """The readout bug that hid the detector's own horn rejection.

    ``p_siren`` was computed as ``1 - P(none)``. That is correct only while
    every non-silent class is an emergency class. Once the confuser has its own
    label it is wrong in the worst possible way: a detector that has confidently
    and *correctly* identified a road horn puts its mass on ``horn``, so
    ``1 - P(none)`` reads ~1.0 and the controller is told a siren is present.

    Measured on the retrained detector, the old readout called a horn a siren
    with probability 1.000 at every SNR above 0 dB, which reads as a network
    that cannot tell them apart. Summing the siren classes instead gives 0.001
    in the same bin. The network had learned the distinction all along.
    """
    from atsc.sensing.belief import siren_probability

    cm = CFG.class_map
    horn = np.zeros(len(cm))
    horn[cm["horn"]] = 0.95
    horn[cm["none"]] = 0.05
    assert siren_probability(horn, cm) < 0.01
    assert 1.0 - horn[cm["none"]] > 0.9, "the old readout really was this wrong"

    siren = np.zeros(len(cm))
    siren[cm["ambulance"]] = 0.9
    siren[cm["none"]] = 0.1
    assert siren_probability(siren, cm) > 0.85

    batch = np.stack([horn, siren])
    got = siren_probability(batch, cm)
    assert got.shape == (2,) and got[0] < 0.01 < got[1]


def test_fitted_horn_curve_rejects_horns_where_it_matters():
    """The shipped error model must actually reject horns at audible SNR.

    A horn is sounded from the queue, metres from the mast, so it arrives at
    20 dB SNR and above. That is the band the guarantee acts on, and a
    false-alarm rate there is a forced green on the wrong approach.
    """
    from atsc.config import MODELS_DIR

    path = MODELS_DIR / "sensor_error_model.json"
    if not path.exists():
        return
    em = SensorErrorModel.load(path)
    if not em.horn_logit_mean:
        return
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))
    for b in range(len(em.horn_logit_mean)):
        lo = em.snr_edges[b - 1] if 0 < b <= len(em.snr_edges) else -np.inf
        if lo < 0.0:
            continue
        p_horn = sig(em.horn_logit_mean[b])
        p_siren = sig(em.siren_logit_mean[b])
        assert p_horn < 0.10, f"bin {b}: horn fires at {p_horn:.3f}"
        assert p_siren > 0.90, f"bin {b}: siren only fires at {p_siren:.3f}"
