"""Property tests: invariants that must hold under randomised input.

The example-based tests elsewhere check specific behaviours. These sweep the
input space instead, which is what catches the bugs nobody thought to write a
test for. They are cheap because none of them needs SUMO.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from atsc.agents.replay import PrioritizedReplay, Transition
from atsc.config import AcousticConfig, BeliefConfig
from atsc.sensing.belief import BeliefTracker
from atsc.sensing.sensor import AcousticSensor, SensorErrorModel, SourceState

APPROACHES = ["N", "E", "S", "W"]


def test_belief_state_stays_well_formed_under_random_input():
    """No sequence of detector outputs may produce an invalid posterior."""
    cfg = BeliefConfig()
    bt = BeliefTracker(APPROACHES, cfg)
    rng = np.random.default_rng(0)

    for _ in range(400):
        bt.predict(float(rng.uniform(0.5, 12.0)))
        for a in APPROACHES:
            has_range = rng.random() < 0.7
            r = float(rng.uniform(1.0, 400.0)) if has_range else None
            bt.update(
                a,
                p_siren=float(rng.random()),
                class_probs={"ambulance": 0.5, "fire": 0.3, "police": 0.2},
                range_measurement_m=r,
                range_measurement_var=(0.45 * (r or 1.0)) ** 2,
                rate_hint_ms=float(rng.uniform(-25.0, 10.0)),
            )
        for a in APPROACHES:
            b = bt.belief(a)
            assert 0.0 <= b.presence <= 1.0, b.presence
            assert abs(sum(b.class_probs.values()) - 1.0) < 1e-6
            assert 0.0 <= b.eta_s <= cfg.max_eta_s, b.eta_s
            assert math.isfinite(b.eta_var) and b.eta_var > 0
            assert b.range_m >= 0.0
    assert np.isfinite(bt.feature_vector()).all()


def test_sensor_outputs_are_well_formed_in_every_mode():
    rng = np.random.default_rng(1)
    for mode in ("oracle", "surrogate", "none"):
        sensor = AcousticSensor(APPROACHES[:2], AcousticConfig(), mode=mode,
                                error_model=SensorErrorModel.default(), seed=1)
        for _ in range(200):
            scene = {
                "N": [SourceState("ambulance",
                                  float(rng.uniform(2.0, 400.0)),
                                  float(rng.uniform(-30.0, 20.0)),
                                  float(rng.uniform(0.0, np.pi)), 14.0)],
                "E": [],
            }
            for reading in sensor.sense(scene).values():
                assert 0.0 <= reading.p_siren <= 1.0, reading.p_siren
                if reading.class_probs:
                    assert abs(sum(reading.class_probs.values()) - 1.0) < 1e-6
                if reading.distance_m is not None:
                    assert 0.0 < reading.distance_m < 1e4
                    assert reading.range_var is not None and reading.range_var > 0


def test_nstep_returns_match_the_closed_form_for_every_n_and_gamma():
    """Both the reward and the cost must discount identically."""
    rewards = [0.5, -1.0, 2.0, 0.25, -0.75, 1.5]
    costs = [0.1, 0.2, 0.0, 0.4, 0.3, 0.05]
    for n, gamma in itertools.product((1, 2, 3, 5), (0.9, 0.95, 0.99)):
        buf = PrioritizedReplay(64, 4, 3, n_step=n, gamma=gamma, seed=0)
        for i in range(n):
            buf.push(Transition(
                np.full(4, i, np.float32), i % 3, rewards[i], costs[i],
                np.full(4, i + 1, np.float32), False, np.ones(3, np.float32)))
        expected_r = sum(gamma ** k * rewards[k] for k in range(n))
        expected_c = sum(gamma ** k * costs[k] for k in range(n))
        assert abs(float(buf.rewards[0]) - expected_r) < 1e-5, (n, gamma)
        assert abs(float(buf.costs[0]) - expected_c) < 1e-5, (n, gamma)


def test_sum_tree_samples_the_correct_bucket_everywhere():
    """Proportional sampling must land in the interval each priority owns."""
    buf = PrioritizedReplay(64, 4, 3, seed=0)
    priorities = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    for p in priorities:
        buf.tree.add(p)
    assert abs(buf.tree.total - sum(priorities)) < 1e-9
    cumulative = np.cumsum(priorities)
    for i, edge in enumerate(cumulative):
        assert buf.tree.sample_index(float(edge) - 1e-6) == i, i
        lower = 0.0 if i == 0 else float(cumulative[i - 1])
        assert buf.tree.sample_index(lower + 1e-6) == i, i


def test_importance_weights_are_bounded_and_normalised():
    buf = PrioritizedReplay(256, 4, 3, n_step=1, gamma=0.9, seed=0)
    rng = np.random.default_rng(0)
    for i in range(120):
        buf.push(Transition(rng.standard_normal(4).astype(np.float32), i % 3,
                            float(rng.normal()), float(abs(rng.normal())),
                            rng.standard_normal(4).astype(np.float32), False,
                            np.ones(3, np.float32)))
    buf.update_priorities(np.arange(len(buf)), rng.uniform(0.01, 10.0, len(buf)))
    for step in (0, 1000, 100000):
        *_, weights, idx = buf.sample(32, step=step)
        assert weights.shape == (32,)
        assert np.all(weights > 0) and float(weights.max()) <= 1.0 + 1e-6
        assert np.all(idx < len(buf))
