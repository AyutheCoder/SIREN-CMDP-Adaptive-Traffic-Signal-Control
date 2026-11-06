"""Replay buffer, Lagrangian agent, and Webster timing."""

from __future__ import annotations

import numpy as np

from atsc.agents.baselines import webster_green_splits
from atsc.agents.dueling_ddqn import LagrangianDuelingDDQN
from atsc.agents.replay import PrioritizedReplay, SumTree, Transition
from atsc.config import AgentConfig

STATE_DIM, N_ACTIONS = 12, 4


def _t(i: int, reward: float = 1.0, cost: float = 0.5, done: bool = False) -> Transition:
    return Transition(
        state=np.full(STATE_DIM, i, dtype=np.float32),
        action=i % N_ACTIONS,
        reward=reward,
        cost=cost,
        next_state=np.full(STATE_DIM, i + 1, dtype=np.float32),
        done=done,
        next_mask=np.ones(N_ACTIONS, dtype=np.float32),
    )


# ------------------------------------------------------------------- sum tree
def test_sumtree_total_and_sampling():
    tree = SumTree(8)
    for p in (1.0, 3.0, 2.0, 4.0):
        tree.add(p)
    assert abs(tree.total - 10.0) < 1e-9
    assert tree.sample_index(0.5) == 0
    assert tree.sample_index(2.0) == 1
    assert tree.sample_index(5.0) == 2
    assert tree.sample_index(9.0) == 3


def test_sumtree_update_propagates():
    tree = SumTree(4)
    for _ in range(4):
        tree.add(1.0)
    tree.update(2, 5.0)
    assert abs(tree.total - 8.0) < 1e-9


def test_sumtree_wraps_at_capacity():
    tree = SumTree(4)
    for _ in range(6):
        tree.add(1.0)
    assert tree.size == 4 and abs(tree.total - 4.0) < 1e-9


# ---------------------------------------------------------------------- replay
def test_nstep_returns_discount_both_signals():
    gamma, n = 0.9, 3
    buf = PrioritizedReplay(64, STATE_DIM, N_ACTIONS, n_step=n, gamma=gamma)
    for i in range(n):
        buf.push(_t(i, reward=1.0, cost=0.5))
    assert len(buf) == 1
    expected_r = 1.0 + gamma * 1.0 + gamma ** 2 * 1.0
    expected_c = 0.5 + gamma * 0.5 + gamma ** 2 * 0.5
    assert abs(float(buf.rewards[0]) - expected_r) < 1e-5
    assert abs(float(buf.costs[0]) - expected_c) < 1e-5
    assert float(buf.n_steps[0]) == n


def test_nstep_transition_spans_correct_states():
    buf = PrioritizedReplay(64, STATE_DIM, N_ACTIONS, n_step=3, gamma=0.9)
    for i in range(3):
        buf.push(_t(i))
    assert float(buf.states[0][0]) == 0.0
    assert float(buf.next_states[0][0]) == 3.0


def test_done_flushes_remaining_transitions():
    buf = PrioritizedReplay(64, STATE_DIM, N_ACTIONS, n_step=5, gamma=0.9)
    for i in range(3):
        buf.push(_t(i, done=(i == 2)))
    # A terminal transition emits the partial windows rather than dropping them.
    assert len(buf) == 3
    assert float(buf.dones[0]) == 1.0


def test_sampling_returns_consistent_shapes_and_weights():
    buf = PrioritizedReplay(256, STATE_DIM, N_ACTIONS, n_step=1, gamma=0.9, seed=0)
    for i in range(100):
        buf.push(_t(i))
    (s, a, r, c, s2, d, masks, ns, w, idx) = buf.sample(16, step=10)
    assert s.shape == (16, STATE_DIM) and s2.shape == (16, STATE_DIM)
    assert a.shape == r.shape == c.shape == d.shape == (16,)
    assert masks.shape == (16, N_ACTIONS)
    assert w.shape == (16,) and float(w.max()) <= 1.0 + 1e-6
    assert idx.shape == (16,)


def test_priority_update_biases_sampling():
    buf = PrioritizedReplay(64, STATE_DIM, N_ACTIONS, n_step=1, gamma=0.9, seed=1)
    for i in range(32):
        buf.push(_t(i))
    idx = np.arange(32)
    td = np.full(32, 0.01)
    td[7] = 50.0
    buf.update_priorities(idx, td)
    counts = np.zeros(32)
    for _ in range(40):
        _, _, _, _, _, _, _, _, _, sampled = buf.sample(8, step=1)
        for j in sampled:
            counts[j] += 1
    assert counts[7] == counts.max() and counts[7] > 20, counts[7]


def test_beta_anneals_to_one():
    buf = PrioritizedReplay(8, STATE_DIM, N_ACTIONS, beta_start=0.4,
                            beta_final=1.0, beta_steps=100)
    assert abs(buf.beta(0) - 0.4) < 1e-9
    assert abs(buf.beta(100) - 1.0) < 1e-9
    assert abs(buf.beta(1000) - 1.0) < 1e-9


# ----------------------------------------------------------------------- agent
def test_agent_respects_action_mask():
    agent = LagrangianDuelingDDQN(STATE_DIM, N_ACTIONS, AgentConfig(seed=0))
    state = np.random.default_rng(0).standard_normal(STATE_DIM).astype(np.float32)
    mask = np.array([False, True, False, True])
    for _ in range(50):
        assert agent.act(state, mask) in (1, 3)


def test_agent_single_legal_action_is_forced():
    agent = LagrangianDuelingDDQN(STATE_DIM, N_ACTIONS, AgentConfig(seed=0))
    state = np.zeros(STATE_DIM, dtype=np.float32)
    mask = np.array([False, False, True, False])
    assert agent.act(state, mask) == 2


def test_agent_outputs_two_critics():
    agent = LagrangianDuelingDDQN(STATE_DIM, N_ACTIONS, AgentConfig(seed=0))
    qr, qc = agent.q_values(np.zeros(STATE_DIM, dtype=np.float32))
    assert qr.shape == (N_ACTIONS,) and qc.shape == (N_ACTIONS,)
    assert np.isfinite(qr).all() and np.isfinite(qc).all()


def test_lambda_rises_above_budget_and_falls_below():
    # Warmup disabled: this test is about the sign of the dual step, not its
    # schedule (see test_lambda_is_held_fixed_during_warmup for that).
    cfg = AgentConfig(constrained=True, cost_budget=0.1, lambda_init=1.0,
                      lambda_lr=0.5, cost_ema=0.0, lambda_warmup_steps=0, seed=0)
    agent = LagrangianDuelingDDQN(STATE_DIM, N_ACTIONS, cfg)

    agent.observe_cost(0.9)          # well over budget
    for _ in range(5):
        agent.update_lambda()
    assert float(agent.lam.numpy()) > 1.0

    high = float(agent.lam.numpy())
    agent.observe_cost(0.0)          # comfortably under budget
    for _ in range(5):
        agent.update_lambda()
    assert float(agent.lam.numpy()) < high


def test_lambda_is_clipped_to_nonnegative_and_bounded():
    cfg = AgentConfig(constrained=True, cost_budget=1.0, lambda_init=0.1,
                      lambda_lr=10.0, lambda_max=5.0, cost_ema=0.0,
                      lambda_warmup_steps=0, seed=0)
    agent = LagrangianDuelingDDQN(STATE_DIM, N_ACTIONS, cfg)
    agent.observe_cost(0.0)
    for _ in range(20):
        agent.update_lambda()
    assert float(agent.lam.numpy()) == 0.0

    agent.observe_cost(100.0)
    for _ in range(20):
        agent.update_lambda()
    assert float(agent.lam.numpy()) == 5.0


def test_unconstrained_agent_keeps_lambda_at_zero():
    cfg = AgentConfig(constrained=False, seed=0)
    agent = LagrangianDuelingDDQN(STATE_DIM, N_ACTIONS, cfg)
    agent.observe_cost(10.0)
    for _ in range(10):
        agent.update_lambda()
    assert float(agent.lam.numpy()) == 0.0


def test_learning_step_runs_and_changes_weights():
    cfg = AgentConfig(batch_size=8, warmup_transitions=8, seed=0)
    agent = LagrangianDuelingDDQN(STATE_DIM, N_ACTIONS, cfg)
    buf = PrioritizedReplay(256, STATE_DIM, N_ACTIONS, n_step=1,
                            gamma=cfg.gamma, seed=0)
    rng = np.random.default_rng(0)
    for i in range(64):
        buf.push(Transition(
            state=rng.standard_normal(STATE_DIM).astype(np.float32),
            action=int(rng.integers(N_ACTIONS)),
            reward=float(rng.normal()),
            cost=float(abs(rng.normal())),
            next_state=rng.standard_normal(STATE_DIM).astype(np.float32),
            done=False,
            next_mask=np.ones(N_ACTIONS, dtype=np.float32),
        ))
    before = [w.copy() for w in agent.online.get_weights()]
    losses = [agent.learn(buf) for _ in range(5)]
    assert all(l is not None and np.isfinite(l) for l in losses)
    after = agent.online.get_weights()
    assert any(not np.allclose(a, b) for a, b in zip(before, after))


def test_target_network_syncs_on_schedule():
    cfg = AgentConfig(batch_size=4, warmup_transitions=4, target_sync_steps=3, seed=0)
    agent = LagrangianDuelingDDQN(STATE_DIM, N_ACTIONS, cfg)
    buf = PrioritizedReplay(64, STATE_DIM, N_ACTIONS, n_step=1, gamma=cfg.gamma)
    for i in range(16):
        buf.push(_t(i))
    for _ in range(3):
        agent.learn(buf)
    for a, b in zip(agent.online.get_weights(), agent.target.get_weights()):
        assert np.allclose(a, b)


def test_agent_save_load_roundtrip():
    import tempfile
    from pathlib import Path
    cfg = AgentConfig(seed=0)
    agent = LagrangianDuelingDDQN(STATE_DIM, N_ACTIONS, cfg)
    agent.lam.assign(2.75)
    agent.cost_ema = 0.33
    state = np.random.default_rng(1).standard_normal(STATE_DIM).astype(np.float32)
    q_before = agent.composite_q(state)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "agent.h5"
        agent.save(path)
        restored = LagrangianDuelingDDQN(STATE_DIM, N_ACTIONS, cfg).load(path)

    assert abs(float(restored.lam.numpy()) - 2.75) < 1e-6
    assert abs(restored.cost_ema - 0.33) < 1e-9
    assert np.allclose(q_before, restored.composite_q(state), atol=1e-5)


# --------------------------------------------------------------------- Webster
def test_webster_cycle_grows_with_demand():
    lanes = [2, 1, 2, 1]
    light, _ = webster_green_splits([300, 90, 300, 90], lanes, 5.0)
    heavy, _ = webster_green_splits([1100, 260, 1100, 260], lanes, 5.0)
    assert heavy > light


def test_webster_allocates_green_by_flow_ratio():
    """Green follows the flow *ratio* (flow per lane), not raw flow."""
    _, greens = webster_green_splits([900, 150, 500, 150], [2, 1, 2, 1], 5.0)
    assert greens[0] > greens[2] > greens[1]

    # 300 veh/h across two lanes loads a phase exactly as hard as 150 across
    # one, so those two phases must receive equal green.
    _, equal = webster_green_splits([900, 150, 300, 150], [2, 1, 2, 1], 5.0)
    assert abs(equal[1] - equal[2]) < 1e-9
    assert abs(equal[1] - equal[3]) < 1e-9


def test_webster_clamps_oversaturated_demand():
    cycle, greens = webster_green_splits([5000, 5000, 5000, 5000], [1, 1, 1, 1], 5.0)
    assert cycle == 180.0
    assert all(np.isfinite(g) and g > 0 for g in greens)


def test_lambda_is_held_fixed_during_warmup():
    """Dual ascent must not start before the cost critic is worth listening to."""
    cfg = AgentConfig(constrained=True, cost_budget=0.01, lambda_init=1.0,
                      lambda_lr=1.0, cost_ema=0.0, lambda_warmup_steps=100, seed=0)
    agent = LagrangianDuelingDDQN(STATE_DIM, N_ACTIONS, cfg)
    agent.observe_cost(5.0)                       # wildly infeasible

    agent.env_steps = 50
    for _ in range(10):
        agent.update_lambda()
    assert float(agent.lam.numpy()) == 1.0, "lambda moved during warmup"

    agent.env_steps = 500
    agent.update_lambda()
    assert float(agent.lam.numpy()) > 1.0, "lambda did not resume after warmup"


def test_relative_violation_makes_dual_step_budget_invariant():
    """Halving the budget must not halve the speed the multiplier responds at."""
    steps = {}
    for budget in (0.05, 0.005):
        cfg = AgentConfig(constrained=True, cost_budget=budget, lambda_init=0.0,
                          lambda_lr=0.1, cost_ema=0.0, lambda_warmup_steps=0, seed=0)
        agent = LagrangianDuelingDDQN(STATE_DIM, N_ACTIONS, cfg)
        agent.observe_cost(budget * 2.0)          # same *relative* violation
        agent.update_lambda()
        steps[budget] = float(agent.lam.numpy())
    assert abs(steps[0.05] - steps[0.005]) < 1e-9, steps
