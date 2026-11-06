"""Prioritised n-step replay carrying both a reward and a constraint cost.

Two departures from a textbook PER buffer, both required by the CMDP setting:

* Every transition stores **two** returns, ``r`` and ``c``, accumulated over the
  same n-step window with the same discount. The critic learns ``Q_r`` and
  ``Q_c`` from the identical trajectory, so the Lagrangian combination
  ``Q_r - lambda * Q_c`` is consistent.

* Priorities come from the TD error of the **composite** ``Q_r - lambda*Q_c``,
  not of the reward alone. Emergency-vehicle events are rare; prioritising on
  reward error alone would let the buffer starve the cost critic of exactly the
  transitions it needs.

The sum tree gives O(log N) proportional sampling and O(log N) priority update.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    cost: float
    next_state: np.ndarray
    done: bool
    next_mask: np.ndarray
    n_step: int = 1


class SumTree:
    """Fixed-capacity binary tree over priorities; internal nodes hold subtree sums."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.size = 0
        self.cursor = 0

    def add(self, priority: float) -> int:
        idx = self.cursor
        self.update(idx, priority)
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return idx

    def update(self, idx: int, priority: float) -> None:
        node = idx + self.capacity
        delta = priority - self.tree[node]
        self.tree[node] = priority
        node //= 2
        while node >= 1:
            self.tree[node] += delta
            node //= 2

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def sample_index(self, value: float) -> int:
        node = 1
        while node < self.capacity:
            left = 2 * node
            if value <= self.tree[left]:
                node = left
            else:
                value -= self.tree[left]
                node = left + 1
        return node - self.capacity


class PrioritizedReplay:
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        n_actions: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_final: float = 1.0,
        beta_steps: int = 100_000,
        eps: float = 1e-3,
        n_step: int = 3,
        gamma: float = 0.95,
        seed: int = 0,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_final = beta_final
        self.beta_steps = max(beta_steps, 1)
        self.eps = eps
        self.n_step = max(n_step, 1)
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.costs = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.next_masks = np.ones((capacity, n_actions), dtype=np.float32)
        self.n_steps = np.ones(capacity, dtype=np.float32)

        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        self._pending: Deque[Transition] = deque()
        self._n_added = 0

    def __len__(self) -> int:
        return self.tree.size

    # ------------------------------------------------------------------ insert
    def push(self, t: Transition) -> None:
        """Buffer a raw transition; emit the n-step aggregate once available."""
        self._pending.append(t)
        if t.done:
            while self._pending:
                self._commit(len(self._pending))
                self._pending.popleft()
            return
        if len(self._pending) >= self.n_step:
            self._commit(self.n_step)
            self._pending.popleft()

    def _commit(self, horizon: int) -> None:
        items = list(self._pending)[:horizon]
        head, tail = items[0], items[-1]
        r = c = 0.0
        discount = 1.0
        for it in items:
            r += discount * it.reward
            c += discount * it.cost
            discount *= self.gamma
            if it.done:
                break
        self._store(
            Transition(
                state=head.state,
                action=head.action,
                reward=r,
                cost=c,
                next_state=tail.next_state,
                done=tail.done,
                next_mask=tail.next_mask,
                n_step=len(items),
            )
        )

    def _store(self, t: Transition) -> None:
        idx = self.tree.add(self.max_priority ** self.alpha)
        self.states[idx] = t.state
        self.next_states[idx] = t.next_state
        self.actions[idx] = t.action
        self.rewards[idx] = t.reward
        self.costs[idx] = t.cost
        self.dones[idx] = float(t.done)
        self.next_masks[idx] = t.next_mask.astype(np.float32)
        self.n_steps[idx] = float(t.n_step)
        self._n_added += 1

    def flush(self) -> None:
        """Emit any partially-accumulated transitions (call at episode end)."""
        while self._pending:
            self._commit(len(self._pending))
            self._pending.popleft()

    # ------------------------------------------------------------------ sample
    def beta(self, step: int) -> float:
        frac = min(step / self.beta_steps, 1.0)
        return self.beta_start + frac * (self.beta_final - self.beta_start)

    def sample(self, batch_size: int, step: int):
        total = self.tree.total
        if self.tree.size == 0 or total <= 0:
            raise ValueError("cannot sample from an empty buffer")

        segment = total / batch_size
        idx = np.empty(batch_size, dtype=np.int64)
        for i in range(batch_size):
            v = self.rng.uniform(i * segment, (i + 1) * segment)
            j = self.tree.sample_index(v)
            # Guard against the rare case of landing on an unfilled slot.
            idx[i] = j if j < self.tree.size else self.rng.integers(self.tree.size)

        priorities = self.tree.tree[idx + self.capacity]
        probs = priorities / total
        beta = self.beta(step)
        weights = (self.tree.size * np.maximum(probs, 1e-12)) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)

        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.costs[idx],
            self.next_states[idx],
            self.dones[idx],
            self.next_masks[idx],
            self.n_steps[idx],
            weights,
            idx,
        )

    def update_priorities(self, idx: np.ndarray, td_errors: np.ndarray) -> None:
        p = (np.abs(td_errors) + self.eps) ** self.alpha
        self.max_priority = max(self.max_priority, float(np.max(np.abs(td_errors) + self.eps)))
        for i, pri in zip(idx, p):
            self.tree.update(int(i), float(pri))
