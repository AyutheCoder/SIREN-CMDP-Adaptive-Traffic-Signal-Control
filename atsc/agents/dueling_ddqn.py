"""Lagrangian dueling Double DQN for the constrained signal-control MDP.

The problem being solved is

    maximise   E[ sum_t gamma^t r_t ]
    subject to E[ sum_t gamma_c^t c_t ] <= d0

with ``r`` network efficiency and ``c`` priority-weighted emergency delay. The
saddle-point form of the Lagrangian is

    max_pi min_{lambda >= 0}   J_r(pi) - lambda * ( J_c(pi) - d0 )

so the algorithm keeps **two** critics on a shared trunk, ``Q_r`` and ``Q_c``,
and acts greedily with respect to the composite ``Q_r - lambda * Q_c``. The
multiplier is updated by projected dual ascent on the constraint violation,

    lambda <- clip( lambda + eta * ( Jhat_c - d0 ), 0, lambda_max )

on a slower timescale than the critics, which is the standard two-timescale
condition for this class of methods to converge to a saddle point.

Why this beats a hand-tuned penalty
-----------------------------------
Writing ``r - k*c`` and tuning ``k`` gives a controller whose emergency-vehicle
behaviour is whatever the tuning happened to produce, and ``k`` has to be
re-tuned for every demand level. Here the *requirement* is the input and the
trade-off weight is discovered: when emergency delay sits under budget lambda
decays and the agent recovers ordinary throughput; when it exceeds budget lambda
climbs until the policy yields. Sweeping ``d0`` traces the achievable frontier
directly (see ``scripts/frontier.py``).

Both critics are dueling and trained with Double-DQN targets, prioritised n-step
replay, and action masking supplied by the safety shield, so the argmax is only
ever taken over actions the shield would actually permit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ..config import AgentConfig

NEG_INF = -1e9


def _huber(error: tf.Tensor, delta: float = 1.0) -> tf.Tensor:
    abs_e = tf.abs(error)
    quad = tf.minimum(abs_e, delta)
    lin = abs_e - quad
    return 0.5 * quad ** 2 + delta * lin


class LagrangianDuelingDDQN:
    def __init__(self, state_dim: int, n_actions: int, cfg: Optional[AgentConfig] = None):
        self.cfg = cfg or AgentConfig()
        self.state_dim = state_dim
        self.n_actions = n_actions

        tf.random.set_seed(self.cfg.seed)
        self.rng = np.random.default_rng(self.cfg.seed)

        self.online = self._build()
        self.target = self._build()
        self.target.set_weights(self.online.get_weights())
        self.optimizer = keras.optimizers.Adam(learning_rate=self.cfg.learning_rate)

        # Held as a tf.Variable so changing it does not retrace the train step.
        self.lam = tf.Variable(
            self.cfg.lambda_init if self.cfg.constrained else 0.0,
            dtype=tf.float32, trainable=False, name="lambda",
        )
        self.cost_ema = float("nan")
        self.train_steps = 0
        self.env_steps = 0
        self.lambda_history: list = []

    # ------------------------------------------------------------------ model
    def _build(self) -> keras.Model:
        cfg = self.cfg
        inp = keras.Input(shape=(self.state_dim,), name="obs")
        h = inp
        for i, units in enumerate(cfg.hidden_units):
            h = layers.Dense(units, activation="relu", name=f"trunk{i}")(h)

        def stream(prefix: str):
            v = layers.Dense(128, activation="relu", name=f"{prefix}_v_h")(h)
            v = layers.Dense(1, name=f"{prefix}_v")(v)
            a = layers.Dense(128, activation="relu", name=f"{prefix}_a_h")(h)
            a = layers.Dense(self.n_actions, name=f"{prefix}_a")(a)
            a_centred = a - tf.reduce_mean(a, axis=1, keepdims=True)
            return layers.Add(name=f"{prefix}_q")([v, a_centred])

        return keras.Model(inp, [stream("r"), stream("c")], name="lagrangian_dueling_ddqn")

    # ----------------------------------------------------------------- acting
    @property
    def epsilon(self) -> float:
        cfg = self.cfg
        frac = min(self.env_steps / max(cfg.epsilon_decay_steps, 1), 1.0)
        return cfg.epsilon_start + frac * (cfg.epsilon_final - cfg.epsilon_start)

    def q_values(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s = np.asarray(state, dtype=np.float32).reshape(1, -1)
        qr, qc = self.online(s, training=False)
        return np.asarray(qr)[0], np.asarray(qc)[0]

    def composite_q(self, state: np.ndarray) -> np.ndarray:
        qr, qc = self.q_values(state)
        return qr - float(self.lam.numpy()) * qc

    def act(self, state: np.ndarray, mask: np.ndarray, greedy: bool = False) -> int:
        allowed = np.flatnonzero(mask)
        if allowed.size == 0:
            raise ValueError("safety shield left no legal action")
        if allowed.size == 1:
            return int(allowed[0])
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.choice(allowed))
        q = self.composite_q(state)
        q = np.where(mask, q, NEG_INF)
        return int(np.argmax(q))

    # ---------------------------------------------------------------- training
    @tf.function(reduce_retracing=True)
    def _train_step(
        self,
        states, actions, rewards, costs, next_states, dones, next_masks, n_steps, weights,
    ):
        cfg = self.cfg
        gamma_r = tf.pow(tf.constant(cfg.gamma, tf.float32), n_steps)
        gamma_c = tf.pow(tf.constant(cfg.cost_gamma, tf.float32), n_steps)

        # --- Double DQN action selection on the composite, under the mask ---
        qr_next_online, qc_next_online = self.online(next_states, training=False)
        composite_next = qr_next_online - self.lam * qc_next_online
        masked = tf.where(next_masks > 0.5, composite_next, tf.fill(tf.shape(composite_next), NEG_INF))
        best = tf.argmax(masked, axis=1, output_type=tf.int32)
        sel = tf.stack([tf.range(tf.shape(best)[0]), best], axis=1)

        qr_next_target, qc_next_target = self.target(next_states, training=False)
        not_done = 1.0 - dones
        y_r = tf.stop_gradient(rewards + gamma_r * not_done * tf.gather_nd(qr_next_target, sel))
        y_c = tf.stop_gradient(costs + gamma_c * not_done * tf.gather_nd(qc_next_target, sel))

        idx = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
        with tf.GradientTape() as tape:
            qr, qc = self.online(states, training=True)
            qr_a = tf.gather_nd(qr, idx)
            qc_a = tf.gather_nd(qc, idx)
            err_r = qr_a - y_r
            err_c = qc_a - y_c
            loss = tf.reduce_mean(weights * (_huber(err_r) + _huber(err_c)))

        grads = tape.gradient(loss, self.online.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, cfg.grad_clip_norm)
        self.optimizer.apply_gradients(zip(grads, self.online.trainable_variables))

        # Priority signal: error in the quantity the policy actually maximises.
        td = tf.abs(err_r - self.lam * err_c)
        return loss, td

    def learn(self, buffer) -> Optional[float]:
        cfg = self.cfg
        if len(buffer) < max(cfg.warmup_transitions, cfg.batch_size):
            return None
        batch = buffer.sample(cfg.batch_size, self.train_steps)
        (states, actions, rewards, costs, next_states, dones,
         next_masks, n_steps, weights, idx) = batch

        loss, td = self._train_step(
            tf.convert_to_tensor(states, tf.float32),
            tf.convert_to_tensor(actions, tf.int32),
            tf.convert_to_tensor(rewards, tf.float32),
            tf.convert_to_tensor(costs, tf.float32),
            tf.convert_to_tensor(next_states, tf.float32),
            tf.convert_to_tensor(dones, tf.float32),
            tf.convert_to_tensor(next_masks, tf.float32),
            tf.convert_to_tensor(n_steps, tf.float32),
            tf.convert_to_tensor(weights, tf.float32),
        )
        buffer.update_priorities(idx, np.asarray(td))
        self.train_steps += 1
        if self.train_steps % cfg.target_sync_steps == 0:
            self.target.set_weights(self.online.get_weights())
        return float(loss)

    # ------------------------------------------------------------ dual ascent
    def observe_cost(self, cost: float) -> None:
        """Track the running per-decision cost that the dual variable reacts to."""
        if np.isnan(self.cost_ema):
            self.cost_ema = float(cost)
        else:
            a = self.cfg.cost_ema
            self.cost_ema = a * self.cost_ema + (1.0 - a) * float(cost)

    def update_lambda(self) -> float:
        """One projected dual-ascent step on the *relative* constraint violation.

        The step uses ``(Jhat_c - d0) / d0`` rather than the raw difference. With
        the raw difference the effective learning rate scales with the budget, so
        a step size tuned at one budget crawls at a tighter one: halving ``d0``
        halves the typical violation and therefore halves the speed at which the
        multiplier can respond. Since the whole point of the formulation is that
        the operator gets to choose ``d0`` freely, and since the frontier is
        traced by varying it, the dual dynamics must not depend on its scale.
        """
        cfg = self.cfg
        if not cfg.constrained or np.isnan(self.cost_ema):
            return float(self.lam.numpy())
        if self.env_steps < cfg.lambda_warmup_steps:
            # Critics are still noise; ascending now would peg the multiplier.
            return float(self.lam.numpy())
        denom = max(cfg.cost_budget, 1e-6)
        violation = (self.cost_ema - cfg.cost_budget) / denom
        # Clip the relative violation so an early, wildly-infeasible policy
        # cannot slam the multiplier to its ceiling before the critics are usable.
        violation = float(np.clip(violation, -5.0, 5.0))
        new = float(self.lam.numpy()) + cfg.lambda_lr * violation
        new = float(np.clip(new, 0.0, cfg.lambda_max))
        self.lam.assign(new)
        self.lambda_history.append(new)
        return new

    # ------------------------------------------------------------------- i/o
    def save(self, path: Path, extra: Optional[Dict] = None) -> None:
        """Write weights plus the metadata needed to evaluate this agent correctly.

        ``extra`` records training-time environment settings - above all the
        sensing mode. An agent trained with emergency-vehicle sensing disabled
        has only ever seen zeroed belief features, so evaluating it with sensing
        switched on is a train/test mismatch that would misattribute the
        resulting damage to the ablation rather than to the mismatch.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.online.save_weights(str(path))
        meta = {
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "lambda": float(self.lam.numpy()),
            "cost_ema": None if np.isnan(self.cost_ema) else self.cost_ema,
            "train_steps": self.train_steps,
            "env_steps": self.env_steps,
            "cost_budget": self.cfg.cost_budget,
            "constrained": self.cfg.constrained,
        }
        meta.update(extra or {})
        path.with_suffix(".agent.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")

    def load(self, path: Path) -> "LagrangianDuelingDDQN":
        path = Path(path)
        self.online.load_weights(str(path))
        self.target.set_weights(self.online.get_weights())
        meta_path = path.with_suffix(".agent.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.lam.assign(float(meta.get("lambda", 0.0)))
            if meta.get("cost_ema") is not None:
                self.cost_ema = float(meta["cost_ema"])
        return self

    def summary(self) -> Dict[str, float]:
        return {
            "lambda": float(self.lam.numpy()),
            "cost_ema": self.cost_ema,
            "epsilon": self.epsilon,
            "train_steps": self.train_steps,
            "env_steps": self.env_steps,
        }
