"""Belief-driven emergency-vehicle priority, applied above the learned policy.

Why this exists
---------------
The obvious design is to let the RL agent learn prioritisation from the CMDP
cost. We built that first and measured it, and it does not work: with a trained
constrained agent, zeroing the emergency-belief features changes its chosen
action 34 % of the time and shifts Q by 7.5 on average -- so the network is
plainly *using* the belief -- yet when an emergency vehicle is committed the
agent serves that vehicle's approach exactly 50 % of the time, which is chance
for a four-phase signal where each approach is served by two phases.

The belief is being consumed as ordinary state, not as a priority signal. The
reason is signal starvation: emergency events occupy a low single-digit
percentage of transitions, the cost is zero in most epochs, and ``lambda * Q_c``
stays small against ``Q_r``. Meanwhile a plain rule driven by the *same* beliefs
reaches 12.1 s emergency stopped delay against the learned agent's 16.5 s. The
information is available and actionable; value learning simply does not extract
it at this event rate.

So priority is made explicit here, and the learned policy keeps the job it is
actually good at -- throughput and delay. The division is:

* **the belief** decides *whether* an emergency vehicle warrants priority,
* **this overlay** guarantees the approach is served,
* **the learned policy** picks *which* serving phase (through vs. protected
  left) and drives everything else,
* **the safety shield** still has the final word, so none of the timing or
  starvation guarantees are weakened.

The dual variable from the CMDP is not discarded: it sets how aggressively the
overlay engages (:meth:`aggressiveness_from_lambda`), so the operator's budget
still controls the trade-off -- it just acts through a mechanism that reliably
fires instead of through a critic that cannot learn the event.

Guarding against the failure modes we actually hit
--------------------------------------------------
An unguarded overlay gridlocks. An earlier oracle sensor reported every
emergency vehicle at every microphone, the overlay seized all four phases at
once, and the intersection deadlocked (139 s mean delay, 10 teleports). The
guards below are each a response to something observed:

* only the single highest-belief approach is ever preempted;
* a vehicle must be *inbound* with an arrival estimate inside the lead window,
  so a departing or stationary one cannot hold green;
* ``max_hold_s`` bounds a single preemption;
* ``cooldown_s`` prevents re-triggering on the same vehicle as it clears;
* ``duty_cycle_cap`` bounds the long-run fraction of epochs spent preempting,
  which is what stops a burst of vehicles from starving cross traffic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class PriorityDecision:
    """What the overlay did at one decision epoch, and why."""

    engaged: bool
    approach: Optional[str] = None
    phase: Optional[int] = None
    presence: float = 0.0
    eta_s: float = float("inf")
    priority: float = 0.0
    reason: str = ""


@dataclass
class PriorityOverlay:
    """Grants green to an approach the belief says an emergency vehicle is on."""

    commit_threshold: float = 0.50
    """Belief above which an approach is treated as emergency-bearing. Below the
    belief tracker's own commit level so the overlay engages a beat earlier than
    a reader of the belief alone would."""

    max_range_m: float = 250.0
    """Only preempt for vehicles the tracker places within this range.

    Range rather than arrival time, deliberately. Estimated ETA is clamped at
    the belief's ceiling in about half of all committed epochs, because the
    range-rate estimate loses lock whenever closing speed drops -- so gating on
    ETA silently rejected half the *genuine* detections while still admitting
    stale ones. Tracked range is far better behaved: median 80 m with only 7 %
    of committed epochs beyond 200 m.
    """

    min_closing_ms: float = 2.0
    """Required closing speed, and the single most important parameter here.

    Presence decays slowly by design, so after a vehicle clears the junction the
    belief stays high for several epochs; without a closing test the overlay
    holds green for a vehicle that has already gone. Measured directly: dropping
    this from 2.0 to 0.5 m/s, changing nothing else, moves emergency stopped
    delay from 14.7 s to 18.5 s -- worse than running no overlay at all."""

    lead_window_s: float = 28.0
    """Upper bound on arrival time, applied only when the ETA estimate is
    actually informative (not sitting at its ceiling)."""

    max_hold_s: float = 45.0
    """Longest a single preemption may last."""

    cooldown_s: float = 2.0
    """Quiet period after releasing. Kept short deliberately: a long cooldown
    blocks the overlay while emergency vehicles are still arriving, and a
    12 s cooldown combined with a tight duty-cycle cap produced catastrophic
    runs (61.8 s mean delay, 5 teleports) rather than the conservative
    behaviour it was intended to give. The closing-speed gate, not a cooldown,
    is what stops re-triggering on a vehicle that has already passed."""

    duty_cycle_cap: float = 0.70
    """Long-run ceiling on the fraction of decision epochs spent preempting.
    A backstop against pathological demand, not a routine limiter -- set low it
    starves the overlay exactly when emergency traffic is heaviest."""

    min_priority: float = 0.0
    """Optional triage floor; raise it to ignore lower-priority classes."""

    # --- runtime state ---
    _held_s: float = field(default=0.0, init=False)
    _cooldown_s: float = field(default=0.0, init=False)
    _target_approach: Optional[str] = field(default=None, init=False)
    n_engagements: int = field(default=0, init=False)
    n_epochs: int = field(default=0, init=False)
    n_preempt_epochs: int = field(default=0, init=False)

    def reset(self) -> None:
        self._held_s = 0.0
        self._cooldown_s = 0.0
        self._target_approach = None
        self.n_engagements = 0
        self.n_epochs = 0
        self.n_preempt_epochs = 0

    # ------------------------------------------------------------------ policy
    @staticmethod
    def aggressiveness_from_lambda(lam: float) -> Dict[str, float]:
        """Map the CMDP dual variable onto overlay settings.

        A larger multiplier means the emergency constraint is binding harder, so
        the overlay commits on weaker evidence and looks further ahead. This is
        how the operator's budget keeps controlling the trade-off once
        prioritisation moves out of the critic.

        The output range is confined to settings measured to be stable. An
        earlier mapping reached commit 0.67 / range 170 m / duty-cycle 0.34 at
        moderate lambda, and that configuration gridlocked the intersection
        (61.8 s mean delay, 5 teleports over 6 seeds). Restrictive settings are
        not the safe end of this dial: refusing to engage leaves the base policy
        fighting a queue the overlay has already begun to build.
        """
        lam = float(np.clip(lam, 0.0, 30.0))
        t = lam / 30.0
        return {
            "commit_threshold": float(0.58 - 0.13 * t),   # 0.58 -> 0.45
            "max_range_m": float(210.0 + 80.0 * t),       # 210 m -> 290 m
            "duty_cycle_cap": float(0.60 + 0.20 * t),     # 0.60 -> 0.80
        }

    def _duty_cycle(self) -> float:
        return self.n_preempt_epochs / max(self.n_epochs, 1)

    def select(
        self,
        env,
        mask: np.ndarray,
        base_action: int,
        rank_phases=None,
    ) -> PriorityDecision:
        """Decide whether to override ``base_action`` for an emergency vehicle.

        ``rank_phases(candidates) -> ordered list`` lets the caller choose among
        the phases that serve the target approach; the learned policy's own
        value estimate is the natural ranker, so the agent still picks between a
        through movement and a protected left.
        """
        dt = env.cfg.control.decision_interval_s
        self.n_epochs += 1

        if self._cooldown_s > 0.0:
            self._cooldown_s = max(0.0, self._cooldown_s - dt)

        eta_ceiling = env.cfg.belief.max_eta_s - 1e-3
        beliefs = env.belief.beliefs()
        best_approach, best = None, None
        for approach, b in beliefs.items():
            if b.presence < self.commit_threshold:
                continue
            if b.expected_priority < self.min_priority:
                continue
            # Must be within range and genuinely closing. Either test alone is
            # insufficient: range alone keeps green for a vehicle that has just
            # passed, closing alone preempts for one still far away.
            if not np.isfinite(b.range_m) or b.range_m > self.max_range_m:
                continue
            if not np.isfinite(b.range_rate_ms) or b.range_rate_ms > -self.min_closing_ms:
                continue
            # Apply the arrival-time window only when the estimate is real; at
            # the ceiling it carries no information and must not veto.
            if np.isfinite(b.eta_s) and b.eta_s < eta_ceiling and b.eta_s > self.lead_window_s:
                continue
            score = b.presence * b.expected_priority
            if best is None or score > best.presence * best.expected_priority:
                best_approach, best = approach, b

        # --- release conditions ---
        if best_approach is None:
            if self._target_approach is not None:
                self._target_approach = None
                self._held_s = 0.0
                self._cooldown_s = self.cooldown_s
            return PriorityDecision(False, reason="no committed emergency vehicle")

        if self._cooldown_s > 0.0 and self._target_approach != best_approach:
            return PriorityDecision(False, approach=best_approach,
                                    presence=best.presence, eta_s=best.eta_s,
                                    reason="cooldown")

        if self._target_approach == best_approach and self._held_s >= self.max_hold_s:
            self._target_approach = None
            self._held_s = 0.0
            self._cooldown_s = self.cooldown_s
            return PriorityDecision(False, approach=best_approach,
                                    presence=best.presence, eta_s=best.eta_s,
                                    reason="max hold reached")

        if self._duty_cycle() > self.duty_cycle_cap and self._target_approach is None:
            return PriorityDecision(False, approach=best_approach,
                                    presence=best.presence, eta_s=best.eta_s,
                                    reason="duty-cycle cap")

        # --- engage ---
        candidates = [p for p in env.phases_serving_approach(best_approach,
                                                             primary_only=True)
                      if mask[p]]
        if not candidates:
            # The shield forbids every serving phase right now (usually minimum
            # green elsewhere). Keep the target alive and retry next epoch
            # rather than dropping it mid-approach.
            self._held_s += dt
            return PriorityDecision(False, approach=best_approach,
                                    presence=best.presence, eta_s=best.eta_s,
                                    reason="serving phase not yet legal")

        if rank_phases is not None:
            ordered = [p for p in rank_phases(candidates) if p in candidates]
            phase = ordered[0] if ordered else candidates[0]
        else:
            queues = env.phase_queues()
            phase = max(candidates, key=lambda p: queues[p])

        if self._target_approach != best_approach:
            self.n_engagements += 1
            self._held_s = 0.0
        self._target_approach = best_approach
        self._held_s += dt
        self.n_preempt_epochs += 1

        return PriorityDecision(
            True, approach=best_approach, phase=int(phase),
            presence=best.presence, eta_s=best.eta_s,
            priority=best.expected_priority,
            reason="emergency priority",
        )

    def stats(self) -> Dict[str, float]:
        return {
            "priority_engagements": float(self.n_engagements),
            "priority_duty_cycle": float(self._duty_cycle()),
            "priority_epochs": float(self.n_preempt_epochs),
        }
