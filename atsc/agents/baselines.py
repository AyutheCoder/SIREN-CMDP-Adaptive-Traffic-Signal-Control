"""Baseline signal controllers.

The comparison set is deliberately strong. Beating a badly-timed fixed plan
proves nothing, so this includes Webster-optimal fixed timing computed from the
*known* generated demand, SUMO's own vehicle-actuated logic, and max-pressure,
which is provably throughput-optimal in the store-and-forward model and is the
standard modern yardstick.

The most important comparison is :class:`HardPreemption`. Real deployed
emergency-vehicle preemption (Opticom-style) works exactly this way: on
detection, jump to the phase serving the vehicle and hold. It is very good at
the thing it optimises and pathological about everything else - the queues it
builds on the cross street outlast the emergency vehicle by minutes. Showing
that a constrained policy attains comparable emergency delay at a fraction of
that collateral cost is the empirical claim of this work.

Every controller here goes through the same safety shield as the learned policy,
so all reported numbers respect identical minimum-green, changeover and
starvation guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..config import DemandConfig

SATURATION_FLOW_VPHPL = 1800.0
"""Ideal saturation flow per lane, veh/h. Standard HCM planning value."""


class BaseController:
    """Common interface: propose an action; the shield decides what is legal."""

    name: str = "base"
    passive: bool = False

    def interrupt(self, env):
        """Optional 1 Hz override, polled by the environment inside a green.

        Returning ``None`` -- the default for every controller -- means the
        green runs to the end of the decision epoch, which is the behaviour
        every baseline here has. Only the corridor scheduler uses this, and
        only the shield can grant what it asks for.
        """
        return None
    """If True the runner steps SUMO without setting the signal state, because
    SUMO's own program is driving (used for the native actuated baseline)."""
    program_id: Optional[str] = None

    def reset(self, env) -> None:  # noqa: D401
        pass

    def act(self, env, obs: np.ndarray, mask: np.ndarray) -> int:
        raise NotImplementedError

    @staticmethod
    def _first_allowed(mask: np.ndarray, order: Sequence[int]) -> int:
        for a in order:
            if mask[a]:
                return int(a)
        return int(np.flatnonzero(mask)[0])


# ----------------------------------------------------------------------------
# Fixed time
# ----------------------------------------------------------------------------
def webster_green_splits(
    phase_movement_flows: Sequence[float],
    lanes_per_phase: Sequence[int],
    lost_time_per_phase_s: float,
    saturation_flow: float = SATURATION_FLOW_VPHPL,
) -> Tuple[float, List[float]]:
    """Webster's optimal cycle length and green splits.

    ``C = (1.5 L + 5) / (1 - Y)`` with ``L`` total lost time and ``Y`` the sum of
    critical flow ratios; effective green is apportioned in proportion to those
    ratios. If demand exceeds capacity (``Y >= 1``) the formula is undefined, so
    the cycle is clamped to 180 s and splits fall back to the flow proportions.
    """
    n = len(phase_movement_flows)
    ratios = [
        float(flow) / max(saturation_flow * max(lanes, 1), 1e-9)
        for flow, lanes in zip(phase_movement_flows, lanes_per_phase)
    ]
    total_lost = lost_time_per_phase_s * n
    Y = sum(ratios)
    if Y >= 0.95:
        cycle = 180.0
    else:
        cycle = (1.5 * total_lost + 5.0) / (1.0 - Y)
        cycle = float(np.clip(cycle, 40.0, 180.0))
    effective_green = max(cycle - total_lost, n * 5.0)
    denom = max(Y, 1e-9)
    greens = [effective_green * r / denom for r in ratios]
    greens = [float(max(g, 6.0)) for g in greens]
    return cycle, greens


@dataclass
class FixedTime(BaseController):
    """Cycles the phases in program order with fixed green durations."""

    green_durations: List[float]
    name: str = "fixed_time"

    _elapsed: float = field(default=0.0, init=False)
    _index: int = field(default=0, init=False)

    def reset(self, env) -> None:
        self._elapsed = 0.0
        self._index = 0

    def act(self, env, obs: np.ndarray, mask: np.ndarray) -> int:
        dt = env.cfg.control.decision_interval_s
        self._elapsed += dt
        if self._elapsed >= self.green_durations[self._index % len(self.green_durations)]:
            self._elapsed = 0.0
            self._index += 1
        target = self._index % env.n_actions
        if mask[target]:
            return int(target)
        # Shield says no (usually minimum green); hold and retry next epoch.
        self._elapsed = 0.0
        return self._first_allowed(mask, [env.current_phase, target])


def fixed_time_from_demand(
    env, demand: DemandConfig, lost_time_per_phase_s: Optional[float] = None
) -> FixedTime:
    """Build a Webster-timed plan from the demand the generator will actually emit.

    This gives fixed timing every advantage: it is optimised against the true
    arrival rates rather than a guess, so any gap to the adaptive controllers is
    attributable to adaptivity and not to a mis-timed plan.
    """
    ctrl = env.cfg.control
    if lost_time_per_phase_s is None:
        lost_time_per_phase_s = ctrl.yellow_s + ctrl.all_red_s

    p_left, p_through, p_right = demand.turn_ratios
    flows: List[float] = []
    lanes: List[int] = []
    for phase in env.phases:
        critical_flow = 0.0
        critical_lanes = 1
        for approach in phase.approaches:
            approach_flow = demand.base_flow_vph
            green_lanes = [ln for ln in phase.green_lanes if env.lane_to_approach[ln] == approach]
            n_green = len(green_lanes)
            total_lanes = sum(1 for ln in env.controlled_lanes
                              if env.lane_to_approach[ln] == approach)
            # A phase that greens the leftmost lane only is serving left turns.
            is_left_only = n_green == 1 and total_lanes > 1
            share = p_left if is_left_only else (p_through + p_right)
            movement_flow = approach_flow * share
            if movement_flow > critical_flow:
                critical_flow = movement_flow
                critical_lanes = max(n_green, 1)
        flows.append(critical_flow)
        lanes.append(critical_lanes)

    cycle, greens = webster_green_splits(flows, lanes, lost_time_per_phase_s)
    greens = [float(np.clip(g, ctrl.min_green_s, ctrl.max_green_s)) for g in greens]
    plan = FixedTime(green_durations=greens)
    plan.cycle_length = cycle  # type: ignore[attr-defined]
    return plan


# ----------------------------------------------------------------------------
# Actuated (SUMO native)
# ----------------------------------------------------------------------------
@dataclass
class SumoActuated(BaseController):
    """Hands control to SUMO's own vehicle-actuated program."""

    name: str = "actuated"
    passive: bool = True
    program_id: str = "actuated"

    def act(self, env, obs: np.ndarray, mask: np.ndarray) -> int:
        return int(env.current_phase)


# ----------------------------------------------------------------------------
# Max pressure
# ----------------------------------------------------------------------------
@dataclass
class MaxPressure(BaseController):
    """Serve the phase with the greatest upstream-minus-downstream pressure."""

    name: str = "max_pressure"

    def act(self, env, obs: np.ndarray, mask: np.ndarray) -> int:
        pressure = env.phase_pressure()
        scores = np.where(mask, pressure, -np.inf)
        return int(np.argmax(scores))


@dataclass
class GreedyQueue(BaseController):
    """Serve the phase with the longest standing queue.

    A classic two-phase heuristic, generalised from two
    phases to the discovered phase set. It is included because it is the
    obvious thing to do and it is worth showing what it costs.
    """

    name: str = "greedy_queue"

    def act(self, env, obs: np.ndarray, mask: np.ndarray) -> int:
        queues = env.phase_queues()
        scores = np.where(mask, queues, -np.inf)
        return int(np.argmax(scores))


# ----------------------------------------------------------------------------
# Hard preemption overlay
# ----------------------------------------------------------------------------
@dataclass
class HardPreemption(BaseController):
    """Classic emergency-vehicle preemption layered over any base controller.

    On a detection above ``commit_threshold`` it seizes the phase serving that
    approach and holds it until the vehicle clears, exactly as deployed
    preemption systems do. Where two phases serve the approach, the one with the
    larger queue is taken, since the turn intent is not observable acoustically.

    ``hold_after_clear_s`` models the dwell most controllers apply before
    returning to normal operation.
    """

    base: BaseController
    commit_threshold: float = 0.6
    hold_after_clear_s: float = 4.0
    name: str = "hard_preemption"

    _hold_remaining: float = field(default=0.0, init=False)
    _target: Optional[int] = field(default=None, init=False)
    n_preemptions: int = field(default=0, init=False)

    def reset(self, env) -> None:
        self.base.reset(env)
        self._hold_remaining = 0.0
        self._target = None
        self.n_preemptions = 0
        env.clear_ev_request()

    def act(self, env, obs: np.ndarray, mask: np.ndarray) -> int:
        dt = env.cfg.control.decision_interval_s
        beliefs = env.belief.beliefs()

        best_approach, best_score = None, self.commit_threshold
        for approach, b in beliefs.items():
            if b.presence >= best_score:
                best_score = b.presence
                best_approach = approach

        if best_approach is not None:
            # Primary phases only: a protected-left phase greens one lane in
            # three and does not serve a vehicle whose movement is unknown.
            # Without this the "max queue among serving phases" rule below
            # actively prefers it whenever the left lane is the busy one.
            candidates = env.phases_serving_approach(best_approach,
                                                     primary_only=True)
            if candidates:
                queues = env.phase_queues()
                target = max(candidates, key=lambda p: queues[p])
                if self._target != target:
                    self.n_preemptions += 1
                self._target = target
                self._hold_remaining = self.hold_after_clear_s
                # Deployed preemption holds the phase until the vehicle clears,
                # so the shield's gap-out rule must not release it the moment
                # the queue empties. Declaring the call gives this baseline the
                # same protection the corridor gets; without it the comparison
                # would flatter us for a reason that has nothing to do with the
                # method.
                env.emergency_call = True
                # The same standing request the corridor raises, so S6's
                # deadline applies to this baseline too. Without it the shield
                # would be doing strictly more for one controller than for the
                # other, and the comparison would measure the shield rather
                # than the method.
                env.set_ev_request(candidates)
                if mask[target]:
                    return int(target)
                return self._first_allowed(mask, [env.current_phase, target])

        if self._hold_remaining > 0.0 and self._target is not None:
            self._hold_remaining -= dt
            env.emergency_call = True
            if mask[self._target]:
                return int(self._target)
            # The target is momentarily illegal (usually another phase is still
            # inside its minimum green). Keep the dwell alive and retry next
            # epoch rather than abandoning it -- dropping the target here ends
            # the preemption early and hands control back to the base
            # controller while the emergency vehicle is still crossing.
            return self._first_allowed(mask, [env.current_phase, self._target])

        self._target = None
        env.emergency_call = False
        env.set_ev_request(())
        return self.base.act(env, obs, mask)


# ----------------------------------------------------------------------------
# Learned-policy adapter
# ----------------------------------------------------------------------------
@dataclass
class LearnedPolicy(BaseController):
    """Wraps a trained agent so it can be evaluated through the same runner."""

    agent: object
    name: str = "siren_cmdp"
    greedy: bool = True

    def act(self, env, obs: np.ndarray, mask: np.ndarray) -> int:
        return int(self.agent.act(obs, mask, greedy=self.greedy))


@dataclass
class PriorityAwarePolicy(BaseController):
    """Learned policy for efficiency, explicit belief-driven overlay for priority.

    The learned agent runs the intersection; when the acoustic belief commits to
    an inbound emergency vehicle the overlay takes the approach, and the agent
    still chooses *which* of that approach's phases to run via its own value
    estimate. See ``atsc/control/priority.py`` for why prioritisation is explicit
    rather than learned -- in short, the critic demonstrably fails to extract it
    at realistic emergency-vehicle rates, while the same beliefs drive a rule
    perfectly well.

    If ``adaptive`` is set, the overlay's aggressiveness is derived from the
    agent's converged dual variable, so the CMDP budget still governs the
    trade-off.
    """

    agent: object
    overlay: object = None
    name: str = "siren_priority"
    greedy: bool = True
    adaptive: bool = True

    n_preemptions: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.overlay is None:
            from ..control.priority import PriorityOverlay
            self.overlay = PriorityOverlay()
        if self.adaptive and hasattr(self.agent, "lam"):
            for key, value in self.overlay.aggressiveness_from_lambda(
                    float(self.agent.lam.numpy())).items():
                setattr(self.overlay, key, value)

    def reset(self, env) -> None:
        self.overlay.reset()
        self.n_preemptions = 0

    def act(self, env, obs: np.ndarray, mask: np.ndarray) -> int:
        base = int(self.agent.act(obs, mask, greedy=self.greedy))

        def rank(candidates):
            q = self.agent.composite_q(obs)
            return sorted(candidates, key=lambda p: -q[p])

        decision = self.overlay.select(env, mask, base, rank_phases=rank)
        env.emergency_call = bool(decision.engaged)
        if decision.engaged and decision.phase is not None:
            self.n_preemptions = self.overlay.n_engagements
            return int(decision.phase)
        return base


@dataclass
class CorridorPolicy(BaseController):
    """Learned policy for efficiency, risk-calibrated corridors for priority.

    This is the deployable configuration. The learned agent runs the
    intersection; :class:`~atsc.control.corridor.CorridorScheduler` decides,
    from the posterior and the current queue state, whether opening a corridor
    buys more prioritised delay than it costs ordinary traffic, and in what
    order to serve several emergency vehicles that arrive at once. The agent
    still chooses among phases the schedule ranks equally, so it keeps the
    through/protected-left decision it is good at, and the safety shield still
    has the final word on every action.

    Unlike :class:`PriorityAwarePolicy`, which it replaces, there is no tuned
    commitment threshold: the threshold is derived per epoch from the delay each
    option actually buys and costs. See ``atsc/control/corridor.py``.
    """

    agent: object
    scheduler: object = None
    name: str = "siren_arc"
    greedy: bool = True
    adaptive: bool = False
    """Take the willingness-to-preempt from the agent's converged dual variable,
    so the CMDP budget the policy was trained under also governs the overlay.

    Appealing, and measured to be wrong. Both quantities do express willingness
    to pay for emergency delay, but they are not in the same units: gamma is an
    exchange rate in vehicle-seconds, while lambda is whatever scale makes
    ``Q_r - lambda Q_c`` balance for that particular pair of critics. On a
    constrained agent lambda converges near 8.6, which multiplies the corridor's
    exchange rate by nearly ten and pushes it well past the point where
    preemption pays -- emergency stopped delay got *worse*, not better. The
    operator's knob is gamma; lambda governs the policy. Set False to decouple
    them, which is what the results recommend."""

    calibrate: bool = True
    """Load the fitted presence calibration if one exists. Set False for the
    ablation that measures what the raw, uncalibrated posterior costs."""

    calibration_path: object = None
    """Which calibration map to load. Each presence filter has its own: the map
    is a monotone correction fitted to one filter's score distribution, and
    applying it to another filter's scores produces a number that is neither the
    raw score nor a probability. Defaults to the joint filter's map."""

    dwell_s: float = 4.0
    """How long the call is held after the last committed request clears.

    A request clears when the vehicle reaches the stop line, at which point it
    still has the junction to cross. Without a dwell the shield can take the
    green back during that crossing."""

    n_preemptions: int = field(default=0, init=False)
    last_decision: object = field(default=None, init=False)
    _dwell: float = field(default=0.0, init=False)
    _obs: object = field(default=None, init=False)
    _base_action: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.scheduler is None:
            from ..control.corridor import CorridorScheduler
            self.scheduler = CorridorScheduler()
        if self.scheduler.calibrator is None and self.calibrate:
            from ..config import MODELS_DIR
            from ..sensing.calibration import PresenceCalibrator
            path = Path(self.calibration_path) if self.calibration_path                 else MODELS_DIR / "presence_calibration.json"
            if path.exists():
                self.scheduler.calibrator = PresenceCalibrator.load(path)
        if self.adaptive and hasattr(self.agent, "lam"):
            self.scheduler.set_lambda(float(self.agent.lam.numpy()))

    def reset(self, env) -> None:
        self.scheduler.reset(env)
        self.n_preemptions = 0
        self.last_decision = None
        self._dwell = 0.0
        self._obs = None
        self._base_action = 0
        env.clear_ev_request()

    def act(self, env, obs: np.ndarray, mask: np.ndarray) -> int:
        self._obs = obs
        base = int(self.agent.act(obs, mask, greedy=self.greedy))
        self._base_action = base
        decision = self.scheduler.decide(env, mask, base,
                                         rank_phases=self._ranker(obs))
        self.last_decision = decision
        # Tell the environment a call is active, so the shield does not release
        # the corridor's own green under its gap-out rule the instant the queue
        # the corridor was clearing is gone. The call is held for a short dwell
        # after the request clears, because the vehicle is at the stop line at
        # that moment and still has the junction to cross -- the same dwell
        # deployed preemption controllers apply.
        # The call suspends gap-out, so it is raised only while the vehicle is
        # close enough that releasing the phase would strand it -- not for the
        # whole time a corridor is open. See CorridorDecision.imminent.
        near = bool(decision.engaged and decision.imminent)
        self._dwell = (self.dwell_s if near else
                       max(0.0, self._dwell - env.cfg.control.decision_interval_s))
        env.emergency_call = bool(near or self._dwell > 0.0)
        # Hand the shield the evidence-based service set, so a vehicle the risk
        # rule declines is still guaranteed a green within S6's deadline.
        env.set_ev_request(decision.service_phases)
        if decision.engaged and decision.phase is not None:
            self.n_preemptions = self.scheduler.n_engagements
            return int(decision.phase)
        return base

    def _ranker(self, obs):
        def rank(candidates):
            q = self.agent.composite_q(obs)
            return sorted(candidates, key=lambda p: -q[p])
        return rank

    def interrupt(self, env):
        """Re-run the corridor decision inside a green, once per second.

        The learned policy is not re-queried here -- its action is a plan for
        the epoch and re-evaluating it at 1 Hz would be both expensive and a
        different controller from the one that was trained. What is re-run is
        the risk calculation, whose inputs (posterior, range, queue) change
        every second and whose whole value is in acting the moment the sums
        turn favourable rather than up to an epoch later.
        """
        if self._obs is None:
            return None
        mask = env.action_mask(emergency=True)
        decision = self.scheduler.decide(env, mask, self._base_action,
                                         rank_phases=self._ranker(self._obs),
                                         poll_dt=float(env.cfg.step_length_s))
        near = bool(decision.engaged and decision.imminent)
        self._dwell = (self.dwell_s if near else
                       max(0.0, self._dwell - float(env.cfg.step_length_s)))
        env.emergency_call = bool(near or self._dwell > 0.0)
        env.set_ev_request(decision.service_phases)
        if decision.engaged and decision.phase is not None:
            self.last_decision = decision
            self.n_preemptions = self.scheduler.n_engagements
            return int(decision.phase)
        return None


def build_baseline(name: str, env, demand: DemandConfig) -> BaseController:
    """Factory used by the evaluation runner."""
    if name == "fixed_time":
        return fixed_time_from_demand(env, demand)
    if name == "actuated":
        return SumoActuated()
    if name == "max_pressure":
        return MaxPressure()
    if name == "greedy_queue":
        return GreedyQueue()
    if name == "max_pressure_preempt":
        return HardPreemption(base=MaxPressure(), name="max_pressure_preempt")
    if name == "fixed_time_preempt":
        return HardPreemption(base=fixed_time_from_demand(env, demand), name="fixed_time_preempt")
    raise ValueError(f"unknown baseline {name!r}")


BASELINE_NAMES = (
    "fixed_time",
    "actuated",
    "greedy_queue",
    "max_pressure",
    "max_pressure_preempt",
)
