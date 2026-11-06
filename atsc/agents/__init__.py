"""Learned and hand-designed controllers."""

from .replay import PrioritizedReplay, Transition
from .dueling_ddqn import LagrangianDuelingDDQN

__all__ = ["PrioritizedReplay", "Transition", "LagrangianDuelingDDQN"]
