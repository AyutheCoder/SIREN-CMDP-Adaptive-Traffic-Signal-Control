"""
SIREN-CMDP: Siren-Informed Risk-Constrained Emergency-Aware Adaptive Signal Control.

A research codebase for adaptive traffic signal control that (a) senses emergency
vehicles acoustically rather than assuming perfect V2I detection, and (b) treats
emergency-vehicle delay as a hard *constraint* in a Constrained MDP rather than as
another weighted reward term.
"""

__version__ = "1.0.0"

__all__ = ["config", "netgen", "demand", "env", "sensing", "agents", "control", "evaluation"]
