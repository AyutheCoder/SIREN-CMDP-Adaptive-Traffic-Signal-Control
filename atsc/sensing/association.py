"""Which approach is the siren actually on? Geometric association across mics.

The problem
-----------
Four microphones, one per approach, sit within a couple of hundred metres of
each other. A siren on the eastbound arm is plainly audible at the northbound
microphone -- quieter, because of distance and the cardioid's rear rejection,
but far above the noise floor. Each microphone's detector correctly answers the
question it was asked ("is a siren audible here?") with *yes*, and if each
approach then runs its own independent presence filter, all four approaches
convince themselves an emergency vehicle is inbound on them.

That is not a hypothetical. Measured against ground truth over an episode, the
independent per-approach filter reaches presence > 0.9 on 21 % of approach-epochs
where no emergency vehicle was inbound at all, and the live demo shows all four
approaches pinned at 1.00 while a single ambulance crosses. The posterior was
still *informative* -- AUC 0.93 against truth -- but it was not usable as a
probability, and a controller that acts on it pays for cross-talk with real
green time on empty arms.

The fix
-------
Cross-talk is not noise to be filtered; it is a *consistency constraint*, and it
is the reason to have four microphones rather than one. If a source really is at
range ``r`` up the eastbound arm, then its position is determined, and every
other microphone's range estimate is determined with it -- geometrically, from
known mast positions. So the readings are tested against each other:

* take the microphone reporting the nearest source; that is the best candidate
  host, since it is the one whose cardioid is pointed at the source;
* place a hypothetical source on its axis at its estimated range;
* predict what every other microphone should be reporting if that single source
  were the whole story;
* any microphone whose reading is *explained* by that prediction gets no
  independent credit, because its evidence has already been accounted for;
* any microphone whose reading is *not* explained -- a genuinely closer, second
  source -- keeps its evidence and becomes a host in its own right.

The output is one weight per approach in [0, 1]: the probability that this
approach carries a source of its own rather than an echo of somebody else's.
Evidence is then tempered by that weight before it reaches the presence filter
(``L**w``, so an unexplained reading is used in full and a fully explained one
is inert).

This keeps the case the naive filter gets wrong *and* the case a plain
winner-takes-all rule would get wrong: two genuine emergency vehicles on
different approaches are both retained, because neither hypothesis explains the
other's range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

RANGE_LOG_SIGMA = 0.45
"""Standard deviation of the detector's range error in log space, matching the
value the tracker uses for :attr:`SensorReading.range_var`. Residuals are scored
in the same units the range estimate is actually noisy in, so a 20 m
disagreement at 30 m is significant and the same disagreement at 250 m is not."""

EVIDENCE_FLOOR = 0.25
"""Detector posterior below which a reading carries no evidence worth
associating, so the hypothesis set stays confined to microphones that actually
heard something.

Deliberately below the point where a reading starts to *raise* the presence
posterior. At 0.5 the test missed nearly half the cross-talk it should have
caught, because the case that matters is precisely the one where the on-axis
microphone had a marginal frame and an off-axis one did not: the true host was
then not even in the candidate set, so there was no hypothesis to explain the
leak away."""


@dataclass
class Association:
    """Per-approach association weights and the reasoning behind them."""

    weights: Dict[str, float]
    host: Optional[str]
    explained_by: Dict[str, Optional[str]]

    def describe(self) -> str:
        parts = []
        for a, w in sorted(self.weights.items()):
            src = self.explained_by.get(a)
            parts.append(f"{a}:{w:.2f}" + (f"<-{src}" if src else ""))
        return " ".join(parts)


def _source_position(mic: Sequence[float], axis: Sequence[float],
                     range_m: float) -> Tuple[float, float]:
    """Where a source at ``range_m`` on this microphone's axis would be."""
    return (float(mic[0]) + float(axis[0]) * range_m,
            float(mic[1]) + float(axis[1]) * range_m)


def associate(
    readings: Mapping[str, object],
    mic_pos: Mapping[str, Sequence[float]],
    mic_axis: Mapping[str, Sequence[float]],
    log_sigma: float = RANGE_LOG_SIGMA,
    evidence_floor: float = EVIDENCE_FLOOR,
) -> Association:
    """Decide which approaches carry a source of their own.

    ``readings`` maps approach -> object with ``p_siren`` and ``distance_m``
    attributes (a :class:`~atsc.sensing.sensor.SensorReading`). Approaches with
    no usable range estimate are left at weight 1: there is nothing to test
    them against, and silently suppressing them would lose real detections
    whenever the range head fails.
    """
    weights: Dict[str, float] = {a: 1.0 for a in readings}
    explained: Dict[str, Optional[str]] = {a: None for a in readings}

    candidates = []
    for approach, r in readings.items():
        p = float(getattr(r, "p_siren", 0.0) or 0.0)
        d = getattr(r, "distance_m", None)
        if p < evidence_floor or d is None:
            continue
        d = float(d)
        if not math.isfinite(d) or d <= 0.0:
            continue
        if approach not in mic_pos or approach not in mic_axis:
            continue
        candidates.append((d, approach))

    if len(candidates) < 2:
        return Association(weights, candidates[0][1] if candidates else None, explained)

    # Nearest first: the microphone reporting the shortest range is the one
    # most likely to be pointed at the source rather than hearing it off-axis.
    candidates.sort()
    hosts: list = []
    host_positions: Dict[str, Tuple[float, float]] = {}

    for d, approach in candidates:
        best_expl, best_src = 0.0, None
        for h in hosts:
            hx, hy = host_positions[h]
            mx, my = mic_pos[approach]
            predicted = math.hypot(hx - float(mx), hy - float(my))
            if predicted <= 0.0:
                continue
            # How well does the already-accepted source explain this reading?
            z = (math.log(d) - math.log(predicted)) / max(log_sigma, 1e-6)
            expl = math.exp(-0.5 * z * z)
            if expl > best_expl:
                best_expl, best_src = expl, h
        weights[approach] = float(np.clip(1.0 - best_expl, 0.0, 1.0))
        if best_expl > 0.5:
            explained[approach] = best_src
        else:
            hosts.append(approach)
            host_positions[approach] = _source_position(
                mic_pos[approach], mic_axis[approach], d)

    return Association(weights, hosts[0] if hosts else None, explained)


def temper(p_siren: float, weight: float, prior_odds_correction: float = 1.0,
           eps: float = 1e-6) -> float:
    """Damp a detector posterior toward uninformative by ``weight``.

    Works on the likelihood ratio rather than the probability, because the
    likelihood ratio is what the presence filter multiplies: ``L**w`` is the
    evidence of ``w`` independent looks, so ``w = 1`` passes the reading
    through unchanged and ``w = 0`` makes it exactly uninformative (``L = 1``),
    with a smooth interpolation between. Returns a probability again so callers
    need not know how the filter consumes it.
    """
    p = float(np.clip(p_siren, eps, 1.0 - eps))
    w = float(np.clip(weight, 0.0, 1.0))
    if w >= 1.0:
        return p
    lr = (p / (1.0 - p)) * prior_odds_correction
    lr_w = lr ** w
    # Undo the prior correction so the caller's own correction still applies.
    lr_out = lr_w / max(prior_odds_correction, eps)
    return float(np.clip(lr_out / (1.0 + lr_out), eps, 1.0 - eps))
