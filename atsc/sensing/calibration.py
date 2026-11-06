"""Turning the presence posterior into an actual probability.

Why this is needed
------------------
The corridor scheduler is a Bayes decision rule: it preempts when
``p * gamma * B > C``. That arithmetic is only meaningful if ``p`` is a
probability. The recursive filter's presence output is not one.

Measured against ground truth over an hour of traffic, the filter *ranks*
beautifully -- AUC 0.95 -- and is badly *calibrated*: at the 0.9 level it fires
on 80 % of approach-epochs that genuinely carry an inbound emergency vehicle and
on 17 % of those that do not. With a true occupancy of 5.3 %, the probability
that an approach reading 0.9 actually has an emergency vehicle on it is

    0.797 * 0.053 / (0.797 * 0.053 + 0.167 * 0.947) = 0.21,

not 0.9. A decision rule handed 0.9 in place of 0.21 over-values preemption by
more than four times, and we measured the consequence: with the raw posterior
the corridor made emergency stopped delay *worse* than the learned policy alone
(15.4 s against 14.5 s), while the same scheduler on oracle sensing improved it
(13.9 s). The scheduler was not wrong; its input was.

Ranking well and being calibrated are different properties, and the second is
the one a decision rule needs. This module supplies the missing monotone map
from filter output to probability, fitted by isotonic regression on episodes
held out from every reported result.

Why isotonic rather than a sigmoid
----------------------------------
Platt scaling assumes the miscalibration has a particular two-parameter shape.
Here it does not: the filter is roughly right at low presence, badly
over-confident in the 0.6-0.95 band where the HMM has integrated a few
cross-talk frames, and roughly right again at 1.0 where the evidence really is
overwhelming. That is not a logistic distortion. Isotonic regression assumes
only monotonicity -- more filter output should never mean less probability --
which is exactly the property we are willing to assert and nothing more.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


def pool_adjacent_violators(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Least-squares isotonic fit: the classic PAVA, in a dozen lines.

    ``y`` are bin means and ``w`` bin counts, both ordered by increasing score.
    Returns the monotone non-decreasing sequence minimising the weighted squared
    error, which is the maximum-likelihood calibration map under the only
    assumption we are making.
    """
    y = np.asarray(y, dtype=np.float64).copy()
    w = np.asarray(w, dtype=np.float64).copy()
    n = len(y)
    # Each block is [start, end) with a pooled value and weight.
    vals: List[float] = []
    wts: List[float] = []
    sizes: List[int] = []
    for i in range(n):
        vals.append(float(y[i]))
        wts.append(float(w[i]))
        sizes.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, s2 = vals.pop(), wts.pop(), sizes.pop()
            v1, w1, s1 = vals.pop(), wts.pop(), sizes.pop()
            tw = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / tw if tw > 0 else 0.5 * (v1 + v2))
            wts.append(tw)
            sizes.append(s1 + s2)
    out = np.empty(n, dtype=np.float64)
    idx = 0
    for v, s in zip(vals, sizes):
        out[idx:idx + s] = v
        idx += s
    return out


@dataclass
class PresenceCalibrator:
    """Monotone map from raw filter presence to calibrated probability."""

    edges: np.ndarray
    """Right edges of the score bins, ascending, last one 1.0."""

    probs: np.ndarray
    """Calibrated probability for each bin, non-decreasing."""

    counts: np.ndarray
    """Samples per bin; kept so a thin bin can be recognised as such."""

    raw_means: np.ndarray
    """Mean raw score per bin, for the reliability diagram."""

    n_samples: int = 0
    occupancy: float = 0.0
    """Base rate the map was fitted at. A deployment with a very different
    emergency-vehicle rate should refit rather than reuse this."""

    # ------------------------------------------------------------------- apply
    def __call__(self, p: float) -> float:
        return self.transform(p)

    def transform(self, p: float) -> float:
        p = float(np.clip(p, 0.0, 1.0))
        i = int(np.searchsorted(self.edges, p, side="left"))
        i = int(np.clip(i, 0, len(self.probs) - 1))
        return float(self.probs[i])

    def transform_many(self, p: Sequence[float]) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
        i = np.clip(np.searchsorted(self.edges, p, side="left"),
                    0, len(self.probs) - 1)
        return self.probs[i]

    # --------------------------------------------------------------------- fit
    @classmethod
    def fit(cls, scores: Sequence[float], truth: Sequence[bool],
            n_bins: int = 20) -> "PresenceCalibrator":
        s = np.asarray(scores, dtype=np.float64)
        y = np.asarray(truth, dtype=np.float64)
        if s.size == 0:
            raise ValueError("no samples to calibrate on")

        # Equal-count bins, so the fit spends its resolution where the data is.
        # The filter piles mass at 0 and at 1, and equal-width bins would put
        # almost every sample in two of them.
        qs = np.linspace(0.0, 1.0, n_bins + 1)[1:]
        edges = np.unique(np.quantile(s, qs))
        edges = np.clip(edges, 0.0, 1.0)
        if edges[-1] < 1.0:
            edges = np.append(edges, 1.0)

        idx = np.clip(np.searchsorted(edges, s, side="left"), 0, len(edges) - 1)
        counts = np.zeros(len(edges))
        means = np.zeros(len(edges))
        raw = np.zeros(len(edges))
        for b in range(len(edges)):
            m = idx == b
            counts[b] = m.sum()
            means[b] = y[m].mean() if counts[b] else 0.0
            raw[b] = s[m].mean() if counts[b] else edges[b]

        probs = pool_adjacent_violators(means, counts)
        probs = np.clip(probs, 1e-4, 1.0 - 1e-4)
        return cls(edges=edges, probs=probs, counts=counts, raw_means=raw,
                   n_samples=int(s.size), occupancy=float(y.mean()))

    # ----------------------------------------------------------------- quality
    @staticmethod
    def expected_calibration_error(scores: Sequence[float], truth: Sequence[bool],
                                   n_bins: int = 15) -> float:
        s = np.asarray(scores, dtype=np.float64)
        y = np.asarray(truth, dtype=np.float64)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.clip(np.digitize(s, edges[1:-1]), 0, n_bins - 1)
        ece = 0.0
        for b in range(n_bins):
            m = idx == b
            if not m.any():
                continue
            ece += (m.mean()) * abs(y[m].mean() - s[m].mean())
        return float(ece)

    @staticmethod
    def brier(scores: Sequence[float], truth: Sequence[bool]) -> float:
        s = np.asarray(scores, dtype=np.float64)
        y = np.asarray(truth, dtype=np.float64)
        return float(np.mean((s - y) ** 2))

    # ------------------------------------------------------------------- (de)serialise
    def to_dict(self) -> Dict[str, object]:
        return {
            "edges": self.edges.tolist(),
            "probs": self.probs.tolist(),
            "counts": self.counts.tolist(),
            "raw_means": self.raw_means.tolist(),
            "n_samples": int(self.n_samples),
            "occupancy": float(self.occupancy),
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "PresenceCalibrator":
        return cls(
            edges=np.asarray(d["edges"], dtype=np.float64),
            probs=np.asarray(d["probs"], dtype=np.float64),
            counts=np.asarray(d.get("counts", []), dtype=np.float64),
            raw_means=np.asarray(d.get("raw_means", []), dtype=np.float64),
            n_samples=int(d.get("n_samples", 0)),
            occupancy=float(d.get("occupancy", 0.0)),
        )

    @classmethod
    def load(cls, path: Path) -> Optional["PresenceCalibrator"]:
        path = Path(path)
        if not path.exists():
            return None
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
