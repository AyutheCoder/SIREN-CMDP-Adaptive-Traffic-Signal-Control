"""Metrics computed from SUMO's own trip records.

Per-vehicle numbers come from ``tripinfo-output`` rather than from TraCI polling.
That matters for credibility: SUMO computes ``timeLoss`` against each vehicle's
own free-flow reference speed including its individual speed factor, which is
the standard definition of delay in the traffic literature and is not something
a step-loop accumulator reproduces exactly.

Reported quantities
-------------------
``mean_time_loss_s``      average delay relative to free flow - the headline.
``p95_time_loss_s``       tail delay; a controller that helps the mean while
                          wrecking the tail is not an improvement.
``ev_*``                  the same quantities restricted to emergency vehicles.
``jain_fairness``         Jain's index over per-approach mean delay. Detects the
                          classic failure where a controller buys a good average
                          by permanently sacrificing one approach.
``co2_g`` / ``fuel_l``    HBEFA3 emissions, since idling at red is the point.
"""

from __future__ import annotations

import math
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

EMERGENCY_VTYPES = ("ambulance", "fire", "police")


def jains_fairness(values: Sequence[float]) -> float:
    """Jain's fairness index in ``(0, 1]``; 1.0 means perfectly equal treatment.

    Applied to *service quality* (negated delay is not used directly because the
    index requires non-negative inputs); here it is applied to per-approach mean
    delay, so a **lower** spread gives a value closer to 1.
    """
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return float("nan")
    v = np.maximum(v, 0.0)
    denom = v.size * float(np.sum(v ** 2))
    if denom <= 0:
        return 1.0
    return float(np.sum(v) ** 2 / denom)


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def parse_tripinfo(
    path: Path,
    approach_of: Optional[Dict[str, str]] = None,
    emergency_ids: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """Aggregate a SUMO ``tripinfo`` file into the reported metric set.

    ``approach_of`` maps an edge id to an approach label, used for the fairness
    index; ``emergency_ids`` names the prioritised vehicles.
    """
    path = Path(path)
    if not path.exists():
        return {"n_trips": 0}

    ev_ids = set(emergency_ids or ())
    all_loss: List[float] = []
    all_duration: List[float] = []
    all_wait: List[float] = []
    all_stops: List[float] = []
    ev_loss: List[float] = []
    ev_duration: List[float] = []
    ev_wait: List[float] = []
    ev_stops: List[float] = []
    ev_speed: List[float] = []
    ev_class: List[str] = []
    ev_by_class: Dict[str, List[float]] = defaultdict(list)
    ev_stops_by_class: Dict[str, List[float]] = defaultdict(list)
    by_approach: Dict[str, List[float]] = defaultdict(list)
    co2_mg = 0.0
    fuel_mg = 0.0
    nox_mg = 0.0

    for _, elem in ET.iterparse(str(path), events=("end",)):
        if elem.tag != "tripinfo":
            continue
        vid = elem.get("id", "")
        vtype = elem.get("vType", "")
        loss = float(elem.get("timeLoss", 0.0))
        duration = float(elem.get("duration", 0.0))
        wait = float(elem.get("waitingTime", 0.0))
        stops = float(elem.get("waitingCount", 0.0))
        route_length = float(elem.get("routeLength", 0.0))

        is_ev = vid in ev_ids or vtype in EMERGENCY_VTYPES
        if is_ev:
            ev_loss.append(loss)
            ev_duration.append(duration)
            ev_wait.append(wait)
            ev_stops.append(stops)
            # Triage is only meaningful if it shows up per class: the scheduler
            # ranks an ambulance above a police car, so realised delay should be
            # ordered the same way, and the only way to see that is to keep the
            # class alongside the vehicle rather than pooling them.
            cls = vtype if vtype in EMERGENCY_VTYPES else "unknown"
            ev_class.append(cls)
            ev_by_class[cls].append(wait)
            ev_stops_by_class[cls].append(stops)
            if duration > 0:
                ev_speed.append(route_length / duration)
        else:
            all_loss.append(loss)
            all_duration.append(duration)
            all_wait.append(wait)
            all_stops.append(stops)

        if approach_of is not None:
            lane = elem.get("departLane", "")
            edge = lane.rsplit("_", 1)[0] if lane else ""
            label = approach_of.get(edge)
            if label:
                by_approach[label].append(loss)

        em = elem.find("emissions")
        if em is not None:
            co2_mg += float(em.get("CO2_abs", 0.0))
            fuel_mg += float(em.get("fuel_abs", 0.0))
            nox_mg += float(em.get("NOx_abs", 0.0))
        elem.clear()

    n_total = len(all_loss) + len(ev_loss)
    approach_means = [statistics.fmean(v) for v in by_approach.values() if v]

    out: Dict[str, float] = {
        "n_trips": n_total,
        "n_background_trips": len(all_loss),
        "n_emergency_trips": len(ev_loss),
        "mean_time_loss_s": statistics.fmean(all_loss) if all_loss else float("nan"),
        "median_time_loss_s": statistics.median(all_loss) if all_loss else float("nan"),
        "p95_time_loss_s": _percentile(all_loss, 95),
        "mean_travel_time_s": statistics.fmean(all_duration) if all_duration else float("nan"),
        "mean_waiting_time_s": statistics.fmean(all_wait) if all_wait else float("nan"),
        "mean_stops": statistics.fmean(all_stops) if all_stops else float("nan"),
        "jain_fairness": jains_fairness(approach_means),
        "approach_delay_spread_s": (max(approach_means) - min(approach_means))
        if len(approach_means) > 1 else 0.0,
        "co2_g": co2_mg / 1000.0,
        "fuel_l": fuel_mg / 1000.0 / 745.0,  # mg -> l, petrol density 745 g/l
        "nox_g": nox_mg / 1000.0,
        # NB: teleports are not recorded in tripinfo. They are counted live in
        # SumoSignalEnv and reported from the episode summary; an earlier
        # version emitted a hard-coded zero here, which read as "no vehicle ever
        # teleported" in precisely the gridlocked runs where many did.
    }

    if ev_loss:
        out.update({
            "ev_mean_time_loss_s": statistics.fmean(ev_loss),
            "ev_max_time_loss_s": max(ev_loss),
            "ev_p95_time_loss_s": _percentile(ev_loss, 95),
            "ev_mean_travel_time_s": statistics.fmean(ev_duration),
            "ev_mean_waiting_time_s": statistics.fmean(ev_wait),
            "ev_mean_stops": statistics.fmean(ev_stops),
            "ev_zero_stop_rate": float(np.mean([s == 0 for s in ev_stops])),
            "ev_mean_speed_ms": statistics.fmean(ev_speed) if ev_speed else float("nan"),
            "ev_p95_waiting_time_s": _percentile(ev_wait, 95),
            "ev_max_waiting_time_s": max(ev_wait),
            # Kept per vehicle so the distribution -- not only its mean -- can be
            # drawn and tested. The mean of a delay distribution with this much
            # skew is not a summary of it.
            "ev_waiting_times_s": [float(x) for x in ev_wait],
            "ev_stop_counts": [float(x) for x in ev_stops],
            "ev_classes": list(ev_class),
        })
        for cls in EMERGENCY_VTYPES:
            vals = ev_by_class.get(cls, [])
            stops_c = ev_stops_by_class.get(cls, [])
            out[f"ev_{cls}_n"] = float(len(vals))
            out[f"ev_{cls}_mean_waiting_time_s"] = (
                statistics.fmean(vals) if vals else float("nan"))
            out[f"ev_{cls}_zero_stop_rate"] = (
                float(np.mean([x == 0 for x in stops_c])) if stops_c else float("nan"))
    else:
        for k in ("ev_mean_time_loss_s", "ev_max_time_loss_s", "ev_p95_time_loss_s",
                  "ev_mean_travel_time_s", "ev_mean_waiting_time_s", "ev_mean_stops",
                  "ev_zero_stop_rate", "ev_mean_speed_ms", "ev_p95_waiting_time_s",
                  "ev_max_waiting_time_s"):
            out[k] = float("nan")
        out["ev_waiting_times_s"] = []
        out["ev_stop_counts"] = []
        out["ev_classes"] = []

    if n_total:
        out["co2_g_per_vehicle"] = out["co2_g"] / n_total
        out["fuel_l_per_vehicle"] = out["fuel_l"] / n_total
    return out


def aggregate_runs(runs: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Mean, median, spread and failure rate across repeated seeds.

    The **median** and ``jam_rate`` are reported alongside the mean because the
    delay distribution here has a heavy tail: an intersection run near capacity
    occasionally tips into spillback, and one such seed reaches ~290 s against a
    ~23 s median for the same controller. A mean over twelve seeds then says
    more about whether that one run happened than about the controller. Quoting
    the mean alone would have max-pressure at 45.6 s when eleven of its twelve
    runs are at 22.8 s.
    """
    if not runs:
        return {}
    keys = sorted({k for r in runs for k in r})
    out: Dict[str, float] = {"n_runs": len(runs)}

    # Fraction of runs that gridlocked, identified by SUMO having had to
    # teleport stuck vehicles. This is the number to read next to the median.
    tele = [float(r.get("teleports", 0.0)) for r in runs if "teleports" in r]
    if tele:
        out["jam_rate"] = float(np.mean([t > 0 for t in tele]))
        out["n_jammed"] = float(sum(t > 0 for t in tele))
    for k in keys:
        vals = [float(r[k]) for r in runs if k in r and isinstance(r[k], (int, float))
                and math.isfinite(float(r[k]))]
        if not vals:
            continue
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_median"] = float(np.median(vals))
        out[f"{k}_max"] = float(np.max(vals))
        if len(vals) > 1:
            sd = float(np.std(vals, ddof=1))
            out[f"{k}_std"] = sd
            # Normal approximation is adequate at the seed counts used here.
            out[f"{k}_ci95"] = 1.96 * sd / math.sqrt(len(vals))
        else:
            out[f"{k}_std"] = 0.0
            out[f"{k}_ci95"] = 0.0

    # Per-vehicle emergency delays pooled across seeds. The seed-level mean is
    # the right unit for a paired test between controllers; the pooled vehicles
    # are the right unit for drawing the distribution, and quoting only the
    # former hides that the improvement is concentrated in the tail.
    pooled = [x for r in runs for x in (r.get("ev_waiting_times_s") or [])]
    if pooled:
        out["ev_waiting_times_pooled_s"] = [float(x) for x in pooled]
        out["ev_waiting_pooled_median_s"] = float(np.median(pooled))
        out["ev_waiting_pooled_p90_s"] = float(np.percentile(pooled, 90))
        out["ev_n_vehicles"] = float(len(pooled))
        classes = [c for r in runs for c in (r.get("ev_classes") or [])]
        if len(classes) == len(pooled):
            out["ev_classes_pooled"] = classes
    return out


def paired_bootstrap_pvalue(
    a: Sequence[float], b: Sequence[float], n_boot: int = 20000, seed: int = 0
) -> Dict[str, float]:
    """Two-sided paired bootstrap on the mean difference ``a - b``.

    Runs are paired by seed - both controllers see the identical demand
    realisation - so pairing removes scenario variance and makes the comparison
    far tighter than an unpaired test at the same number of runs.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.size != b_arr.size or a_arr.size < 2:
        return {"diff": float("nan"), "p_value": float("nan")}
    diff = a_arr - b_arr
    observed = float(np.mean(diff))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    boot = diff[idx].mean(axis=1)
    centred = boot - observed
    p = float(np.mean(np.abs(centred) >= abs(observed)))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "diff": observed,
        "p_value": max(p, 1.0 / n_boot),
        "ci_low": float(lo),
        "ci_high": float(hi),
    }
