"""Score the acoustic belief against ground truth, filter by filter.

Everything the controller does downstream is a function of three numbers per
approach -- presence, range, and expected triage priority -- so those three are
what have to be measured, not the CNN's frame-level accuracy. This script runs
the simulator with ground truth alongside the filter and reports:

* **Presence.** AUC, expected calibration error, Brier score, and precision /
  recall / F1 at the operating points the corridor actually uses.
* **Localisation.** Given that some approach really is carrying an inbound
  emergency vehicle, how often is the argmax of the presence vector the right
  arm? This is the number the per-approach filter cannot do well on by
  construction, because nothing in it makes the four answers compete.
* **Cross-talk.** Mean number of approaches simultaneously above 0.5 when
  exactly one is truly occupied. One is perfect; the independent filter's
  failure shows up here directly.
* **Range and arrival time.** Absolute error against the true distance to the
  stop line, restricted to approach-epochs that really are occupied, and broken
  out by distance band -- the 100-220 m band is where the controller has to
  commit, so that is where accuracy matters.
* **Warning time.** How many seconds before the vehicle reaches the stop line
  the belief first crosses each threshold. A detector that is right only once
  the vehicle has arrived is worth nothing to a signal controller, and this is
  the metric that says so.
* **Class.** Confusion over {ambulance, fire, police} and triage-weight error.
* **Cost.** Wall-clock microseconds per decision epoch.

    python scripts/sensing_benchmark.py --episodes 6
    python scripts/sensing_benchmark.py --modes joint,independent --sensing oracle
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from atsc.agents.baselines import build_baseline
from atsc.config import (
    DATA_DIR, DemandConfig, EnvConfig, MODELS_DIR, RESULTS_DIR, RUNS_DIR, ensure_dirs)
from atsc.env import SumoSignalEnv
from atsc.netgen import build_network
from atsc.sensing.belief import SIREN_CLASSES
from atsc.sensing.calibration import PresenceCalibrator
from atsc.sensing.sensor import AcousticSensor, SensorErrorModel

BENCH_SEED_BASE = 700000
"""Disjoint from training (0..), evaluation (50000..) and calibration (900000..)."""

THRESHOLDS = (0.25, 0.5, 0.75)


def ev_world_state(env) -> list:
    """Every emergency vehicle on the network, with where it is relative to us.

    Used to decompose false positives: a detection with no vehicle anywhere is a
    hallucination, one with a vehicle on another arm is mis-localisation, and
    one with a vehicle that has just crossed this arm is a stale tail. Those
    three call for completely different fixes, so lumping them into a single
    false-alarm rate hides the only thing the number could have told us.
    """
    evs = getattr(env.scenario, "emergency_vehicles", None) or {}
    out = []
    if not evs:
        return out
    conn = env.conn
    for vid in set(conn.vehicle.getIDList()) & set(evs):
        try:
            road = conn.vehicle.getRoadID(vid)
            px, py = conn.vehicle.getPosition(vid)
        except Exception:
            continue
        for a in env.approaches:
            sx, sy = env.stop_line[a]
            out.append((vid, a, road == a, float(np.hypot(px - sx, py - sy))))
    return out


def classify_false_positive(env, approach, radius) -> str:
    """Why did this approach light up when nothing was inbound on it?

    The four arms' stop lines sit within about thirty metres of one another, so
    "is there an emergency vehicle near this stop line" is true of every arm
    whenever it is true of one. The categories therefore have to be decided from
    each vehicle's *route*, not from distance: a vehicle whose route contains
    this arm and which is now past it is a stale tail, one inbound on a
    different arm is a mis-localisation, and only a genuinely empty
    neighbourhood is a hallucination.
    """
    evs = getattr(env.scenario, "emergency_vehicles", None) or {}
    conn = env.conn
    live = set(conn.vehicle.getIDList()) & set(evs)
    if not live:
        return "hallucination"
    crossed_here = elsewhere = False
    for vid in live:
        ev = evs[vid]
        try:
            road = conn.vehicle.getRoadID(vid)
            px, py = conn.vehicle.getPosition(vid)
        except Exception:
            continue
        sx, sy = env.stop_line[approach]
        d = float(np.hypot(px - sx, py - sy))
        if approach in ev.route_edges and road != approach and d <= radius:
            crossed_here = True
        elif road in env.approaches and road != approach and d <= radius:
            elsewhere = True
    if crossed_here:
        return "stale_tail"
    if elsewhere:
        return "mislocalised"
    return "hallucination"


def truth_state(env, radius: float) -> dict:
    """Per approach: is an emergency vehicle inbound, and where exactly is it?

    Returns ``{approach: (present, distance_to_stop_line_m, closing, class)}``.
    'Inbound' is the event a green light can still help with: the vehicle is on
    the approach edge, inside the detection radius, and has not crossed.
    """
    out = {a: (False, float("nan"), False, None) for a in env.approaches}
    evs = getattr(env.scenario, "emergency_vehicles", None) or {}
    if not evs:
        return out
    conn = env.conn
    for vid in set(conn.vehicle.getIDList()) & set(evs):
        try:
            road = conn.vehicle.getRoadID(vid)
            if road not in out:
                continue
            px, py = conn.vehicle.getPosition(vid)
            speed = float(conn.vehicle.getSpeed(vid))
        except Exception:
            continue
        sx, sy = env.stop_line[road]
        d = float(np.hypot(px - sx, py - sy))
        if d > radius:
            continue
        prev = out[road]
        # Nearest vehicle wins when two share an approach: it is the one whose
        # arrival the controller has to plan for.
        if prev[0] and np.isfinite(prev[1]) and prev[1] <= d:
            continue
        out[road] = (True, d, speed >= 0.0, evs[vid].veh_class)
    return out


def run_episode(cfg, net_file, error_model, seed, controller_name, records):
    sensor = AcousticSensor([], cfg.acoustic, mode=cfg.sensing_mode,
                            error_model=error_model, seed=seed)
    env = SumoSignalEnv(cfg, net_file=net_file,
                        work_dir=RUNS_DIR / "sensebench",
                        label=f"sb_{seed}", sensor=sensor)
    obs = env.reset(seed=seed)
    controller = build_baseline(controller_name, env, cfg.demand)
    controller.reset(env)
    # "Inbound" is on the arm and not yet across, which for this network is the
    # whole arm: it is 320 m long and the mast hears to 300 m. Scoring against
    # the detection radius instead would count every correct detection made in
    # the first 20 m of the arm as an error.
    radius = float(cfg.network.arm_length_m) + 5.0
    done = False
    epoch_us = []
    first_seen: dict = {}

    while not done:
        mask = env.action_mask()
        action = controller.act(env, obs, mask)
        t = truth_state(env, radius)
        beliefs = env.belief.beliefs()
        pres = np.array([beliefs[a].presence for a in env.approaches])
        truth = np.array([t[a][0] for a in env.approaches])

        # Ground-truth concurrency, which is what actually justifies truncating
        # the hypothesis set at two simultaneous sources. The filter's own
        # posterior mass on two-source hypotheses measures its uncertainty, not
        # whether a third vehicle was there to be missed; only the simulator can
        # answer that.
        records["n_inbound"].append(int(sum(1 for a in env.approaches if t[a][0])))

        for i, a in enumerate(env.approaches):
            if pres[i] >= 0.5 and not t[a][0]:
                records["fp_kind"].append(
                    classify_false_positive(env, a, radius))
            b = beliefs[a]
            present, dist, _closing, cls = t[a]
            records["score"].append(float(b.presence))
            records["truth"].append(bool(present))
            records["approach"].append(a)
            records["true_dist"].append(float(dist))
            records["est_range"].append(float(b.range_m))
            records["est_eta"].append(float(b.eta_s))
            records["true_class"].append(cls)
            records["pred_class"].append(
                max(b.class_probs, key=b.class_probs.get) if b.class_probs else None)
            records["pred_priority"].append(float(b.expected_priority))
            records["true_priority"].append(
                float(cfg.demand.ev_priority.get(cls, np.nan)) if cls else float("nan"))
            records["true_eta"].append(_true_eta(env, a, dist) if present else float("nan"))

        # Cross-talk: how many arms light up when exactly one is truly occupied.
        if truth.sum() == 1:
            for th in THRESHOLDS:
                records[f"crosstalk@{th}"].append(int((pres >= th).sum()))
            records["argmax_correct"].append(
                bool(env.approaches[int(np.argmax(pres))] ==
                     env.approaches[int(np.argmax(truth))]))

        # Warning distance: how far out the belief first crosses each threshold,
        # per *arrival*. An arrival that is never detected at all is recorded as
        # a miss rather than as a zero, because averaging misses in as zeros
        # conflates "detected late" with "not detected", and only one of those
        # is a sensing failure the controller can do anything about.
        for i, a in enumerate(env.approaches):
            present, dist, _c, _cl = t[a]
            if present:
                for th in THRESHOLDS:
                    key = (a, th)
                    if pres[i] >= th and key not in first_seen:
                        first_seen[key] = dist
                first_seen[(a, "active")] = True
            elif first_seen.pop((a, "active"), False):
                for th in THRESHOLDS:
                    d = first_seen.pop((a, th), None)
                    records[f"warn_m@{th}"].append(
                        float("nan") if d is None else float(d))

        t0 = time.perf_counter()
        obs, _, _, done, _ = env.step(action)
        epoch_us.append((time.perf_counter() - t0) * 1e6)

    two_source = float(getattr(env.belief, "two_source_mass", float("nan")))
    records["two_source_mass"].append(two_source)
    env.close()
    return float(np.median(epoch_us)) if epoch_us else float("nan")


def _true_eta(env, approach, dist_m) -> float:
    """Seconds to the stop line at the free-flow speed of the approach."""
    if not np.isfinite(dist_m):
        return float("nan")
    v = 13.89
    try:
        lanes = [ln for ln in env.controlled_lanes
                 if env.lane_to_approach.get(ln) == approach]
        if lanes:
            v = max(env._lane_speed.get(lanes[0], v), 1.0)
    except Exception:
        pass
    return float(dist_m / (v * 1.25))


def auc(scores, truth):
    scores = np.asarray(scores, dtype=float)
    truth = np.asarray(truth, dtype=bool)
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks over ties, or a near-binary score is unfairly penalised.
    uniq, inv = np.unique(scores, return_inverse=True)
    for u in range(len(uniq)):
        m = inv == u
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    n1, n0 = truth.sum(), (~truth).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[truth].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def prf(scores, truth, th):
    s = np.asarray(scores) >= th
    y = np.asarray(truth, dtype=bool)
    tp = float((s & y).sum())
    fp = float((s & ~y).sum())
    fn = float((~s & y).sum())
    prec = tp / max(tp + fp, 1e-9)
    rec = tp / max(tp + fn, 1e-9)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"precision": prec, "recall": rec, "f1": f1,
            "false_alarms_per_hour": fp}


def summarise(records, epoch_us, cal: PresenceCalibrator | None) -> dict:
    score = np.asarray(records["score"], dtype=float)
    truth = np.asarray(records["truth"], dtype=bool)
    out = {
        "n_approach_epochs": int(score.size),
        "occupancy": float(truth.mean()),
        "auc": auc(score, truth),
        "ece": float(PresenceCalibrator.expected_calibration_error(score, truth)),
        "brier": float(PresenceCalibrator.brier(score, truth)),
        "median_epoch_us": float(np.median(epoch_us)),
    }
    if cal is not None:
        c = cal.transform_many(score)
        out["ece_calibrated"] = float(
            PresenceCalibrator.expected_calibration_error(c, truth))
        out["brier_calibrated"] = float(PresenceCalibrator.brier(c, truth))
        out["max_calibrated_p"] = float(np.max(c))
    for th in THRESHOLDS:
        out[f"prf@{th}"] = prf(score, truth, th)
        ct = records.get(f"crosstalk@{th}", [])
        if ct:
            out[f"arms_above@{th}"] = float(np.mean(ct))
        wm = records.get(f"warn_m@{th}", [])
        if wm:
            wm = np.asarray(wm, dtype=float)
            seen = np.isfinite(wm)
            out[f"arrival_detection_rate@{th}"] = float(seen.mean())
            if seen.any():
                out[f"warning_range_m@{th}"] = float(np.mean(wm[seen]))
                out[f"warning_range_median_m@{th}"] = float(np.median(wm[seen]))
    conc = np.asarray(records.get("n_inbound", []), dtype=float)
    if conc.size:
        out["epochs_with_1_inbound"] = float(np.mean(conc == 1))
        out["epochs_with_2_inbound"] = float(np.mean(conc == 2))
        out["epochs_with_3plus_inbound"] = float(np.mean(conc >= 3))
    tsm = [v for v in records.get("two_source_mass", []) if np.isfinite(v)]
    if tsm:
        out["two_source_posterior_mass"] = float(np.mean(tsm))
    if records.get("argmax_correct"):
        out["localisation_top1"] = float(np.mean(records["argmax_correct"]))
    fp = records.get("fp_kind", [])
    if fp:
        kinds, counts = np.unique(np.asarray(fp), return_counts=True)
        out["false_positive_kinds"] = {
            str(k): float(c) / len(fp) for k, c in zip(kinds, counts)}
        out["false_positive_total"] = int(len(fp))

    # Range / ETA accuracy, restricted to genuinely occupied approach-epochs.
    td = np.asarray(records["true_dist"], dtype=float)
    er = np.asarray(records["est_range"], dtype=float)
    te = np.asarray(records["true_eta"], dtype=float)
    ee = np.asarray(records["est_eta"], dtype=float)
    m = truth & np.isfinite(te) & np.isfinite(ee) & (te < 40.0)
    if m.any():
        out["eta_mae_s"] = float(np.mean(np.abs(ee[m] - te[m])))
        out["eta_median_ae_s"] = float(np.median(np.abs(ee[m] - te[m])))
    ok = truth & np.isfinite(td) & np.isfinite(er)
    if ok.any():
        err = np.abs(er[ok] - td[ok])
        out["range_mae_m"] = float(np.mean(err))
        out["range_median_ae_m"] = float(np.median(err))
        for lo, hi in ((0, 100), (100, 220), (220, 320)):
            band = ok & (td >= lo) & (td < hi)
            if band.sum() > 5:
                out[f"range_mae_m_{lo}_{hi}"] = float(
                    np.mean(np.abs(er[band] - td[band])))

    # Class: confusion and triage-weight error where a vehicle is present.
    tc_ = np.asarray(records["true_class"], dtype=object)
    pc = np.asarray(records["pred_class"], dtype=object)
    cls_ok = truth & np.array([c is not None for c in tc_])
    if cls_ok.any():
        out["class_accuracy"] = float(np.mean(tc_[cls_ok] == pc[cls_ok]))
        conf = {a: {b: 0 for b in SIREN_CLASSES} for a in SIREN_CLASSES}
        for a, b in zip(tc_[cls_ok], pc[cls_ok]):
            if a in conf and b in conf[a]:
                conf[a][b] += 1
        out["class_confusion"] = conf
        tp_ = np.asarray(records["true_priority"], dtype=float)
        pp = np.asarray(records["pred_priority"], dtype=float)
        m = cls_ok & np.isfinite(tp_) & np.isfinite(pp)
        if m.any():
            out["triage_weight_mae"] = float(np.mean(np.abs(pp[m] - tp_[m])))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--horizon", type=int, default=3600)
    p.add_argument("--ev-rate", type=float, default=40.0)
    p.add_argument("--regime", default="balanced")
    p.add_argument("--sensing", default="surrogate")
    p.add_argument("--controller", default="max_pressure")
    p.add_argument("--modes", default="joint,independent",
                   help="comma-separated belief modes to compare")
    p.add_argument("--calibration", default=str(MODELS_DIR / "presence_calibration.json"))
    p.add_argument("--out", default=str(RESULTS_DIR / "sensing_benchmark.json"))
    args = p.parse_args()

    ensure_dirs()
    net_cfg = EnvConfig()
    net_file, _ = build_network(net_cfg.network, DATA_DIR / "networks")
    em_path = MODELS_DIR / "sensor_error_model.json"
    error_model = SensorErrorModel.load(em_path) if em_path.exists() else None
    cal_path = Path(args.calibration)
    cal = PresenceCalibrator.load(cal_path) if cal_path.exists() else None

    seeds = [BENCH_SEED_BASE + i for i in range(args.episodes)]
    results = {}
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        cfg = EnvConfig()
        cfg.demand = DemandConfig(horizon_s=args.horizon, regime=args.regime,
                                  ev_rate_per_hour=args.ev_rate)
        cfg.sensing_mode = args.sensing
        cfg.belief_mode = mode
        cfg.gui = False
        cfg.collect_emissions = False

        records = defaultdict(list)
        per_epoch = []
        print(f"\n=== belief_mode={mode} sensing={args.sensing} ===", flush=True)
        for seed in seeds:
            us = run_episode(cfg, net_file, error_model, seed, args.controller, records)
            per_epoch.append(us)
            print(f"  seed {seed}: {len(records['score'])} approach-epochs, "
                  f"{us:.0f} us/epoch", flush=True)
        # Only the joint filter's own calibration map is meaningful for it; the
        # independent filter is reported raw plus its own map elsewhere.
        summary = summarise(records, per_epoch, cal if mode == "joint" else None)
        results[mode] = summary
        for k, v in summary.items():
            if isinstance(v, dict):
                print(f"  {k}: " + " ".join(f"{kk}={vv:.3f}" if isinstance(vv, float)
                                            else f"{kk}={vv}" for kk, vv in v.items()))
            elif isinstance(v, float):
                print(f"  {k:28s} {v:10.4f}")
            else:
                print(f"  {k:28s} {v}")

    out = Path(args.out)
    out.write_text(json.dumps({
        "config": {"episodes": args.episodes, "horizon_s": args.horizon,
                   "ev_rate_per_hour": args.ev_rate, "regime": args.regime,
                   "sensing": args.sensing, "controller": args.controller,
                   "seeds": seeds},
        "modes": results,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
