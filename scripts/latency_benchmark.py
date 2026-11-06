"""Per-stage compute cost of SIREN-ARC on one CPU, for the deployment question.

    python scripts/latency_benchmark.py

Times the parts that would run at a real intersection, not the parts that only
exist because the intersection is simulated:

* every second: log-mel front end for four one-second clips, the CNN on all
  four, and one predict/update of the joint filter;
* every decision (5 s, plus a 1 s re-check inside greens): the learned
  controller and the corridor rule together.

The filter and controller are timed inside a real SUMO run, so their inputs are
realistic; the sound synthesis and the simulator step are excluded. Run it on
an otherwise idle machine.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atsc.agents.baselines import CorridorPolicy
from atsc.config import DATA_DIR, DemandConfig, EnvConfig, MODELS_DIR, RESULTS_DIR, RUNS_DIR
from atsc.control.corridor import CorridorScheduler
from atsc.env import SumoSignalEnv
from atsc.evaluation.runner import run_episode
from atsc.netgen import build_network
from atsc.sensing.detector import SirenDetector
from atsc.sensing.sensor import AcousticSensor, SensorErrorModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import load_agent  # noqa: E402

T = {"filter_ms": [], "sensor_sim_ms": [], "controller_ms": []}


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def front_end(n_iter: int = 300) -> dict:
    det = SirenDetector.from_saved(MODELS_DIR / "siren_detector.h5")
    sr = det.acoustic.sample_rate
    rng = np.random.default_rng(0)
    clips = [0.02 * rng.standard_normal(int(sr * det.acoustic.frame_s)) for _ in range(4)]
    mel_ms, cnn_ms = [], []
    for i in range(n_iter + 20):
        t0 = time.perf_counter()
        x = np.stack([det.mel(c)[..., None] for c in clips]).astype(np.float32)
        a = _ms(t0)
        t0 = time.perf_counter()
        det.model(x, training=False)
        b = _ms(t0)
        if i >= 20:  # discard warm-up
            mel_ms.append(a)
            cnn_ms.append(b)
    return {"mel_ms_median": float(np.median(mel_ms)), "mel_ms_p95": float(np.percentile(mel_ms, 95)),
            "cnn_ms_median": float(np.median(cnn_ms)), "cnn_ms_p95": float(np.percentile(cnn_ms, 95)),
            "detector_params": int(det.model.count_params())}


def main() -> int:
    out = {"front_end_4_mics": front_end()}

    cfg = EnvConfig()
    cfg.demand = DemandConfig(horizon_s=1800, regime="balanced",
                              base_flow_vph=700.0, ev_rate_per_hour=30.0)
    cfg.sensing_mode = "surrogate"
    cfg.belief_mode = "joint"
    net_file, _ = build_network(cfg.network, DATA_DIR / "networks")
    sensor = AcousticSensor([], cfg.acoustic, mode="surrogate",
                            error_model=SensorErrorModel.load(MODELS_DIR / "sensor_error_model.json"),
                            seed=50000)
    orig_sense = sensor.sense

    def timed_sense(*a, **k):
        t0 = time.perf_counter()
        r = orig_sense(*a, **k)
        T["sensor_sim_ms"].append(_ms(t0))
        return r
    sensor.sense = timed_sense

    orig_env_sense = SumoSignalEnv._sense

    def timed_env_sense(self, dt=None):
        n_before = len(T["sensor_sim_ms"])
        t0 = time.perf_counter()
        orig_env_sense(self, dt)
        total = _ms(t0)
        sim = sum(T["sensor_sim_ms"][n_before:])
        T["filter_ms"].append(total - sim)
    SumoSignalEnv._sense = timed_env_sense

    env = SumoSignalEnv(cfg, net_file=net_file, work_dir=RUNS_DIR / "latency",
                        label="latency", sensor=sensor)
    env.reset(seed=50000)
    agent, _ = load_agent(MODELS_DIR / "siren_arc.h5", env.observation_dim, env.n_actions,
                          run_name="siren_arc")
    env.close()

    class Timed(CorridorPolicy):
        def act(self, env, obs, mask):
            t0 = time.perf_counter()
            a = super().act(env, obs, mask)
            T["controller_ms"].append(_ms(t0))
            return a

        def interrupt(self, env):
            t0 = time.perf_counter()
            a = super().interrupt(env)
            T["controller_ms"].append(_ms(t0))
            return a

    run_env = SumoSignalEnv(cfg, net_file=net_file, work_dir=RUNS_DIR / "latency",
                            label="latency_run", sensor=sensor)
    run_episode(run_env, lambda e: Timed(
        agent=agent, name="siren_arc",
        calibration_path=MODELS_DIR / "presence_calibration.json",
        scheduler=CorridorScheduler(ev_value_ratio=50.0)), seed=50000)

    for k in ("filter_ms", "controller_ms"):
        v = np.asarray(T[k][5:])
        out[k] = {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
                  "max": float(v.max()), "n": int(v.size)}
    fe = out["front_end_4_mics"]
    out["per_second_total_ms_median"] = (fe["mel_ms_median"] + fe["cnn_ms_median"]
                                         + out["filter_ms"]["median"])
    out["per_second_total_ms_p95"] = (fe["mel_ms_p95"] + fe["cnn_ms_p95"]
                                      + out["filter_ms"]["p95"])
    try:
        ps = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-Process -Id {os.getpid()}).PeakWorkingSet64"],
            capture_output=True, text=True, timeout=30)
        out["peak_memory_mb"] = int(ps.stdout.strip()) / 1e6
    except Exception:  # noqa: BLE001
        out["peak_memory_mb"] = None
    path = RESULTS_DIR / "camera_ready" / "latency.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
