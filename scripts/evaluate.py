"""Evaluate every controller on identical, seeded demand and emit the results table.

    python scripts/evaluate.py --seeds 10 --regime balanced \
        --agent siren_cmdp=models/siren_cmdp.h5 \
        --agent unconstrained=models/unconstrained.h5

Every controller sees exactly the same demand realisation for a given seed, so
differences are attributable to control rather than to scenario luck, and the
per-seed pairing makes the bootstrap test in the results table valid.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atsc.agents.baselines import (
    BASELINE_NAMES, CorridorPolicy, LearnedPolicy, PriorityAwarePolicy,
    build_baseline)
from atsc.agents.dueling_ddqn import LagrangianDuelingDDQN
from atsc.config import (
    AgentConfig, DATA_DIR, DemandConfig, EnvConfig, MODELS_DIR, RESULTS_DIR, RUNS_DIR, ensure_dirs,
)
from atsc.control.corridor import CorridorScheduler
from atsc.env import SumoSignalEnv
from atsc.evaluation.metrics import paired_bootstrap_pvalue
from atsc.evaluation.runner import evaluate_controller
from atsc.netgen import build_network, write_actuated_program
from atsc.sensing.sensor import AcousticSensor, SensorErrorModel

# Emergency-vehicle performance is led by *stopped* delay rather than time loss.
# SUMO's timeLoss is measured against each vehicle's own desired speed, and
# emergency vehicles are configured to want to travel faster than surrounding
# traffic, so they accrue time loss simply by being behind law-abiding cars --
# a component signal control cannot remove. Stopped delay and stop count are the
# parts the controller actually governs; timeLoss is retained for completeness.
REPORT_KEYS = [
    ("mean_time_loss_s", "delay (s)", 2),
    ("p95_time_loss_s", "p95 delay (s)", 2),
    ("mean_queue", "queue (veh)", 2),
    ("ev_mean_waiting_time_s", "EV stopped (s)", 2),
    ("ev_mean_stops", "EV stops", 2),
    ("ev_zero_stop_rate", "EV clean pass", 3),
    ("ev_mean_time_loss_s", "EV timeLoss (s)", 2),
    ("throughput", "throughput", 0),
    ("jain_fairness", "fairness", 4),
    ("max_movement_wait_s", "max wait (s)", 0),
    ("teleports", "teleports", 1),
    ("co2_g_per_vehicle", "CO2 (g/veh)", 1),
    ("fuel_l_per_vehicle", "fuel (l/veh)", 4),
]


def load_agent(path: Path, state_dim: int, n_actions: int, run_name: str):
    """Load a checkpoint together with the training settings it needs at test time."""
    meta_path = Path(path).with_suffix(".agent.json")
    cfg = AgentConfig()
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cfg.constrained = bool(meta.get("constrained", True))
        cfg.cost_budget = float(meta.get("cost_budget", cfg.cost_budget))

    # Checkpoints written before the sensing mode was recorded alongside the
    # weights fall back to the environment config that the training run saved,
    # which is authoritative either way.
    if "sensing_mode" not in meta:
        env_cfg_path = RESULTS_DIR / "training" / run_name / "env_config.json"
        if env_cfg_path.exists():
            recorded = json.loads(env_cfg_path.read_text(encoding="utf-8"))
            if "sensing_mode" in recorded:
                meta["sensing_mode"] = recorded["sensing_mode"]

    agent = LagrangianDuelingDDQN(state_dim, n_actions, cfg)
    agent.load(Path(path))
    return agent, meta


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--seed-start", type=int, default=50000)
    p.add_argument("--horizon", type=int, default=3600)
    p.add_argument("--regime", type=str, default="balanced",
                   choices=["balanced", "asymmetric", "peak", "surge"])
    p.add_argument("--flow", type=float, default=700.0)
    p.add_argument("--ev-rate", type=float, default=24.0,
                   help="emergency vehicles per hour. Higher than a typical "
                        "intersection sees, chosen so the emergency-delay "
                        "statistics have adequate sample size; at <1%% of total "
                        "flow it does not materially change the background "
                        "traffic problem.")
    p.add_argument("--sensing-interval", type=float, default=None,
                   help="seconds between microphone readings; defaults to "
                        "EnvConfig.sensing_interval_s. Exposed so the sensing "
                        "rate can be compared against the controller's decision "
                        "interval on identical seeds.")
    p.add_argument("--sensing", type=str, default="surrogate",
                   choices=["surrogate", "oracle", "none", "end_to_end"])
    p.add_argument("--belief-mode", type=str, default="joint",
                   choices=["joint", "independent"],
                   help="presence filter used by every entry that does not "
                        "override it with the ':indep' flag")
    p.add_argument("--agent", action="append", default=[],
                   metavar="NAME=PATH", help="trained checkpoint, policy alone")
    p.add_argument("--priority", action="append", default=[],
                   metavar="NAME=PATH",
                   help="trained checkpoint evaluated with the emergency-priority "
                        "overlay -- this is the deployable configuration")
    p.add_argument("--corridor", action="append", default=[],
                   metavar="NAME=PATH",
                   help="trained checkpoint evaluated with the risk-calibrated "
                        "corridor scheduler -- this is the deployable "
                        "configuration (see atsc/control/corridor.py)")
    p.add_argument("--value-ratio", type=float, default=50.0,
                   help="emergency/background exchange rate used by every "
                        "corridor entry; see scripts/value_sweep.py")
    p.add_argument("--baselines", type=str,
                   default=",".join(BASELINE_NAMES) + ",max_pressure_preempt:oracle",
                   help="comma-separated; append ':<sensing>' to run a baseline "
                        "under a different sensing mode. 'none' runs no "
                        "baselines at all, which is what a run that only adds "
                        "learned controllers to an existing evaluation wants. "
                        "A sentinel rather than an empty string because "
                        "PowerShell drops an empty argument when it builds a "
                        "child command line, so --baselines '' arrives as no "
                        "value at all and argparse rejects the option.")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--reference", type=str, default="max_pressure",
                   help="controller the significance tests compare against")
    args = p.parse_args()

    ensure_dirs()
    cfg = EnvConfig()
    cfg.demand = DemandConfig(
        horizon_s=args.horizon, regime=args.regime,
        base_flow_vph=args.flow, ev_rate_per_hour=args.ev_rate,
    )
    cfg.sensing_mode = args.sensing
    if args.sensing_interval is not None:
        cfg.sensing_interval_s = float(args.sensing_interval)
    cfg.belief_mode = args.belief_mode
    cfg.collect_emissions = True

    seeds = [args.seed_start + i for i in range(args.seeds)]
    net_file, _ = build_network(cfg.network, DATA_DIR / "networks")
    actuated_add = write_actuated_program(
        net_file, DATA_DIR / "networks" / "actuated.add.xml",
        min_dur=cfg.control.min_green_s, max_dur=cfg.control.max_green_s,
    )

    em_path = MODELS_DIR / "sensor_error_model.json"
    error_model = SensorErrorModel.load(em_path) if em_path.exists() else None
    if error_model is None:
        print("WARNING: no fitted sensor error model; using the fallback.")

    def make_sensor() -> AcousticSensor:
        return AcousticSensor([], cfg.acoustic, mode=cfg.sensing_mode,
                              error_model=error_model, seed=args.seed_start)

    # Probe the environment once to learn the observation/action dimensions.
    probe = SumoSignalEnv(cfg, net_file=net_file,
                          work_dir=RUNS_DIR / "probe", label="probe")
    probe.reset(seed=seeds[0])
    state_dim, n_actions = probe.observation_dim, probe.n_actions
    probe.close()

    results = {}
    print(f"\nregime={args.regime}  flow={args.flow} veh/h/approach  "
          f"ev_rate={args.ev_rate}/h  horizon={args.horizon}s  seeds={len(seeds)}\n")

    # A baseline may be written "name:sensing" to run it under a different
    # sensing mode. The important case is "max_pressure_preempt:oracle": hard
    # preemption with perfect detection is the *ceiling* on what any signal
    # controller can do for emergency vehicles at this intersection. Without it
    # there is no way to tell whether a modest improvement means weak control or
    # a genuinely short frontier.
    _skip = {"", "none"}
    for spec in [b for b in args.baselines.split(",")
                 if b.strip().lower() not in _skip]:
        name, _, sensing = spec.partition(":")
        label = spec.replace(":", "_")
        base_cfg = cfg
        if sensing:
            base_cfg = copy.deepcopy(cfg)
            base_cfg.sensing_mode = sensing
        adds = [actuated_add] if name == "actuated" else None
        results[label] = evaluate_controller(
            lambda env, n=name: build_baseline(n, env, cfg.demand),
            base_cfg, seeds, label, net_file=net_file,
            additional_files=adds,
            sensor=AcousticSensor([], base_cfg.acoustic, mode=base_cfg.sensing_mode,
                                  error_model=error_model, seed=args.seed_start),
            work_dir=RUNS_DIR / f"{args.regime}_{label}",
        )

    SENSING_MODES = ("surrogate", "oracle", "none", "end_to_end")
    for spec, mode in ([(s, "plain") for s in args.agent]
                       + [(s, "overlay") for s in args.priority]
                       + [(s, "corridor") for s in args.corridor]):
        if "=" not in spec:
            raise SystemExit(f"--agent/--priority expects NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        # NAME=PATH:sensing overrides the sensing mode for this entry. Because
        # priority is an evaluation-time overlay rather than something baked
        # into the weights, the whole sensing ablation runs off one trained
        # base -- no retraining, and no risk of the variants differing by
        # anything other than the sensor.
        sensing_override = None
        corridor_flags = set()
        # ``adapt`` couples the corridor's exchange rate to the policy's dual
        # variable; ``noadapt`` is its explicit opposite and the default. The
        # set held only ``noadapt`` while run_all.py asked for ``:adapt``, so
        # that suffix was never recognised, stayed on the path, and the lookup
        # for "models/siren_cmdp.h5:adapt" failed -- skipping the coupled
        # ablation entirely, with one line of output in the middle of an
        # hour-long stage.
        CORRIDOR_FLAGS = {"nocal", "greedy", "noassoc", "indep", "myopic",
                          "adapt", "noadapt"}
        while True:
            head, sep, tail = path.rpartition(":")
            if sep and tail in SENSING_MODES:
                path, sensing_override = head, tail
                continue
            if sep and tail in CORRIDOR_FLAGS:
                path = head
                corridor_flags.add(tail)
                continue
            break
        ckpt = Path(path)
        if not ckpt.exists():
            print(f"  skipping {name}: {ckpt} not found")
            continue
        agent, meta = load_agent(ckpt, state_dim, n_actions, run_name=name)

        # Each agent is evaluated under the sensing regime it was trained with.
        # A checkpoint trained with sensing disabled has only ever seen zeroed
        # belief features; feeding it live beliefs at test time would measure a
        # train/test mismatch rather than the ablation it is meant to isolate.
        agent_cfg = copy.deepcopy(cfg)
        agent_cfg.sensing_mode = (sensing_override
                                  or meta.get("sensing_mode", cfg.sensing_mode))
        # Same argument as the sensing mode: a policy's belief features come
        # from a particular filter, and running it against the other one
        # measures a train/test mismatch rather than the ablation intended.
        # The ':indep' flag below is the deliberate exception.
        agent_cfg.belief_mode = meta.get("belief_mode", cfg.belief_mode)
        if "noassoc" in corridor_flags:
            agent_cfg.use_association = False
        if "indep" in corridor_flags:
            # The sensing ablation: four independent per-approach filters with
            # the geometric association test, i.e. the previous design.
            agent_cfg.belief_mode = "independent"
        note = "" if agent_cfg.sensing_mode == cfg.sensing_mode else \
            f", sensing={agent_cfg.sensing_mode} as trained"
        print(f"  loaded {name} (lambda={float(agent.lam.numpy()):.3f}{note})")

        agent_sensor = AcousticSensor([], agent_cfg.acoustic,
                                      mode=agent_cfg.sensing_mode,
                                      error_model=error_model, seed=args.seed_start)
        if mode == "corridor":
            # NAME=PATH may carry corridor ablation flags after the sensing
            # override, e.g. "arc_raw=models/x.h5:nocal". Each isolates one
            # component so the table can attribute the result rather than
            # assert it.
            # Each filter has its own calibration map; see CorridorPolicy.
            cal_path = MODELS_DIR / (
                "presence_calibration_independent.json"
                if agent_cfg.belief_mode == "independent"
                else "presence_calibration.json")
            # ':adapt' re-couples the corridor's exchange rate to the
            # policy's converged dual variable. That coupling is defensible in
            # principle -- both express willingness to pay for emergency
            # delay -- and measurably wrong in practice, because lambda is
            # calibrated against the cost critic's scale rather than against
            # vehicle-seconds. It is off by default and kept as an ablation.
            # Decoupled unless ``:adapt`` is asked for. That matches
            # CorridorPolicy's own default and what the results recommend.
            # Written as ``"noadapt" not in f`` it defaulted the other way, so
            # the row reported as the *decoupled* configuration was in fact
            # scaling gamma by (1 + lambda) -- 5.6x on the constrained policy --
            # which is the coupling the paper reports as measurably worse. With
            # the flag above unrecognised as well, the two CMDP rows came out
            # as one mislabelled controller and one that never ran at all.
            factory = (lambda env, a=agent, n=name, f=corridor_flags, c=cal_path:
                       CorridorPolicy(
                           agent=a, name=n, adaptive="adapt" in f,
                           calibrate="nocal" not in f,
                           calibration_path=c,
                           scheduler=CorridorScheduler(
                               ev_value_ratio=args.value_ratio,
                               just_in_time="myopic" not in f,
                               plan_horizon=1 if "greedy" in f else 4)))
        elif mode == "overlay":
            factory = (lambda env, a=agent, n=name:
                       PriorityAwarePolicy(agent=a, name=n, adaptive=True))
        else:
            factory = (lambda env, a=agent, n=name:
                       LearnedPolicy(a, name=n, greedy=True))
        results[name] = evaluate_controller(
            factory, agent_cfg, seeds, name, net_file=net_file, sensor=agent_sensor,
            work_dir=RUNS_DIR / f"{args.regime}_{name}",
        )

    # ------------------------------------------------------------- reporting
    print("\n" + "=" * 118)
    header = f"{'controller':<24}" + "".join(f"{label:>15}" for _, label, _ in REPORT_KEYS[:6])
    print(header)
    print("-" * 118)
    for name, res in results.items():
        agg = res["aggregate"]
        row = f"{name:<24}"
        for key, _, places in REPORT_KEYS[:6]:
            mean = agg.get(f"{key}_mean", float("nan"))
            ci = agg.get(f"{key}_ci95", 0.0)
            row += f"{mean:>10.{places}f}+-{ci:<3.{places}f}"
        med = agg.get("mean_time_loss_s_median", float("nan"))
        njam = agg.get("n_jammed", 0.0)
        row += f"   med={med:6.2f} jam={njam:.0f}/{agg.get('n_runs', 0):.0f}"
        print(row)
    print("=" * 118)

    ref = args.reference
    tests = {}
    if ref in results:
        print(f"\npaired bootstrap against {ref} (negative difference = better):")
        for name, res in results.items():
            if name == ref:
                continue
            tests[name] = {}
            for key in ("mean_time_loss_s", "ev_mean_waiting_time_s"):
                a = [r.get(key, float("nan")) for r in res["runs"]]
                b = [r.get(key, float("nan")) for r in results[ref]["runs"]]
                pair = [(x, y) for x, y in zip(a, b)
                        if np.isfinite(x) and np.isfinite(y)]
                if len(pair) < 2:
                    continue
                stat = paired_bootstrap_pvalue([x for x, _ in pair], [y for _, y in pair])
                tests[name][key] = stat
                print(f"  {name:<22} {key:<24} diff={stat['diff']:+8.2f}s  "
                      f"p={stat['p_value']:.4f}")

    out_path = Path(args.out) if args.out else RESULTS_DIR / f"evaluation_{args.regime}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": {
            "regime": args.regime, "flow": args.flow, "ev_rate": args.ev_rate,
            "horizon": args.horizon, "seeds": seeds, "sensing": args.sensing,
        },
        "results": results,
        "significance": tests,
    }, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out_path}")

    csv_path = out_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["controller", "delay_median", "jam_rate", "n_jammed"]
                        + [k for k, _, _ in REPORT_KEYS]
                        + [f"{k}_ci95" for k, _, _ in REPORT_KEYS])
        for name, res in results.items():
            agg = res["aggregate"]
            writer.writerow(
                [name,
                 agg.get("mean_time_loss_s_median", ""),
                 agg.get("jam_rate", ""),
                 agg.get("n_jammed", "")]
                + [agg.get(f"{k}_mean", "") for k, _, _ in REPORT_KEYS]
                + [agg.get(f"{k}_ci95", "") for k, _, _ in REPORT_KEYS]
            )
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
