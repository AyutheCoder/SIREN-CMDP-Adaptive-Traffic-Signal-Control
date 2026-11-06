"""Episode and experiment runners.

All controllers - learned and hand-designed - are driven through the same loop
against the same seeded demand realisations, and their per-vehicle metrics come
from the same tripinfo parser. Seeds are shared across controllers so results
can be compared pairwise, which is what makes the bootstrap test in
``metrics.paired_bootstrap_pvalue`` meaningful.

Controllers are built by a factory taking the *reset* environment, because
several of them (Webster fixed timing, max-pressure) need the discovered phase
structure, which only exists once SUMO is running.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from ..config import EnvConfig, RUNS_DIR
from ..env import SumoSignalEnv
from .metrics import aggregate_runs, parse_tripinfo


@dataclass
class EpisodeResult:
    controller: str
    seed: int
    metrics: Dict[str, float] = field(default_factory=dict)
    wall_time_s: float = 0.0


def run_episode(
    env: SumoSignalEnv,
    controller_factory: Callable[[SumoSignalEnv], object],
    seed: int,
    max_decisions: int = 5000,
    on_transition: Optional[Callable] = None,
    parse_trips: bool = True,
) -> Dict[str, float]:
    """Run one full episode and return merged in-loop and per-vehicle metrics.

    ``on_transition(obs, action, reward, cost, next_obs, done, next_mask)`` is
    called after every decision; training uses it to fill the replay buffer.
    """
    obs = env.reset(seed=seed)
    controller = controller_factory(env)
    controller.reset(env)
    if getattr(controller, "program_id", None):
        env.set_program(controller.program_id)

    passive = getattr(controller, "passive", False)
    done = False
    n = 0
    while not done and n < max_decisions:
        mask = env.action_mask()
        if passive:
            action = int(env.current_phase)
            next_obs, reward, cost, done, info = env.step_passive()
        else:
            action = controller.act(env, obs, mask)
            next_obs, reward, cost, done, info = env.step(action, interrupt=controller.interrupt)
        if on_transition is not None:
            # Store the action that was *executed*, not the one proposed. The
            # safety shield -- and, when active, the priority overlay -- can
            # override the policy's choice, and Q-learning must update
            # Q(s, a_executed) against the transition that action produced.
            # Recording the proposal instead silently trains the critic on
            # outcomes another action caused.
            executed = int(info.get("phase", action))
            on_transition(obs, executed, reward, cost, next_obs, done, env.action_mask())
        obs = next_obs
        n += 1

    summary = env.episode_summary()
    summary["controller"] = getattr(controller, "name", "unknown")
    if hasattr(controller, "n_preemptions"):
        summary["n_preemptions"] = float(getattr(controller, "n_preemptions"))
    # The scheduler's own diagnostics -- how often it engaged, how much lead
    # time it left, how much of the false-preemption budget it spent -- belong
    # in the run record. Reading them off the console was how the previous
    # version's timing bug survived as long as it did.
    sched = getattr(controller, "scheduler", None)
    if sched is not None and hasattr(sched, "stats"):
        for k, v in sched.stats().items():
            summary[f"corridor_{k}"] = float(v)
    ev_ids = set(env.scenario.emergency_vehicles) if env.scenario else set()
    approach_of = dict(env.approach_direction)
    tripinfo_path = env.tripinfo_path
    env.close()

    if parse_trips and tripinfo_path is not None:
        summary.update(parse_tripinfo(tripinfo_path, approach_of, ev_ids))
    return summary


def evaluate_controller(
    controller_factory: Callable[[SumoSignalEnv], object],
    cfg: EnvConfig,
    seeds: Sequence[int],
    label: str,
    net_file: Optional[Path] = None,
    additional_files: Optional[List[Path]] = None,
    sensor: Optional[object] = None,
    work_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Run one controller over several seeds; return per-run and aggregate metrics."""
    runs: List[Dict[str, float]] = []
    work_dir = Path(work_dir or (RUNS_DIR / label))
    work_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        tripinfo = work_dir / f"tripinfo_{label}_{seed}.xml"
        env = SumoSignalEnv(
            cfg,
            net_file=net_file,
            work_dir=work_dir,
            label=f"{label}_{seed}",
            tripinfo_path=tripinfo,
            additional_files=additional_files,
            sensor=sensor,
        )
        t0 = time.time()
        metrics = run_episode(env, controller_factory, seed=seed)
        metrics["wall_time_s"] = time.time() - t0
        metrics["seed"] = seed
        runs.append(metrics)
        if verbose:
            print(
                f"  [{label}] seed={seed}  "
                f"delay={metrics.get('mean_time_loss_s', float('nan')):7.2f}s  "
                f"ev_stopped={metrics.get('ev_mean_waiting_time_s', float('nan')):6.2f}s  "
                f"ev_stops={metrics.get('ev_mean_stops', float('nan')):5.2f}  "
                f"queue={metrics.get('mean_queue', float('nan')):6.2f}  "
                f"({metrics['wall_time_s']:.1f}s)",
                flush=True,
            )

    return {"label": label, "runs": runs, "aggregate": aggregate_runs(runs)}
