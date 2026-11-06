"""The SUMO signal-control environment, formulated as a Constrained MDP.

Two things distinguish this from the usual RL-for-traffic-signals environment.

**It returns a cost as well as a reward.** ``step`` yields ``(obs, reward,
cost, done, info)``. The reward is ordinary network efficiency; the cost is
priority-weighted emergency-vehicle delay. Keeping them separate is the whole
point: a scalarised reward ``r - k*c`` forces you to pick a trade-off weight
before training and gives no guarantee about the result, whereas a CMDP lets you
state the requirement directly ("expected emergency delay at most d0") and let
the dual variable find the weight.

**It is network-agnostic.** Nothing is hard-coded about four approaches or two
phases. The controller discovers the signal's green phases and their controlled
lanes from the loaded network at reset, derives each approach's compass bearing
from lane geometry, and builds its action space from that. The same code drives
the generated intersection, a grid, or an imported OSM junction.

Phase transitions always run green -> yellow -> all-red -> green, so no two
conflicting movements can be green simultaneously by construction.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Sequence

import numpy as np

import traci
import traci.constants as tc
from sumolib import checkBinary

from .config import EnvConfig, DATA_DIR, RUNS_DIR
from .control.shield import SafetyShield
from .demand import DemandGenerator, Scenario, write_sumocfg
from .netgen import NetworkBuilder, build_network
from .sensing.association import Association, associate, temper
from .sensing.belief import BeliefTracker
from .sensing.joint_belief import JointBeliefTracker, MicArray
from .sensing.sensor import AcousticSensor, SensorErrorModel, SourceState

COMPASS = ("N", "E", "S", "W")

EV_PRIMARY_LANE_FRACTION = 0.5
"""Share of an approach's lanes a phase must green to count as *serving* it for
emergency purposes. See :meth:`SumoSignalEnv.phases_serving_approach`."""
_EMERGENCY_TYPES = ("ambulance", "fire", "police")


@dataclass
class PhaseSpec:
    """One green phase of the signal, discovered from the loaded program."""

    action_index: int
    program_index: int
    state: str
    green_link_indices: Tuple[int, ...]
    green_lanes: Tuple[str, ...]
    approaches: Tuple[str, ...]
    """Incoming edges that receive at least one green movement in this phase."""


WINDOWS_MAX_PATH = 260
"""Length at which SUMO stops being able to open a file that Python can.

Not a limit this code imposes, and not one it can raise. Python on Windows
resolves paths through the long-path API, so it writes the scenario, the route
file and the tripinfo output without complaint and ``Path.exists()`` returns
True. SUMO is a C++ binary using the legacy API, and past 260 characters it
fails with "Could not access configuration" -- naming a file that is provably
there. The contradiction is the whole difficulty: it reads as a race or a
permissions problem, and it is neither.

Run directories are the variable part, which is why ``ATSC_RUNS_DIR`` exists.
Measured here: a run directory 100 characters deep put ``max_pressure`` at 248
characters (fine) and ``max_pressure_preempt`` at 264 (broken), so the failure
selects controllers by the length of their name -- which is as confusing as it
sounds when it lands in the middle of a twelve-hour study.
"""


def _unit(vx: float, vy: float) -> Tuple[float, float]:
    n = math.hypot(vx, vy)
    if n < 1e-9:
        return (0.0, 0.0)
    return (vx / n, vy / n)


def _bearing_to_compass(dx: float, dy: float) -> str:
    """Map a direction vector to the nearest cardinal label."""
    ang = math.degrees(math.atan2(dy, dx)) % 360.0
    # atan2 gives 0 deg = east, 90 deg = north.
    if 45.0 <= ang < 135.0:
        return "N"
    if 135.0 <= ang < 225.0:
        return "W"
    if 225.0 <= ang < 315.0:
        return "S"
    return "E"


class SumoSignalEnv:
    """Single signalised intersection under acoustic emergency-vehicle sensing."""

    def __init__(
        self,
        cfg: Optional[EnvConfig] = None,
        net_file: Optional[Path] = None,
        work_dir: Optional[Path] = None,
        label: str = "atsc",
        sensor: Optional[AcousticSensor] = None,
        tripinfo_path: Optional[Path] = None,
        additional_files: Optional[List[Path]] = None,
    ):
        self.cfg = cfg or EnvConfig()
        self.additional_files = [Path(p) for p in (additional_files or [])]
        self.work_dir = Path(work_dir or (RUNS_DIR / label))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.label = label
        self.tripinfo_path = Path(tripinfo_path) if tripinfo_path else None

        if net_file is None:
            net_file, self.builder = build_network(self.cfg.network, DATA_DIR / "networks")
        else:
            net_file = Path(net_file)
            self.builder = NetworkBuilder(self.cfg.network, DATA_DIR / "networks")
        self.net_file = Path(net_file)

        self._demand = DemandGenerator(self.builder, self.cfg.demand)
        self._external_sensor = sensor

        self.conn = None
        self.emergency_call = False
        """Is a priority layer currently holding a corridor?

        Set by the controller each epoch. The shield stays *unaware* of
        emergency vehicles -- it never reads a belief -- but it does need to
        know that a call is in progress, because its gap-out rule would
        otherwise release the very green the call is holding. See
        SafetyShield.allowed_actions."""

        self.ev_request_phases: Set[int] = set()
        """Phases that would serve a standing emergency request, if any.

        Distinct from :attr:`emergency_call`, and the distinction is the whole
        point of S6. ``emergency_call`` says "a corridor is open, do not gap it
        out" -- it is about the green that is *already* running. This says "an
        emergency vehicle is waiting for one of these phases and has not been
        served", which is what the deadline is measured against. A request that
        is never granted keeps ageing; a call that is never raised has nothing
        to age.

        Set by the priority layer; consumed only through
        :meth:`ev_request_age_s`. The shield sees the ages, never the belief."""

        self._ev_request_since: Dict[int, float] = {}
        """Simulation time at which each phase's standing request was raised."""
        self._ev_request_seen: Dict[int, float] = {}
        """Simulation time each phase's request was last actually declared, so a
        dropout can be told from a release. See :meth:`set_ev_request`."""
        self._frame_deadline: Optional[float] = None
        """Wall-clock time the next simulation step is allowed to finish at.

        ``None`` outside a paced GUI run; see :meth:`_pace`."""
        self.step_hook = None
        """Optional callable invoked after every simulation step.

        The demo's painter needs to run at the simulation's frame rate, not at
        the 5 s decision epoch: a beacon that flashes once per decision is not a
        beacon, and a badge that follows its vehicle in 70 m jumps is worse than
        no badge. Nothing in the control loop uses this, and it stays ``None``
        for every batch run."""
        self._sense_accum_s = 0.0
        """Simulated seconds elapsed since the microphones were last read."""
        self._horn_until: Dict[str, float] = {}
        """Ordinary vehicles currently sounding their horn, and until when."""
        self._horn_rng = np.random.default_rng(0)
        self.last_readings: Dict[str, object] = {}
        self.scenario: Optional[Scenario] = None
        self.phases: List[PhaseSpec] = []
        self.tls_id: Optional[str] = None
        self._episode = 0
        self._built = False

    # ================================================================= startup
    def _sumo_binary(self) -> str:
        return checkBinary("sumo-gui" if self.cfg.gui else "sumo")

    def _start_sumo(self, seed: int) -> None:
        scenario = self._demand.generate(
            self.work_dir / f"routes_{self.label}.rou.xml", seed=seed,
            bluelight=self.cfg.bluelight,
        )
        self.scenario = scenario
        sumocfg = write_sumocfg(
            self.work_dir / f"scenario_{self.label}.sumocfg",
            self.net_file,
            scenario.route_file,
            step_length=self.cfg.step_length_s,
            teleport_s=self.cfg.teleport_s,
            emissions=self.cfg.collect_emissions,
            additional_files=self.additional_files,
        )
        cmd = [
            self._sumo_binary(),
            "-c", str(sumocfg),
            "--seed", str(seed),
            "--waiting-time-memory", "10000",
            "--time-to-impatience", "300",
        ]
        if self.tripinfo_path is not None:
            self.tripinfo_path.parent.mkdir(parents=True, exist_ok=True)
            cmd += ["--tripinfo-output", str(self.tripinfo_path)]
            if self.cfg.collect_emissions:
                cmd += ["--device.emissions.probability", "1.0"]
        if self.cfg.gui:
            cmd += ["--start"]
            if self.cfg.gui_quit_on_end:
                cmd += ["--quit-on-end"]
            # Passed so the window's own delay box agrees with the playback
            # rate if a viewer pauses and drives the GUI by hand. It does not
            # pace this run: while TraCI is stepping, SUMO never consults it
            # (measured -- see _pace, which is what actually holds the rate).
            if self.cfg.gui_delay_ms > 0:
                cmd += ["--delay", str(int(self.cfg.gui_delay_ms))]

        self._start_with_retry(cmd)
        self.conn = traci.getConnection(self.label)

    SUMO_START_ATTEMPTS = 3
    SUMO_START_BACKOFF_S = 0.5
    """How hard to try before giving up on launching SUMO.

    Ordinary robustness for a study that runs unattended for hours: a
    subprocess launch can fail for reasons that do not recur, and losing an
    hour-long stage to one of them is a poor trade against a second of backoff.
    It is deliberately a small number, because the failure this guard was first
    written for turned out not to be transient at all -- see
    :data:`WINDOWS_MAX_PATH`, and note that retrying a deterministic failure
    only delays the report of it.
    """

    def _start_with_retry(self, cmd: List[str]) -> None:
        # Checked *before* the first launch, not after the last failure.
        #
        # The diagnostic below existed already and was unreachable, which is
        # worse than not having it: `traci.start` does not raise on a config
        # SUMO cannot open. It launches SUMO, SUMO exits with "Could not access
        # configuration", TraCI reports "TraCI server already finished" and
        # retries on a *different port* -- dozens of times, each attempt
        # launching and losing another process. The client sits at zero CPU
        # with no SUMO alive, which reads as a hang rather than as an error,
        # and the run stalls on whichever controller first has a long enough
        # name. Measured: this cost an overnight evaluation that stopped dead
        # on its fifth controller and reported nothing at all.
        #
        # A deterministic failure must be reported before it is retried.
        cfg_path = str(cmd[cmd.index("-c") + 1]) if "-c" in cmd else ""
        if len(cfg_path) >= WINDOWS_MAX_PATH:
            raise RuntimeError(
                f"scenario path is {len(cfg_path)} characters, at or over "
                f"the Windows {WINDOWS_MAX_PATH}-character limit, so SUMO "
                f"cannot open it: {cfg_path} -- Python writes and reads such "
                "a path happily (it uses the long-path API), so the file "
                "exists and Path.exists() is True, but SUMO opens it through "
                "the legacy API and reports 'Could not access "
                "configuration'. Set ATSC_RUNS_DIR to a short directory; a "
                "folder near the drive root leaves room for every controller "
                "name."
            )
        last = None
        for attempt in range(self.SUMO_START_ATTEMPTS):
            try:
                traci.start(cmd, label=self.label)
                return
            except Exception as exc:                      # noqa: BLE001
                last = exc
                if attempt == self.SUMO_START_ATTEMPTS - 1:
                    break
                # Close anything half-open before trying again, or the retry
                # inherits a dead connection under the same label.
                try:
                    traci.close(False)
                except Exception:
                    pass
                time.sleep(self.SUMO_START_BACKOFF_S * (attempt + 1))
        hint = ""
        cfg_path = str(cmd[cmd.index("-c") + 1]) if "-c" in cmd else ""
        if len(cfg_path) >= WINDOWS_MAX_PATH:
            hint = (
                f"\n\nThe scenario path is {len(cfg_path)} characters, at or over "
                f"the Windows {WINDOWS_MAX_PATH}-character limit. Python writes "
                "and reads such a path happily -- it uses the long-path API -- so "
                "the file exists and looks fine, but SUMO opens it through the "
                "legacy API and reports 'Could not access configuration'. Set "
                "ATSC_RUNS_DIR to a short directory."
            )
        raise RuntimeError(
            f"SUMO would not start after {self.SUMO_START_ATTEMPTS} attempts "
            f"({last}); command was: {' '.join(str(c) for c in cmd)}{hint}"
        ) from last

    # ======================================================== topology discovery
    def _discover(self) -> None:
        """Read the signal program and network geometry; build the action space."""
        conn = self.conn
        tls_ids = conn.trafficlight.getIDList()
        if not tls_ids:
            raise RuntimeError("network contains no traffic lights")
        self.tls_id = tls_ids[0]

        logic = conn.trafficlight.getAllProgramLogics(self.tls_id)[0]
        links = conn.trafficlight.getControlledLinks(self.tls_id)
        self.n_links = len(links)

        # link index -> (incoming lane, outgoing lane)
        self._link_in_lane: List[Optional[str]] = []
        self._link_out_lane: List[Optional[str]] = []
        for entry in links:
            self._link_in_lane.append(entry[0][0] if entry else None)
            self._link_out_lane.append(entry[0][1] if entry else None)

        # --- controlled lanes and their parent approaches ---
        lanes: List[str] = []
        for lane in self._link_in_lane:
            if lane is not None and lane not in lanes:
                lanes.append(lane)
        self.controlled_lanes = lanes
        self.lane_to_approach = {ln: conn.lane.getEdgeID(ln) for ln in lanes}

        approaches: List[str] = []
        for ln in lanes:
            edge = self.lane_to_approach[ln]
            if edge not in approaches:
                approaches.append(edge)

        # --- approach geometry: bearing, microphone pose ---
        self.approach_direction: Dict[str, str] = {}
        self.mic_pos: Dict[str, Tuple[float, float]] = {}
        self.mic_axis: Dict[str, Tuple[float, float]] = {}
        self.stop_line: Dict[str, Tuple[float, float]] = {}
        for edge in approaches:
            lane = next(ln for ln in lanes if self.lane_to_approach[ln] == edge)
            shape = conn.lane.getShape(lane)
            start, end = shape[0], shape[-1]
            heading = _unit(end[0] - start[0], end[1] - start[1])
            # The approach is named for the direction traffic comes *from*.
            self.approach_direction[edge] = _bearing_to_compass(-heading[0], -heading[1])
            self.stop_line[edge] = (float(end[0]), float(end[1]))
            offset = min(self.cfg.acoustic.mic_offset_m, conn.lane.getLength(lane) * 0.8)
            self.mic_pos[edge] = (end[0] - heading[0] * offset, end[1] - heading[1] * offset)
            # The microphone looks back up its own approach, into oncoming traffic.
            self.mic_axis[edge] = (-heading[0], -heading[1])

        # Order approaches N, E, S, W so the observation layout is stable across
        # networks and runs.
        order = {d: i for i, d in enumerate(COMPASS)}
        self.approaches = sorted(
            approaches, key=lambda e: (order.get(self.approach_direction[e], 99), e)
        )

        # --- green phases ---
        self.phases = []
        for pi, phase in enumerate(logic.phases):
            state = phase.state
            if "y" in state.lower() or not any(c in "Gg" for c in state):
                continue
            idx = tuple(i for i, c in enumerate(state) if c in "Gg")
            plns, papproaches = [], []
            for i in idx:
                ln = self._link_in_lane[i]
                if ln is None:
                    continue
                if ln not in plns:
                    plns.append(ln)
                ap = self.lane_to_approach[ln]
                if ap not in papproaches:
                    papproaches.append(ap)
            self.phases.append(
                PhaseSpec(
                    action_index=len(self.phases),
                    program_index=pi,
                    state=state,
                    green_link_indices=idx,
                    green_lanes=tuple(plns),
                    approaches=tuple(papproaches),
                )
            )
        if len(self.phases) < 2:
            raise RuntimeError(
                f"discovered only {len(self.phases)} green phase(s) at {self.tls_id}; "
                "the network's signal program must define at least two."
            )

        self.n_actions = len(self.phases)
        self._all_red = "r" * self.n_links
        self._lane_speed = {ln: conn.lane.getMaxSpeed(ln) for ln in lanes}
        self._lane_green_mask = np.zeros((self.n_actions, len(lanes)), dtype=np.float32)
        for p in self.phases:
            for j, ln in enumerate(lanes):
                if ln in p.green_lanes:
                    self._lane_green_mask[p.action_index, j] = 1.0

        # Downstream lanes are needed for max-pressure control; subscribe to them
        # too so pressure costs no extra round-trips.
        self._downstream_lanes = sorted(
            {ln for ln in self._link_out_lane if ln is not None and ln not in lanes}
        )
        for ln in self._downstream_lanes:
            conn.lane.subscribe(ln, [tc.LAST_STEP_VEHICLE_NUMBER])

        # link index -> approach, for deriving which approaches are actually
        # green from the live signal state (see _collect_step).
        self._link_approach: List[Optional[str]] = [
            self.lane_to_approach.get(ln) if ln is not None else None
            for ln in self._link_in_lane
        ]
        conn.trafficlight.subscribe(self.tls_id, [tc.TL_RED_YELLOW_GREEN_STATE])

        # --- lane subscriptions: one round-trip per step instead of 4*n_lanes ---
        for ln in lanes:
            conn.lane.subscribe(ln, [
                tc.LAST_STEP_VEHICLE_NUMBER,
                tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
                tc.LAST_STEP_MEAN_SPEED,
                tc.VAR_WAITING_TIME,
            ])
        conn.simulation.subscribe([
            tc.VAR_DEPARTED_VEHICLES_IDS,
            tc.VAR_ARRIVED_VEHICLES_IDS,
            tc.VAR_TELEPORT_STARTING_VEHICLES_NUMBER,
            tc.VAR_TIME,
        ])
        self._built = True

    def _yellow_state(self, from_phase: PhaseSpec, to_phase: PhaseSpec) -> str:
        """Movements losing right-of-way go amber; continuing greens are kept."""
        chars = []
        to_green = set(to_phase.green_link_indices)
        for i, c in enumerate(from_phase.state):
            if c in "Gg":
                chars.append("G" if i in to_green else "y")
            else:
                chars.append("r")
        return "".join(chars)

    # ==================================================================== reset
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        self.close()
        episode_seed = self.cfg.seed + self._episode if seed is None else seed
        self._episode += 1
        self._start_sumo(episode_seed)
        self._discover()

        cfg = self.cfg
        self.shield = SafetyShield(
            n_phases=self.n_actions,
            min_green_s=cfg.control.min_green_s,
            max_green_s=cfg.control.max_green_s,
            yellow_s=cfg.control.yellow_s,
            all_red_s=cfg.control.all_red_s,
            decision_interval_s=cfg.control.decision_interval_s,
            max_starvation_s=cfg.control.max_starvation_s,
            gap_out_queue=cfg.control.gap_out_queue,
            gap_out_min_waiting=cfg.control.gap_out_min_waiting,
            gap_out_max_load=cfg.control.gap_out_max_load,
            gridlock_queue=cfg.control.gridlock_queue,
            gridlock_epochs=cfg.control.gridlock_epochs,
            gridlock_clear_queue=cfg.control.gridlock_clear_queue,
            gridlock_min_discharge=cfg.control.gridlock_min_discharge,
            congestion_queue=cfg.control.congestion_queue,
            congested_max_green_s=cfg.control.congested_max_green_s,
            emergency_min_green_s=cfg.control.ev_min_green_s,
            ev_service_deadline_s=cfg.control.ev_service_deadline_s,
            ev_poll_interval_s=cfg.control.ev_poll_interval_s,
        )
        self.sensor = self._external_sensor or AcousticSensor(
            self.approaches, cfg.acoustic, mode=cfg.sensing_mode, seed=episode_seed
        )
        self.sensor.approaches = self.approaches
        self.sensor.reset_stats()
        prior = getattr(self.sensor.error_model, "detector_prior", 0.5)
        if cfg.belief_mode == "joint":
            # One filter over "which arm holds the siren", not four filters that
            # each answer "is a siren audible here?" and all say yes. See
            # atsc/sensing/joint_belief.py.
            self.mic_array = MicArray(
                approaches=list(self.approaches),
                stop_line=dict(self.stop_line),
                pos=dict(self.mic_pos),
                axis=dict(self.mic_axis),
                front_back_db=cfg.acoustic.directivity_front_back_db,
                dual_element=bool(cfg.acoustic.dual_element_mast),
                detection_radius_m=cfg.acoustic.detection_radius_m,
                # The state space spans the arm, not the detection radius: a
                # vehicle 330 m from the stop line is only 260 m from the mast
                # and is plainly audible, so a grid that stops at 300 m has
                # nowhere to put it and forces the mass onto a closer hypothesis.
                max_range_m=float(cfg.network.arm_length_m),
            )
            # The filter's likelihood must describe the sensor that is
            # actually running, not the one it was written for.
            if cfg.sensing_mode == "oracle":
                em, gate = SensorErrorModel.oracle(), "on_arm"
            else:
                em, gate = self.sensor.error_model, "acoustic"
            self.belief = JointBeliefTracker(
                self.approaches,
                cfg.belief,
                priority_map=cfg.demand.ev_priority,
                detector_prior=prior,
                mics=self.mic_array,
                error_model=em,
                acoustic=cfg.acoustic,
                detection_gate=gate,
            )
        else:
            self.mic_array = None
            offsets = [
                float(np.hypot(self.mic_pos[a][0] - self.stop_line[a][0],
                               self.mic_pos[a][1] - self.stop_line[a][1]))
                for a in self.approaches]
            self.belief = BeliefTracker(
                self.approaches,
                cfg.belief,
                priority_map=cfg.demand.ev_priority,
                detector_prior=prior,
                stop_line_offset_m=float(np.mean(offsets)) if offsets else 0.0,
            )

        # A filter that is not told which masts have failed treats their
        # silence as evidence that nothing is inbound near them. See
        # JointBeliefTracker.disabled_mics.
        dead = set(getattr(self.sensor, "dead_approaches", None) or ())
        if dead and hasattr(self.belief, "disabled_mics"):
            self.belief.disabled_mics = dead

        self.emergency_call = False
        self.ev_request_phases = set()
        self._ev_request_since = {}
        self._ev_request_seen = {}
        # Horn presses are part of the scene, so they are reseeded per episode
        # from the demand seed: two runs of the same seed hear the same horns,
        # which is what makes the paired comparisons paired.
        self._horn_until = {}
        self._horn_rng = np.random.default_rng(int(self.cfg.demand.seed) ^ 0x484F524E)
        self.current_phase = 0
        self.green_elapsed_s = 0.0
        self.time_since_served = np.zeros(self.n_actions, dtype=np.float64)
        self.sim_time = 0.0
        self._ev_waiting = {}
        self._ev_speed = {}
        self._ev_seen: Set[str] = set()
        self._done = False
        # Start the playback clock fresh: episode setup takes far longer than
        # one frame, and a deadline left over from it would be spent catching
        # up at full speed for the first seconds of the new episode.
        self._frame_deadline = None
        self._serving_approaches: Set[str] = set()
        self._step_queue = self._step_veh = self._step_wait = 0.0
        self._epoch_arrived = 0
        self._epoch_ev_delay = 0.0
        self._epoch_ev_weighted_delay = 0.0
        self._epoch_ev_stops = 0.0
        self._epoch_ev_unserved = 0.0
        self._sense_accum_s = 0.0
        self.last_readings = {}
        self._reset_episode_metrics()

        self.conn.trafficlight.setRedYellowGreenState(self.tls_id, self.phases[0].state)
        self._advance(cfg.control.min_green_s, green=True)
        self.green_elapsed_s = cfg.control.min_green_s
        self._sense_flush()
        return self._observe()

    def _reset_episode_metrics(self) -> None:
        self.metrics = {
            "steps": 0,
            "decisions": 0,
            "sum_queue": 0.0,
            "sum_vehicles": 0.0,
            "sum_wait": 0.0,
            "max_queue": 0.0,
            "arrived": 0,
            "departed": 0,
            "teleports": 0,
            "phase_switches": 0,
            "interrupts": 0,
            "reward": 0.0,
            "cost": 0.0,
            "ev_delay_s": 0.0,
            "ev_stops": 0,
            "ev_weighted_delay": 0.0,
            "ev_unserved_s": 0.0,
            "max_movement_wait_s": 0.0,
        }

    # ===================================================================== step
    def _advance(self, seconds: float, green: bool = False) -> None:
        """Run the simulation forward, accumulating per-step statistics.

        ``green`` says whether the current phase is actually showing green, as
        opposed to running a changeover. Starvation clocks are advanced here,
        one simulated second at a time, rather than in bulk at the end of a
        decision epoch, because the corridor scheduler may now cut an epoch
        short (see :meth:`_hold_green`) and the shield's S3 guarantee has to
        stay exact when it does. Charging every phase for changeover time --
        including the one about to be served -- is conservative relative to the
        bound that was proved, so the guarantee is if anything tighter.
        """
        n = max(int(round(seconds / self.cfg.step_length_s)), 1)
        conn = self.conn
        dt = self.cfg.step_length_s
        for _ in range(n):
            if conn.simulation.getMinExpectedNumber() <= 0:
                self._done = True
                return
            conn.simulationStep()
            self._collect_step()
            self._sense_tick(dt)
            self.time_since_served += dt
            if green:
                self.time_since_served[self.current_phase] = 0.0
                # S6's clock measures time *unserved*, so a request whose phase
                # is actually showing green is satisfied and its deadline
                # restarts. Without this the age would keep climbing while the
                # vehicle was being served, and the shield would go on forcing
                # a phase it had already granted -- pinning the green until the
                # controller happened to drop the request.
                #
                # The whole request is satisfied, not just the green phase. A
                # request names every phase that serves one approach -- usually
                # two, the through movement and the protected left -- and
                # serving either of them serves the approach and the vehicle on
                # it. Resetting only the green one leaves its sibling ageing
                # while the vehicle is being released, so the sibling comes due
                # and S6 forces a switch from a phase that is already carrying
                # the emergency vehicle to another phase on the same approach,
                # costing a full changeover to arrive where it already was.
                if self.current_phase in self.ev_request_phases:
                    for p in self.ev_request_phases:
                        self._ev_request_since.pop(p, None)
                else:
                    self._ev_request_since.pop(self.current_phase, None)
            self.metrics["max_movement_wait_s"] = max(
                self.metrics["max_movement_wait_s"],
                float(self.time_since_served.max()))
            self._pace()

    def _pace(self) -> None:
        """Hold a GUI run to the playback rate that was asked for.

        SUMO's own ``--delay`` paces *its* run loop -- the one behind the play
        button. When TraCI is driving, each step is executed as soon as the
        client asks for it and that delay is never consulted, so the picture
        runs at whatever rate the machine can manage. Measured on this network:
        asking for ``--delay 100`` with a 0.05 s step, SUMO returned steps in
        15 ms, i.e. 3.3x real time where 0.5x was requested -- the delay was
        not applied at all. That is the "vehicles move too fast" failure, and
        ``--delay`` cannot fix it because it is not being read. Sleeping here is the only thing in the loop that actually
        controls playback speed, so the flag is kept (it still paces the window
        if a viewer pauses and uses the GUI's own controls) and this is what
        the demo relies on.

        The deadline advances by a fixed budget from the previous deadline
        rather than from the moment this returns, so the time spent stepping
        SUMO and painting the overlay comes *out* of the frame's budget instead
        of being added on top of it, and the rate does not drift slow. When a
        frame overruns its budget the deadline is reset to now rather than
        allowed to fall behind: repaying a hitch by running the next frames
        with no sleep produces exactly the burst of too-fast motion this
        exists to remove.
        """
        budget = self.cfg.gui_delay_ms / 1000.0
        if not self.cfg.gui or budget <= 0.0:
            return
        now = time.monotonic()
        deadline = self._frame_deadline
        if deadline is None:
            self._frame_deadline = now + budget
            return
        remaining = deadline - now
        if remaining > 0.0:
            time.sleep(remaining)
            self._frame_deadline = deadline + budget
        else:
            self._frame_deadline = now + budget

    def _changeover(self, to_phase: int) -> None:
        """Amber, all-red, then green on ``to_phase``, with the clocks kept."""
        ctrl = self.cfg.control
        frm, to = self.phases[self.current_phase], self.phases[int(to_phase)]
        self.conn.trafficlight.setRedYellowGreenState(
            self.tls_id, self._yellow_state(frm, to))
        self._advance(ctrl.yellow_s)
        if self._done:
            return
        self.conn.trafficlight.setRedYellowGreenState(self.tls_id, self._all_red)
        self._advance(ctrl.all_red_s)
        if self._done:
            return
        self.conn.trafficlight.setRedYellowGreenState(self.tls_id, to.state)
        self.current_phase = int(to_phase)
        self.green_elapsed_s = 0.0
        self.metrics["phase_switches"] += 1

    def _hold_green(self, seconds: float, interrupt=None) -> None:
        """Hold the current green for ``seconds``, pollable once per second.

        This is the dual-rate part of the controller. The learned policy plans
        on a 5 s decision epoch, which is the right granularity for queue
        management and the granularity it was trained at, but an emergency
        vehicle closing at 14 m/s covers 70 m in that time. Waiting for the next
        epoch boundary was costing up to a full interval of pure latency on
        every preemption -- with the changeover on top, up to 10 s, which is
        most of the delay a corridor exists to remove.

        So the corridor scheduler is polled every simulated second and may cut
        the hold short. It cannot do anything the shield would not allow at a
        decision epoch: ``interrupt`` is honoured only if the shield's own mask
        permits the phase, so minimum green, maximum green and the starvation
        bound are enforced identically whether a switch originates from the
        policy or from an interrupt.
        """
        dt = self.cfg.step_length_s
        remaining = float(seconds)
        while remaining > 1e-9 and not self._done:
            self._advance(dt, green=True)
            self.green_elapsed_s += dt
            remaining -= dt
            if interrupt is None or self._done or remaining <= 1e-9:
                continue
            want = interrupt(self)
            if want is None:
                continue
            want = int(want)
            if want == self.current_phase:
                continue
            # An interrupt only ever originates from a committed emergency
            # call, so the reduced minimum green applies to it.
            mask = self.shield.mask(self.current_phase, self.green_elapsed_s,
                                    self.time_since_served, self.phase_queues(),
                                    self._recent_discharge(), emergency=True,
                                    ev_request_age_s=self.ev_request_age_s())
            if not mask[want]:
                continue
            self._changeover(want)
            self.metrics["interrupts"] += 1

    def _collect_step(self) -> None:
        conn = self.conn
        sub = conn.lane.getAllSubscriptionResults()
        queue = veh = wait = 0.0
        for ln in self.controlled_lanes:
            d = sub.get(ln, {})
            queue += d.get(tc.LAST_STEP_VEHICLE_HALTING_NUMBER, 0)
            veh += d.get(tc.LAST_STEP_VEHICLE_NUMBER, 0)
            wait += d.get(tc.VAR_WAITING_TIME, 0.0)
        self._step_queue = queue
        self._step_veh = veh
        self._step_wait = wait

        m = self.metrics
        m["steps"] += 1
        m["sum_queue"] += queue
        m["sum_vehicles"] += veh
        m["sum_wait"] += wait
        m["max_queue"] = max(m["max_queue"], queue)

        # Which approaches are green *right now*, read from the live signal
        # state rather than inferred from self.current_phase. During a switching
        # epoch current_phase still holds the outgoing phase while the incoming
        # green is already running, and during yellow/all-red no phase is
        # serving at all. Attributing emergency exposure to the wrong phase
        # silently un-credits the agent for the exact action the constraint is
        # meant to encourage, so this has to come from the signal itself.
        tl_sub = conn.trafficlight.getAllSubscriptionResults().get(self.tls_id, {})
        state = tl_sub.get(tc.TL_RED_YELLOW_GREEN_STATE)
        if state:
            self._serving_approaches = {
                ap for i, ch in enumerate(state)
                if ch in "Gg" and i < len(self._link_approach)
                and (ap := self._link_approach[i]) is not None
            }
        else:
            self._serving_approaches = set()

        sim = conn.simulation.getAllSubscriptionResults().get("", {})
        arrived = sim.get(tc.VAR_ARRIVED_VEHICLES_IDS, ())
        departed = sim.get(tc.VAR_DEPARTED_VEHICLES_IDS, ())
        m["arrived"] += len(arrived)
        m["departed"] += len(departed)
        # SUMO removes vehicles that have been stuck past --time-to-teleport and
        # reinserts them downstream. That silently rescues throughput in exactly
        # the gridlocked runs whose failure we most need to see, so the count is
        # tracked and reported rather than assumed to be zero.
        m["teleports"] += int(sim.get(tc.VAR_TELEPORT_STARTING_VEHICLES_NUMBER, 0))
        self.sim_time = conn.simulation.getTime()

        self._epoch_arrived += len(arrived)
        self._track_emergency(set(arrived))

        if self.step_hook is not None:
            self.step_hook(self)

    # ------------------------------------------------------- emergency tracking
    def _emergency_ids(self) -> List[str]:
        return list(self._ev_waiting.keys())

    def _track_emergency(self, arrived: Set[str]) -> None:
        """Accrue priority-weighted delay for every emergency vehicle on the net.

        Two quantities are accumulated. The *realised* terms (waiting time,
        stops) record what actually happened to the vehicle. The *anticipatory*
        term records, at every simulation step, that a prioritised vehicle is
        inbound on an approach the signal is currently not serving, weighted by
        how close it is. Only the second is available early enough for the
        controller to act on; see ``RewardConfig.w_ev_unserved``.
        """
        conn = self.conn
        scenario = self.scenario
        assert scenario is not None

        present = set(conn.vehicle.getIDList()) if scenario.emergency_vehicles else set()
        active = present & set(scenario.emergency_vehicles.keys())

        served_approaches = self._serving_approaches
        ramp = max(self.cfg.reward.ev_imminence_range_m, 1.0)
        dt = self.cfg.step_length_s

        for vid in active:
            ev = scenario.emergency_vehicles[vid]
            try:
                wait = conn.vehicle.getAccumulatedWaitingTime(vid)
                speed = conn.vehicle.getSpeed(vid)
                edge = conn.vehicle.getRoadID(vid)
                lane_pos = conn.vehicle.getLanePosition(vid)
                lane_id = conn.vehicle.getLaneID(vid)
            except traci.TraCIException:
                continue

            # Anticipatory term: inbound on an approach that is not green.
            if edge in self.approaches and edge not in served_approaches:
                try:
                    remaining = max(conn.lane.getLength(lane_id) - lane_pos, 0.0)
                except traci.TraCIException:
                    remaining = ramp
                imminence = float(np.clip(1.0 - remaining / ramp, 0.0, 1.0))
                self._epoch_ev_unserved += ev.priority * imminence * dt
                self.metrics["ev_unserved_s"] += ev.priority * imminence * dt
            prev_wait = self._ev_waiting.get(vid, wait)
            prev_speed = self._ev_speed.get(vid, speed)
            delta = max(wait - prev_wait, 0.0)

            self._epoch_ev_delay += delta
            self._epoch_ev_weighted_delay += delta * ev.priority
            self.metrics["ev_delay_s"] += delta
            self.metrics["ev_weighted_delay"] += delta * ev.priority

            if prev_speed > 0.3 and speed <= 0.1:
                self._epoch_ev_stops += ev.priority
                self.metrics["ev_stops"] += 1

            self._ev_waiting[vid] = wait
            self._ev_speed[vid] = speed
            self._ev_seen.add(vid)

        for vid in arrived:
            self._ev_waiting.pop(vid, None)
            self._ev_speed.pop(vid, None)

    # ------------------------------------------------------------------ sensing
    def _acoustic_scene(self) -> Dict[str, List[SourceState]]:
        """Everything the masts can hear: emergency sirens *and* ordinary horns.

        The horns are not decoration. A road horn is the one urban sound
        genuinely confusable with a siren -- 100-112 dB at 1 m against a siren's
        109-116, harmonic, with its tones sitting on the low end of the siren
        band -- and until it was added here the detector had never been asked to
        reject one. Evaluated at 30 m, a horn reaches the mast at 23 dB SNR
        against an ambulance's 15.5 dB, so **the confuser is the louder signal**
        and no level threshold separates them.
        """
        conn = self.conn
        scenario = self.scenario
        assert scenario is not None
        scene: Dict[str, List[SourceState]] = {a: [] for a in self.approaches}
        self._emit_horns(scene)
        if not scenario.emergency_vehicles:
            return scene

        present = set(conn.vehicle.getIDList())
        active = present & set(scenario.emergency_vehicles.keys())
        if not active:
            return scene

        fb = self.cfg.acoustic.directivity_front_back_db
        dual = bool(self.cfg.acoustic.dual_element_mast)
        for vid in active:
            ev = scenario.emergency_vehicles[vid]
            try:
                px, py = conn.vehicle.getPosition(vid)
                speed = conn.vehicle.getSpeed(vid)
                angle = conn.vehicle.getAngle(vid)
            except traci.TraCIException:
                continue
            # SUMO reports heading in degrees clockwise from north.
            rad = math.radians(angle)
            vx, vy = speed * math.sin(rad), speed * math.cos(rad)

            for approach in self.approaches:
                mx, my = self.mic_pos[approach]
                ax, ay = self.mic_axis[approach]
                dx, dy = px - mx, py - my
                dist = math.hypot(dx, dy)
                if dist < 1.0:
                    dist = 1.0
                # Range rate: positive when the separation is growing.
                v_r = (dx * vx + dy * vy) / dist
                ux, uy = dx / dist, dy / dist
                cos_off = max(-1.0, min(1.0, ux * ax + uy * ay))
                scene[approach].append(
                    SourceState(
                        veh_class=ev.veh_class,
                        distance_m=dist,
                        radial_velocity_ms=v_r,
                        off_axis_rad=math.acos(cos_off),
                        front_back_db=fb,
                        dual_element=dual,
                    )
                )
        return scene

    def _emit_horns(self, scene: Dict[str, List[SourceState]]) -> None:
        """Sound the horns of frustrated drivers, and add them to the scene.

        Honking is modelled per unit of *queued* time rather than per vehicle or
        per distance travelled, because that is what it is a function of: nobody
        sounds a horn on an open road. A stopped ordinary vehicle presses at
        ``AcousticConfig.horn_rate_per_veh_hour``, so the horn rate rises and
        falls with congestion -- which means the detector faces its hardest
        negatives exactly when the intersection is busiest, and an emergency
        vehicle arriving into a jam arrives into a chorus.

        Presses are held for a sampled duration rather than sounded for a single
        step, so a press spans several sensing frames the way a real one does.
        """
        cfg = self.cfg.acoustic
        rate = float(getattr(cfg, "horn_rate_per_veh_hour", 0.0))
        if rate <= 0.0:
            return
        conn = self.conn
        now = float(self.sim_time)
        dt = float(self.cfg.step_length_s)
        scenario = self.scenario
        ev_ids = set(scenario.emergency_vehicles) if scenario else set()

        # Retire finished presses first, so a vehicle can honk again later.
        for vid in [v for v, until in self._horn_until.items() if until <= now]:
            del self._horn_until[vid]

        try:
            halting = [v for v in conn.vehicle.getIDList() if v not in ev_ids]
        except traci.TraCIException:
            return

        p_press = rate * dt / 3600.0
        for vid in halting:
            if vid in self._horn_until:
                continue
            try:
                if conn.vehicle.getSpeed(vid) > 0.5:
                    continue
                edge = conn.vehicle.getRoadID(vid)
            except traci.TraCIException:
                continue
            if edge not in self.approaches:
                continue
            if self._horn_rng.random() < p_press:
                self._horn_until[vid] = now + float(self._horn_rng.uniform(0.3, 1.6))

        if not self._horn_until:
            return

        fb = cfg.directivity_front_back_db
        dual = bool(cfg.dual_element_mast)
        for vid in list(self._horn_until):
            try:
                px, py = conn.vehicle.getPosition(vid)
            except traci.TraCIException:
                del self._horn_until[vid]
                continue
            for approach in self.approaches:
                mx, my = self.mic_pos[approach]
                ax, ay = self.mic_axis[approach]
                dx, dy = px - mx, py - my
                dist = max(math.hypot(dx, dy), 1.0)
                ux, uy = dx / dist, dy / dist
                cos_off = max(-1.0, min(1.0, ux * ax + uy * ay))
                scene[approach].append(SourceState(
                    veh_class="horn",
                    distance_m=dist,
                    # A honking vehicle is, by construction, stopped.
                    radial_velocity_ms=0.0,
                    off_axis_rad=math.acos(cos_off),
                    front_back_db=fb,
                    dual_element=dual,
                ))

    def _sense_tick(self, dt: float) -> None:
        """Advance the sensing clock by ``dt`` of simulated time.

        Called once per simulation step, from :meth:`_advance`, so the
        microphones are read on their own schedule rather than on the
        controller's. The two were the same thing until now, and that is why
        the corridor's 1 Hz re-check was re-pricing a posterior that could not
        have changed: the belief was only ever advanced at the epoch boundary,
        so within an epoch the "closeness and loudness" the decision turns on
        were frozen while the vehicle covered up to seventy metres.

        Sensing also runs *through the changeover*. Yellow plus all-red is five
        seconds during which the vehicle keeps approaching, and a filter that
        is blind for those five seconds comes out of them with a stale range.
        """
        if self.cfg.sensing_mode == "none":
            return
        interval = float(self.cfg.sensing_interval_s)
        if interval <= 0.0:
            self._sense(dt)
            return
        self._sense_accum_s += float(dt)
        # A step longer than the frame interval (batch runs use a 1 s step and
        # the demo 0.05 s) is sensed once for the whole of it rather than
        # looped, so the number of measurements never exceeds the number of
        # frames the detector could actually have produced.
        if self._sense_accum_s + 1e-9 >= interval:
            elapsed, self._sense_accum_s = self._sense_accum_s, 0.0
            self._sense(elapsed)

    def _sense_flush(self) -> None:
        """Consume any part-frame so the policy's observation is current."""
        if self.cfg.sensing_mode == "none" or self._sense_accum_s <= 1e-9:
            return
        elapsed, self._sense_accum_s = self._sense_accum_s, 0.0
        self._sense(elapsed)

    def _sense(self, dt: Optional[float] = None) -> None:
        """One predict/update cycle of the belief filter over ``dt`` seconds."""
        if self.cfg.sensing_mode == "none":
            return
        elapsed = float(dt if dt is not None else self.cfg.control.decision_interval_s)
        self.belief.predict(elapsed)
        readings = self.sensor.sense(self._acoustic_scene(), dt=elapsed)
        self.last_readings = readings
        if self.cfg.belief_mode == "joint":
            # The joint filter consumes every mast at once: cross-talk is only
            # explainable as one source heard from several places, which is
            # information the per-approach update throws away by construction.
            self.belief.update_joint(readings)
            self.last_association = Association(
                {a: 1.0 for a in readings}, None, {a: None for a in readings})
            return
        # Cross-talk between the four masts is a consistency constraint, not
        # noise: a reading a nearer microphone already explains geometrically
        # carries no independent evidence. Without this every approach convinces
        # itself of the same siren -- measured at 21%% of approach-epochs with
        # presence > 0.9 and no vehicle inbound. See atsc/sensing/association.py.
        if self.cfg.use_association:
            assoc = associate(readings, self.mic_pos, self.mic_axis)
        else:
            assoc = Association({a: 1.0 for a in readings}, None,
                                {a: None for a in readings})
        self.last_association = assoc
        for approach, r in readings.items():
            self.belief.update(
                approach,
                temper(r.p_siren, assoc.weights.get(approach, 1.0),
                       self.belief._prior_odds_correction),
                r.class_probs,
                range_measurement_m=r.distance_m,
                range_measurement_var=r.range_var,
                rate_hint_ms=r.radial_velocity_ms,
            )

    # ------------------------------------------------------------- observation
    def _observe(self) -> np.ndarray:
        conn = self.conn
        sub = conn.lane.getAllSubscriptionResults()
        feats: List[float] = []
        for ln in self.controlled_lanes:
            d = sub.get(ln, {})
            n_veh = d.get(tc.LAST_STEP_VEHICLE_NUMBER, 0)
            halt = d.get(tc.LAST_STEP_VEHICLE_HALTING_NUMBER, 0)
            speed = d.get(tc.LAST_STEP_MEAN_SPEED, 0.0)
            wait = d.get(tc.VAR_WAITING_TIME, 0.0)
            vmax = max(self._lane_speed.get(ln, 13.89), 1e-3)
            feats.extend([
                min(halt / 10.0, 2.0),
                min(n_veh / 10.0, 2.0),
                float(np.clip(speed / vmax, 0.0, 1.0)),
                min(wait / 60.0, 3.0),
            ])
        feats.extend(self._lane_green_mask[self.current_phase].tolist())

        onehot = [0.0] * self.n_actions
        onehot[self.current_phase] = 1.0
        feats.extend(onehot)
        feats.append(min(self.green_elapsed_s / self.cfg.control.max_green_s, 2.0))
        feats.extend(
            (self.time_since_served / self.cfg.control.max_starvation_s).clip(0.0, 2.0).tolist()
        )

        n_belief = len(self.approaches) * BeliefTracker.N_FEATURES_PER_APPROACH
        if self.cfg.sensing_mode == "none":
            feats.extend([0.0] * (n_belief + self.n_actions))
        else:
            feats.extend(self.belief.feature_vector(
                self.cfg.acoustic.detection_radius_m).tolist())
            # Beliefs are per *approach*, but actions are per *phase*. Supplying
            # the phase-level aggregate directly means the policy does not have
            # to rediscover which phase serves which approach from reward alone
            # -- a mapping that is fixed, already known to the environment, and
            # expensive to learn from a signal as sparse as emergency delay.
            feats.extend(self.phase_urgency().tolist())

        return np.asarray(feats, dtype=np.float32)

    def phase_urgency(self) -> np.ndarray:
        """Per-phase emergency urgency: how much this phase is needed, and how soon.

        For each phase, the maximum over the approaches it serves of
        ``presence * (1 - eta/eta_max) * priority``. Taking the maximum rather
        than the sum is deliberate: two emergency vehicles on one phase are not
        twice as urgent as one, but a phase serving any imminent vehicle is
        maximally urgent.
        """
        out = np.zeros(self.n_actions, dtype=np.float32)
        if self.cfg.sensing_mode == "none":
            return out
        max_eta = self.cfg.belief.max_eta_s
        radius = self.cfg.acoustic.detection_radius_m
        beliefs = self.belief.beliefs()
        for phase in self.phases:
            best = 0.0
            for approach in phase.approaches:
                b = beliefs.get(approach)
                if b is None:
                    continue
                # Imminence falls back on range when the arrival estimate is
                # uninformative, which it is for every vehicle stopped at a red
                # -- the estimate divides by closing speed. See
                # ApproachBelief.imminence: without the fallback this feature
                # was identically zero exactly when the phase was most needed.
                best = max(best, b.presence * b.imminence(max_eta, radius)
                           * b.expected_priority)
            out[phase.action_index] = best
        return out

    @property
    def observation_dim(self) -> int:
        n_lanes = len(self.controlled_lanes)
        return (
            n_lanes * 5                                                   # per-lane traffic
            + self.n_actions * 2                                          # phase one-hot + starvation clocks
            + 1                                                           # green elapsed
            + len(self.approaches) * BeliefTracker.N_FEATURES_PER_APPROACH  # per-approach belief
            + self.n_actions                                              # per-phase urgency
        )

    # -------------------------------------------------- emergency service (S6)
    def set_ev_request(self, phases: Optional[Sequence[int]]) -> None:
        """Declare (or clear) the standing emergency request the shield must meet.

        Called by the priority layer every poll. Two rules, and each closes a way
        the deadline could otherwise be evaded:

        **Re-declaring a live request does not reset its age.** Phases already
        requested keep the timestamp they were raised with. Otherwise a layer
        that simply keeps asking would postpone the deadline indefinitely and
        S6 would be vacuous.

        **A request survives a short dropout.** The posterior flickers -- it is
        a filtered detection, and at close range the filter routinely moves its
        mass past the stop line for an epoch and back again. Deleting the
        request the instant it dips below the floor resets its age, so the exact
        behaviour S6 exists to bound (a vehicle waiting while the evidence comes
        and goes) would restart the clock on every flicker and the deadline
        would never be reached. The request is therefore held for
        ``ev_request_grace_s`` past its last declaration, and only then dropped.
        This is the same dwell deployed preemption controllers apply, and for
        the same reason.
        """
        want = {int(p) for p in (phases or ()) if 0 <= int(p) < self.n_actions}
        now = float(self.sim_time)
        grace = float(self.cfg.control.ev_request_grace_s)
        for p in want:
            self._ev_request_since.setdefault(p, now)
            self._ev_request_seen[p] = now
        for p in list(self._ev_request_since):
            if p in want:
                continue
            if now - self._ev_request_seen.get(p, now) >= grace:
                del self._ev_request_since[p]
                self._ev_request_seen.pop(p, None)
            else:
                want.add(p)
        self.ev_request_phases = want

    def clear_ev_request(self, phase: Optional[int] = None) -> None:
        """Mark a request served. Called when its phase actually goes green."""
        if phase is None:
            self._ev_request_since.clear()
            self._ev_request_seen.clear()
            self.ev_request_phases = set()
            return
        self._ev_request_since.pop(int(phase), None)
        self._ev_request_seen.pop(int(phase), None)
        self.ev_request_phases.discard(int(phase))

    def ev_request_age_s(self) -> np.ndarray:
        """Seconds each phase's standing emergency request has gone unserved."""
        ages = np.zeros(self.n_actions, dtype=np.float64)
        now = float(self.sim_time)
        for p, t0 in self._ev_request_since.items():
            ages[p] = max(0.0, now - t0)
        return ages

    # ------------------------------------------------------------- action mask
    def action_mask(self, emergency: Optional[bool] = None) -> np.ndarray:
        # Queues are supplied so the shield can apply its gap-out rule (S4) and
        # scale the green ceiling with congestion; without them it falls back to
        # a fixed ceiling, which is what let adaptive controllers spill back
        # under saturated demand.
        if emergency is None:
            emergency = bool(self.emergency_call)
        return self.shield.mask(self.current_phase, self.green_elapsed_s,
                                self.time_since_served, self.phase_queues(),
                                self._recent_discharge(), emergency,
                                self.ev_request_age_s())

    # ------------------------------------------------------------------- step
    def step(self, action: int,
             interrupt=None) -> Tuple[np.ndarray, float, float, bool, Dict]:
        if not self._built:
            raise RuntimeError("call reset() before step()")
        cfg = self.cfg
        ctrl = cfg.control
        self._done = False
        self._epoch_arrived = 0
        self._epoch_ev_delay = 0.0
        self._epoch_ev_weighted_delay = 0.0
        self._epoch_ev_stops = 0.0
        self._epoch_ev_unserved = 0.0

        report = self.shield.project(
            int(action), self.current_phase, self.green_elapsed_s,
            self.time_since_served, self.phase_queues(),
            self._recent_discharge(), emergency=bool(self.emergency_call),
            ev_request_age_s=self.ev_request_age_s(),
        )
        applied = int(report.applied)

        start_steps = self.metrics["steps"]

        switched = applied != self.current_phase
        if switched:
            self._changeover(applied)
        if not self._done:
            self._hold_green(ctrl.decision_interval_s, interrupt)

        elapsed_steps = max(self.metrics["steps"] - start_steps, 1)
        dt = elapsed_steps * cfg.step_length_s

        # Sensing already ran inside _advance, once per detector frame. All
        # that is left here is to flush any part-frame so the observation the
        # policy sees is not older than the frame interval.
        self._sense_flush()
        obs = self._observe()

        reward = self._reward(switched)
        cost = self._cost()
        self.metrics["reward"] += reward
        self.metrics["cost"] += cost
        self.metrics["decisions"] += 1

        done = self._done or self.sim_time >= cfg.demand.horizon_s + 900
        info = {
            "sim_time": self.sim_time,
            "phase": self.current_phase,
            "shield_overridden": report.overridden,
            "shield_reason": report.forced_reason,
            "queue": self._step_queue,
            "dt": dt,
            "ev_delay": self._epoch_ev_delay,
            "ev_active": len(self._ev_waiting),
            "arrived": self._epoch_arrived,
            "interrupts": self.metrics["interrupts"],
        }
        return obs, reward, cost, done, info

    def step_passive(self) -> Tuple[np.ndarray, float, float, bool, Dict]:
        """Advance one decision interval without touching the signal.

        Used by the SUMO-native actuated baseline, where SUMO's own gap-out
        logic drives the signal. Metrics, sensing and cost accounting are
        identical to :meth:`step`, so the comparison stays like-for-like.
        """
        cfg = self.cfg
        self._done = False
        self._epoch_arrived = 0
        self._epoch_ev_delay = 0.0
        self._epoch_ev_weighted_delay = 0.0
        self._epoch_ev_stops = 0.0
        self._epoch_ev_unserved = 0.0

        start_steps = self.metrics["steps"]
        self._advance(cfg.control.decision_interval_s)
        elapsed_steps = max(self.metrics["steps"] - start_steps, 1)
        dt = elapsed_steps * cfg.step_length_s

        # Infer which phase SUMO chose so the served/starved clocks stay correct.
        # _advance has already ticked every clock once per simulated second; all
        # that is left is to zero the one SUMO actually served.
        state = self.conn.trafficlight.getRedYellowGreenState(self.tls_id)
        served = self._match_phase(state)
        if served is not None:
            if served != self.current_phase:
                self.metrics["phase_switches"] += 1
                self.green_elapsed_s = dt
            else:
                self.green_elapsed_s += dt
            self.current_phase = served
            self.time_since_served[served] = 0.0
        self.metrics["max_movement_wait_s"] = max(
            self.metrics["max_movement_wait_s"], float(self.time_since_served.max())
        )

        self._sense_flush()
        obs = self._observe()
        reward = self._reward(False)
        cost = self._cost()
        self.metrics["reward"] += reward
        self.metrics["cost"] += cost
        self.metrics["decisions"] += 1
        done = self._done or self.sim_time >= cfg.demand.horizon_s + 900
        info = {"sim_time": self.sim_time, "phase": self.current_phase, "dt": dt,
                "shield_overridden": False, "shield_reason": None,
                "queue": self._step_queue, "ev_delay": self._epoch_ev_delay,
                "ev_active": len(self._ev_waiting), "arrived": self._epoch_arrived}
        return obs, reward, cost, done, info

    def _match_phase(self, state: str) -> Optional[int]:
        """Best-matching discovered green phase for a raw signal state string."""
        green = {i for i, c in enumerate(state) if c in "Gg"}
        if not green:
            return None
        best, best_score = None, -1.0
        for p in self.phases:
            ref = set(p.green_link_indices)
            score = len(green & ref) / max(len(green | ref), 1)
            if score > best_score:
                best, best_score = p.action_index, score
        return best

    def set_program(self, program_id: str) -> None:
        self.conn.trafficlight.setProgram(self.tls_id, program_id)

    # ------------------------------------------------------- controller inputs
    def lane_vehicle_counts(self) -> Dict[str, float]:
        """Vehicle counts on every controlled and downstream lane (one round-trip)."""
        sub = self.conn.lane.getAllSubscriptionResults()
        return {
            ln: float(sub.get(ln, {}).get(tc.LAST_STEP_VEHICLE_NUMBER, 0))
            for ln in list(self.controlled_lanes) + list(self._downstream_lanes)
        }

    def phase_pressure(self) -> np.ndarray:
        """Max-pressure score per phase: upstream minus downstream queue."""
        counts = self.lane_vehicle_counts()
        out = np.zeros(self.n_actions, dtype=np.float64)
        for p in self.phases:
            total = 0.0
            for i in p.green_link_indices:
                inl, outl = self._link_in_lane[i], self._link_out_lane[i]
                if inl is None:
                    continue
                total += counts.get(inl, 0.0) - counts.get(outl, 0.0)
            out[p.action_index] = total
        return out

    def _recent_discharge(self) -> float:
        """Vehicles that completed their trip during the last decision epoch.

        The shield uses this to tell a *jammed* intersection from a merely busy
        one: both have long queues, but only the jammed one has stopped moving
        traffic. Gating recovery on queue alone made a demand surge markedly
        worse, because heavy-but-healthy operation looks identical to gridlock
        if you only look at queue length.
        """
        return float(self._epoch_arrived)

    def phase_queues(self) -> np.ndarray:
        """Halting vehicles served by each phase."""
        sub = self.conn.lane.getAllSubscriptionResults()
        out = np.zeros(self.n_actions, dtype=np.float64)
        for p in self.phases:
            out[p.action_index] = sum(
                sub.get(ln, {}).get(tc.LAST_STEP_VEHICLE_HALTING_NUMBER, 0)
                for ln in p.green_lanes
            )
        return out

    def approach_queues(self) -> Dict[str, float]:
        """Halting vehicles waiting on each approach.

        The corridor scheduler needs this rather than :meth:`phase_queues`
        because the quantity that determines how long an emergency vehicle is
        held up is the queue standing in front of *it*, on its own arm, not the
        total a phase happens to discharge.
        """
        sub = self.conn.lane.getAllSubscriptionResults()
        out: Dict[str, float] = {a: 0.0 for a in self.approaches}
        for ln in self.controlled_lanes:
            approach = self.lane_to_approach.get(ln)
            if approach in out:
                out[approach] += float(
                    sub.get(ln, {}).get(tc.LAST_STEP_VEHICLE_HALTING_NUMBER, 0))
        return out

    def phases_serving_approach(self, approach: str,
                                primary_only: bool = False) -> List[int]:
        """Phases that show green to ``approach``.

        ``primary_only`` restricts the answer to phases that release *most of
        the approach* rather than merely one movement on it, and it is what any
        emergency reasoning must use. The distinction is not cosmetic on this
        signal: a four-phase program gives N/S all their movements in phase 0
        and a **protected left only** in phase 1, so the unrestricted answer for
        approach N is ``[0, 1]`` and phase 1 discharges one lane in three.

        Handing that set to the corridor, or to the shield's S6 guarantee, lets
        both of them satisfy an emergency request for a vehicle going *straight*
        by granting the protected left. Measured before this existed: the
        corridor committed to the left-turn phase with an ambulance approaching
        in a through lane, the overlay drew its green band over the approach,
        S6 recorded the request as served -- and the vehicle sat at a red the
        whole time. It then cost a second full changeover when the signal moved
        to the phase that actually carried it, which is where the stop came
        from.

        The two phases are also *siblings*: both "serve" the approach, so a
        planner ranking them can swap between them and pay a changeover to
        arrive where it already was. Restricting to the primary phase removes
        that move from the search entirely.

        The test is by lane rather than by link, because links are per-movement
        and would count a permissive left as a whole extra lane's worth of
        service. A phase qualifies when it greens at least
        :attr:`EV_PRIMARY_LANE_FRACTION` of the approach's controlled lanes.
        Since the acoustic layer cannot see which way a vehicle intends to
        turn, this is the correct conservative reading: the all-movements phase
        serves every turn, including a permissive left, so it is right for a
        vehicle whose movement is unknown.
        """
        out = [p.action_index for p in self.phases if approach in p.approaches]
        if not primary_only or not out:
            return out
        lanes = [ln for ln in self.controlled_lanes
                 if self.lane_to_approach.get(ln) == approach]
        if not lanes:
            return out
        best = max(
            len(set(self.phases[p].green_lanes) & set(lanes)) for p in out)
        primary = [p for p in out
                   if len(set(self.phases[p].green_lanes) & set(lanes))
                   >= max(EV_PRIMARY_LANE_FRACTION * len(lanes), 1.0)]
        # Never answer "nothing". If no phase clears the bar -- a program where
        # every phase serves one movement at a time -- fall back to whichever
        # serves the most, so the guarantee still has something to grant.
        if primary:
            return primary
        return [p for p in out
                if len(set(self.phases[p].green_lanes) & set(lanes)) == best]

    def _reward(self, switched: bool) -> float:
        """Efficiency reward for the epoch just completed.

        Queue and waiting terms are read at the *end* of the epoch rather than
        averaged over it, so the reward describes the state the next decision is
        made from. Throughput is a true sum over the epoch, since discharge is a
        flow rather than a state.
        """
        r = self.cfg.reward
        queue_term = self._step_queue / r.queue_norm
        wait_term = self._step_wait / r.wait_norm
        throughput = self._epoch_arrived / r.throughput_norm
        return float(
            -r.w_queue * queue_term
            - r.w_wait * wait_term
            + r.w_throughput * throughput
            - (r.w_switch if switched else 0.0)
        )

    def _cost(self) -> float:
        """CMDP constraint signal: priority-weighted emergency-vehicle exposure.

        Sum of a realised component (delay actually suffered, plus a penalty per
        full stop) and an anticipatory component (an emergency vehicle inbound on
        an approach that is not being served, weighted by proximity). The
        anticipatory component is what makes the constraint learnable: the
        realised component alone only fires after the damage is done.
        """
        r = self.cfg.reward
        c = self._epoch_ev_weighted_delay / r.ev_delay_norm
        c += r.ev_stop_cost * self._epoch_ev_stops / max(r.ev_delay_norm / 4.0, 1e-6)
        c += r.w_ev_unserved * self._epoch_ev_unserved / max(r.ev_unserved_norm, 1e-6)
        return float(c)

    # ==================================================================== close
    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
            self._built = False

    def episode_summary(self) -> Dict[str, float]:
        m = dict(self.metrics)
        steps = max(m["steps"], 1)
        decisions = max(m["decisions"], 1)
        out = {
            "mean_queue": m["sum_queue"] / steps,
            "mean_vehicles": m["sum_vehicles"] / steps,
            "mean_waiting_veh_s": m["sum_wait"] / steps,
            "max_queue": m["max_queue"],
            "throughput": m["arrived"],
            "departed": m["departed"],
            "teleports": m["teleports"],
            "teleports_per_1k_veh": 1000.0 * m["teleports"] / max(m["departed"], 1),
            "phase_switches": m["phase_switches"],
            "switch_rate": m["phase_switches"] / decisions,
            "interrupts": m["interrupts"],
            "interrupt_rate": m["interrupts"] / max(m["phase_switches"], 1),
            "total_reward": m["reward"],
            "total_cost": m["cost"],
            "mean_cost": m["cost"] / decisions,
            "ev_delay_s": m["ev_delay_s"],
            "ev_weighted_delay_s": m["ev_weighted_delay"],
            "ev_unserved_s": m["ev_unserved_s"],
            "ev_stops": m["ev_stops"],
            "n_emergency": len(self._ev_seen),
            "ev_delay_per_vehicle_s": m["ev_delay_s"] / max(len(self._ev_seen), 1),
            "max_movement_wait_s": m["max_movement_wait_s"],
            "shield_override_rate": self.shield.override_rate,
            "shield_gap_outs": float(self.shield.n_gap_outs),
            "shield_recoveries": float(self.shield.n_recoveries),
            "shield_recovery_epochs": float(self.shield.n_recovery_epochs),
            "sim_time": self.sim_time,
            "decisions": m["decisions"],
        }
        out.update(self.sensor.stats())
        return out

    def __enter__(self) -> "SumoSignalEnv":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
