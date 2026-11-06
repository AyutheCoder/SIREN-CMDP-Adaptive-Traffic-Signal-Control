"""Run the controller live, with commentary and (optionally) the SUMO GUI.

This is the app: an adaptive signal controller *hearing* emergency vehicles and
clearing a path for them. It prints a running commentary of what the acoustic
belief thinks is out there, what each corridor would buy and cost, when it is
due to be opened, and what the scheduler does about it, so the behaviour is
legible rather than buried in a summary table.

    python scripts/demo.py                     # GUI, real time, smooth
    python scripts/demo.py --speed 2           # twice real time
    python scripts/demo.py --follow-ev         # camera pans with each emergency vehicle
    python scripts/demo.py --no-clear-lane     # without the lane clearing, for contrast
    python scripts/demo.py --no-gui            # headless, just the commentary
    python scripts/demo.py --controller max_pressure_preempt
    python scripts/demo.py --screenshot-ev     # save a frame on each commitment
    python scripts/demo.py --no-sound          # silence the sirens (they play by default)
    python scripts/demo.py --sound-out run.wav # and/or record them to a file

What you are looking at
-----------------------
* The camera is framed on the junction and stays there. It is set by a
  boundary in metres rather than by a zoom level, so the same command gives the
  same picture on any window, and nothing resizes it mid-run -- an emergency
  vehicle arriving pans nothing and zooms nothing. ``--follow-ev`` pans the
  frame with the vehicle being served; it still never resizes it.
* Emergency vehicles carry a class badge -- a red cross for an ambulance, a
  flame for a fire appliance, a chequer-and-beacon for police -- and flash
  between their two livery colours, so which vehicle the controller is
  prioritising, and what triage weight that implies, is visible on the map.
* The approach the scheduler has committed to is overlaid with a green band and
  its ordinary traffic turns green. The band appears when the *decision* is
  taken, which is before the signal changes: the corridor opens ahead of the
  vehicle rather than around it.
* **The lane ahead empties.** Watch the cars in front of an approaching
  emergency vehicle slide into the next lane over. Signal preemption opens the
  junction; it does nothing about the queue standing between the vehicle and
  the stop line, and a green it cannot reach is not a green. Every car ahead of
  it in its lane is asked to move aside -- only sideways, never slowed, and
  only into a lane its own route can still use, which is the check that stops
  this from jamming the approach it is trying to clear. ``--no-clear-lane``
  turns it off, and the difference is the point.
* The commentary line for each commitment shows the red the vehicle is
  projected to meet at its arrival, the delay the corridor saves, the slack
  left before it must be opened, and what it costs cross traffic.

Emergency vehicles arrive spread evenly across the run rather than as the
Poisson process the evaluations use. Same count, same rate; the gaps are just
not exponential, so the corridor is exercised from the start of the episode to
the end of it instead of three times in the first two minutes and never again.
``--ev-schedule poisson`` restores the process every reported number is
measured under.

Playback
--------
``--speed`` is simulated seconds per wall-clock second; the default is 1.5.
Motion is smooth at 10 fps whatever it is set to, so this only trades how much
you see against how long you sit there: 0.5 is watchable but makes a 600 s
episode twenty minutes. ``--step`` is how much simulated time passes per redraw, and it
is what makes the picture smooth: SUMO redraws once per step, so at a 1 s step a
car at 50 km/h teleports fourteen metres between frames and then freezes. The
default scales the step with the speed to hold 10 frames per second, so asking
for a slower demo gives you a slower one rather than a choppier one.

Both are enforced by the client, not by SUMO: while TraCI is driving the
simulation, SUMO never consults the ``--delay`` it was launched with, so the
window used to run at whatever rate the machine managed -- measured at 3.3x
real time where 0.5x was asked for, which is why the traffic looked too fast.
The wait now happens in the step loop (``SumoSignalEnv._pace``) and ``--speed``
means what it says.

The controller is untouched by either. It still decides every 5 s, and still
re-checks the corridor on every step.

Press Ctrl+C at any time; the episode summary still prints.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atsc.agents.baselines import (
    CorridorPolicy, LearnedPolicy, PriorityAwarePolicy, build_baseline)
from atsc.agents.dueling_ddqn import LagrangianDuelingDDQN
from atsc.config import (
    AgentConfig, DATA_DIR, DemandConfig, EnvConfig, FIGURES_DIR, MODELS_DIR, ensure_dirs,
)
from atsc.control.clearance import CorridorClearance
from atsc.control.corridor import CorridorScheduler
from atsc.env import SumoSignalEnv
from atsc.evaluation.audio import SirenPlayer, scene_from_env
from atsc.evaluation.gui import GuiPainter
from atsc.evaluation.icons import CLASS_LABEL
from atsc.netgen import build_network, write_actuated_program
from atsc.sensing.sensor import AcousticSensor, SensorErrorModel

try:
    from traci.exceptions import FatalTraCIError
except ImportError:                                   # pragma: no cover
    class FatalTraCIError(Exception):
        """Stand-in so the demo still imports without TraCI on the path."""

BAR = "=" * 132


def nearest_ev(env) -> str:
    """Ground truth about the closest emergency vehicle, for the caption.

    Deliberately not the belief. The belief is already on the line to the left
    of this, and the question a viewer keeps asking -- is there really one out
    there, and is it moving? -- is a question *about* the belief that the
    belief cannot answer. Reading the simulator here is what lets the caption
    show the filter being right and the filter being wrong.
    """
    scenario = getattr(env, "scenario", None)
    conn = getattr(env, "conn", None)
    blank = "-".rjust(16)
    if scenario is None or conn is None:
        return blank
    try:
        present = set(conn.vehicle.getIDList())
    except Exception:
        return blank
    best = None
    for vid in present & set(scenario.emergency_vehicles or ()):
        try:
            edge = conn.vehicle.getRoadID(vid)
            if edge not in env.approaches:
                continue
            lane = conn.vehicle.getLaneID(vid)
            gap = conn.lane.getLength(lane) - conn.vehicle.getLanePosition(vid)
            speed = conn.vehicle.getSpeed(vid)
        except Exception:
            continue
        if best is None or gap < best[0]:
            best = (float(gap), env.approach_direction.get(edge, edge), float(speed))
    if best is None:
        return blank
    gap, d, speed = best
    # A stopped emergency vehicle is the failure this whole system exists to
    # prevent, so it is named rather than left to be inferred from a zero.
    tag = "STOPPED" if speed < 0.3 else f"{speed:4.1f}m/s"
    return f"{d}{gap:6.0f}m {tag:>8}"


def bar(value: float, scale: float, width: int = 10, ch: str = "#") -> str:
    n = int(np.clip(value / max(scale, 1e-9), 0.0, 1.0) * width)
    return ch * n + "." * (width - n)


def build_controller(args, env, cfg):
    """Return (controller, description). Handles trained runs and baselines."""
    ckpt = MODELS_DIR / f"{args.controller}.h5"
    if not ckpt.exists():
        return build_baseline(args.controller, env, cfg.demand), \
            f"baseline controller '{args.controller}'"

    agent = LagrangianDuelingDDQN(env.observation_dim, env.n_actions, AgentConfig())
    agent.load(ckpt)
    lam = float(agent.lam.numpy())
    if args.priority == "none":
        return LearnedPolicy(agent, name=args.controller, greedy=True), \
            f"learned policy {ckpt.name} (lambda={lam:.2f}), no priority layer"
    if args.priority == "overlay":
        return PriorityAwarePolicy(agent=agent, name=args.controller), \
            f"learned policy {ckpt.name} + fixed-threshold overlay (ablation)"

    sched = CorridorScheduler(ev_value_ratio=args.value_ratio,
                              just_in_time=not args.no_jit,
                              commit_range_m=max(args.commit_range, 0.0),
                              service_request_presence=args.service_floor)
    ctrl = CorridorPolicy(agent=agent, name=args.controller, scheduler=sched)
    extra = ""
    if args.commit_range > 0.0:
        extra += f", served outright inside {args.commit_range:g}m"
    if args.service_floor != CorridorScheduler.service_request_presence:
        extra += f", service floor {args.service_floor:g}"
    return ctrl, (f"learned policy {ckpt.name} (lambda={lam:.2f}) + "
                  f"risk-calibrated just-in-time corridor "
                  f"(gamma={args.value_ratio:g}{extra})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--controller", default="siren_arc",
                   help="a trained run name (siren_priority, unconstrained, ...) "
                        "or a baseline (max_pressure, max_pressure_preempt, ...)")
    p.add_argument("--priority", default="corridor",
                   choices=["corridor", "overlay", "none"],
                   help="which priority layer to run above the learned policy")
    p.add_argument("--value-ratio", type=float, default=50.0,
                   help="background vehicle-seconds one emergency vehicle-second "
                        "is worth; the scheduler's only normative parameter")
    p.add_argument("--no-jit", action="store_true",
                   help="open the corridor as soon as it is worth it rather than "
                        "at the last moment it can still be opened in time")
    p.add_argument("--commit-range", type=float, default=150.0, metavar="M",
                   help="serve any emergency vehicle the sensor is confident "
                        "about once it is within this many metres of the stop "
                        "line, whatever the cross-street queue is worth. This "
                        "is the operator's rule rather than the Bayes rule, "
                        "and it is what makes the approach *clear* before the "
                        "vehicle arrives instead of as it arrives: 150 m is "
                        "about eight seconds at emergency speed, which is one "
                        "changeover plus the discharge of a typical queue. "
                        "Set 0 for the paper's behaviour, where the economics "
                        "decide alone.")
    p.add_argument("--no-clear-lane", action="store_true",
                   help="do not move ordinary traffic out of the emergency "
                        "vehicle's lane. Clearing is on by default: signal "
                        "preemption opens the junction but does nothing about "
                        "the cars standing between the vehicle and the stop "
                        "line, and a green it cannot reach is not a green. "
                        "Every car ahead of it in its own lane, within "
                        "--clear-ahead metres, is asked to move to another "
                        "lane of the same approach -- only sideways, never "
                        "slowed, only into a lane its own route can use, and "
                        "only while the vehicle is actually coming.")
    p.add_argument("--clear-range", type=float, default=220.0, metavar="M",
                   help="distance from the stop line at which lane clearing "
                        "starts. About eleven seconds at emergency speed, "
                        "which is a lane change from a standstill plus room "
                        "for the car in front to move first.")
    p.add_argument("--clear-ahead", type=float, default=180.0, metavar="M",
                   help="how far ahead of the emergency vehicle to clear")
    p.add_argument("--service-floor", type=float, default=0.55, metavar="P",
                   help="evidence at which an approach is *owed* a green, as "
                        "a fraction of the largest posterior this sensor can "
                        "emit. This is the input to the shield's S6 deadline: "
                        "below it a vehicle is not owed anything and waits on "
                        "the economics. Exposed because it is the policy "
                        "choice in the guarantee, not because it is a tuning "
                        "knob -- measured over twelve seeds, dropping it to "
                        "0.35 changed nothing at all, because the calibrated "
                        "posterior almost never lands between the two.")
    p.add_argument("--ev-deadline", type=float, default=None, metavar="S",
                   help="S6: guaranteed worst-case seconds from a standing "
                        "emergency request to a green serving it (default 16). "
                        "It cannot go below emergency-minimum-green + poll + "
                        "yellow + all-red, which is 14 s here, and it should "
                        "not go near it: measured at 14 s the guarantee "
                        "thrashes -- a forced changeover costs the whole "
                        "budget and the request comes due again as it lands -- "
                        "firing hundreds of times an episode and taking mean "
                        "queue from 11 to 18 veh and emergency delay from 16 s "
                        "to 51 s. Tighter is not better here.")
    p.add_argument("--no-gui", action="store_true", help="run headless")
    p.add_argument("--seed", type=int, default=2029,
                   help="demand seed. Chosen for exposition, not for results: "
                        "emergency vehicles arrive as a Poisson process, so the "
                        "wait for the first one is exponential and varies wildly "
                        "between seeds -- the previous default sat through 266 s "
                        "of ordinary traffic before anything happened, which is "
                        "four minutes of watching nothing at 1.5x. This one puts "
                        "the first vehicle at 72 s, with a second right behind "
                        "it. Nothing reported anywhere depends on it; every "
                        "evaluation seed is 50000+ and disjoint from this.")
    p.add_argument("--horizon", type=int, default=1200)
    p.add_argument("--ev-schedule", default="spread",
                   choices=["spread", "poisson"],
                   help="how emergency arrivals are placed in time. Poisson is "
                        "the correct model and is what every evaluation uses, "
                        "but its gaps are exponential, so at any given rate a "
                        "demonstration gets two vehicles ten seconds apart and "
                        "then six minutes of ordinary traffic. 'spread' keeps "
                        "the same count over the horizon and places them at "
                        "even intervals with jitter, so the corridor is "
                        "exercised from the start of the run to the end of it.")
    p.add_argument("--ev-rate", type=float, default=24.0,
                   help="emergency vehicles per hour. Still well above a "
                        "realistic 12/h, so that something happens while you "
                        "are watching, but below the 45/h this used to default "
                        "to. 24/h puts a vehicle in roughly every 150 s, which "
                        "over the default 1200 s horizon is seven or eight of "
                        "them: enough to watch the corridor open, hold and "
                        "release repeatedly, and to see two conflicting "
                        "requests scheduled against each other, without the "
                        "intersection spending most of the episode in "
                        "preemption. At 45/h arrivals overlap often enough "
                        "that the background queue never fully recovers "
                        "between events, which makes the demo read as a "
                        "congestion study rather than a priority one.")
    p.add_argument("--speed", type=float, default=1.5,
                   help="playback speed relative to real time. 1 = real time, "
                        "2 = twice as fast, 0.5 = half. The default is 1.5x: "
                        "the motion is smooth at 10 fps whatever this is set "
                        "to, so the only question is how long you want to wait, "
                        "and real time makes a 600 s episode a ten-minute sit.")
    p.add_argument("--step", type=float, default=None,
                   help="simulation step in seconds, i.e. how much simulated "
                        "time passes per redraw. Defaults to 0.05 with the GUI "
                        "and 1.0 headless. Frames per second is speed/step, so "
                        "the GUI defaults give 10 fps however --speed is set.")
    p.add_argument("--delay-ms", type=int, default=None,
                   help="raw override for --speed: wall-clock ms per simulation step")
    p.add_argument("--keep-open", action="store_true",
                   help="leave the GUI window open when the episode ends")
    p.add_argument("--sensing", default="surrogate",
                   choices=["surrogate", "oracle", "none", "end_to_end"])
    p.add_argument("--belief", default="joint", choices=["joint", "independent"],
                   help="presence filter: the exclusive joint filter, or the "
                        "previous four independent per-approach filters")
    p.add_argument("--screenshot-seq", type=int, default=0, metavar="N",
                   help="once the first corridor commits, save a frame on each "
                        "of the next N decision epochs. Unlike --screenshot-ev, "
                        "which saves one frame per approach, this follows a "
                        "single emergency vehicle through commitment, green and "
                        "crossing, which is the sequence a figure needs.")
    p.add_argument("--screenshot-at", default=None, metavar="T1,T2,...",
                   help="save a frame the first time the simulation clock "
                        "passes each of these times, in simulated seconds. "
                        "Deterministic, so the same times on two runs that "
                        "differ only in --priority give directly comparable "
                        "frames -- which is what a with/without figure needs.")
    p.add_argument("--screenshot-tag", default="at", metavar="NAME",
                   help="filename prefix for --screenshot-at frames, so two "
                        "runs do not overwrite each other.")
    p.add_argument("--screenshot-ev", action="store_true",
                   help="save a GUI frame the first time each approach is committed to")
    p.add_argument("--follow-ev", action="store_true",
                   help="pan the camera with the emergency vehicle being "
                        "served. Off by default: the view is framed on the "
                        "junction, which is where the thing being demonstrated "
                        "happens. This only moves that frame -- it never "
                        "resizes it.")
    p.add_argument("--view-radius", type=float, default=260.0, metavar="M",
                   help="half-width of the framed view in metres, centred on "
                        "the junction. The camera is set by boundary rather "
                        "than by zoom level so the same command frames the "
                        "same picture on any window and never lurches "
                        "mid-run.")
    p.add_argument("--max-epochs", type=int, default=100000)
    p.add_argument("--quiet", action="store_true",
                   help="print only emergency events, not routine epochs")
    p.add_argument("--sound", action="store_true", default=None,
                   help="play the sirens out loud. On by default whenever the "
                        "GUI is up, because the controller's only sensor is a "
                        "microphone and a silent demonstration of a system "
                        "that hears is missing the input. The waveform is the "
                        "same physical model the detector is trained on -- "
                        "level from spherical spreading plus ISO 9613-1 "
                        "absorption, Doppler from the closing speed, panned by "
                        "the vehicle's true bearing -- so what you hear is what "
                        "the microphones are hearing, and the corridor opens "
                        "as it gets louder. Ground truth, not the belief: when "
                        "the filter is wrong you can hear that it is wrong.")
    p.add_argument("--no-sound", dest="sound", action="store_false",
                   help="run the GUI silently")
    p.add_argument("--sound-out", default=None, metavar="FILE.wav",
                   help="record the soundtrack to a WAV file. Works with or "
                        "without --sound and with no audio device at all.")
    p.add_argument("--sound-backend", default="auto",
                   choices=["auto", "winsound", "command", "null"],
                   help="output device to use for --sound")
    p.add_argument("--horn-rate", type=float, default=None,
                   help="horn presses per vehicle-hour of queued time. Ordinary "
                        "drivers sound their horns when stopped, and a road "
                        "horn is the one urban sound genuinely confusable with "
                        "a siren -- louder at the mast than a distant siren, "
                        "harmonic, and overlapping the siren band. The default "
                        "(15/h) makes a busy junction honk about every fifteen "
                        "seconds. Set 0 for silent traffic, which is what this "
                        "project modelled before horns existed and is the "
                        "ablation the detector's horn class is measured "
                        "against.")
    p.add_argument("--bluelight", action="store_true",
                   help="let ordinary traffic pull aside for emergency "
                        "vehicles (SUMO's bluelight device). Off by default "
                        "for two reasons: yielding drivers would confound the "
                        "signal-control effect the study measures, and the "
                        "device deadlocks this intersection outright on some "
                        "seeds -- measured, 1 run in 6 went from 24 s of delay "
                        "to 213 s with a 95-vehicle queue. Worth watching, not "
                        "worth defaulting to.")
    args = p.parse_args()
    # Sound follows the GUI unless it was asked for either way explicitly.
    # Headless stays silent: a batch run goes at forty times real time and
    # there is nothing for the audio to be in step with.
    if args.sound is None:
        args.sound = not args.no_gui

    ensure_dirs()
    cfg = EnvConfig()
    cfg.demand = DemandConfig(horizon_s=args.horizon, seed=args.seed,
                              ev_rate_per_hour=args.ev_rate,
                              ev_schedule=args.ev_schedule)
    cfg.sensing_mode = args.sensing
    cfg.belief_mode = args.belief
    if args.horn_rate is not None:
        cfg.acoustic.horn_rate_per_veh_hour = max(float(args.horn_rate), 0.0)
    if args.ev_deadline is not None:
        cfg.control.ev_service_deadline_s = float(args.ev_deadline)
    cfg.gui = not args.no_gui
    cfg.bluelight = args.bluelight
    cfg.collect_emissions = False
    # Frames per second is speed/step. Scaling the step with the speed keeps
    # the frame rate fixed at 10 fps whatever playback speed is asked for, so
    # slowing the demo down makes it slower rather than choppier.
    if args.step is not None:
        cfg.step_length_s = float(args.step)
    elif cfg.gui:
        cfg.step_length_s = float(np.clip(max(args.speed, 1e-3) / 10.0, 0.02, 0.5))
    else:
        cfg.step_length_s = 1.0
    if args.delay_ms is not None:
        cfg.gui_delay_ms = max(args.delay_ms, 0)
    else:
        speed = max(args.speed, 1e-3)
        cfg.gui_delay_ms = int(round(1000.0 * cfg.step_length_s / speed))
    cfg.gui_quit_on_end = not args.keep_open

    net_file, _ = build_network(cfg.network, DATA_DIR / "networks")
    additional = None
    if args.controller == "actuated":
        additional = [write_actuated_program(
            net_file, DATA_DIR / "networks" / "actuated.add.xml",
            min_dur=cfg.control.min_green_s, max_dur=cfg.control.max_green_s)]

    em_path = MODELS_DIR / "sensor_error_model.json"
    sensor = AcousticSensor(
        [], cfg.acoustic, mode=cfg.sensing_mode,
        error_model=SensorErrorModel.load(em_path) if em_path.exists() else None,
        seed=args.seed)

    env = SumoSignalEnv(cfg, net_file=net_file, work_dir=DATA_DIR / "runs" / "demo",
                        label="demo", sensor=sensor, additional_files=additional)

    print(BAR)
    print(f"SIREN-ARC live demo   controller={args.controller}   "
          f"priority={args.priority}   sensing={args.sensing}   "
          f"belief={args.belief}   seed={args.seed}")
    print(BAR, flush=True)

    obs = env.reset(seed=args.seed)
    painter = GuiPainter(env, enabled=cfg.gui, icons_dir=FIGURES_DIR / "icons",
                         view_radius_m=args.view_radius)
    painter.setup()
    # The other half of "give it a clear run": the signal opens the junction,
    # this empties the lane leading to it. Headless too -- it is actuation on
    # the simulation, not decoration on the picture, and the emergency delay in
    # the summary below is only meaningful if both halves ran.
    clearance = CorridorClearance(env, enabled=not args.no_clear_lane,
                                  trigger_m=args.clear_range,
                                  ahead_m=args.clear_ahead)
    clearance.reset()
    # Live playback follows the wall clock, which is right when the run is
    # paced to real time and wrong otherwise: a headless episode goes at forty
    # times real time, and a wall-clock recording of it would be thirteen
    # seconds of audio for eight minutes of simulation. Recording alone
    # therefore renders in the simulation's own time base.
    siren = SirenPlayer(cfg.acoustic, enabled=args.sound,
                        backend=args.sound_backend,
                        record_path=Path(args.sound_out) if args.sound_out else None,
                        mode="live" if args.sound else "offline",
                        seed=args.seed).start()

    def _frame(_env):
        # All of these run at the simulation's frame rate rather than at the
        # 5 s decision epoch. A beacon that flashes once per decision is not a
        # beacon, and a siren whose level steps once per decision is not a
        # siren -- the level has to move with the vehicle for the thing the
        # controller is reacting to to be audible.
        painter.paint_emergency()
        if args.follow_ev:
            # The camera too, and for a sharper reason: releasing has to be
            # prompt. The vehicle crosses the junction in about two seconds,
            # and a camera re-evaluated only on epoch boundaries spends most of
            # that still locked to it, chasing it down the far arm.
            painter.refresh_follow()
        clearance.update()
        siren.update(scene_from_env(_env))
        siren.advance(cfg.step_length_s)

    env.step_hook = _frame

    controller, description = build_controller(args, env, cfg)
    controller.reset(env)
    print(description)
    dirs = [env.approach_direction[a] for a in env.approaches]
    playback = cfg.step_length_s / max(cfg.gui_delay_ms / 1000.0, 1e-6)
    rate = (f"playback: {playback:.2f}x real time at "
            f"{1000.0 / max(cfg.gui_delay_ms, 1):.0f} fps"
            if cfg.gui else "headless, as fast as it will go")
    print(f"approaches: {dirs}   phases: {env.n_actions}   "
          f"shield bound: {cfg.control.max_starvation_s:.0f}s   "
          f"step: {cfg.step_length_s:g}s   {rate}")
    if cfg.gui:
        print("map key: " + "   ".join(
            f"{CLASS_LABEL[c]}" for c in ("ambulance", "fire", "police"))
            + "   |  amber band = siren heard  |  green band = corridor "
              "committed  |  green cars = released")
        if not args.no_clear_lane:
            print(f"lane clearing: on, from {args.clear_range:.0f}m out "
                  f"({args.clear_ahead:.0f}m ahead of the vehicle)")
    print(f"\n{'time':>6} {'ph':>3} {'queue N/E/S/W':>22} "
          f"{'belief N/E/S/W':>22} {'nearest EV':>16}  act  event")
    print("-" * 132, flush=True)

    shot_dir = FIGURES_DIR / "demo"
    seen_committed: set = set()
    seq_left = None   # None until the first corridor commits, then counts down
    seq_index = 0
    shot_times = sorted(
        float(t) for t in str(args.screenshot_at).split(",") if t.strip()
    ) if args.screenshot_at else []
    epoch = 0
    done = False
    served_epochs = corridor_epochs = conflict_epochs = deferred_epochs = 0
    t0 = time.time()

    try:
        while not done and epoch < args.max_epochs:
            mask = env.action_mask()
            action = controller.act(env, obs, mask)
            prev_phase = env.current_phase
            interrupt = getattr(controller, "interrupt", None)
            obs, reward, cost, done, info = env.step(action, interrupt=interrupt)
            epoch += 1

            decision = getattr(controller, "last_decision", None)
            beliefs = env.belief.beliefs()
            pres = [beliefs[a].presence for a in env.approaches]
            queues = env.approach_queues()

            event = ""
            target_approach = None
            if decision is not None and decision.engaged and decision.target is not None:
                corridor_epochs += 1
                target_approach = decision.target.approach
                # Green *for this vehicle*, which is the primary phase of its
                # approach -- not merely green somewhere on the approach.
                # ``_serving_approaches`` counts any link, and every through
                # phase here shows a permissive left, so an approach reads as
                # served while the movement the vehicle needs is at a red. The
                # caption said [GREEN] over a stationary ambulance.
                served = int(env.current_phase) in set(decision.target.phases)
                served_epochs += int(served)
                r = decision.target
                held = not decision.committed
                event = (f"{'HOLDING ' if held else 'CORRIDOR '}{r.direction} <- "
                         f"{r.describe()} "
                         f"{'[GREEN]' if served else '[opening]'}")
                if decision.conflict:
                    conflict_epochs += 1
                    others = ", ".join(x.direction for x in decision.committed
                                       if x.approach != target_approach)
                    event += f"   CONFLICT with {others}; plan={list(decision.schedule)}"
            elif decision is not None and decision.committed:
                deferred_epochs += 1
                event = f"deferred: {decision.reason}"
            elif decision is not None and any(
                    not r.due and r.benefit_s > 0 for r in decision.requests):
                pending = [r for r in decision.requests if not r.due and r.benefit_s > 0]
                r = max(pending, key=lambda x: x.presence)
                event = (f"watching {r.direction}: p={r.presence:.2f} eta={r.eta_s:4.1f}s "
                         f"-- corridor due in {r.slack_s:4.1f}s")
            elif info.get("shield_reason") == "ev_service_deadline":
                # S6 fired: the layers above declined or were too slow, and the
                # shield's guarantee is what is delivering the green. Worth
                # showing separately from an ordinary override, because it is
                # the backstop doing exactly the job it was added for.
                event = (f"S6 GUARANTEE: shield forced emergency service "
                         f"(deadline {cfg.control.ev_service_deadline_s:.0f}s)")
            elif info["shield_overridden"]:
                event = f"shield override ({info['shield_reason']})"
            elif prev_phase != env.current_phase:
                event = "phase change"

            # Amber for what is heard, green for what is acted on. A request
            # that survived screening is a vehicle the filter is tracking; the
            # scheduler may still be waiting for the last moment at which
            # opening the corridor works, and that wait is the mechanism.
            heard = [] if decision is None else [
                r.approach for r in decision.requests
                if r.approach != target_approach and r.presence > 0.05]
            painter.paint_corridor(target_approach, heard=heard)
            if args.follow_ev:
                painter.follow(target_approach)
            # The screenshot is taken *after* the overlay is drawn, or it
            # captures the previous epoch's corridor and shows the band on the
            # wrong arm -- which is exactly the confusion the overlay exists to
            # remove.
            if (args.screenshot_ev and cfg.gui and target_approach is not None
                    and decision.target.direction not in seen_committed):
                seen_committed.add(decision.target.direction)
                shot_dir.mkdir(parents=True, exist_ok=True)
                path = (shot_dir /
                        f"corridor_{decision.target.direction}_{int(env.sim_time)}s.png")
                if painter.screenshot(path):
                    print(f"       [screenshot {path.name}]", flush=True)

            # Sequence capture. The frame is taken every epoch once the first
            # corridor commits, so the same vehicle is followed from commitment
            # through green to crossing. The counter runs on after the corridor
            # closes, which is the half of the story a commitment-triggered
            # frame cannot show.
            # Fixed-time capture. Fires the first time the clock passes each
            # requested time, so two runs differing only in the priority layer
            # produce frames of the same instant.
            if shot_times and cfg.gui:
                while shot_times and env.sim_time >= shot_times[0]:
                    want = shot_times.pop(0)
                    shot_dir.mkdir(parents=True, exist_ok=True)
                    path = (shot_dir /
                            f"{args.screenshot_tag}_t{int(want):04d}s.png")
                    if painter.screenshot(path):
                        print(f"       [shot {path.name}]", flush=True)

            if args.screenshot_seq > 0 and cfg.gui:
                if seq_left is None and target_approach is not None:
                    seq_left = args.screenshot_seq
                    seq_index = 0
                if seq_left:
                    shot_dir.mkdir(parents=True, exist_ok=True)
                    path = (shot_dir /
                            f"seq_{seq_index:02d}_t{int(env.sim_time)}s.png")
                    if painter.screenshot(path):
                        print(f"       [seq {path.name}]", flush=True)
                    seq_index += 1
                    seq_left -= 1

            important = bool(event and not event.startswith("phase change"))
            if important or (not args.quiet and epoch % 4 == 0):
                qs = "/".join(f"{queues[a]:4.0f}" for a in env.approaches)
                bs = "/".join(f"{v:4.2f}" for v in pres)
                print(f"{env.sim_time:6.0f} {env.current_phase:3d} {qs:>22} "
                      f"{bs:>22} {nearest_ev(env):>16} {action:4d}  {event}",
                      flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted -- printing summary so far", flush=True)
    except FatalTraCIError:
        # Closing the SUMO window is a normal way to end a demo, and from the
        # client it looks exactly like this: the next simulationStep finds the
        # connection gone. A traceback for it says something broke when nothing
        # did. The summary below reads only counters this process already
        # holds, so it stays correct up to the moment the window went away --
        # note that it therefore describes a *partial* episode, which is why it
        # says so rather than being printed as if the run had finished.
        print("\nSUMO window closed -- printing summary of the partial run",
              flush=True)

    summary = env.episode_summary()
    # Read off the shield before the environment is closed.
    ev_forced = float(getattr(env.shield, "n_ev_forced", 0.0))
    sched_stats = {}
    if hasattr(controller, "scheduler") and hasattr(controller.scheduler, "stats"):
        sched_stats = controller.scheduler.stats()
    clear_stats = clearance.stats()
    painter.close()
    clearance.close()
    siren.close()
    env.close()

    print(BAR)
    print(f"ran {epoch} decision epochs over {summary['sim_time']:.0f}s of "
          f"simulation in {time.time() - t0:.1f}s wall")
    print(BAR)
    for label, key, unit in (
        ("background delay proxy (mean queue)", "mean_queue", "veh"),
        ("throughput", "throughput", "veh"),
        ("emergency vehicles seen", "n_emergency", ""),
        ("emergency stopped delay / vehicle", "ev_delay_per_vehicle_s", "s"),
        ("emergency full stops", "ev_stops", ""),
        ("phase switches", "phase_switches", ""),
        ("of which mid-epoch interrupts", "interrupts", ""),
        ("longest any movement waited", "max_movement_wait_s", "s"),
        ("teleports (0 = no gridlock)", "teleports", ""),
        ("shield override rate", "shield_override_rate", ""),
    ):
        v = summary.get(key)
        if v is None:
            continue
        print(f"  {label:38s} {v:10.2f} {unit}")

    if corridor_epochs:
        print(f"\n  corridor epochs                        {corridor_epochs:10d}")
        print(f"  of which the target was already green  {served_epochs:10d} "
              f"({100.0 * served_epochs / corridor_epochs:.0f}%)")
        print(f"  epochs with conflicting requests       {conflict_epochs:10d} "
              f"({100.0 * conflict_epochs / corridor_epochs:.0f}%)")
        print(f"  epochs the shield deferred             {deferred_epochs:10d}")
    for key, label in (("priority_engagements", "distinct corridors opened"),
                       ("expected_false_preemptions", "expected false preemptions")):
        if key in sched_stats:
            print(f"  {label:38s} {sched_stats[key]:10.2f}")

    if not args.no_clear_lane:
        print("")
        print(f"  approaches cleared ahead of a vehicle    "
              f"{clear_stats['clearance_events']:8.0f}")
        print(f"  cars moved out of the way                "
              f"{clear_stats['vehicles_moved_aside']:8.0f}")

    bound = cfg.control.max_starvation_s
    held = summary["max_movement_wait_s"] <= bound
    print(f"\n  S3 starvation bound  (<= {bound:.0f}s): "
          f"{'HELD' if held else 'VIOLATED'} "
          f"(worst {summary['max_movement_wait_s']:.0f}s)")
    deadline = cfg.control.ev_service_deadline_s
    print(f"  S6 emergency service (<= {deadline:.0f}s): "
          f"shield had to force it {ev_forced:.0f} times")
    if args.screenshot_ev and seen_committed:
        print(f"  screenshots -> {shot_dir}")
    if args.sound or args.sound_out:
        print(f"  audio backend                          {siren.backend:>10s}")
        if args.sound_out:
            print(f"  soundtrack -> {args.sound_out}")
    print(BAR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
