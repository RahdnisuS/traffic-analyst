"""One dashboard session: the live simulation(s) its browsers watch, plus the streaming loop.

Locally there is one session shared by every browser; the public demo gives each browser tab its
own (``server.sessions: per_visitor``, managed by ``server.sessions.SessionManager``). Either way the
heavy, read-only parts (the RL network, the queue predictor) are loaded once per server
(``SharedModels``) and used by every session.

Design:
  * The sim is advanced by a single asyncio task at FRAME_HZ (20 Hz). Each tick
    owes ``speed * 0.05`` simulated seconds; we run as many fixed ``dt`` steps as
    that covers and carry the remainder ("time debt"), so speed is exact on average.
  * If the machine can't keep up, we cap steps per tick and DROP the debt rather
    than trying to catch up. The sim then runs slower than requested instead of
    freezing in a catch-up spiral.
  * Each browser has an outbox: position frames are "latest wins" (a slow phone
    skips frames instead of falling seconds behind), while metrics, events and
    status messages are queued reliably and in order (a chart never loses a point).
  * Each streamed simulation is a ``LiveRun``: the sim, its MetricsCollector and its run log.
    The same collector output goes to the log (disk) and the browsers (live), so the
    dashboard and the logs can never disagree.

Phase 8, AI in the UI:
  * AI on/off (``controller`` command) swaps the live run's signal policy in place: the new
    controller takes over from the current state. Safe by construction, because a policy can
    only REQUEST switches (min green, amber, all-red and the pedestrian phase stay with the
    SignalController). The swap is logged as a ``controller_change`` event.
  * Comparison (``compare`` command) runs a TWIN: both runs restart at t = 0 with the same seed
    and settings, the main run on the fixed timer and the twin on the AI, stepped in lockstep,
    and every settings change is applied to both at the same simulated instant. Why that is a
    fair comparison: every random draw (arrivals, vehicle types, pedestrians) happens on every
    step regardless of the signals, so both runs see exactly the same traffic; only the
    controller differs. Each run is logged separately, tagged with a shared comparison id.

Phase 9, congestion forecast: if a trained predictor exists (``python -m predict.train_predictor``),
the main run forecasts every approach's queue 2 minutes ahead once per simulated second, from the
same metrics rows the chart shows. Forecasts go out with the metrics (``forecast``) and in ``init``
(``forecast_history``, ``predictor``). Not during a comparison: its charts show totals for two
controllers, and a third kind of line would muddle that story.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from datetime import datetime

from fastapi import WebSocket

from bench.live import LiveStressTest
from bench.stress import latest_summary
from rl.features import waiting_now
from sim.config import Config
from sim.engine import Simulation
from sim.logger import RunLogger
from sim.metrics import MetricsCollector
from sim.signals import FixedTimePolicy

from .schemas import (GREEN_MAX_S, PED_RATE_MAX, SPEEDS, VOLUME_MAX_VPH, CompareCmd, ControllerCmd, PauseCmd,
                      PlayCmd, ResetCmd, SettingsCmd, SpeedCmd, StressCmd)

log = logging.getLogger("traffic.session")

FRAME_HZ = 20
MAX_STEPS_PER_TICK = 400       # 20x speed at dt=0.1 needs 10/tick; this is a generous safety cap
MIN_RUN_TO_KEEP_S = 10.0       # dashboard runs shorter than this are discarded on reset
HISTORY_ON_CONNECT = 300       # seconds of chart history sent to a newly connected browser
CONTROLLER_NAMES = {"fixed": "fixed timer", "rl": "AI"}


def _dumps(msg: dict) -> str:
    return json.dumps(msg, separators=(",", ":"))


def _pct(new: float | None, old: float | None) -> float | None:
    """Relative change new vs old in %, negative = lower. None while there is nothing to compare."""
    if new is None or not old:
        return None
    return round(100.0 * (new - old) / old, 1)


class Client:
    """Per-connection outbox with two lanes: droppable frames, and reliable messages."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.reliable: deque[str] = deque(maxlen=5000)
        self.frame: str | None = None
        self.dropped_frames = 0
        self._wake = asyncio.Event()

    def send_frame(self, text: str) -> None:
        if self.frame is not None:
            self.dropped_frames += 1  # the previous frame was never sent; replace it
        self.frame = text
        self._wake.set()

    def send(self, text: str) -> None:
        self.reliable.append(text)
        self._wake.set()

    async def pump(self) -> None:
        """Runs as its own task for the lifetime of the connection."""
        try:
            while True:
                await self._wake.wait()
                self._wake.clear()
                while self.reliable:
                    await self.ws.send_text(self.reliable.popleft())
                if self.frame is not None:
                    frame, self.frame = self.frame, None
                    await self.ws.send_text(frame)
        except Exception:  # socket closed mid-send; the receive loop handles cleanup
            return


class LiveRun:
    """One streamed simulation: the sim, its collector, its run log and its outgoing buffers.
    It is the sim's observer. It also keeps running totals for the side-by-side comparison,
    measured exactly as ``rl.evaluate`` does (waiting = stopped vehicles + backlog, per step)."""

    def __init__(self, sim: Simulation, controller: str, log_runs: bool, runs_dir, tags: dict):
        self.sim = sim
        self.controller = controller  # who drives it NOW ("fixed" | "rl")
        self.collector = MetricsCollector(sim, sim.cfg.metrics)
        self.logger = (RunLogger(sim, controller=controller, runs_dir=runs_dir, tags=tags,
                                 collector=self.collector, attach=False)
                       if log_runs else None)
        self.stress: LiveStressTest | None = None  # a stress test belongs to one sim
        self.recent_events: deque[dict] = deque(maxlen=100)
        self.pending_rows: list[dict] = []
        self.pending_events: list[dict] = []
        self.status_dirty = False
        # Queue forecasts (Phase 9): set by the session for the run whose charts are shown.
        self.forecaster = None
        self.forecasts: deque[dict] = deque(maxlen=HISTORY_ON_CONNECT)
        self.pending_forecasts: list[dict] = []
        # Comparison totals since t = 0.
        self.waited_veh_s = 0.0
        self.delay_sum, self.trips = 0.0, 0
        self.ped_wait_sum, self.ped_trips = 0.0, 0
        sim.observers.append(self)

    # ------------------------------------------------------------------ observer
    def after_step(self, sim) -> None:
        row, events, trips, ped_trips = self.collector.after_step(sim)
        if self.stress is not None and self.stress.active:
            # Same event stream as the collector's: logged to disk and shown in the dashboard.
            stress_events = self.stress.after_step(sim)
            if stress_events:
                events = events + stress_events
                self.status_dirty = True  # step finished: browsers need the new stress status
        if self.logger:
            self.logger.record(row, events, trips, ped_trips)
        if row is not None:
            self.pending_rows.append(row)
            if self.forecaster is not None:
                self._forecast(sim)
        self.pending_events.extend(events)
        self.waited_veh_s += waiting_now(sim) * sim.dt
        for t in trips:
            self.delay_sum += t.delay
            self.trips += 1
        for p in ped_trips:
            self.ped_wait_sum += p.wait_time
            self.ped_trips += 1

    def _forecast(self, sim) -> None:
        try:
            f = self.forecaster.forecast(self.collector.history, sim, self.controller)
        except Exception:  # a forecast is a nice-to-have: never let it stop the simulation
            log.exception("queue forecast failed; forecasting turned off for this run")
            self.forecaster = None
            return
        self.forecasts.append(f)
        self.pending_forecasts.append(f)

    def log_events(self, events: list[dict]) -> list[dict]:
        """Events raised outside a sim step (stopping the stress test, a controller swap) still
        go to the log, and to the browsers with the next tick."""
        if self.logger and events:
            self.logger.record(None, events, [])
        self.pending_events.extend(events)
        return events

    def close(self) -> None:
        if self.logger and not self.logger.closed:
            if self.sim.time < MIN_RUN_TO_KEEP_S:
                self.logger.discard()
            else:
                self.logger.close()
        if self in self.sim.observers:
            self.sim.observers.remove(self)

    # ------------------------------------------------------------------ read-outs
    def totals(self) -> dict:
        return {"controller": self.controller,
                "waiting_veh_s": round(self.waited_veh_s, 1),
                "mean_delay_s": round(self.delay_sum / self.trips, 1) if self.trips else None,
                "trips": self.trips,
                "mean_ped_wait_s": round(self.ped_wait_sum / self.ped_trips, 1) if self.ped_trips else None,
                "waiting_now": waiting_now(self.sim)}

    def frame(self) -> dict:
        sim = self.sim
        tindex = {name: i for i, name in enumerate(sorted(sim.cfg.vehicle_types))}
        vehicles = []
        for eid in sim.junction.edge_order:
            lane = sim.lanes[eid]
            if not lane:
                continue
            xs, ys, hs = sim.lane_poses(eid)
            for veh, x, y, h in zip(lane, xs, ys, hs):
                # Compact arrays instead of dicts: ~3x smaller on the wire at 20 Hz.
                vehicles.append([veh.id, round(float(x), 2), round(float(y), 2),
                                 round(float(h), 3), round(veh.v, 2), tindex[veh.vtype]])
        # Pedestrians: [id, x, y, state] with state 0 approach, 1 wait, 2 cross, 3 leave.
        peds = [[p.id, round(float(p.pos[0]), 2), round(float(p.pos[1]), 2), int(p.state)]
                for p in sim.peds.peds]
        return {"t": round(sim.time, 2), "phase": sim.signals.phase_name,
                "stage": sim.signals.stage.value,
                "signals": sim.signals.arm_states(sim.junction.inbound),
                "ped_signal": sim.signals.ped_signal(), "controller": self.controller,
                "backlog": sim.spawner.backlog_size(), "vehicles": vehicles, "peds": peds}


class SharedModels:
    """Loaded once per server and shared by every session: the trained PPO network (torch, ~2 s to
    import, so on first use and in a worker thread) and the queue predictor (small, at start-up).
    Sharing is safe because both are only READ: each sim gets its own policy object and decision
    clock around the one network."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rl = None
        self._lock = asyncio.Lock()
        self.forecaster = self._load_forecaster()

    def ai_available(self) -> bool:
        return self.cfg.rl_model_path().is_file()

    def _load_forecaster(self):
        """The queue predictor, if one has been trained. It is small (a text file of trees) and
        lightgbm imports quickly, so unlike the RL model it loads at start-up."""
        if not self.cfg.predict_model_path().is_file():
            return None
        try:
            from predict.forecaster import QueueForecaster
            return QueueForecaster.load(self.cfg)
        except Exception:  # no lightgbm, or a model from an older feature list: run without it
            log.exception("queue predictor not loaded; the dashboard runs without forecasts")
            return None

    async def ensure_rl(self):
        """Load the trained network once, in a worker thread (importing torch takes seconds and
        must not freeze the stream for everyone)."""
        async with self._lock:
            if self.rl is None:
                if not self.ai_available():
                    raise ValueError("no trained AI model yet: run `python -m rl.train`")
                from rl.policy import RLPolicy
                try:
                    self.rl = await asyncio.to_thread(RLPolicy.load, self.cfg)
                except ImportError as exc:
                    raise ValueError(f"AI needs the RL dependencies (torch, sb3-contrib): {exc}") from exc
        return self.rl


class SimSession:
    def __init__(self, cfg: Config, log_runs: bool = True, runs_dir=None, shared: SharedModels | None = None,
                 private: bool = False):
        """``private``: this session belongs to one visitor (public demo), so the analyst may only
        look at runs made in it, never another visitor's."""
        self.base_cfg = cfg
        self.shared = shared or SharedModels(cfg)
        self.private = private
        self.run_dirs: list = []  # every run log this session has opened (oldest first)
        self.log_runs = log_runs
        self.runs_dir = runs_dir
        self.clients: set[Client] = set()
        self.playing = False
        self.speed = 1.0
        self.controller = cfg.signals.controller  # the AI toggle: what drives the run outside a comparison
        self.last_error: str | None = None
        self._debt = 0.0
        self.twin: LiveRun | None = None   # comparison mode: the AI run (main run = fixed timer)
        self.compare_id: str | None = None
        self.forecaster = self.shared.forecaster
        self.analyst = None  # server.analyst.AnalystService, attached by the app (Phase 12)
        if self.controller == "rl":
            # The model loads asynchronously (in a thread) and there is no event loop yet, so the
            # dashboard always starts on the fixed timer; the AI is one click away.
            self.controller = "fixed"
        self.run = self._new_run(cfg.sim.seed, None, self.controller)
        self._attach_forecaster()

    # convenience: most callers mean the main run
    @property
    def sim(self) -> Simulation:
        return self.run.sim

    @property
    def stress(self) -> LiveStressTest | None:
        return self.run.stress

    # ------------------------------------------------------------------ lifecycle
    def ai_available(self) -> bool:
        return self.shared.ai_available()

    def _attach_forecaster(self) -> None:
        """Forecast for the main run only, and not while comparing (see the module docstring)."""
        self.run.forecaster = None if self.twin else self.forecaster
        if self.twin:
            self.twin.forecaster = None

    async def _ensure_model(self):
        return await self.shared.ensure_rl()

    def _policy(self, controller: str, sim: Simulation):
        if controller == "rl":
            from rl.policy import RLPolicy
            base = self.shared.rl
            # One network, but each sim gets its own policy: the decision clock is per sim.
            return RLPolicy(base.model, self.base_cfg.rl.decision_interval_s, model_id=base.model_id,
                            max_green_s=self.base_cfg.rl.max_green_s)
        return FixedTimePolicy(sim.cfg.signals.green_times(len(sim.junction.phases)))

    def _new_run(self, seed: int, settings: dict | None, controller: str, tags: dict | None = None) -> LiveRun:
        sim = Simulation(self.base_cfg, seed=seed)
        if settings:
            # Applied before the logger exists, so the manifest config already includes them.
            self._apply_settings(sim, settings, source="reset")
        sim.policy = self._policy(controller, sim)
        sim.cfg.signals.controller = controller
        self._debt = 0.0
        run = LiveRun(sim, controller, self.log_runs, self.runs_dir, {"source": "dashboard", **(tags or {})})
        if run.logger:
            self.run_dirs.append(run.logger.dir)
        return run

    def runs(self) -> list[LiveRun]:
        return [self.run] + ([self.twin] if self.twin else [])

    def close_run(self) -> None:
        for r in self.runs():
            r.close()

    def shutdown(self) -> None:
        self.close_run()

    def own_runs(self) -> list[str]:
        """Ids of this session's run logs still on disk, oldest first (short runs are discarded)."""
        return [d.name for d in self.run_dirs if (d / "manifest.json").is_file()]

    # ------------------------------------------------------------------ settings & status
    def settings(self) -> dict:
        sim = self.sim
        return {"volume_vph": round(sim.mean_volume_vph(), 1),
                "aggressiveness": sim.cfg.driver.aggressiveness,
                # From the config, not the policy: the fixed-time plan keeps its green time while
                # the AI drives, and gets it back when the AI is switched off.
                "green_s": sim.cfg.signals.green_times(len(sim.junction.phases))[0],
                # As percentages, which is what the sliders show.
                "vehicle_mix": {t: round(100 * w / sum(sim.cfg.demand.vehicle_mix.values()), 1)
                                for t, w in sorted(sim.cfg.demand.vehicle_mix.items())},
                "ped_rate_ph": sim.cfg.pedestrians.rate_ph}

    @staticmethod
    def _apply_settings(sim: Simulation, s: dict, source: str) -> None:
        s = {k: v for k, v in s.items() if v is not None}
        green = s.get("green_s")
        if green is not None and not hasattr(sim.policy, "green_times"):
            # The AI has no green time to set; store it for the fixed-time plan instead.
            if green < sim.cfg.signals.min_green_s:
                raise ValueError(f"green_s must be >= min_green_s ({sim.cfg.signals.min_green_s})")
            sim.cfg.signals.green_s = float(green)
            del s["green_s"]
        if s:
            sim.apply_settings(**s, source=source)

    def comparison(self) -> dict | None:
        """Totals of both runs since t = 0 and the AI's change vs the fixed timer.
        Negative = the AI is better (less waiting, less delay)."""
        if not self.twin:
            return None
        f, a = self.run.totals(), self.twin.totals()
        return {"active": True, "id": self.compare_id, "t": round(self.sim.time, 1), "fixed": f, "rl": a,
                "change_pct": {"waiting": _pct(a["waiting_veh_s"], f["waiting_veh_s"]),
                               "delay": _pct(a["mean_delay_s"], f["mean_delay_s"]),
                               "ped_wait": _pct(a["mean_ped_wait_s"], f["mean_ped_wait_s"])}}

    def status(self) -> dict:
        return {"type": "status", "playing": self.playing, "speed": self.speed, "seed": self.sim.seed,
                "run_id": self.run.logger.run_id if self.run.logger else None,
                "twin_run_id": self.twin.logger.run_id if self.twin and self.twin.logger else None,
                "controller": self.controller, "ai_available": self.ai_available(),
                "compare": self.comparison(),
                "settings": self.settings(), "error": self.last_error,
                "stress": self.stress.status() if self.stress else None}

    def benchmark_brief(self) -> dict | None:
        """The latest saved `python -m bench` result, trimmed to what the dashboard shows."""
        b = latest_summary(self.base_cfg.benchmark_path())
        if not b:
            return None
        return {"id": b["id"], "created_at": b["created_at"], "seeds": b["seeds"], "score": b["score"],
                "config_hash": b["config_hash"], "definition": b["definition"]}

    def init_message(self) -> dict:
        sig = self.sim.cfg.signals
        n = len(self.sim.junction.phases)
        return {
            "type": "init",
            "layout": self.sim.junction.describe(),
            "vehicle_types": sorted(self.sim.cfg.vehicle_types),
            "vehicle_dims": {k: [p.length, p.width] for k, p in sorted(self.sim.cfg.vehicle_types.items())},
            "limits": {"volume_vph": [0, VOLUME_MAX_VPH], "green_s": [sig.min_green_s, GREEN_MAX_S],
                       "ped_rate_ph": [0, PED_RATE_MAX],
                       "speeds": list(SPEEDS), "amber_s": sig.amber_s, "all_red_s": sig.all_red_s,
                       "n_phases": n, "walk_s": self.sim.cfg.pedestrians.walk_s,
                       "ped_clear_s": self.sim.ped_clear_s},
            "ped_radius": self.sim.cfg.pedestrians.radius,
            "benchmark": self.benchmark_brief(),
            "predictor": self.forecaster.brief() if self.forecaster else None,
            "analyst": self.analyst.status() if self.analyst else None,
            "forecast_history": list(self.run.forecasts) if self.run.forecaster else [],
            "history": list(self.run.collector.history)[-HISTORY_ON_CONNECT:],
            "twin_history": list(self.twin.collector.history)[-HISTORY_ON_CONNECT:] if self.twin else None,
            "events": list(self.run.recent_events),
            "twin_events": list(self.twin.recent_events) if self.twin else None,
            **{k: v for k, v in self.status().items() if k != "type"},
        }

    # ------------------------------------------------------------------ clients
    def add_client(self, client: Client) -> None:
        self.clients.add(client)
        client.send(_dumps(self.init_message()))
        client.send_frame(_dumps(self._frame()))

    def remove_client(self, client: Client) -> None:
        self.clients.discard(client)

    def _broadcast(self, msg: dict) -> None:
        text = _dumps(msg)
        for c in self.clients:
            c.send(text)

    # ------------------------------------------------------------------ commands
    async def handle(self, cmd) -> None:
        """Apply one validated command. Raises ValueError for semantically invalid ones."""
        if isinstance(cmd, PlayCmd):
            self.playing = True
            self.last_error = None
        elif isinstance(cmd, PauseCmd):
            self.playing = False
        elif isinstance(cmd, SpeedCmd):
            if cmd.value not in SPEEDS:
                raise ValueError(f"speed must be one of {list(SPEEDS)}")
            self.speed = cmd.value
        elif isinstance(cmd, StressCmd):
            if cmd.action == "start":
                if self.stress is not None and self.stress.active:
                    raise ValueError("a stress test is already running")
                if self.twin:
                    raise ValueError("stop the comparison before running a stress test")
                self.run.stress = LiveStressTest(self.sim, self.run.collector, self.sim.cfg.benchmark,
                                                 VOLUME_MAX_VPH)
                self.playing = True  # a ramp only makes sense while the sim runs
            elif self.stress is not None and self.stress.active:
                self.run.log_events(self.stress.stop(self.sim))
        elif isinstance(cmd, ControllerCmd):
            await self._set_controller(cmd.value)
        elif isinstance(cmd, CompareCmd):
            if cmd.action == "start":
                await self._start_compare()
            else:
                self._stop_compare(cmd.keep)
        elif isinstance(cmd, ResetCmd):
            if self.stress is not None and self.stress.active:
                # Restore the user's volume first, so the reset keeps THEIR settings, not a ramp level.
                self.run.log_events(self.stress.stop(self.sim))
            keep = self.settings()
            seed = cmd.seed if cmd.seed is not None else self.sim.seed
            self.close_run()
            if self.twin:  # a reset keeps the comparison going, on the new seed
                self._open_compare(seed, keep)
            else:
                self.run = self._new_run(seed, keep, self.controller)
            self._attach_forecaster()
            self.last_error = None
            self._broadcast(self.init_message())
        elif isinstance(cmd, SettingsCmd):
            if self.stress is not None and self.stress.active:
                # The ramp owns the demand while it runs; any other change would muddle its result.
                raise ValueError("stop the stress test before changing settings")
            s = cmd.model_dump(exclude={"cmd"})
            # Both runs of a comparison change at the same simulated instant (they are in lockstep),
            # so they keep seeing identical traffic. Validated on the main run first: if it raises,
            # neither run has changed.
            for r in self.runs():
                self._apply_settings(r.sim, s, source="dashboard")
        self._broadcast(self.status())

    async def _set_controller(self, value: str) -> None:
        if self.twin:
            raise ValueError("the comparison already runs both controllers: stop it to choose one")
        if value == self.controller:
            return
        if value == "rl":
            await self._ensure_model()
        old, run = self.controller, self.run
        run.sim.policy = self._policy(value, run.sim)
        run.sim.cfg.signals.controller = value
        run.controller = self.controller = value
        run.log_events([run.collector.custom_event("controller_change", run.sim.time, {
            "from": old, "to": value, "source": "dashboard"})])

    def _open_compare(self, seed: int, settings: dict) -> None:
        self.compare_id = datetime.now().strftime("cmp-%Y%m%d-%H%M%S")
        tags = {"compare_id": self.compare_id}
        self.run = self._new_run(seed, settings, "fixed", {**tags, "compare_side": "fixed"})
        self.twin = self._new_run(seed, settings, "rl", {**tags, "compare_side": "rl"})
        self._attach_forecaster()

    async def _start_compare(self) -> None:
        if self.twin:
            raise ValueError("a comparison is already running")
        if self.stress is not None and self.stress.active:
            raise ValueError("stop the stress test before starting a comparison")
        await self._ensure_model()
        keep, seed = self.settings(), self.sim.seed
        self.close_run()
        # Both from an empty junction at t = 0: neither controller inherits the other's queues.
        self._open_compare(seed, keep)
        self.playing = True
        self.last_error = None
        self._broadcast(self.init_message())

    def _stop_compare(self, keep_side: str | None = None) -> None:
        if not self.twin:
            return
        # The user picks which side keeps running (default: the AI switch's setting); the other
        # run's log is closed. The kept side's controller becomes the AI switch's new state.
        if keep_side is not None:
            self.controller = keep_side
        keep, drop = (self.twin, self.run) if self.controller == "rl" else (self.run, self.twin)
        drop.close()
        self.run, self.twin, self.compare_id = keep, None, None
        self._attach_forecaster()
        self._broadcast(self.init_message())

    # ------------------------------------------------------------------ main loop
    def _frame(self) -> dict:
        frame = {"type": "frame", **self.run.frame()}
        if self.twin:
            frame["twin"] = self.twin.frame()
        return frame

    def tick(self, period: float) -> None:
        """Advance the sim(s) by one wall-clock tick and push updates. Synchronous and fast."""
        runs = self.runs()
        if self.playing:
            self._debt += self.speed * period
            steps = int(self._debt / self.sim.dt + 1e-9)
            if steps > MAX_STEPS_PER_TICK:
                steps, self._debt = MAX_STEPS_PER_TICK, 0.0  # can't keep up: slow down, don't spiral
            else:
                self._debt -= steps * self.sim.dt
            try:
                for _ in range(steps):
                    for r in runs:  # lockstep: both runs are always at the same simulated time
                        r.sim.step()
            except Exception as exc:  # never let a sim bug kill the server loop
                log.exception("simulation step failed")
                self.playing = False
                self.last_error = f"simulation error: {type(exc).__name__}: {exc}"
                self._broadcast(self.status())
        main, twin = self.run, self.twin
        if main.pending_rows or (twin and twin.pending_rows):
            msg = {"type": "metrics", "rows": main.pending_rows}
            if twin:
                msg["twin_rows"] = twin.pending_rows
                msg["compare"] = self.comparison()
            if main.pending_forecasts:
                msg["forecast"] = main.pending_forecasts
            self._broadcast(msg)
            for r in runs:
                r.pending_rows = []
                r.pending_forecasts = []
        if main.pending_events or (twin and twin.pending_events):
            msg = {"type": "events", "events": main.pending_events}
            if twin:
                msg["twin_events"] = twin.pending_events
            for r in runs:
                r.recent_events.extend(e for e in r.pending_events if e["type"] != "phase_change")
                r.pending_events = []
            self._broadcast(msg)
        if any(r.status_dirty for r in runs):
            for r in runs:
                r.status_dirty = False
            self._broadcast(self.status())
        if self.clients:
            text = _dumps(self._frame())  # serialise once, share with every client
            for c in self.clients:
                c.send_frame(text)

    async def run_loop(self) -> None:
        period = 1.0 / FRAME_HZ
        next_t = time.perf_counter()
        while True:
            self.tick(period)
            next_t += period
            delay = next_t - time.perf_counter()
            if delay < 0:  # we fell behind (e.g. laptop asleep): resync instead of bursting
                next_t, delay = time.perf_counter(), 0.0
            await asyncio.sleep(delay)
