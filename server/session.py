"""One live simulation shared by every connected browser, plus its streaming loop.

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
  * The session is the sim's only observer. It feeds the same MetricsCollector
    output to the run logger (disk) and to the browsers (live), so the dashboard
    and the logs can never disagree.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque

from fastapi import WebSocket

from sim.config import Config
from sim.engine import Simulation
from sim.logger import RunLogger
from sim.metrics import MetricsCollector

from .schemas import GREEN_MAX_S, SPEEDS, VOLUME_MAX_VPH, PauseCmd, PlayCmd, ResetCmd, SettingsCmd, SpeedCmd

log = logging.getLogger("traffic.session")

FRAME_HZ = 20
MAX_STEPS_PER_TICK = 400       # 20x speed at dt=0.1 needs 10/tick; this is a generous safety cap
MIN_RUN_TO_KEEP_S = 10.0       # dashboard runs shorter than this are discarded on reset
HISTORY_ON_CONNECT = 300       # seconds of chart history sent to a newly connected browser


def _dumps(msg: dict) -> str:
    return json.dumps(msg, separators=(",", ":"))


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


class SimSession:
    def __init__(self, cfg: Config, log_runs: bool = True, runs_dir=None):
        self.base_cfg = cfg
        self.log_runs = log_runs
        self.runs_dir = runs_dir
        self.clients: set[Client] = set()
        self.playing = False
        self.speed = 1.0
        self.seed = cfg.sim.seed
        self.recent_events: deque[dict] = deque(maxlen=100)
        self._debt = 0.0
        self._pending_rows: list[dict] = []
        self._pending_events: list[dict] = []
        self.last_error: str | None = None
        self._new_sim(self.seed, settings=None)

    # ------------------------------------------------------------------ lifecycle
    def _new_sim(self, seed: int, settings: dict | None) -> None:
        self.sim = Simulation(self.base_cfg, seed=seed)
        if settings:
            # Applied before the logger exists, so the manifest config already includes them.
            self.sim.apply_settings(**settings, source="reset")
        self.collector = MetricsCollector(self.sim, self.sim.cfg.metrics)
        self.logger = (RunLogger(self.sim, controller=self.sim.cfg.signals.controller,
                                 runs_dir=self.runs_dir, tags={"source": "dashboard"},
                                 collector=self.collector, attach=False)
                       if self.log_runs else None)
        self.sim.observers.append(self)
        self.recent_events.clear()
        self._pending_rows.clear()
        self._pending_events.clear()
        self._debt = 0.0

    def close_run(self) -> None:
        if self.logger and not self.logger.closed:
            if self.sim.time < MIN_RUN_TO_KEEP_S:
                self.logger.discard()
            else:
                self.logger.close()

    def shutdown(self) -> None:
        self.close_run()

    # ------------------------------------------------------------------ observer
    def after_step(self, sim) -> None:
        row, events, trips = self.collector.after_step(sim)
        if self.logger:
            self.logger.record(row, events, trips)
        if row is not None:
            self._pending_rows.append(row)
        self._pending_events.extend(events)

    # ------------------------------------------------------------------ settings & status
    def settings(self) -> dict:
        sim = self.sim
        greens = getattr(sim.policy, "green_times", None)
        return {"volume_vph": round(sim.mean_volume_vph(), 1),
                "aggressiveness": sim.cfg.driver.aggressiveness,
                "green_s": greens[0] if greens else None}

    def status(self) -> dict:
        return {"type": "status", "playing": self.playing, "speed": self.speed, "seed": self.sim.seed,
                "run_id": self.logger.run_id if self.logger else None,
                "settings": self.settings(), "error": self.last_error}

    def init_message(self) -> dict:
        sig = self.sim.cfg.signals
        n = len(self.sim.junction.phases)
        return {
            "type": "init",
            "layout": self.sim.junction.describe(),
            "vehicle_types": sorted(self.sim.cfg.vehicle_types),
            "vehicle_dims": {k: [p.length, p.width] for k, p in sorted(self.sim.cfg.vehicle_types.items())},
            "limits": {"volume_vph": [0, VOLUME_MAX_VPH], "green_s": [sig.min_green_s, GREEN_MAX_S],
                       "speeds": list(SPEEDS), "amber_s": sig.amber_s, "all_red_s": sig.all_red_s,
                       "n_phases": n},
            "history": list(self.collector.history)[-HISTORY_ON_CONNECT:],
            "events": list(self.recent_events),
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
        elif isinstance(cmd, ResetCmd):
            keep = self.settings()
            self.close_run()
            self._new_sim(cmd.seed if cmd.seed is not None else self.sim.seed, keep)
            self.last_error = None
            self._broadcast(self.init_message())
        elif isinstance(cmd, SettingsCmd):
            self.sim.apply_settings(volume_vph=cmd.volume_vph, aggressiveness=cmd.aggressiveness,
                                    green_s=cmd.green_s, source="dashboard")
        self._broadcast(self.status())

    # ------------------------------------------------------------------ main loop
    def _frame(self) -> dict:
        sim = self.sim
        tindex = {name: i for i, name in enumerate(sorted(sim.cfg.vehicle_types))}
        vehicles = []
        for eid in sim.junction.edge_order:
            lane = sim.lanes[eid]
            if not lane:
                continue
            xs, ys, hs = sim.junction.edges[eid].path.poses_at([v.s for v in lane])
            for veh, x, y, h in zip(lane, xs, ys, hs):
                # Compact arrays instead of dicts: ~3x smaller on the wire at 20 Hz.
                vehicles.append([veh.id, round(float(x), 2), round(float(y), 2),
                                 round(float(h), 3), round(veh.v, 2), tindex[veh.vtype]])
        return {"type": "frame", "t": round(sim.time, 2), "phase": sim.signals.phase.name,
                "stage": sim.signals.stage.value,
                "signals": sim.signals.arm_states(sim.junction.inbound),
                "backlog": sim.spawner.backlog_size(), "vehicles": vehicles}

    def tick(self, period: float) -> None:
        """Advance the sim by one wall-clock tick and push updates. Synchronous and fast."""
        if self.playing:
            self._debt += self.speed * period
            steps = int(self._debt / self.sim.dt + 1e-9)
            if steps > MAX_STEPS_PER_TICK:
                steps, self._debt = MAX_STEPS_PER_TICK, 0.0  # can't keep up: slow down, don't spiral
            else:
                self._debt -= steps * self.sim.dt
            try:
                for _ in range(steps):
                    self.sim.step()
            except Exception as exc:  # never let a sim bug kill the server loop
                log.exception("simulation step failed")
                self.playing = False
                self.last_error = f"simulation error: {type(exc).__name__}: {exc}"
                self._broadcast(self.status())
        if self._pending_rows:
            self._broadcast({"type": "metrics", "rows": self._pending_rows})
            self._pending_rows = []
        if self._pending_events:
            notable = [e for e in self._pending_events if e["type"] != "phase_change"]
            self.recent_events.extend(notable)
            self._broadcast({"type": "events", "events": self._pending_events})
            self._pending_events = []
        if self.clients:
            text = _dumps(self._frame())  # serialise once, share with every client
            for c in self.clients:
                c.send_frame(text)

    async def run(self) -> None:
        period = 1.0 / FRAME_HZ
        next_t = time.perf_counter()
        while True:
            self.tick(period)
            next_t += period
            delay = next_t - time.perf_counter()
            if delay < 0:  # we fell behind (e.g. laptop asleep): resync instead of bursting
                next_t, delay = time.perf_counter(), 0.0
            await asyncio.sleep(delay)
