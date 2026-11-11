"""Which simulation a browser gets (Phase 14).

``server.sessions: shared`` (local use): one session for everybody, as in Phases 3-13. Every browser
sees and drives the same junction, which is what you want on your own machine (and what the
headless screenshot scripts rely on).

``server.sessions: per_visitor`` (the public demo, deploy/public.yaml): each browser TAB gets its own
session, keyed by an id the page keeps in sessionStorage (so a reload reconnects to the same sim,
while a new tab starts a new one). Why: on a public link, two strangers would otherwise move each
other's sliders. Limits, because every sim costs CPU on a 2-vCPU free host:
  * at most ``max_sessions`` at once; a further visitor gets a "busy" message and retries later;
  * a session whose last tab closed survives ``idle_close_s`` (a reload or a flaky phone network
    reconnects to it), then is closed; with ``keep_runs: false`` its run logs are deleted and
    dropped from the analyst's index, so the disk and the index don't grow with every visitor.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import time
from collections import deque
from contextlib import suppress

from sim.config import Config

from .analyst import AnalystEngine, AnalystService
from .session import SharedModels, SimSession

log = logging.getLogger("traffic.sessions")
SID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
SHARED_ID = "shared"


class Busy(Exception):
    """Every per-visitor slot is taken."""


class SessionManager:
    def __init__(self, cfg: Config, log_runs: bool, runs_dir):
        """Create inside a running event loop (each session's stream is an asyncio task)."""
        self.cfg = cfg
        self.log_runs = log_runs
        self.runs_dir = runs_dir
        self.mode = cfg.server.sessions
        self.shared = SharedModels(cfg)
        self.engine = AnalystEngine(cfg)
        self.sessions: dict[str, SimSession] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._closers: dict[str, asyncio.Task] = {}
        self._layout = None
        if self.mode == "shared":
            self._open(SHARED_ID)

    # ------------------------------------------------------------------ lookup
    def _open(self, sid: str) -> SimSession:
        s = SimSession(self.cfg, log_runs=self.log_runs, runs_dir=self.runs_dir, shared=self.shared,
                       private=self.mode == "per_visitor")
        s.analyst = AnalystService(s, self.engine)
        self.sessions[sid] = s
        self._tasks[sid] = asyncio.create_task(s.run_loop())
        log.info("session %s opened (%d active)", sid, len(self.sessions))
        return s

    def get(self, sid: str | None) -> SimSession | None:
        """The session a request belongs to (the chat), or None if it isn't running."""
        if self.mode == "shared":
            return self.sessions[SHARED_ID]
        return self.sessions.get(sid) if sid else None

    @property
    def default(self) -> SimSession | None:
        return self.sessions.get(SHARED_ID)

    def attach(self, sid: str | None) -> SimSession:
        """The session for a new WebSocket. Raises ValueError (bad id) or Busy (no free slot)."""
        if self.mode == "shared":
            return self.sessions[SHARED_ID]
        if not sid or not SID_RE.match(sid):
            raise ValueError("missing or invalid session id")
        closer = self._closers.pop(sid, None)
        if closer:
            closer.cancel()  # the tab came back (a reload) before its sim was closed
        s = self.sessions.get(sid)
        if s is None:
            if len(self.sessions) >= self.cfg.server.max_sessions:
                raise Busy()
            s = self._open(sid)
        return s

    def detach(self, sid: str | None, session: SimSession) -> None:
        """A WebSocket closed. When a per-visitor session has no tab left, close it after the grace time."""
        if self.mode == "shared" or session.clients or sid not in self.sessions:
            return
        if sid not in self._closers:
            self._closers[sid] = asyncio.create_task(self._close_later(sid))

    async def _close_later(self, sid: str) -> None:
        await asyncio.sleep(self.cfg.server.idle_close_s)
        self._closers.pop(sid, None)
        s = self.sessions.get(sid)
        if s is not None and not s.clients:
            await self.close(sid)

    # ------------------------------------------------------------------ closing
    async def close(self, sid: str) -> None:
        s = self.sessions.pop(sid, None)
        task = self._tasks.pop(sid, None)
        if task:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        if s is None:
            return
        s.shutdown()  # closes the run logs cleanly
        if not self.cfg.server.keep_runs:
            ids = [d.name for d in s.run_dirs]
            await self.engine.forget_runs(ids)
            for d in s.run_dirs:
                shutil.rmtree(d, ignore_errors=True)
        log.info("session %s closed (%d active)", sid, len(self.sessions))

    async def shutdown(self) -> None:
        for t in list(self._closers.values()):
            t.cancel()
        self._closers.clear()
        for sid in list(self.sessions):
            await self.close(sid)

    # ------------------------------------------------------------------ read-outs
    def layout(self) -> dict:
        """Junction geometry (the same for every session)."""
        if self._layout is None:
            from sim.engine import Simulation
            self._layout = Simulation(self.cfg, seed=0).junction.describe()
        return self._layout

    def health(self) -> dict:
        out = {"ok": True, "mode": self.mode, "sessions": len(self.sessions),
               "max_sessions": self.cfg.server.max_sessions if self.mode == "per_visitor" else None}
        s = self.default
        if s is not None:  # shared mode: the one sim's state, as before
            out.update(sim_time=round(s.sim.time, 1), playing=s.playing, clients=len(s.clients),
                       controller=s.controller, comparing=s.twin is not None)
        return out


class RateLimiter:
    """At most ``per_hour`` events per key in any rolling hour (the analyst on the public demo: the
    free Groq tier allows ~200k tokens a day, about 60 questions, for everybody together)."""

    def __init__(self, per_hour: int, window_s: float = 3600.0):
        self.per_hour = per_hour
        self.window_s = window_s
        self._hits: dict[str, deque] = {}

    def check(self, key: str, now: float | None = None) -> float | None:
        """Record one event for ``key``; None if allowed, else the seconds until the next is."""
        if self.per_hour <= 0:
            return None
        now = time.monotonic() if now is None else now
        q = self._hits.setdefault(key, deque())
        while q and q[0] <= now - self.window_s:
            q.popleft()
        if len(q) >= self.per_hour:
            return q[0] + self.window_s - now
        q.append(now)
        if len(self._hits) > 10_000:  # forget visitors with no recent questions
            self._hits = {k: v for k, v in self._hits.items() if v and v[-1] > now - self.window_s}
        return None
