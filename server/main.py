"""FastAPI app: REST for static data, one WebSocket for the live simulation.

Run:  python -m server          (http://127.0.0.1:8000)

Routes
  GET  /api/health            liveness probe (the Docker HEALTHCHECK)
  GET  /api/layout            junction geometry for renderers
  GET  /api/runs              index of logged runs (local use only: not on the public demo)
  GET  /api/runs/{run_id}     one run's manifest (local use only)
  GET  /api/benchmark         the latest saved capacity benchmark
  POST /api/chat              ask the analyst ({question, history, session})
  GET  /api/analyst           is the analyst available?
  WS   /ws/sim?session=<id>   live frames + metrics + events; accepts commands
  /                           the dashboard (static files from web/)

Which simulation a WebSocket or chat question belongs to is decided by ``SessionManager``
(``server.sessions``: one shared sim locally, one per browser tab on the public demo). The config
is config.yaml plus, if the TRAFFIC_OVERLAY environment variable names one, an overlay file
(deploy/public.yaml for the public demo).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from sim.config import PROJECT_ROOT, Config, load_config
from bench.stress import latest_summary
from sim.logger import list_runs

from .schemas import ChatRequest, command_adapter
from .session import Client
from .sessions import Busy, RateLimiter, SessionManager

log = logging.getLogger("traffic.server")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")  # blocks path traversal like "../../etc"


def create_app(cfg: Config | None = None, runs_dir: str | Path | None = None,
               log_runs: bool | None = None) -> FastAPI:
    cfg = cfg or load_config(overlay=os.environ.get("TRAFFIC_OVERLAY") or None)
    runs_path = Path(runs_dir) if runs_dir is not None else cfg.runs_path()
    public = cfg.server.sessions == "per_visitor"
    limiter = RateLimiter(cfg.server.chat_per_hour)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager = SessionManager(cfg, log_runs=cfg.logging.enabled if log_runs is None else log_runs,
                                 runs_dir=runs_path)
        app.state.sessions = manager
        app.state.session = manager.default  # shared mode: the one session (None per visitor)
        try:
            yield
        finally:
            await manager.shutdown()  # close every run's log cleanly

    app = FastAPI(title="Traffic Analyst", lifespan=lifespan)

    def visitor(request: Request) -> str:
        """Who is asking, for the per-visitor question limit. Behind the host's proxy the client's
        address is the first X-Forwarded-For entry. A client can forge that header, so this limit is
        a courtesy brake on a demo, not security; Groq's own daily cap is the hard limit."""
        fwd = request.headers.get("x-forwarded-for", "")
        return fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")

    @app.get("/api/health")
    async def health():
        return app.state.sessions.health()

    @app.get("/api/layout")
    async def layout():
        return app.state.sessions.layout()

    @app.get("/api/runs")
    async def runs():
        if public:  # visitors' runs are private to them
            raise HTTPException(404, "run logs are not listed on the public demo")
        return list_runs(runs_path) if runs_path.exists() else []

    @app.get("/api/runs/{run_id}")
    async def run_detail(run_id: str):
        if public:
            raise HTTPException(404, "run logs are not listed on the public demo")
        if not RUN_ID_RE.match(run_id):
            raise HTTPException(400, "invalid run id")
        mf = runs_path / run_id / "manifest.json"
        if not mf.is_file():
            raise HTTPException(404, "run not found")
        return json.loads(mf.read_text(encoding="utf-8"))

    @app.get("/api/benchmark")
    async def benchmark():
        """Full summary of the most recent `python -m bench` run (404 if none has been saved)."""
        b = latest_summary(cfg.benchmark_path())
        if not b:
            raise HTTPException(404, "no benchmark has been run yet: python -m bench")
        return b

    @app.post("/api/chat")
    async def chat(req: ChatRequest, request: Request):
        """Ask the analyst. Returns {"type": "answer" | "scenario" | "error", ...}; errors come back as
        a normal reply (shown in the chat), never as a server error page."""
        manager: SessionManager = app.state.sessions
        s = manager.get(req.session)
        if s is None:
            return {"type": "error", "message": "This chat isn't connected to a running simulation "
                                                "(it closes a few minutes after its tab does). Reload the page."}
        st = manager.engine.status()
        if not st["available"]:
            return {"type": "error", "message": st["reason"]}
        wait = limiter.check(visitor(request))  # counted only for questions that reach the model
        if wait is not None:
            return {"type": "error", "message": (
                f"The public demo allows {cfg.server.chat_per_hour} questions per hour per visitor, to keep "
                f"the free language-model quota for everyone. Try again in about {max(1, round(wait / 60))} min, "
                "or run it locally with your own key (see the README).")}
        return await s.analyst.ask(req.question.strip(), [t.model_dump() for t in req.history])

    @app.get("/api/analyst")
    async def analyst_status():
        return {**app.state.sessions.engine.status(), "per_hour": cfg.server.chat_per_hour or None}

    @app.websocket("/ws/sim")
    async def ws_sim(ws: WebSocket):
        await ws.accept()
        manager: SessionManager = app.state.sessions
        sid = ws.query_params.get("session")
        try:
            session = manager.attach(sid)
        except Busy:
            n = cfg.server.max_sessions
            await ws.send_text(json.dumps({"type": "busy", "message": (
                f"The demo is busy: {n} visitor{'s are' if n != 1 else ' is'} already running a simulation "
                "(each one gets their own). It retries automatically every 20 seconds.")}))
            await ws.close(code=1013)  # 1013 = "try again later"
            return
        except ValueError as exc:
            await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
            await ws.close(code=1008)
            return
        client = Client(ws)
        pump = asyncio.create_task(client.pump())
        session.add_client(client)
        try:
            while True:
                text = await ws.receive_text()
                try:
                    cmd = command_adapter.validate_python(json.loads(text))
                    await session.handle(cmd)
                except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                    # Tell this client what was wrong; the sim is untouched.
                    msg = exc.errors(include_url=False)[0]["msg"] if isinstance(exc, ValidationError) else str(exc)
                    client.send(json.dumps({"type": "error", "message": msg}))
        except WebSocketDisconnect:
            pass
        finally:
            session.remove_client(client)
            pump.cancel()
            manager.detach(sid, session)

    # Mounted last so /api and /ws routes take priority over static files.
    app.mount("/", StaticFiles(directory=PROJECT_ROOT / "web", html=True), name="web")
    return app


app = create_app()
