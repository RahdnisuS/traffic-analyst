"""FastAPI app: REST for static data, one WebSocket for the live simulation.

Run:  python -m server          (http://127.0.0.1:8000)

Routes
  GET  /api/health            liveness probe (used by Docker/Render later)
  GET  /api/layout            junction geometry for renderers
  GET  /api/runs              index of logged runs
  GET  /api/runs/{run_id}     one run's manifest
  WS   /ws/sim                live frames + metrics + events; accepts commands
  /                           the dashboard (static files from web/)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from sim.config import PROJECT_ROOT, Config, load_config
from bench.stress import latest_summary
from sim.logger import list_runs

from .schemas import command_adapter
from .session import Client, SimSession

log = logging.getLogger("traffic.server")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")  # blocks path traversal like "../../etc"


def create_app(cfg: Config | None = None, runs_dir: str | Path | None = None,
               log_runs: bool | None = None) -> FastAPI:
    cfg = cfg or load_config()
    runs_path = Path(runs_dir) if runs_dir is not None else cfg.runs_path()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        session = SimSession(cfg, log_runs=cfg.logging.enabled if log_runs is None else log_runs,
                             runs_dir=runs_path)
        app.state.session = session
        task = asyncio.create_task(session.run())
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            session.shutdown()  # close the current run's log cleanly

    app = FastAPI(title="Traffic Analyst", lifespan=lifespan)

    @app.get("/api/health")
    async def health():
        s: SimSession = app.state.session
        return {"ok": True, "sim_time": round(s.sim.time, 1), "playing": s.playing,
                "clients": len(s.clients)}

    @app.get("/api/layout")
    async def layout():
        return app.state.session.sim.junction.describe()

    @app.get("/api/runs")
    async def runs():
        return list_runs(runs_path) if runs_path.exists() else []

    @app.get("/api/runs/{run_id}")
    async def run_detail(run_id: str):
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

    @app.websocket("/ws/sim")
    async def ws_sim(ws: WebSocket):
        await ws.accept()
        session: SimSession = app.state.session
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

    # Mounted last so /api and /ws routes take priority over static files.
    app.mount("/", StaticFiles(directory=PROJECT_ROOT / "web", html=True), name="web")
    return app


app = create_app()
