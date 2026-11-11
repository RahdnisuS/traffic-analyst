"""Phase 14: per-visitor sessions, the question limit and the config overlay (public demo)."""

import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from rag.analyst import Analyst, AnalystContext
from server.main import create_app
from server.sessions import RateLimiter
from sim.config import load_config


def recv_until(ws, kind, limit=400):
    for _ in range(limit):
        msg = ws.receive_json()
        if msg["type"] == kind:
            return msg
    raise AssertionError(f"no {kind} message received")


def public_cfg(tmp_path, **server):
    cfg = load_config(overlay="deploy/public.yaml")
    cfg.rag.chroma_dir = str(tmp_path / "chroma")
    for k, v in server.items():
        setattr(cfg.server, k, v)
    return cfg


def test_overlay_changes_only_what_it_names():
    base, pub = load_config(), load_config(overlay="deploy/public.yaml")
    assert base.server.sessions == "shared" and base.server.chat_per_hour == 0
    assert pub.server.sessions == "per_visitor" and pub.server.chat_per_hour > 0 and not pub.server.keep_runs
    assert pub.model_dump(exclude={"server"}) == base.model_dump(exclude={"server"})


def test_each_tab_gets_its_own_sim(tmp_path):
    app = create_app(public_cfg(tmp_path), runs_dir=tmp_path / "runs", log_runs=False)
    with TestClient(app) as c:
        with c.websocket_connect("/ws/sim?session=tab-aaaaaaaa") as a, \
             c.websocket_connect("/ws/sim?session=tab-bbbbbbbb") as b:
            recv_until(a, "init")
            recv_until(b, "init")
            a.send_json({"cmd": "settings", "volume_vph": 123})
            assert recv_until(a, "status")["settings"]["volume_vph"] == pytest.approx(123, abs=1)
            sb = c.app.state.sessions.get("tab-bbbbbbbb")
            assert sb.settings()["volume_vph"] != pytest.approx(123, abs=1)   # B is untouched
            assert c.get("/api/health").json()["sessions"] == 2
        assert c.get("/api/runs").status_code == 404  # visitors' runs are private


def test_busy_when_every_slot_is_taken_and_bad_ids_rejected(tmp_path):
    app = create_app(public_cfg(tmp_path, max_sessions=1), runs_dir=tmp_path / "runs", log_runs=False)
    with TestClient(app) as c:
        with c.websocket_connect("/ws/sim?session=first-visitor") as a:
            recv_until(a, "init")
            with c.websocket_connect("/ws/sim?session=second-visitor") as b:
                assert b.receive_json()["type"] == "busy"
                with pytest.raises(WebSocketDisconnect):
                    b.receive_json()
            # The same tab reconnecting (a reload) is not a new visitor.
            with c.websocket_connect("/ws/sim?session=first-visitor") as again:
                assert recv_until(again, "init")["type"] == "init"
        with c.websocket_connect("/ws/sim?session=../x") as bad:
            assert bad.receive_json()["type"] == "error"


def test_closed_session_frees_its_slot_and_deletes_its_runs(tmp_path):
    app = create_app(public_cfg(tmp_path, idle_close_s=0), runs_dir=tmp_path / "runs", log_runs=True)
    with TestClient(app) as c:
        mgr = c.app.state.sessions
        with c.websocket_connect("/ws/sim?session=visitor-one") as ws:
            recv_until(ws, "init")
            run_dir = mgr.get("visitor-one").run.logger.dir
            assert run_dir.is_dir()
        for _ in range(100):  # the close runs as a task on the server's event loop
            if "visitor-one" not in mgr.sessions:
                break
            time.sleep(0.02)
        assert "visitor-one" not in mgr.sessions and not run_dir.exists()


def test_chat_needs_a_live_session_and_respects_the_hourly_limit(tmp_path, monkeypatch):
    monkeypatch.setattr("server.analyst.AnalystEngine.status",
                        lambda self: {"available": True, "model": "m", "reason": None, "loaded": True})

    async def fake_ask(self, question, history, make_context):
        return {"type": "answer", "answer": "ok", "sources": []}
    monkeypatch.setattr("server.analyst.AnalystEngine.ask", fake_ask)
    app = create_app(public_cfg(tmp_path, chat_per_hour=2), runs_dir=tmp_path / "runs", log_runs=False)
    with TestClient(app) as c:
        r = c.post("/api/chat", json={"question": "hi", "session": "nobody-here"}).json()
        assert r["type"] == "error" and "Reload" in r["message"]
        with c.websocket_connect("/ws/sim?session=visitor-two") as ws:
            assert recv_until(ws, "init")["analyst"]["per_hour"] == 2
            ask = lambda ip: c.post("/api/chat", json={"question": "hi", "session": "visitor-two"},
                                    headers={"X-Forwarded-For": f"{ip}, 10.0.0.1"}).json()
            assert ask("1.1.1.1")["type"] == "answer" and ask("1.1.1.1")["type"] == "answer"
            limited = ask("1.1.1.1")
            assert limited["type"] == "error" and "2 questions per hour" in limited["message"]
            assert ask("2.2.2.2")["type"] == "answer"   # another visitor has their own allowance


def test_rate_limiter_rolls_over():
    rl = RateLimiter(2, window_s=60)
    assert rl.check("a", 0) is None and rl.check("a", 1) is None
    assert rl.check("a", 2) == pytest.approx(58)
    assert rl.check("a", 61) is None            # the first question left the window
    assert RateLimiter(0).check("a") is None    # 0 = no limit


def test_a_visitor_cannot_name_another_visitors_run():
    a = Analyst.__new__(Analyst)
    a.kb = type("KB", (), {"state": {"runs": {"mine": {}, "theirs": {}}}})()
    ctx = AnalystContext(run_ids=["live"], allowed_runs={"live", "mine"})
    assert a._target_runs({"run_ref": "mine"}, ctx) == ["mine"]
    assert a._target_runs({"run_ref": "theirs"}, ctx) == ["live"]          # falls back to the live run
    assert a._target_runs({"run_ref": "theirs"}, AnalystContext(run_ids=["live"])) == ["theirs"]  # local use
