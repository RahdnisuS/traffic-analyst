import json

import pytest
from fastapi.testclient import TestClient

from server.main import create_app
from server.session import Client
from sim.config import load_config


@pytest.fixture()
def client(tmp_path):
    app = create_app(load_config(), runs_dir=tmp_path, log_runs=True)
    with TestClient(app) as c:
        yield c


def recv_until(ws, kind, limit=400):
    for _ in range(limit):
        msg = ws.receive_json()
        if msg["type"] == kind:
            return msg
    raise AssertionError(f"no {kind} message received")


def test_health_and_layout(client):
    assert client.get("/api/health").json()["ok"] is True
    layout = client.get("/api/layout").json()
    assert layout["name"] == "four_way_2x2" and "in_N_0" in layout["edges"]


def test_dashboard_page_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Traffic Analyst" in r.text


def test_run_id_path_traversal_rejected(client):
    assert client.get("/api/runs/..%2F..%2Fetc").status_code in (400, 404)
    assert client.get("/api/runs/not_a_run").status_code == 404


def test_ws_init_then_play_advances_time(client):
    with client.websocket_connect("/ws/sim") as ws:
        init = ws.receive_json()
        assert init["type"] == "init" and init["playing"] is False
        assert init["vehicle_types"] == ["car"] and init["limits"]["speeds"][-1] == 20.0
        ws.send_json({"cmd": "speed", "value": 20})
        ws.send_json({"cmd": "play"})
        assert recv_until(ws, "status")["speed"] == 20.0
        t0 = recv_until(ws, "frame")["t"]
        rows = recv_until(ws, "metrics")["rows"]
        assert rows and "queue_N" in rows[0]
        assert recv_until(ws, "frame")["t"] > t0


def test_invalid_commands_return_errors_and_dont_crash(client):
    with client.websocket_connect("/ws/sim") as ws:
        ws.receive_json()
        for bad in ("not json", json.dumps({"cmd": "explode"}),
                    json.dumps({"cmd": "settings", "volume_vph": -5}),
                    json.dumps({"cmd": "speed", "value": 3})):
            ws.send_text(bad)
            assert recv_until(ws, "error")["message"]
        ws.send_json({"cmd": "settings", "green_s": 3})  # below min green
        assert "min_green" in recv_until(ws, "error")["message"]
        assert client.get("/api/health").json()["ok"]


def test_settings_apply_and_reset_keeps_them(client):
    with client.websocket_connect("/ws/sim") as ws:
        ws.receive_json()
        ws.send_json({"cmd": "settings", "volume_vph": 600, "aggressiveness": 0.8, "green_s": 30})
        st = recv_until(ws, "status")["settings"]
        assert st == {"volume_vph": 600.0, "aggressiveness": 0.8, "green_s": 30.0}
        ws.send_json({"cmd": "reset", "seed": 99})
        init = recv_until(ws, "init")
        assert init["seed"] == 99 and init["settings"]["green_s"] == 30.0


def test_client_outbox_drops_stale_frames_but_keeps_reliable_messages():
    c = Client(ws=None)
    c.send_frame("f1")
    c.send("m1")
    c.send_frame("f2")
    c.send("m2")
    assert c.frame == "f2" and c.dropped_frames == 1
    assert list(c.reliable) == ["m1", "m2"]
