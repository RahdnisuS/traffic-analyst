"""Phase 14: error handling. Bad inputs must give a clear message and leave everything else working:
broken PDFs and run logs (the knowledge base), language-model API failures (the analyst) and
invalid configs and scenarios (Pydantic)."""

import json

import httpx
import pytest
from pydantic import ValidationError

from rag.llm import GroqClient, LLMError, RateLimited
from sim.config import load_config

from test_rag import FakeEmbedder, PDF_PAGES, make_pdf
from test_rag import run_dir  # noqa: F401  (a fixture: a small logged run)


# ---------------------------------------------------------------------- bad PDFs and run logs
def test_bad_pdfs_are_skipped_with_a_message_and_good_ones_still_index(tmp_path, run_dir):  # noqa: F811
    import pymupdf
    from rag.build import KnowledgeBase

    docs = tmp_path / "docs"
    docs.mkdir()
    make_pdf(docs / "good.pdf", PDF_PAGES)
    (docs / "not_a_pdf.pdf").write_bytes(b"this is a text file with the wrong extension")
    (docs / "truncated.pdf").write_bytes((docs / "good.pdf").read_bytes()[:300])
    blank = pymupdf.open()
    blank.new_page()  # a page with no text layer, like a scanned document
    blank.save(docs / "scanned.pdf")
    enc = pymupdf.open(docs / "good.pdf")
    enc.save(docs / "locked.pdf", encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="secret")

    cfg = load_config()
    cfg.rag.chroma_dir = str(tmp_path / "chroma")
    cfg.rag.docs_dir = str(docs)
    cfg.logging.runs_dir = str(run_dir.parent)
    messages = []
    stats = KnowledgeBase(cfg, embedder=FakeEmbedder()).index_docs(log=messages.append)
    assert stats["indexed"] == 1 and stats["failed"] == 4
    text = "\n".join(messages)
    for name in ("not_a_pdf.pdf", "truncated.pdf", "scanned.pdf", "locked.pdf"):
        assert name in text and "skipped" in text
    assert "OCR" in text  # the scanned PDF's message says what to do


def test_corrupt_run_log_is_skipped(tmp_path, run_dir):  # noqa: F811
    from rag.build import KnowledgeBase

    bad = run_dir.parent / "20990101-000000_fixed_s1"
    bad.mkdir()
    (bad / "manifest.json").write_text("{ not json")
    cfg = load_config()
    cfg.rag.chroma_dir = str(tmp_path / "chroma")
    cfg.logging.runs_dir = str(run_dir.parent)
    messages = []
    stats = KnowledgeBase(cfg, embedder=FakeEmbedder()).index_runs(log=messages.append)
    assert stats["indexed"] == 1 and stats["failed"] == 1 and bad.name in messages[0]


# ---------------------------------------------------------------------- language-model API failures
def _client():
    return GroqClient("m1", "m2", api_key="k")


def _post_returning(*responses):
    calls = []

    def post(url, json=None, timeout=None, headers=None):
        calls.append(json["model"])
        r = responses[min(len(calls), len(responses)) - 1]
        if isinstance(r, Exception):
            raise r
        return r
    return post, calls


@pytest.mark.parametrize("failure,expected", [
    (httpx.ConnectError("dns"), "internet connection"),
    (httpx.ReadTimeout("slow"), "internet connection"),
    (httpx.Response(401, json={"error": {"message": "bad key"}}), "API key was rejected"),
    (httpx.Response(500, json={"error": {"message": "model overloaded"}}), "model overloaded"),
    (httpx.Response(503, text="<html>gateway</html>"), "503"),
])
def test_api_failures_become_readable_errors(monkeypatch, failure, expected):
    post, _ = _post_returning(failure)
    monkeypatch.setattr("rag.llm.httpx.post", post)
    with pytest.raises(LLMError) as e:
        _client().chat([{"role": "user", "content": "hi"}])
    assert expected in str(e.value)


def test_rate_limit_falls_back_then_reports(monkeypatch):
    ok = httpx.Response(200, json={"choices": [{"message": {"content": "fine"}}], "usage": {}})
    limited = httpx.Response(429, headers={"retry-after": "30"},
                             json={"error": {"message": "Rate limit reached ... tokens per day (TPD)"}})
    post, calls = _post_returning(limited, ok)
    monkeypatch.setattr("rag.llm.httpx.post", post)
    assert _client().chat([{"role": "user", "content": "x"}]).text == "fine" and calls == ["m1", "m2"]
    post, calls = _post_returning(limited, limited)
    monkeypatch.setattr("rag.llm.httpx.post", post)
    with pytest.raises(RateLimited) as e:
        _client().chat([{"role": "user", "content": "x"}])
    assert "30 s" in str(e.value) and "TPD" in e.value.detail


def test_malformed_json_and_missing_key(monkeypatch):
    post, _ = _post_returning(httpx.Response(200, json={"choices": [{"message": {"content": "[1, 2]"}}]}))
    monkeypatch.setattr("rag.llm.httpx.post", post)
    with pytest.raises(LLMError, match="not an object"):
        _client().chat_json([{"role": "user", "content": "x"}])
    post, _ = _post_returning(httpx.Response(200, json={"choices": [{"message": {"content": "{oops"}}]}))
    monkeypatch.setattr("rag.llm.httpx.post", post)
    with pytest.raises(LLMError, match="malformed JSON"):
        _client().chat_json([{"role": "user", "content": "x"}])
    monkeypatch.setattr("rag.llm.load_api_key", lambda: None)
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        GroqClient("m1").chat([{"role": "user", "content": "x"}])


# ---------------------------------------------------------------------- invalid configs and scenarios
def test_invalid_config_files_fail_loudly(tmp_path):
    import yaml
    from sim.config import PROJECT_ROOT

    raw = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    for bad, msg in [({"sim": {"dt": -1}}, "greater than 0"), ({"sim": {"dtt": 0.1}}, "Extra inputs"),
                     ({"server": {"sessions": "everyone"}}, "shared"),
                     ({"demand": {"vehicle_mix": {"tank": 1}}}, "undefined vehicle types")]:
        p = tmp_path / "over.yaml"
        p.write_text(yaml.safe_dump(bad))
        with pytest.raises(ValidationError, match=msg):
            load_config(overlay=p)
    assert raw  # the real config itself is valid (load_config() is used everywhere else)


def test_invalid_scenarios_are_rejected_with_a_reason():
    from rag.scenario_parser import ScenarioProposal, parse_scenario
    from test_analyst import ScriptedLLM

    for bad in ({"title": "x", "volume_vph": -5}, {"title": "x", "vehicle_mix": {"tank": 50}},
                {"title": "x", "vehicle_mix": {"car": 0, "bike": 0}}, {"title": "x", "controller": "magic"},
                {"title": "", "volume_vph": 100}):
        with pytest.raises(ValidationError):
            ScenarioProposal.model_validate(bad)
    # Below the minimum green twice in a row (one retry): a readable failure, nothing applied.
    limits = {"volume_vph": [0, 1500], "green_s": [7, 90], "ped_rate_ph": [0, 1200]}
    llm = ScriptedLLM({"title": "Short greens", "green_s": 3}, {"title": "Short greens", "green_s": 4})
    with pytest.raises(LLMError, match="minimum green"):
        parse_scenario(llm, "make the greens 3 seconds", {"green_s": 25.0}, limits, {"capacity": "x"})
    assert len(llm.calls) == 2


def test_bad_websocket_commands_leave_the_sim_untouched(tmp_path):
    from fastapi.testclient import TestClient
    from server.main import create_app

    with TestClient(create_app(load_config(), runs_dir=tmp_path, log_runs=False)) as c:
        with c.websocket_connect("/ws/sim") as ws:
            while ws.receive_json()["type"] != "init":
                pass
            before = c.app.state.session.settings()
            for bad in ("not json", json.dumps({"cmd": "explode"}), json.dumps({"cmd": "settings", "volume_vph": -1}),
                        json.dumps({"cmd": "settings", "green_s": 2}), json.dumps({"cmd": "speed", "value": 7})):
                ws.send_text(bad)
                while (m := ws.receive_json())["type"] != "error":
                    pass
                assert m["message"]
            assert c.app.state.session.settings() == before
