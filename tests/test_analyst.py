"""Phase 12: the analyst (routing, grounding check, scenarios, error handling, chat endpoint).

The language model is replaced by a scripted fake (no network, no API key): each test says what
the "model" replies, and checks what the pipeline does with it.
"""

import json

import pytest

from rag.analyst import Analyst, AnalystContext, normalise_citations, resolve_relative_time
from rag.grounding import bad_citations, unsupported_numbers
from rag.llm import LLMError, LLMReply
from rag.scenario_parser import ScenarioProposal, diff, parse_scenario
from sim.config import load_config


class ScriptedLLM:
    """Replies with the next scripted text (a dict = JSON). Records every request."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def _next(self, messages):
        self.calls.append(messages)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def chat(self, messages, **kw):
        r = self._next(messages)
        return LLMReply(r if isinstance(r, str) else json.dumps(r), "fake", 10, 10, 0.01)

    def chat_json(self, messages, **kw):
        r = self._next(messages)
        return r, LLMReply(json.dumps(r), "fake", 10, 10, 0.01)


class Hit:
    def __init__(self, cid, text, meta):
        self.id, self.text, self.metadata = cid, text, meta


class FakeRetriever:
    """Returns canned chunks and records the filters it was searched with."""

    def __init__(self):
        self.searches = []
        self.run = Hit("run:R1:w180", "Run R1, 3:00-4:00. Queued vehicles per approach: N 6->14 (peak 14 at 3:48). "
                       "Mean control delay 43.8 s.", {"source_type": "run", "kind": "run_window", "run_id": "R1",
                                                      "t_start": 180.0, "t_end": 240.0, "citation": "run R1 @ 3:00-4:00"})
        self.doc = Hit("doc:fhwa:p1:c0", "FHWA Signal Timing Manual, p. 6-28\nCycle lengths should not exceed 120 seconds.",
                       {"source_type": "doc", "kind": "doc_page", "citation": "FHWA Signal Timing Manual, p. 6-28"})

    def search(self, query, k=5, mode="hybrid_rerank", filters=None):
        self.searches.append(filters)
        if filters and filters.get("source_type") == "doc":
            return [self.doc]
        return [self.run]

    def get(self, cid):
        return ("Run R1 summary: mean control delay 59 s.", {"source_type": "run", "kind": "run_summary",
                                                            "run_id": "R1", "citation": "run R1 @ 0:00-10:00"}) \
            if cid == "run:R1:summary" else None


@pytest.fixture()
def cfg(tmp_path):
    c = load_config()
    c.analyst.log_path = str(tmp_path / "log.jsonl")
    return c


CTX = AnalystContext(run_ids=["R1"], live_status="Live dashboard at 5:00. Cycle 120 s.", now_s=300.0,
                     settings={"volume_vph": 400.0, "vehicle_mix": {"car": 35, "bike": 45, "auto": 15, "bus": 5},
                               "ped_rate_ph": 300.0, "aggressiveness": 0.5, "green_s": 25.0, "controller": "fixed"},
                     limits={"volume_vph": [0, 1500], "green_s": [7, 90], "ped_rate_ph": [0, 1200]},
                     capacity="1,790 veh/h", previous_run_id="R0")


# ---------------------------------------------------------------------- grounding
def test_grounding_accepts_quotes_and_rounding_and_rejects_new_numbers():
    src = ["N 6->14 (peak 14 at 2:57); delay 43.8 s; capacity 1,790 veh/h; value -2.567; Section 4D.05"]
    # "1 790": gpt-oss writes thousands with a narrow no-break space. A plain space is NOT a
    # separator ("1 790" could be two numbers).
    assert unsupported_numbers("Peak 14 at 2:57 [1], delay 44 s [1], 1 790 veh/h, 2.567 [1].", src) == []
    assert unsupported_numbers("1 790", src) == ["1", "790"]
    assert unsupported_numbers("It rose by 8 at 2:58, delay 43.9 s.", src) == ["2:58", "8", "43.9"]
    assert unsupported_numbers("1. A point\n2. Another [1]", src) == []           # list numbering
    assert unsupported_numbers("Minute 4 [1].", src, question="what about minute 4?") == []
    assert bad_citations("a [1][3] b [2, 4]", 3) == [4]
    assert normalise_citations("a【1】 b【3†L5-L9】") == "a[1] b[3]"


def test_relative_times_are_computed_in_code():
    assert resolve_relative_time("what happened in the last minute?", 311) == "between 4:11 and 5:11"
    assert resolve_relative_time("queues in the last 3 minutes", 311) == "between 2:11 and 5:11"
    assert resolve_relative_time("why at 3:20?", 311) is None
    assert resolve_relative_time("the last minute", None) is None


# ---------------------------------------------------------------------- pipeline
def test_run_question_is_grounded_and_cites_the_live_run(cfg):
    llm = ScriptedLLM({"intent": "run", "query": "why did N grow", "run_ref": None},
                      "The North queue rose from 6 to 14, peaking at 3:48 [2]; delay 43.8 s [2].")
    ret = FakeRetriever()
    out = Analyst(cfg, ret, llm).ask("why did N grow?", [], CTX)
    assert out["type"] == "answer" and out["grounding"]["unsupported"] == []
    assert ret.searches[0] == {"run_id": ["R1"]}                      # the live run, not every run
    kinds = [s["kind"] for s in out["sources"]]
    assert kinds[0] == "live_status" and "run_summary" in kinds       # "here/now" + the run overview
    assert [s["n"] for s in out["sources"]] == list(range(1, len(kinds) + 1))
    assert json.loads(open(cfg.analyst.log_path).read().splitlines()[-1])["question"] == "why did N grow?"


def test_invented_number_triggers_one_repair(cfg):
    llm = ScriptedLLM({"intent": "run", "query": "q", "run_ref": None},
                      "The queue grew by 8 vehicles [2].",                # 8 is computed, not in a source
                      "The queue rose from 6 to 14 [2].")                 # the repair
    out = Analyst(cfg, FakeRetriever(), llm).ask("how much did it grow?", [], CTX)
    assert out["grounding"] == {"unsupported": [], "bad_citations": [], "repaired": True}
    assert "6 to 14" in out["answer"]
    assert "8" in llm.calls[-1][-1]["content"]                          # the model was told which number


def test_unfixable_numbers_are_reported_not_hidden(cfg):
    llm = ScriptedLLM({"intent": "guideline", "query": "q", "run_ref": None},
                      "Use a 90 s cycle [1].", "Use a 90 s cycle [1].")
    out = Analyst(cfg, FakeRetriever(), llm).ask("what cycle?", [], CTX)
    assert out["grounding"]["unsupported"] == ["90"] and not out["grounding"]["repaired"]


def test_routing_filters(cfg):
    ret = FakeRetriever()
    llm = ScriptedLLM({"intent": "decision", "query": "q", "run_ref": None}, "x",
                      {"intent": "guideline", "query": "q", "run_ref": None}, "y",
                      {"intent": "mixed", "query": "q", "run_ref": None}, "z")
    a = Analyst(cfg, ret, llm)
    a.ask("why did the AI switch?", [], CTX)
    assert ret.searches[-1] == {"run_id": ["R1"], "kind": ["rl_decisions", "run_window"]}
    a.ask("cycle guidance?", [], CTX)
    assert ret.searches[-1] == {"source_type": "doc"}
    ret.searches.clear()
    a.ask("too long here?", [], CTX)
    assert ret.searches == [{"run_id": ["R1"]}, {"source_type": "doc"}]


def test_previous_run_needs_the_users_words(cfg):
    ret = FakeRetriever()
    llm = ScriptedLLM({"intent": "run", "query": "q", "run_ref": "previous"}, "x",
                      {"intent": "run", "query": "q", "run_ref": "previous"}, "y")
    a = Analyst(cfg, ret, llm)
    a.ask("what happened in the last minute?", [], CTX)   # "last minute" is not "last run"
    assert ret.searches[-1] == {"run_id": ["R1"]}
    a.ask("what happened in the previous run?", [], CTX)
    assert ret.searches[-1] == {"run_id": ["R0"]}


def test_other_intent_and_llm_errors(cfg):
    out = Analyst(cfg, FakeRetriever(), ScriptedLLM({"intent": "other", "query": "hi", "run_ref": None})).ask("hi", [], CTX)
    assert out["type"] == "answer" and out["sources"] == [] and "analyst" in out["answer"]
    out = Analyst(cfg, FakeRetriever(), ScriptedLLM(LLMError("rate limit"))).ask("x", [], CTX)
    assert out == {"type": "error", "message": "rate limit", "timing": out["timing"]}


# ---------------------------------------------------------------------- scenarios
def test_scenario_validation_and_diff():
    p = ScenarioProposal.model_validate({"title": "Rush", "volume_vph": 520, "vehicle_mix": {"bike": 60, "car": 26, "auto": 11, "bus": 3}})
    assert sum(p.vehicle_mix.values()) == pytest.approx(100, abs=0.2)
    with pytest.raises(ValueError):
        ScenarioProposal.model_validate({"title": "x", "vehicle_mix": {"bike": 60, "car": 60}})   # 120%
    with pytest.raises(ValueError):
        ScenarioProposal.model_validate({"title": "x", "volume_vph": 99999})                     # over the slider
    with pytest.raises(ValueError):
        ScenarioProposal.model_validate({"title": "x", "teleport": True})                         # unknown key
    rows = diff(ScenarioProposal(title="t", volume_vph=400.0, ped_rate_ph=900.0), CTX.settings)
    assert [r["key"] for r in rows] == ["ped_rate_ph"]  # 400 is already the volume: not a change


def test_parse_scenario_retries_once_on_invalid_json_content():
    llm = ScriptedLLM({"title": "Rush", "volume_vph": 5000},                           # out of range
                      {"title": "Rush", "volume_vph": 520, "ped_rate_ph": 1000, "controller": "rl"})
    res = parse_scenario(llm, "rush hour with the AI", CTX.settings, CTX.limits, {"capacity": "x"})
    assert {r["key"] for r in res["changes"]} == {"volume_vph", "ped_rate_ph", "controller"}
    assert "invalid" in llm.calls[-1][-1]["content"]
    with pytest.raises(LLMError):
        parse_scenario(ScriptedLLM({"title": "x", "green_s": 3}, {"title": "x", "green_s": 2}),
                       "short greens", CTX.settings, CTX.limits, {})                     # below min green, twice


# ---------------------------------------------------------------------- server
def test_chat_endpoint_validates_and_reports_unavailability(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from server.main import create_app

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("rag.llm.load_api_key", lambda: None)
    cfg = load_config()
    cfg.rag.chroma_dir = str(tmp_path / "chroma")
    with TestClient(create_app(cfg, runs_dir=tmp_path, log_runs=False)) as client:
        assert client.post("/api/chat", json={"question": ""}).status_code == 422
        assert client.post("/api/chat", json={"question": "x", "history": [{"role": "system", "content": "x"}]}).status_code == 422
        r = client.post("/api/chat", json={"question": "what happened?"}).json()
        assert r["type"] == "error" and "GROQ_API_KEY" in r["message"]
        assert client.get("/api/analyst").json()["available"] is False


def test_measured_timing_reports_what_the_lights_did():
    from server.analyst import AnalystService
    from sim.engine import Simulation

    cfg = load_config()
    cfg.pedestrians.rate_ph = 0.0
    sim = Simulation(cfg, seed=1)
    sim.run(400)
    txt = AnalystService._measured_timing(sim)
    assert "averaging 25.0 s" in txt and "actual cycle length 120 s" in txt and "0 pedestrian phases" in txt
