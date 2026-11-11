"""Phase 13: the evaluation harness (question set, scoring rules, caching, report) without any LLM."""

import json

import httpx
import pytest
import yaml

import eval.run_rag_eval as ev
from rag.llm import GroqClient, LLMReply, RateLimited
from sim.config import PROJECT_ROOT


def test_question_set_is_consistent():
    qs = ev.load_questions()
    items = qs["questions"]
    ids = [q["id"] for q in items]
    assert len(ids) == len(set(ids))
    by_type = {t: [q for q in items if q["type"] == t] for t in ("run", "decision", "guideline", "unanswerable", "scenario")}
    assert sum(len(v) for t, v in by_type.items() if t != "scenario") >= 40      # the brief: 40+ questions
    assert all(len(by_type[t]) >= 10 for t in ("run", "decision", "guideline", "scenario"))
    doc_ids = {s["id"] for s in yaml.safe_load((PROJECT_ROOT / "rag" / "sources.yaml").read_text(encoding="utf-8"))}
    for q in items:
        if q["type"] in ev.RAG_TYPES:
            assert q["ground_truth"] and q["expected"], q["id"]
            for e in q["expected"]:
                assert (e["doc_id"] in doc_ids) if "doc_id" in e else (e["run_id"] in qs["runs"]), q["id"]
        if q["type"] == "decision":
            assert q["run_id"] == "eval_rl_s11"
        if q["type"] == "scenario":
            assert q["checks"]


def test_source_matching_and_rank():
    q = {"expected": [{"run_id": "R", "kind": "run_window", "t": 221}, {"doc_id": "fhwa-stm", "page_label": "6-28"}]}
    live = {"kind": "live_status", "source_type": "live"}
    wrong = {"run_id": "R", "kind": "run_window", "t_start": 240.0, "t_end": 300.0}
    right = {"run_id": "R", "kind": "run_window", "t_start": 180.0, "t_end": 240.0}
    assert ev.retrieval_scores(q, {"sources": [live, wrong, right]}) == (1.0, 0.5)   # live status isn't ranked
    assert ev.retrieval_scores(q, {"sources": [{"doc_id": "fhwa-stm", "page_label": "6-28"}]}) == (1.0, 1.0)
    assert ev.retrieval_scores(q, {"sources": [wrong]}) == (0.0, 0.0)
    assert ev.source_matches({"run_id": "R", "kind": "run_summary"}, {"run_id": "R", "kind": "run_summary"})


def test_scenario_checks():
    q = {"checks": {"volume_vph": [380, 650], "vehicle_mix.bike": [58, 62], "only": ["volume_vph", "vehicle_mix"]}}
    a = {"type": "scenario", "proposal": {"volume_vph": 500.0, "vehicle_mix": {"bike": 60.0}},
         "changes": [{"key": "volume_vph"}, {"key": "vehicle_mix"}]}
    assert ev.check_scenario(q, a) == (3, 3, [])
    a2 = {**a, "proposal": {"volume_vph": 900.0, "vehicle_mix": {"bike": 60.0}}, "changes": a["changes"] + [{"key": "ped_rate_ph"}]}
    passed, total, fails = ev.check_scenario(q, a2)
    assert (passed, total) == (1, 3) and len(fails) == 2
    assert ev.check_scenario(q, {"type": "error", "message": "x"})[0] == 0
    g = {"checks": {"reject_or_min_green": 7}}
    assert ev.check_scenario(g, {"type": "error"}) == (1, 1, [])
    assert ev.check_scenario(g, {"type": "scenario", "proposal": {"green_s": 7.0}})[0] == 1
    assert ev.check_scenario(g, {"type": "scenario", "proposal": {"green_s": 5.0}})[0] == 0
    c = {"checks": {"controller": ["fixed", None]}}
    assert ev.check_scenario(c, {"type": "scenario", "proposal": {"controller": None}, "changes": []})[0] == 1


def test_abstention_detection():
    for t in ("I don't know: the logs do not record speeds.", "I do not know.", "The sources don't contain emissions.",
              "Weather is not modelled.", "There is no data for 25:00."):
        assert ev.ABSTAIN.search(t), t
    assert not ev.ABSTAIN.search("The queue on N peaked at 24 at 4:44 [1].")
    assert ev.STRICT_IDK.search("I don’t know") and not ev.STRICT_IDK.search("no data for that minute")


def test_mean_ci():
    mu, lo, hi, n = ev.mean_ci([1.0, 1.0, 0.0, float("nan"), None])
    assert n == 3 and mu == pytest.approx(2 / 3) and 0 <= lo <= mu <= hi <= 1
    assert ev.mean_ci([])[3] == 0


def test_llm_cache_and_waiting_on_the_minute_limit(tmp_path, monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None, headers=None):
        calls.append(json["model"])
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "2"}, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}],
                                         "usage": {"prompt_tokens": 3, "completion_tokens": 1}})
    monkeypatch.setattr("rag.llm.httpx.post", fake_post)
    slept = []
    monkeypatch.setattr("rag.llm.time.sleep", slept.append)
    c = GroqClient("m1", "m2", api_key="k", cache_dir=tmp_path, wait_on_limit_s=90)
    r = c.chat([{"role": "user", "content": "hi"}])
    assert r.text == "hello" and calls == ["m1", "m1"] and slept == [3.0]   # waited, same model, no fallback
    r2 = c.chat([{"role": "user", "content": "hi"}])
    assert r2.text == "hello" and len(calls) == 2                            # served from the cache
    # Without waiting (the dashboard), a rate limit falls back to the second model, then gives up.
    calls.clear()
    monkeypatch.setattr("rag.llm.httpx.post", lambda *a, **k: (calls.append(k["json"]["model"]),
                                                              httpx.Response(429, headers={"retry-after": "3600"}, json={}))[1])
    with pytest.raises(RateLimited):
        GroqClient("m1", "m2", api_key="k").chat([{"role": "user", "content": "x"}])
    assert calls == ["m1", "m2"]


def test_report_from_saved_work(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "WORK", tmp_path / "work")
    monkeypatch.setattr(ev, "EVAL_DIR", tmp_path)
    qs = {"runs": {"R": "x"}, "questions": [
        {"id": "r1", "type": "run", "run_id": "R", "question": "q", "ground_truth": "g",
         "expected": [{"run_id": "R", "kind": "run_window", "t": 30}]},
        {"id": "u1", "type": "unanswerable", "run_id": "R", "question": "q", "ground_truth": "idk", "expected": []},
        {"id": "s1", "type": "scenario", "question": "q", "checks": {"volume_vph": [100, 200]}}]}
    src = {"run_id": "R", "kind": "run_window", "t_start": 0.0, "t_end": 60.0, "text": "t", "citation": "c"}
    mode = ev.ANSWER_MODE
    ev.append_jsonl(ev.WORK / "answers.jsonl", {"id": "r1", "mode": mode, "type": "answer", "answer": "A [1]",
                                                 "sources": [src], "grounding": {"unsupported": []}, "wall_s": 3.0})
    ev.append_jsonl(ev.WORK / "answers.jsonl", {"id": "u1", "mode": mode, "type": "answer",
                                                 "answer": "I don't know.", "sources": [], "wall_s": 2.0})
    for m in ev.METRICS:
        ev.append_jsonl(ev.WORK / "scores.jsonl", {"id": "r1", "mode": mode, "metric": m, "value": 0.8})
    ev.append_jsonl(ev.WORK / "answers.jsonl", {"id": "s1", "mode": "scenario", "type": "scenario",
                                                 "proposal": {"volume_vph": 150.0}, "changes": [{"key": "volume_vph"}]})
    retrieval_rows = [{"id": "r1", "type": "run", "mode": m, "hit": float(m != "vector"), "rr": 0.5 if m != "vector" else 0.0,
                       "retrieved": "c"} for m in ev.MODES]
    from sim.config import load_config
    out = ev.stage_report(load_config(), qs, retrieval_rows)
    md = (out / "results.md").read_text(encoding="utf-8")
    assert "**0.80**" in md and "1/1 requests fully right" in md
    assert "| Vector only | 0% / 0.00 |" in md and "| Hybrid + re-ranker | 100% / 0.50 |" in md
    meta = json.loads((out / "meta.json").read_text())
    assert meta["missing_scores"] == 0 and meta["behaviour"]["abstain_correct"] == 1.0
    assert (out / "plot.png").is_file() and (out / "retrieval.csv").is_file()


def test_llm_reply_is_cacheable():
    assert set(LLMReply("t", "m", 1, 2, 0.1).__dict__) == {"text", "model", "prompt_tokens", "completion_tokens", "seconds"}
