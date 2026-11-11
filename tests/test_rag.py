"""Phase 11: knowledge base (chunking, PDF ingestion, run summaries, hybrid retrieval).

The real embedding and re-ranking models are replaced by small deterministic fakes, so these
tests run offline in seconds; what is tested is the pipeline around the models.
"""

import json
import zlib

import numpy as np
import pytest

from rag.chunking import pack_word_spans, pack_words, tokenize
from rag.retrieval import rrf_fuse, time_ranges
from rag.run_indexer import decision_sentence, event_sentence, run_chunks
from sim.config import load_config
from sim.engine import Simulation
from sim.logger import RunLogger

pytest.importorskip("chromadb")
pytest.importorskip("pymupdf")
pytest.importorskip("rank_bm25")


class FakeEmbedder:
    """Bag of words hashed into 384 dimensions: texts sharing words are close. Deterministic."""

    def encode(self, texts, show_progress=False):
        out = np.zeros((len(texts), 384), np.float32)
        for i, t in enumerate(texts):
            for w in tokenize(t):
                out[i, zlib.crc32(w.encode()) % 384] += 1.0
            n = np.linalg.norm(out[i])
            out[i] = out[i] / n if n else out[i]
        return out


class FakeReranker:
    """Score = number of query words in the text (a crude cross-encoder)."""

    def score(self, query, texts):
        q = set(tokenize(query))
        return np.array([len(q & set(tokenize(t))) for t in texts], float)


# ---------------------------------------------------------------------- chunking
def test_pack_words_respects_size_overlap_and_paragraphs():
    paras = [" ".join(f"p{i}w{j}" for j in range(60)) for i in range(5)]  # 300 words
    chunks = pack_words(paras, max_words=100, overlap=20)
    assert all(len(c.split()) <= 100 for c in chunks)
    # Ends at a paragraph break when one is in the second half of the window (60 words in).
    assert chunks[0].split()[-1] == "p0w59"
    # Consecutive chunks overlap by 20 words.
    assert chunks[1].split()[:20] == chunks[0].split()[-20:]
    spans = pack_word_spans(paras, 100, 20)
    assert spans[0][0] == 0 and spans[1][0] == 40
    assert " ".join(chunks).split()[-1] == "p4w59"  # nothing lost at the end
    with pytest.raises(ValueError):
        pack_words(paras, 50, 50)


def test_tokenize_keeps_times_and_numbers():
    assert tokenize("Why did the AI switch at 3:20 with a 120 s cycle (1.5 ratio)?") == \
        ["ai", "switch", "3:20", "120", "s", "cycle", "1.5", "ratio"]


def test_time_ranges():
    assert time_ranges("why did the AI switch at 1:35?") == [(80.0, 110.0)]
    assert time_ranges("between 3:00 and 5:00") == [(180, 300)]
    assert time_ranges("what happened at minute 4") == [(180.0, 300.0)]
    assert time_ranges("the 4th minute") == [(180.0, 300.0)]
    for q in ("is a 120 s cycle too long", "a 2:1 ratio", "at 12:05:30", "section 4D.05"):
        assert time_ranges(q) == [], q


def test_rrf_rewards_agreement_between_retrievers():
    fused = dict(rrf_fuse([["a", "b", "c"], ["c", "b", "d"]], k=60))
    assert fused["b"] == pytest.approx(2 / 62)
    assert fused["b"] > fused["a"] and fused["c"] > fused["a"]  # found by both > first by one
    assert list(dict(rrf_fuse([["a", "b", "c"], ["c", "b", "d"]], k=60)))[0] in ("b", "c")


# ---------------------------------------------------------------------- run summaries
def test_event_and_decision_sentences():
    s = event_sentence({"type": "queue_spike", "t": 152.0, "clock": "2:32", "status": "start", "approach": "W",
                        "value": 13, "phase": "S", "stage": "green", "approach_light": "R",
                        "seconds_since_approach_green": 87.0, "arrivals_last_60s": 11, "backlog": 0, "wait_mean": 24.31})
    assert s.startswith("2:32 queue spike on approach W: 13 vehicles") and "red" in s and "87 s" in s
    assert "pedestrian phase" in event_sentence({"type": "ped_phase", "t": 5, "clock": "0:05", "waiting": 3, "max_wait": 40})
    assert event_sentence({"type": "phase_change", "t": 0, "clock": "0:00"}) is None
    d = {"t": 95.0, "clock": "1:35", "phase": "N", "next_phase": "E", "time_in_green_s": 0.0, "action": "switch",
         "switched": True, "p_switch": 0.86, "value": -2.2, "queue": {"N": 1, "E": 7}, "backlog": {"N": 0, "E": 0},
         "longest_wait_s": {"N": 7.4, "E": 46.9}, "ped_call": True, "obs": {"time_in_green": 0.25}}
    s = decision_sentence(d)
    # Logs from before the fix stored 0 s on switches: the observation's 0.25 x 60 s is used instead.
    assert "green for 15 s" in s and "SWITCH, P(switch) 86%" in s and "E queue 7" in s
    assert "FORCED SWITCH" in decision_sentence({**d, "action": "keep", "forced": "max_green"})


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    cfg = load_config()
    cfg.demand.per_approach_vph = {"N": 600.0, "E": 300.0, "S": 550.0, "W": 300.0}
    cfg.pedestrians.rate_ph = 400.0
    sim = Simulation(cfg, seed=4)
    with RunLogger(sim, controller="fixed", runs_dir=tmp_path_factory.mktemp("runs")) as log:
        sim.run(250)
    # A fake RL decision, as RLPolicy writes them.
    (log.dir / "decisions.jsonl").write_text(json.dumps({
        "t": 95.0, "clock": "1:35", "phase": "N", "next_phase": "E", "time_in_green_s": 12.0, "action": "switch",
        "switched": True, "p_switch": 0.9, "value": -2.0, "queue": {a: 3 for a in "NESW"},
        "backlog": {a: 0 for a in "NESW"}, "longest_wait_s": {a: 10.0 for a in "NESW"}, "ped_call": False}) + "\n")
    return log.dir


def test_run_chunks(run_dir):
    chunks = run_chunks(run_dir, 60.0)
    kinds = [c.metadata["kind"] for c in chunks]
    assert kinds[0] == "run_summary" and kinds.count("run_window") == 5 and kinds.count("rl_decisions") == 1
    w = next(c for c in chunks if c.id.endswith(":w60"))
    assert w.metadata["citation"] == f"run {run_dir.name} @ 1:00-2:00"
    assert w.text.startswith(f"Run {run_dir.name} (fixed timer, seed 4), 1:00-2:00.")
    assert "Queued vehicles per approach" in w.text and "Arrivals:" in w.text
    d = next(c for c in chunks if c.metadata["kind"] == "rl_decisions")
    assert d.metadata["t_start"] == 60.0 and "1:35: phase N green for 12 s" in d.text
    assert all(isinstance(v, (str, int, float, bool)) for c in chunks for v in c.metadata.values())
    assert len({c.id for c in chunks}) == len(chunks)
    # Every event (but phase changes) is told in some window, with its timestamp.
    events = [json.loads(x) for x in (run_dir / "events.jsonl").read_text().splitlines()]
    told = " ".join(c.text for c in chunks)
    for e in events:
        if e["type"] in ("queue_spike", "ped_phase", "spillback") and e.get("status") != "end":
            assert e["clock"] in told


# ---------------------------------------------------------------------- PDFs
def make_pdf(path, pages):
    import pymupdf
    doc = pymupdf.open()
    for i, body in enumerate(pages):
        page = doc.new_page()
        page.insert_text((72, 40), f"Test Manual 2020 Edition                Page {100 + i}", fontsize=9)
        page.insert_textbox(pymupdf.Rect(72, 70, 540, 760), body, fontsize=10)
    doc.save(path)


PDF_PAGES = [
    "Contents\nOverview .................... 1\nCycle length ................ 2\n",
    "Section 1A.01 Cycle Length\nA cycle length that is too long increases delay for every approach. "
    + "Shorter cycles reduce waiting when demand is low. " * 12,
    "The previous section continues here about cycle length and lost time. " * 8
    + "\n\nSection 1A.02 Pedestrian Clearance\nThe pedestrian clearance interval is the crossing distance "
    + "divided by a walking speed of 3.5 feet per second. " * 6,
]


def test_pdf_ingestion(tmp_path):
    from rag.ingest import pdf_chunks
    path = tmp_path / "test_manual.pdf"
    make_pdf(path, PDF_PAGES)
    chunks = pdf_chunks(path, max_words=200, overlap=40)
    pages = {c.metadata["page"] for c in chunks}
    assert pages == {2, 3}  # the contents page (dot leaders) is skipped
    c2 = next(c for c in chunks if c.metadata["page"] == 2)
    assert c2.metadata["page_label"] == "101"            # the printed page number, not the PDF index
    assert c2.metadata["section"] == "Section 1A.01 Cycle Length"
    assert c2.metadata["citation"] == "test manual, p. 101"  # not in the catalogue: titled from the file name
    assert "Test Manual 2020 Edition" not in c2.text     # running header removed
    c3 = [c for c in chunks if c.metadata["page"] == 3]
    assert c3[0].metadata["section"] == "Section 1A.01 Cycle Length"  # page 3 starts in the previous section
    assert any(c.metadata["section"] == "Section 1A.02 Pedestrian Clearance" for c in c3) or \
        "Section 1A.02" in c3[-1].text


# ---------------------------------------------------------------------- knowledge base + retrieval
@pytest.fixture()
def kb(tmp_path, run_dir):
    from rag.build import KnowledgeBase
    cfg = load_config()
    cfg.rag.chroma_dir = str(tmp_path / "chroma")
    cfg.rag.docs_dir = str(tmp_path / "docs")
    cfg.logging.runs_dir = str(run_dir.parent)
    (tmp_path / "docs").mkdir()
    make_pdf(tmp_path / "docs" / "test_manual.pdf", PDF_PAGES)
    return KnowledgeBase(cfg, embedder=FakeEmbedder())


def test_build_is_incremental_and_removes_deleted_sources(kb, tmp_path):
    s = kb.index_docs(log=lambda *_: None)
    r = kb.index_runs(log=lambda *_: None)
    assert s["indexed"] == 1 and r["indexed"] == 1
    total = kb.store.count()
    assert kb.index_docs(log=lambda *_: None)["unchanged"] == 1
    assert kb.index_runs(log=lambda *_: None)["unchanged"] == 1
    assert kb.store.count() == total
    # A run whose logs grew is re-indexed without duplicates.
    run = next(iter(kb.state["runs"]))
    f = kb.cfg.runs_path() / run / "events.jsonl"
    f.write_text(f.read_text() + json.dumps({"id": 999, "type": "ped_phase", "t": 200.0, "clock": "3:20",
                                             "waiting": 9, "max_wait": 80.0}) + "\n")
    import os
    os.utime(f, (f.stat().st_atime, f.stat().st_mtime + 5))
    assert kb.index_run(kb.cfg.runs_path() / run) > 0
    assert kb.store.count() == total
    # A deleted PDF's chunks are removed.
    (tmp_path / "docs" / "test_manual.pdf").unlink()
    assert kb.index_docs(log=lambda *_: None)["removed"] == 1
    assert kb.store.count() < total


def test_hybrid_search_filters_and_time_priority(kb, run_dir):
    from rag.retrieval import Retriever
    kb.index_docs(log=lambda *_: None)
    kb.index_runs(log=lambda *_: None)
    r = Retriever(kb.store, FakeEmbedder(), FakeReranker(), candidates=10, rerank_pool=10, top_k=3)
    hits = r.search("pedestrian clearance interval walking speed", filters={"source_type": "doc"})
    assert hits and hits[0].metadata["source_type"] == "doc" and "p. 102" in hits[0].citation
    assert {"vector", "bm25", "rrf", "rerank"} <= set(hits[0].ranks)
    hits = r.search("cycle length", filters={"source_type": "run"})
    assert all(h.metadata["source_type"] == "run" for h in hits)
    # A time in the question puts the windows covering it first, whatever the text similarity.
    hits = r.search("what happened to the queues at 1:35?", filters={"run_id": run_dir.name})
    assert hits[0].metadata["t_start"] == 60.0 and hits[0].ranks.get("time") == 1
    for mode in ("vector", "bm25", "hybrid"):
        assert r.search("cycle length", mode=mode)
    with pytest.raises(ValueError):
        r.search("x", mode="nope")
