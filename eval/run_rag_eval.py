"""RAG evaluation: ``python -m eval.run_rag_eval`` (resumable; run it again to continue).

WHAT is measured:
  1. RETRIEVAL, all three setups (vector only / hybrid BM25 + vector with RRF / hybrid + cross-encoder
     re-ranker), no LLM: for each of the 36 answerable questions, the analyst's own retrieval code
     (``Analyst._retrieve``, same filters, same k, same extra run summary) is run with the question as
     the query, and scored against the chunks a good retrieval must return (document page, or the
     run window covering the asked time):
       hit rate   share of questions whose expected chunk is among the sources
       MRR        mean reciprocal rank of the first expected chunk (1 = first, 0.5 = second, ...)
  2. ANSWERS, the setup the dashboard uses (hybrid + re-ranker), judged by RAGAS (0..1, higher is better):
       faithfulness       share of the answer's claims supported by the retrieved contexts
       answer_relevancy   how well the answer addresses the question (questions generated back from
                          the answer, compared with the real one by embedding similarity)
       context_precision  are the contexts that matter ranked near the top? (judged vs ground truth)
       context_recall     share of the ground truth's claims found in the contexts
     plus, without an LLM: abstention on the 6 unanswerable questions (does it say "I don't know"?),
     false refusals on the answerable ones, and answers where the grounding check flagged a number.
  3. SCENARIOS (11 requests): rule checks on the proposed settings ("60% bikes" -> bike share 58-62%,
     "turn on the AI" -> controller rl and nothing else, "5 s green" -> rejected or >= the 7 s minimum).

WHY RAGAS judges only one setup (user's choice, Phase 13): the Groq free tier gives each model about
200,000 tokens a DAY (on top of 1,000 requests/day and 8,000 tokens/minute). Answering 42 questions in
3 setups and judging 108 answers (~14k judge tokens each) would have taken ~9 days of daily re-runs.
The setups differ only in RETRIEVAL, which part 1 measures exactly and for free; RAGAS then measures
the answers of the deployed setup.

HOW it stays within the free tier:
  * every LLM call is cached on disk (eval/work/, gitignored), keyed by the exact request, so a run
    stopped by a daily cap resumes where it stopped and nothing is paid for twice;
  * on a per-minute limit the clients WAIT and retry the same model (no fallback: every answer must
    come from the analyst's model, every score from the judge);
  * answers come from gpt-oss-120b (the analyst), judging from gpt-oss-20b (see JUDGE_MODEL).

Stages: ``--stage answers`` (generate), ``judge`` (RAGAS), ``report``, or ``all`` (default).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sim.config import PROJECT_ROOT, load_config

EVAL_DIR = PROJECT_ROOT / "eval"
WORK = EVAL_DIR / "work"
QUESTIONS = EVAL_DIR / "rag_test_questions.json"
MODES = ("vector", "hybrid", "hybrid_rerank")
ANSWER_MODE = "hybrid_rerank"  # the setup the dashboard uses: answered and judged by RAGAS
MODE_LABEL = {"vector": "Vector only", "hybrid": "Hybrid (BM25 + vector, RRF)", "hybrid_rerank": "Hybrid + re-ranker"}
RAG_TYPES = ("run", "decision", "guideline")
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
# The judge. Planned: qwen3.8-27b (another model family, so the analyst isn't grading its own work), but
# on the free tier it has a 1,000 OUTPUT-tokens-per-minute cap that counts the tokens a request MAY
# produce: about one judge call a minute, 15+ hours for one pass. gpt-oss-20b (user's choice) is a
# different, smaller model than the analyst's 120b with its own daily budget; same family, so some
# self-preference bias is possible (stated in the report).
JUDGE_MODEL = "openai/gpt-oss-20b"
ABSTAIN = re.compile(r"\b(?:i\s+(?:do\s+not|don['’]t)\s+know|not\s+(?:recorded|available|logged|modell?ed)|"
                     r"no\s+(?:information|data|record)|(?:sources|logs)\s+(?:do\s+not|don['’]t)\s+(?:contain|include|"
                     r"record|mention))", re.I)
STRICT_IDK = re.compile(r"\bi\s+(?:do\s+not|don['’]t)\s+know\b", re.I)
SETTINGS = {"volume_vph": 400.0, "vehicle_mix": {"auto": 15.0, "bike": 45.0, "bus": 5.0, "car": 35.0},
            "ped_rate_ph": 300.0, "aggressiveness": 0.5, "green_s": 25.0, "controller": "fixed"}


class StopForToday(Exception):
    """A daily rate limit is exhausted: progress is saved, run the command again later."""


# ---------------------------------------------------------------------- storage
def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def load_questions() -> dict:
    return json.loads(QUESTIONS.read_text(encoding="utf-8"))


def check_runs(cfg, qs: dict) -> None:
    from eval.runs import make_runs
    fps = make_runs(cfg)
    bad = {k: (v, fps.get(k)) for k, v in qs["runs"].items() if fps.get(k) != v}
    if bad:
        raise SystemExit(f"evaluation runs differ from the ones the ground truths were written for: {bad}. "
                         "The simulator changed: regenerate the runs and review the ground truths.")


def knowledge(cfg, qs: dict):
    """The knowledge base (with the evaluation runs indexed) and its retriever."""
    from rag.build import KnowledgeBase
    from rag.retrieval import Retriever
    kb = KnowledgeBase(cfg)
    for rid in qs["runs"]:
        kb.index_run(cfg.runs_path() / rid)
    return kb, Retriever.from_config(cfg, store=kb.store, embedder=kb.embedder)


def context_for(cfg, q: dict):
    from rag.analyst import AnalystContext
    return AnalystContext(run_ids=[q["run_id"]] if q.get("run_id") else [], settings=SETTINGS,
                          limits={"volume_vph": [0, 1500], "green_s": [cfg.signals.min_green_s, 90],
                                  "ped_rate_ph": [0, 1200]},
                          capacity="1,790 veh/h in total (about 450 veh/h per approach) with the 25 s fixed timer")


# ---------------------------------------------------------------------- stage 1: answers
def stage_answers(cfg, qs: dict, limit: int | None) -> None:
    from rag.analyst import Analyst
    from rag.llm import GroqClient

    check_runs(cfg, qs)
    kb, retriever = knowledge(cfg, qs)
    # No fallback model, wait out per-minute limits, cache every reply.
    llm = GroqClient(cfg.analyst.model, None, cache_dir=WORK / "llm_cache", wait_on_limit_s=90)
    done = {(r["id"], r["mode"]) for r in load_jsonl(WORK / "answers.jsonl")}
    todo = [(q, ANSWER_MODE) for q in qs["questions"] if q["type"] != "scenario" and (q["id"], ANSWER_MODE) not in done]
    todo += [(q, "scenario") for q in qs["questions"] if q["type"] == "scenario" and (q["id"], "scenario") not in done]
    if limit:
        todo = todo[:limit]
    print(f"answers: {len(done)} stored, {len(todo)} to go", flush=True)
    for n, (q, mode) in enumerate(todo, 1):
        analyst = Analyst(cfg, retriever, llm, kb, retrieval_mode=ANSWER_MODE)
        t0 = time.perf_counter()
        out = analyst.ask(q["question"], [], context_for(cfg, q))
        if out.get("type") == "error" and "rate limit" in out.get("message", ""):
            raise StopForToday(f"{out['message']} [{out.get('detail', '')}]")
        append_jsonl(WORK / "answers.jsonl", {"id": q["id"], "mode": mode, "type": q["type"],
                                              "wall_s": round(time.perf_counter() - t0, 2), **out})
        print(f"  [{n}/{len(todo)}] {q['id']:8s} {mode:14s} {out.get('type')} {out.get('intent', '')}", flush=True)


# ---------------------------------------------------------------------- stage 2: judge
def make_judge():
    from openai import AsyncOpenAI
    from ragas.cache import DiskCacheBackend
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    from rag.llm import load_api_key

    cfg = load_config()
    client = AsyncOpenAI(api_key=load_api_key(), base_url="https://api.groq.com/openai/v1", max_retries=8)
    # One cache per judge model: RAGAS's cache key leaves the model out (measured: a switch from qwen
    # to gpt-oss-20b silently reused qwen's cached judgements).
    cache = DiskCacheBackend(str(WORK / f"ragas_cache_{JUDGE_MODEL.replace('/', '_')}"))
    # Low reasoning effort keeps the hidden reasoning short; 2,048 output tokens leave room for the
    # longest judgement (faithfulness lists a verdict for every statement of the answer).
    llm = llm_factory(JUDGE_MODEL, client=client, cache=cache, max_tokens=2048, reasoning_effort="low")
    emb = HuggingFaceEmbeddings(model=cfg.rag.embedding_model, cache=cache)
    return {"faithfulness": Faithfulness(llm=llm), "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=emb),
            "context_precision": ContextPrecision(llm=llm), "context_recall": ContextRecall(llm=llm)}


async def score_one(metrics: dict, name: str, q: dict, a: dict) -> float:
    ctxs = [s["text"] for s in a.get("sources", [])] or ["(no context retrieved)"]
    m = metrics[name]
    if name == "faithfulness":
        r = await m.ascore(user_input=q["question"], response=a["answer"], retrieved_contexts=ctxs)
    elif name == "answer_relevancy":
        r = await m.ascore(user_input=q["question"], response=a["answer"])
    elif name == "context_precision":
        r = await m.ascore(user_input=q["question"], reference=q["ground_truth"], retrieved_contexts=ctxs)
    else:
        r = await m.ascore(user_input=q["question"], retrieved_contexts=ctxs, reference=q["ground_truth"])
    v = r.value
    return float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else float("nan")


def stage_judge(qs: dict, limit: int | None) -> None:
    from openai import RateLimitError

    by_id = {q["id"]: q for q in qs["questions"]}
    answers = [a for a in load_jsonl(WORK / "answers.jsonl") if a.get("type") == "answer" and a["mode"] == ANSWER_MODE
               and by_id[a["id"]]["type"] in RAG_TYPES]
    done = {(s["id"], s["mode"], s["metric"]) for s in load_jsonl(WORK / "scores.jsonl")}
    todo = [(a, m) for a in answers for m in METRICS if (a["id"], a["mode"], m) not in done]
    if limit:
        todo = todo[:limit]
    print(f"judge: {len(done)} scores stored, {len(todo)} to go (judge {JUDGE_MODEL})", flush=True)
    if not todo:
        return
    metrics = make_judge()

    async def run():
        for n, (a, m) in enumerate(todo, 1):
            t0 = time.perf_counter()
            try:
                v, err = await score_one(metrics, m, by_id[a["id"]], a), None
            except RateLimitError as exc:
                raise StopForToday(f"judge rate limit: {str(exc)[:300]}") from exc
            except Exception as exc:  # a malformed judge reply: record it, don't stop the run
                if "rate_limit" in str(exc) or "429" in str(exc):
                    raise StopForToday(f"judge rate limit: {str(exc)[:300]}") from exc
                v, err = float("nan"), f"{type(exc).__name__}: {str(exc)[:200]}"
            append_jsonl(WORK / "scores.jsonl", {"id": a["id"], "mode": a["mode"], "metric": m, "value": v,
                                                 "error": err, "s": round(time.perf_counter() - t0, 2)})
            print(f"  [{n}/{len(todo)}] {a['id']:8s} {m:18s} {v:.3f}" + (f"  ({err})" if err else ""), flush=True)
    asyncio.run(run())


# ---------------------------------------------------------------------- retrieval (no LLM)
INTENT_OF = {"run": "run", "decision": "decision", "guideline": "guideline"}


def retrieval_eval(cfg, qs: dict) -> list[dict]:
    """Every setup's retrieval for every answerable question, through the analyst's own code
    (``Analyst._retrieve``) with the question as the query: no LLM, so it is exact, free and
    reproducible. The route a real question would get is the question's type."""
    from rag.analyst import Analyst

    kb, retriever = knowledge(cfg, qs)
    rows = []
    for mode in MODES:
        analyst = Analyst(cfg, retriever, llm=None, kb=kb, retrieval_mode=mode)
        for q in qs["questions"]:
            if q["type"] not in RAG_TYPES:
                continue
            route = {"intent": INTENT_OF[q["type"]], "query": q["question"], "run_ref": None}
            sources = analyst._retrieve(route, context_for(cfg, q))
            hit, rr = retrieval_scores(q, {"sources": sources})
            rows.append({"id": q["id"], "type": q["type"], "mode": mode, "hit": hit, "rr": rr,
                         "retrieved": " | ".join(s["citation"] for s in sources)})
    return rows


def source_matches(src: dict, exp: dict) -> bool:
    if "doc_id" in exp:
        return src.get("doc_id") == exp["doc_id"] and str(src.get("page_label")) == str(exp["page_label"])
    if src.get("run_id") != exp["run_id"] or src.get("kind") != exp["kind"]:
        return False
    if "t" in exp:
        return float(src.get("t_start", -1)) < exp["t"] <= float(src.get("t_end", -1))
    return True


def retrieval_scores(q: dict, a: dict) -> tuple[float, float]:
    """(hit, reciprocal rank) of the first expected chunk among the sources the answer was given."""
    srcs = [s for s in a.get("sources", []) if s.get("kind") != "live_status"]
    for rank, s in enumerate(srcs, 1):
        if any(source_matches(s, e) for e in q.get("expected", [])):
            return 1.0, 1.0 / rank
    return 0.0, 0.0


# ---------------------------------------------------------------------- scenarios
def check_scenario(q: dict, a: dict) -> tuple[int, int, list[str]]:
    """(checks passed, checks, failures) for one scenario answer."""
    c = q["checks"]
    if "reject_or_min_green" in c:
        if a.get("type") == "error":
            return 1, 1, []
        g = (a.get("proposal") or {}).get("green_s")
        ok = g is None or g >= c["reject_or_min_green"]
        return int(ok), 1, [] if ok else [f"green_s {g} < minimum"]
    if a.get("type") != "scenario":
        return 0, max(1, len(c)), [f"no proposal ({a.get('type')}: {a.get('message', a.get('intent'))})"]
    p = a["proposal"]
    passed, total, fails = 0, 0, []
    for key, want in c.items():
        if key == "only":
            changed = {ch["key"] for ch in a.get("changes", [])}
            ok = changed <= set(want)
            label = f"only {want} changed (changed: {sorted(changed)})"
        else:
            val = (p.get("vehicle_mix") or {}).get(key.split(".", 1)[1]) if key.startswith("vehicle_mix.") else p.get(key)
            if isinstance(want, list) and len(want) == 2 and all(isinstance(x, (int, float)) for x in want):
                ok = val is not None and want[0] - 1e-6 <= float(val) <= want[1] + 1e-6
            elif isinstance(want, list):
                ok = val in want
            else:
                ok = val == want
            label = f"{key}={val} (want {want})"
        passed += ok
        total += 1
        if not ok:
            fails.append(label)
    return passed, total, fails


# ---------------------------------------------------------------------- report
def mean_ci(vals: list[float], seed: int = 0) -> tuple[float, float, float, int]:
    v = np.array([x for x in vals if x is not None and not math.isnan(x)])
    if not len(v):
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(5000, len(v)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi), len(v)


def stage_report(cfg, qs: dict, retrieval_rows: list[dict] | None = None) -> Path:
    from sim.logger import _git_commit

    answers = load_jsonl(WORK / "answers.jsonl")
    A = {(a["id"], a["mode"]): a for a in answers}
    S = {(s["id"], s["mode"], s["metric"]): s["value"] for s in load_jsonl(WORK / "scores.jsonl")}
    retrieval_rows = retrieval_eval(cfg, qs) if retrieval_rows is None else retrieval_rows
    now = datetime.now(timezone.utc)
    out = EVAL_DIR / "results" / f"{now:%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for q in qs["questions"]:
        if q["type"] not in (*RAG_TYPES, "unanswerable"):
            continue
        a = A.get((q["id"], ANSWER_MODE))
        if not a:
            continue
        g = a.get("grounding") or {}
        text = a.get("answer") or ""
        rows.append({"id": q["id"], "type": q["type"], "answer_type": a.get("type"),
                     **{m: S.get((q["id"], ANSWER_MODE, m)) for m in METRICS},
                     "abstained": bool(ABSTAIN.search(text)), "said_idk": bool(STRICT_IDK.search(text)),
                     "unsupported_numbers": len(g.get("unsupported") or []), "repaired": bool(g.get("repaired")),
                     "n_sources": len(a.get("sources") or []), "wall_s": a.get("wall_s"),
                     "question": q["question"], "answer": (text or a.get("message") or "").replace("\n", " ")})
    ans = [r for r in rows if r["type"] in RAG_TYPES]
    idk = [r for r in rows if r["type"] == "unanswerable"]

    ragas = {}
    for group in ("all", *RAG_TYPES):
        rs = ans if group == "all" else [r for r in ans if r["type"] == group]
        ragas[group] = {m: mean_ci([r[m] for r in rs]) for m in METRICS}
    behaviour = {
        "answered": len(ans), "unanswerable": len(idk),
        "abstain_correct": sum(r["abstained"] for r in idk) / len(idk) if idk else float("nan"),
        "false_idk": sum(r["said_idk"] for r in ans) / len(ans) if ans else float("nan"),
        "grounding_flagged": sum(r["unsupported_numbers"] > 0 for r in ans) / len(ans) if ans else float("nan"),
        "repaired": sum(r["repaired"] for r in ans) / len(ans) if ans else float("nan"),
        "median_wall_s": statistics.median([r["wall_s"] for r in rows if r["wall_s"]]) if rows else float("nan"),
    }
    retrieval = {}
    for mode in MODES:
        for group in ("all", *RAG_TYPES):
            rs = [r for r in retrieval_rows if r["mode"] == mode and (group == "all" or r["type"] == group)]
            retrieval[(mode, group)] = (statistics.fmean(r["hit"] for r in rs) if rs else float("nan"),
                                        statistics.fmean(r["rr"] for r in rs) if rs else float("nan"), len(rs))

    scn = []
    for q in qs["questions"]:
        if q["type"] != "scenario" or (q["id"], "scenario") not in A:
            continue
        p, t, fails = check_scenario(q, A[(q["id"], "scenario")])
        scn.append({"id": q["id"], "request": q["question"], "passed": p, "checks": t, "ok": p == t,
                    "failures": "; ".join(fails), "proposal": json.dumps(A[(q["id"], "scenario")].get("proposal"))})

    def write_csv(name, data):
        if data:
            with open(out / name, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)
    write_csv("answers.csv", rows)
    write_csv("retrieval.csv", retrieval_rows)
    write_csv("scenarios.csv", scn)
    write_csv("summary.csv", [{"group": g, "metric": m, "mean": v[0], "ci_lo": v[1], "ci_hi": v[2], "n": v[3]}
                              for g, d in ragas.items() for m, v in d.items()]
              + [{"group": f"retrieval:{mode}:{g}", "metric": k, "mean": val, "ci_lo": "", "ci_hi": "", "n": n}
                 for (mode, g), (hit, mrr, n) in retrieval.items() for k, val in (("hit_rate", hit), ("mrr", mrr))])
    meta = {"created_at": now.isoformat(timespec="seconds"), "git_commit": _git_commit(),
            "analyst_model": cfg.analyst.model, "judge_model": JUDGE_MODEL, "answer_mode": ANSWER_MODE,
            "embedding_model": cfg.rag.embedding_model, "reranker_model": cfg.rag.reranker_model,
            "questions": {t: sum(q["type"] == t for q in qs["questions"]) for t in (*RAG_TYPES, "unanswerable", "scenario")},
            "answers": len(rows), "scores": sum(1 for k in S if k[1] == ANSWER_MODE),
            "missing_scores": sum(1 for r in ans for m in METRICS if r[m] is None or (isinstance(r[m], float) and math.isnan(r[m]))),
            "behaviour": behaviour, "runs": qs["runs"]}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    plot(out, ragas, retrieval)
    (out / "results.md").write_text(render(ragas, behaviour, retrieval, scn, meta), encoding="utf-8")
    return out


def _f(x, d=2):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def _pct(x):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.0f}%"


def render(ragas: dict, beh: dict, retrieval: dict, scn: list[dict], meta: dict) -> str:
    q = meta["questions"]
    L = ["# RAG evaluation", "",
         f"{meta['created_at']}, commit `{meta['git_commit']}`. Answers by `{meta['analyst_model']}` with "
         f"{MODE_LABEL[meta['answer_mode']].lower()} retrieval, judged by `{meta['judge_model']}` (RAGAS 0.4). "
         f"Retrieval: `{meta['embedding_model']}` + BM25, re-ranker `{meta['reranker_model']}`. "
         f"Questions: {q['run']} about runs, {q['decision']} about AI decisions, {q['guideline']} about guidelines, "
         f"{q['unanswerable']} unanswerable, {q['scenario']} scenario requests."
         + (f" **{meta['missing_scores']} RAGAS scores missing** (judge errors or not yet run)." if meta["missing_scores"] else ""),
         "", "## Retrieval: vector vs hybrid vs hybrid + re-ranker (no LLM)", "",
         "Did the chunk a correct answer needs (the document page, or the run window at the time asked about) come "
         "back among the sources the analyst gets, and at what rank? Same code, filters and k as the analyst.", "",
         "| Retrieval | All (hit rate / MRR) | Runs | AI decisions | Guidelines |", "|---|---|---|---|---|"]
    for mode in MODES:
        cells = [f"{_pct(retrieval[(mode, g)][0])} / {_f(retrieval[(mode, g)][1])}" for g in ("all", *RAG_TYPES)]
        L.append(f"| {MODE_LABEL[mode]} | " + " | ".join(cells) + " |")
    L += ["", f"## Answers (RAGAS, {MODE_LABEL[meta['answer_mode']].lower()}; mean, 95% bootstrap CI)", "",
          "| Questions | n | Faithfulness | Answer relevancy | Context precision | Context recall |", "|---|---|---|---|---|---|"]
    for g in ("all", *RAG_TYPES):
        d = ragas[g]
        n = max(v[3] for v in d.values())
        L.append(f"| {'All answerable' if g == 'all' else g} | {n} | "
                 + " | ".join(f"**{_f(d[m][0])}** [{_f(d[m][1])}, {_f(d[m][2])}]" for m in METRICS) + " |")
    L += ["", "## Behaviour (no LLM judge)", "",
          f"- Says \"I don't know\" (or that the data isn't recorded) on the unanswerable questions: **{_pct(beh['abstain_correct'])}** of {beh['unanswerable']}",
          f"- Wrongly says \"I don't know\" on answerable questions: **{_pct(beh['false_idk'])}** of {beh['answered']}",
          f"- Answers where the grounding check still found a number not in the sources: **{_pct(beh['grounding_flagged'])}** "
          f"(repaired by the second pass: {_pct(beh['repaired'])})",
          f"- Median time per answer: {_f(beh['median_wall_s'], 1)} s (Groq free tier, incl. waits for its per-minute limit)"]
    if scn:
        ok = sum(r["ok"] for r in scn)
        checks = sum(r["passed"] for r in scn), sum(r["checks"] for r in scn)
        L += ["", f"## Scenarios: {ok}/{len(scn)} requests fully right ({checks[0]}/{checks[1]} checks)", "",
              "| Request | Checks | Failures |", "|---|---|---|"]
        L += [f"| {r['request']} | {r['passed']}/{r['checks']} | {r['failures'] or '–'} |" for r in scn]
    L += ["", "![results](plot.png)", "",
          "Notes: the judge (gpt-oss-20b) is from the same model family as the analyst (gpt-oss-120b), so some "
          "self-preference bias is possible; the retrieval and behaviour numbers need no judge. RAGAS judged only "
          "the deployed setup because of the Groq free tier's daily token cap (see eval/run_rag_eval.py).", ""]
    return "\n".join(L)


def plot(out: Path, ragas: dict, retrieval: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#2b2a27", "#6b6a64", "#e6e4de"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    for ax in (a1, a2):
        ax.spines[["top", "right"]].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color(muted)
        ax.tick_params(colors=muted)
        ax.yaxis.grid(True, color=grid)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 1.05)
    groups = ("all", *RAG_TYPES)
    colors = {"vector": "#c9c7c0", "hybrid": "#8a8983", "hybrid_rerank": "#1f63b8"}
    x = np.arange(len(groups))
    w = 0.26
    for i, mode in enumerate(MODES):
        a1.bar(x + (i - 1) * w, [retrieval[(mode, g)][0] for g in groups], w - 0.02, color=colors[mode], label=MODE_LABEL[mode])
    a1.set_xticks(x, ["all", "runs", "AI decisions", "guidelines"])
    a1.set_title("Expected chunk retrieved (hit rate), by setup", color=ink, fontsize=11, loc="left")
    a1.legend(frameon=False, fontsize=8, labelcolor=ink, loc="lower left")
    d = ragas["all"]
    mu = [d[m][0] for m in METRICS]
    err = [[max(0, d[m][0] - d[m][1]) if not math.isnan(d[m][0]) else 0 for m in METRICS],
           [max(0, d[m][2] - d[m][0]) if not math.isnan(d[m][0]) else 0 for m in METRICS]]
    a2.bar(np.arange(len(METRICS)), [0 if math.isnan(v) else v for v in mu], 0.6, yerr=err, color="#1f63b8",
           error_kw={"elinewidth": 1, "capsize": 3, "ecolor": muted})
    a2.set_xticks(np.arange(len(METRICS)), [m.replace("_", " ") for m in METRICS])
    a2.set_title("RAGAS, hybrid + re-ranker (mean, 95% CI)", color=ink, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out / "plot.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the analyst (retrieval metrics + RAGAS). Resumable.")
    ap.add_argument("--stage", choices=["answers", "judge", "report", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="only this many new items per stage (smoke test)")
    ap.add_argument("--until-done", action="store_true",
                    help="on a daily rate limit, wait --retry-min minutes and continue, until everything is done")
    ap.add_argument("--retry-min", type=float, default=15.0)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = load_config()
    qs = load_questions()
    while True:
        try:
            # Judge first: it uses the judge model's budget, so answered questions get scored even
            # while the analyst model's daily budget is spent (and vice versa).
            if args.stage in ("judge", "all"):
                stage_judge(qs, args.limit)
            if args.stage in ("answers", "all"):
                stage_answers(cfg, qs, args.limit)
            if args.stage in ("judge", "all"):
                stage_judge(qs, args.limit)
            break
        except StopForToday as exc:
            print(f"\nStopped: {exc}\nProgress is saved in {WORK}.", flush=True)
            if not args.until_done:
                print("Run the same command again later to continue (or add --until-done).")
                sys.exit(3)
            print(f"Waiting {args.retry_min:g} min for the rate limit ({datetime.now():%H:%M})...", flush=True)
            time.sleep(60 * args.retry_min)
    if args.stage in ("report", "all"):
        print(f"report: {stage_report(cfg, qs)}")


if __name__ == "__main__":
    main()
