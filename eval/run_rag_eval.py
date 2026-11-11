"""RAG evaluation: ``python -m eval.run_rag_eval`` (resumable; run it again to continue).

WHAT is measured, per retrieval setup (vector only / hybrid / hybrid + re-ranker):
  RAGAS (LLM-judged, 0..1, higher is better), on the 36 answerable questions:
    faithfulness       share of the answer's claims that the retrieved contexts support
    answer_relevancy   how well the answer addresses the question (questions generated back
                       from the answer, compared with the real one by embedding similarity)
    context_precision  are the contexts that matter ranked near the top? (judged against the
                       ground truth)
    context_recall     share of the ground truth's claims found in the contexts
  Without an LLM (exact, cheap, reproducible):
    hit rate / MRR     did the expected chunk (document page, or run window at the asked time)
                       come back, and at what rank
    abstention         on the 6 unanswerable questions, did it say "I don't know"? and on the
                       answerable ones, did it wrongly refuse?
    grounding flags    answers where the grounding check found a number not in the sources
  Scenarios (11 requests): rule checks on the proposed settings (e.g. "60% bikes" -> bike share
  58-62%, "turn on the AI" -> controller rl and nothing else changed, "5 s green" -> rejected or
  >= the 7 s minimum green).

HOW it stays within the Groq free tier (per model: 1,000 requests/day, 8,000 tokens/minute):
  * every LLM call is cached on disk (eval/work/, gitignored), keyed by the exact request, so a run
    stopped by the daily cap resumes where it stopped and nothing is paid for twice;
  * on a per-minute limit the clients WAIT and retry the same model (no fallback: every answer must
    come from the analyst's model, every score from the judge);
  * answers come from gpt-oss-120b (the analyst), judging from gpt-oss-20b: a different (smaller)
    model with its own daily budget (see JUDGE_MODEL for why not qwen).

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


class StopForToday(Exception):
    """The daily rate limit is exhausted: progress is saved, run the command again later."""


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


# ---------------------------------------------------------------------- stage 1: answers
def check_runs(cfg, qs: dict) -> None:
    from eval.runs import make_runs
    fps = make_runs(cfg)
    bad = {k: (v, fps.get(k)) for k, v in qs["runs"].items() if fps.get(k) != v}
    if bad:
        raise SystemExit(f"evaluation runs differ from the ones the ground truths were written for: {bad}. "
                         "The simulator changed: regenerate the runs and review the ground truths.")


def stage_answers(cfg, qs: dict, limit: int | None) -> None:
    from rag.analyst import Analyst, AnalystContext
    from rag.build import KnowledgeBase
    from rag.llm import GroqClient
    from rag.retrieval import Retriever

    check_runs(cfg, qs)
    kb = KnowledgeBase(cfg)
    for rid in qs["runs"]:
        kb.index_run(cfg.runs_path() / rid)
    retriever = Retriever.from_config(cfg, store=kb.store, embedder=kb.embedder)
    # No fallback model, wait out per-minute limits, cache every reply.
    llm = GroqClient(cfg.analyst.model, None, cache_dir=WORK / "llm_cache", wait_on_limit_s=90)
    done = {(r["id"], r["mode"]) for r in load_jsonl(WORK / "answers.jsonl")}
    items = [q for q in qs["questions"] if q["type"] != "scenario"]
    scen = [q for q in qs["questions"] if q["type"] == "scenario"]
    todo = [(q, m) for q in items for m in MODES if (q["id"], m) not in done]
    todo += [(q, "scenario") for q in scen if (q["id"], "scenario") not in done]
    if limit:
        todo = todo[:limit]
    print(f"answers: {len(done)} done, {len(todo)} to go")
    settings = {"volume_vph": 400.0, "vehicle_mix": {"auto": 15.0, "bike": 45.0, "bus": 5.0, "car": 35.0},
                "ped_rate_ph": 300.0, "aggressiveness": 0.5, "green_s": 25.0, "controller": "fixed"}
    limits = {"volume_vph": [0, 1500], "green_s": [cfg.signals.min_green_s, 90], "ped_rate_ph": [0, 1200]}
    for n, (q, mode) in enumerate(todo, 1):
        analyst = Analyst(cfg, retriever, llm, kb, retrieval_mode=mode if mode != "scenario" else "hybrid_rerank")
        ctx = AnalystContext(run_ids=[q["run_id"]] if q.get("run_id") else [], settings=settings, limits=limits,
                             capacity="1,790 veh/h in total (about 450 veh/h per approach) with the 25 s fixed timer")
        t0 = time.perf_counter()
        out = analyst.ask(q["question"], [], ctx)
        if out.get("type") == "error" and "rate limit" in out.get("message", ""):
            raise StopForToday(out["message"])
        append_jsonl(WORK / "answers.jsonl", {"id": q["id"], "mode": mode, "type": q["type"],
                                              "wall_s": round(time.perf_counter() - t0, 2), **out})
        print(f"  [{n}/{len(todo)}] {q['id']:8s} {mode:14s} {out.get('type')} {out.get('intent', '')}")


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
    return float(r.value) if r.value is not None and not (isinstance(r.value, float) and math.isnan(r.value)) else float("nan")


def stage_judge(qs: dict, limit: int | None) -> None:
    from openai import RateLimitError

    by_id = {q["id"]: q for q in qs["questions"]}
    answers = [a for a in load_jsonl(WORK / "answers.jsonl") if a.get("type") == "answer" and a["mode"] in MODES
               and by_id[a["id"]]["type"] in RAG_TYPES]
    done = {(s["id"], s["mode"], s["metric"]) for s in load_jsonl(WORK / "scores.jsonl")}
    todo = [(a, m) for a in answers for m in METRICS if (a["id"], a["mode"], m) not in done]
    if limit:
        todo = todo[:limit]
    print(f"judge: {len(done)} scores done, {len(todo)} to go (judge {JUDGE_MODEL})")
    if not todo:
        return
    metrics = make_judge()

    async def run():
        for n, (a, m) in enumerate(todo, 1):
            t0 = time.perf_counter()
            try:
                v = await score_one(metrics, m, by_id[a["id"]], a)
            except RateLimitError as exc:
                raise StopForToday(f"judge rate limit: {exc}") from exc
            except Exception as exc:  # a malformed judge reply: record it, don't stop the run
                v, err = float("nan"), f"{type(exc).__name__}: {str(exc)[:200]}"
            else:
                err = None
            append_jsonl(WORK / "scores.jsonl", {"id": a["id"], "mode": a["mode"], "metric": m, "value": v,
                                                 "error": err, "s": round(time.perf_counter() - t0, 2)})
            print(f"  [{n}/{len(todo)}] {a['id']:8s} {a['mode']:14s} {m:18s} {v:.3f}" + (f"  ({err})" if err else ""))
    asyncio.run(run())


# ---------------------------------------------------------------------- stage 3: report
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
            if key.startswith("vehicle_mix."):
                val = (p.get("vehicle_mix") or {}).get(key.split(".", 1)[1])
            else:
                val = p.get(key)
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


def mean_ci(vals: list[float], seed: int = 0) -> tuple[float, float, float, int]:
    v = np.array([x for x in vals if x is not None and not math.isnan(x)])
    if not len(v):
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(5000, len(v)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi), len(v)


def stage_report(cfg, qs: dict) -> Path:
    from sim.logger import _git_commit

    by_id = {q["id"]: q for q in qs["questions"]}
    answers = load_jsonl(WORK / "answers.jsonl")
    A = {(a["id"], a["mode"]): a for a in answers}
    S = {}
    for s in load_jsonl(WORK / "scores.jsonl"):
        S[(s["id"], s["mode"], s["metric"])] = s["value"]
    now = datetime.now(timezone.utc)
    out = EVAL_DIR / "results" / f"{now:%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    rag_qs = [q for q in qs["questions"] if q["type"] in RAG_TYPES]
    idk_qs = [q for q in qs["questions"] if q["type"] == "unanswerable"]
    scn_qs = [q for q in qs["questions"] if q["type"] == "scenario"]
    rows, summary = [], []
    for q in rag_qs + idk_qs:
        for mode in MODES:
            a = A.get((q["id"], mode))
            if not a:
                continue
            hit, rr = retrieval_scores(q, a) if q["type"] in RAG_TYPES else (None, None)
            g = a.get("grounding") or {}
            rows.append({"id": q["id"], "type": q["type"], "mode": mode, "answer_type": a.get("type"),
                         **{m: S.get((q["id"], mode, m)) for m in METRICS}, "hit": hit, "rr": rr,
                         "abstained": bool(ABSTAIN.search(a.get("answer") or "")),
                         "said_idk": bool(STRICT_IDK.search(a.get("answer") or "")),
                         "unsupported_numbers": len(g.get("unsupported") or []),
                         "n_sources": len(a.get("sources") or []), "wall_s": a.get("wall_s"),
                         "answer": (a.get("answer") or a.get("message") or "").replace("\n", " ")})
    for mode in MODES:
        mr = [r for r in rows if r["mode"] == mode]
        ans = [r for r in mr if r["type"] in RAG_TYPES]
        idk = [r for r in mr if r["type"] == "unanswerable"]
        line = {"mode": mode, "answered": len(ans)}
        for m in METRICS:
            mu, lo, hi, n = mean_ci([r[m] for r in ans])
            line.update({m: mu, f"{m}_lo": lo, f"{m}_hi": hi, f"{m}_n": n})
        line["hit_rate"] = statistics.fmean(r["hit"] for r in ans) if ans else float("nan")
        line["mrr"] = statistics.fmean(r["rr"] for r in ans) if ans else float("nan")
        line["abstain_correct"] = sum(r["abstained"] for r in idk) / len(idk) if idk else float("nan")
        line["false_idk"] = sum(r["said_idk"] for r in ans) / len(ans) if ans else float("nan")
        line["grounding_flagged"] = sum(r["unsupported_numbers"] > 0 for r in ans) / len(ans) if ans else float("nan")
        for t in RAG_TYPES:
            tr = [r for r in ans if r["type"] == t]
            for m in METRICS:
                line[f"{t}:{m}"] = mean_ci([r[m] for r in tr])[0]
            line[f"{t}:hit_rate"] = statistics.fmean(r["hit"] for r in tr) if tr else float("nan")
        summary.append(line)

    scn_rows = []
    for q in scn_qs:
        a = A.get((q["id"], "scenario"))
        if not a:
            continue
        p, t, fails = check_scenario(q, a)
        scn_rows.append({"id": q["id"], "request": q["question"], "passed": p, "checks": t, "ok": p == t,
                         "failures": "; ".join(fails), "proposal": json.dumps(a.get("proposal"))})

    def write_csv(name, data):
        if data:
            with open(out / name, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)
    write_csv("per_question.csv", rows)
    write_csv("summary.csv", summary)
    write_csv("scenarios.csv", scn_rows)
    meta = {"created_at": now.isoformat(timespec="seconds"), "git_commit": _git_commit(),
            "analyst_model": cfg.analyst.model, "judge_model": JUDGE_MODEL, "embedding_model": cfg.rag.embedding_model,
            "reranker_model": cfg.rag.reranker_model, "questions": {t: sum(q["type"] == t for q in qs["questions"])
                                                                   for t in (*RAG_TYPES, "unanswerable", "scenario")},
            "answers": len(answers), "scores": len(S),
            "missing_scores": sum(1 for r in rows if r["type"] in RAG_TYPES for m in METRICS if r[m] is None),
            "runs": qs["runs"]}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    plot(out, summary)
    (out / "results.md").write_text(render(summary, scn_rows, meta), encoding="utf-8")
    return out


def _f(x, d=2):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def render(summary: list[dict], scn: list[dict], meta: dict) -> str:
    L = ["# RAG evaluation: vector vs hybrid vs hybrid + re-ranker", "",
         f"{meta['created_at']}, commit `{meta['git_commit']}`. Answers by `{meta['analyst_model']}`, judged by "
         f"`{meta['judge_model']}` (RAGAS 0.4); retrieval: `{meta['embedding_model']}` + BM25, re-ranker "
         f"`{meta['reranker_model']}`. Questions: {meta['questions']}."
         + (f" **{meta['missing_scores']} scores missing** (judge errors or not yet run)." if meta["missing_scores"] else ""), "",
         "## RAGAS (answerable questions; mean, 95% bootstrap CI)", "",
         "| Retrieval | Faithfulness | Answer relevancy | Context precision | Context recall |", "|---|---|---|---|---|"]
    for s in summary:
        cells = [f"**{_f(s[m])}** [{_f(s[f'{m}_lo'])}, {_f(s[f'{m}_hi'])}]" for m in METRICS]
        L.append(f"| {MODE_LABEL[s['mode']]} | " + " | ".join(cells) + " |")
    L += ["", "## Retrieval and behaviour (no LLM judge)", "",
          "| Retrieval | Expected chunk retrieved | MRR | Says \"I don't know\" when it should | Wrongly says it | Answers with an unverified number |",
          "|---|---|---|---|---|---|"]
    for s in summary:
        L.append(f"| {MODE_LABEL[s['mode']]} | {_f(100 * s['hit_rate'], 0)}% | {_f(s['mrr'])} | "
                 f"{_f(100 * s['abstain_correct'], 0)}% | {_f(100 * s['false_idk'], 0)}% | {_f(100 * s['grounding_flagged'], 0)}% |")
    L += ["", "## By question type", "",
          "| Type | Retrieval | Faithfulness | Answer relevancy | Context precision | Context recall | Expected chunk retrieved |",
          "|---|---|---|---|---|---|---|"]
    for t in RAG_TYPES:
        for s in summary:
            L.append(f"| {t} | {MODE_LABEL[s['mode']]} | " + " | ".join(_f(s[f'{t}:{m}']) for m in METRICS)
                     + f" | {_f(100 * s[f'{t}:hit_rate'], 0)}% |")
    if scn:
        ok = sum(r["ok"] for r in scn)
        checks = sum(r["passed"] for r in scn), sum(r["checks"] for r in scn)
        L += ["", f"## Scenarios: {ok}/{len(scn)} requests fully right ({checks[0]}/{checks[1]} checks)", "",
              "| Request | Checks | Failures |", "|---|---|---|"]
        L += [f"| {r['request']} | {r['passed']}/{r['checks']} | {r['failures'] or '–'} |" for r in scn]
    L += ["", "![results](plot.png)", ""]
    return "\n".join(L)


def plot(out: Path, summary: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#2b2a27", "#6b6a64", "#e6e4de"
    colors = {"vector": "#c9c7c0", "hybrid": "#8a8983", "hybrid_rerank": "#1f63b8"}
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.spines[["top", "right"]].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color(muted)
    ax.tick_params(colors=muted)
    ax.yaxis.grid(True, color=grid)
    ax.set_axisbelow(True)
    x = np.arange(len(METRICS))
    w = 0.26
    for i, s in enumerate(summary):
        mu = [s[m] for m in METRICS]
        err = [[max(0, s[m] - s[f"{m}_lo"]) for m in METRICS], [max(0, s[f"{m}_hi"] - s[m]) for m in METRICS]]
        ax.bar(x + (i - 1) * w, mu, w - 0.02, yerr=err, color=colors[s["mode"]], label=MODE_LABEL[s["mode"]],
               error_kw={"elinewidth": 1, "capsize": 2, "ecolor": muted})
    ax.set_xticks(x, [m.replace("_", " ") for m in METRICS])
    ax.set_ylim(0, 1.05)
    ax.set_title("RAGAS scores by retrieval setup (mean, 95% CI)", color=ink, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=ink, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "plot.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the analyst (RAGAS + retrieval metrics). Resumable.")
    ap.add_argument("--stage", choices=["answers", "judge", "report", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="only this many new items per stage (smoke test)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = load_config()
    qs = load_questions()
    try:
        if args.stage in ("answers", "all"):
            stage_answers(cfg, qs, args.limit)
        if args.stage in ("judge", "all"):
            stage_judge(qs, args.limit)
    except StopForToday as exc:
        print(f"\nStopped: {exc}\nProgress is saved in {WORK}. Run the same command again later to continue.")
        if args.stage == "all":
            print(f"report so far: {stage_report(cfg, qs)}")
        sys.exit(3)
    if args.stage in ("report", "all"):
        print(f"report: {stage_report(cfg, qs)}")


if __name__ == "__main__":
    main()
