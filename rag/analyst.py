"""The Traffic Analyst: question -> route -> retrieve -> answer with citations -> grounding check.

Pipeline for one question (each step is logged with its timing):
  1. ROUTE (one small JSON call): what kind of question is it, and what would a standalone search
     query be? Follow-ups ("and at 4:00?") are rewritten using the conversation, so retrieval sees
     a complete question. Intents:
        run       what happened in a run (queues, delays, events, "at 3:20")
        decision  why the AI kept or switched a green
        guideline traffic-engineering knowledge (cycle length, clearance intervals, PCU, ...)
        mixed     needs both ("is a 120 s cycle too long HERE?")
        scenario  set up / change the simulation -> rag/scenario_parser.py (no answer generated)
        other     greetings, off-topic
  2. RETRIEVE with the hybrid retriever, filtered by intent: the live run(s) (or a run the
     question names) for run/decision questions, documents for guideline questions, both for
     mixed. Run questions also get the run's summary and a LIVE STATUS source (current settings,
     cycle length, latest queues), because "here" and "now" refer to the dashboard, not the log.
  3. ANSWER from numbered sources only, citing [n] after each claim (rules in SYSTEM_ANSWER).
  4. GROUND: every number in the answer must appear in a source (rag/grounding.py). If not, one
     repair pass with the offending numbers named; whatever is still unsupported is returned to
     the UI as a warning next to the answer.

Every exchange is appended to ``analyst.log_path`` (question, route, the sources' ids and texts,
answer, grounding result): what Phase 13 needs to evaluate faithfulness, and what you need to
debug a bad answer.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sim.metrics import clock

from .grounding import bad_citations, unsupported_numbers
from .llm import GroqClient, LLMError
from .scenario_parser import parse_scenario

# gpt-oss sometimes cites in its own style, 【1】 or 【1†L3-L5】: normalise to [1].
_ODD_CITE = re.compile(r"【\s*(\d+)[^】]*】")


def normalise_citations(text: str) -> str:
    return _ODD_CITE.sub(lambda m: f"[{m.group(1)}]", text)


INTENTS = ("run", "decision", "guideline", "mixed", "scenario", "other")

SYSTEM_ROUTE = """You route questions for the analyst of a road-junction traffic simulator.
Return a JSON object: {"intent": ..., "query": ..., "run_ref": ...}
intent, one of:
  "run"       what happened in a simulation run: queues, waiting, delay, events, signals, at a time
  "decision"  why the AI / RL agent switched or kept a green signal
  "guideline" traffic-engineering knowledge, standards or recommendations (no reference to this run)
  "mixed"     needs both this run's data and guidelines, e.g. "is a 120 s cycle too long here?"
  "scenario"  the user wants to SET UP or CHANGE the simulation (traffic level, vehicle mix,
              pedestrians, timing, driver behaviour, AI on/off), e.g. "evening rush, 60% bikes"
  "other"     greetings, thanks, off-topic
query: a standalone search query for the question. Rewrite follow-ups using the conversation
       ("and at 4:00?" -> "what happened to the queues at 4:00"). Keep every time (m:ss), run id
       and number exactly as the user wrote them.
run_ref: a run id if the user names one (like 20261108-140830_rl_s42); "previous" ONLY if they
         explicitly mean an earlier run ("the previous run", "my last simulation"), not for
         "the last minute"; otherwise null."""

_LAST_N_MIN = re.compile(r"\b(?:last|past|previous)\s+(\d{1,2}|two|three|four|five)\s+min(?:ute)?s?\b", re.I)
_LAST_MIN = re.compile(r"\b(?:last|past|this|previous)\s+minute\b|\bjust now\b|\bright now\b|\brecently\b"
                       r"|\bat the moment\b|\bcurrently\b|\bnow\b", re.I)
_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5}


def resolve_relative_time(question: str, now_s: float | None) -> str | None:
    """"the last 3 minutes" at 5:11 -> "between 2:11 and 5:11". Done in code: measured, the model
    turned "the last minute" at 5:11 into "between 5:10 and 5:11"."""
    if now_s is None:
        return None
    m = _LAST_N_MIN.search(question)
    if m:
        n = _WORDS.get(m.group(1).lower()) or int(m.group(1))
    elif _LAST_MIN.search(question):
        n = 1
    else:
        return None
    return f"between {clock(max(0.0, now_s - 60 * n))} and {clock(now_s)}"


_PREVIOUS_RUN = re.compile(r"\b(previous|last|earlier|prior|other|old)\s+(run|simulation|session)\b", re.I)

SYSTEM_ANSWER = """You are the Traffic Analyst for a road-junction simulator: one 4-way junction,
left-hand (Indian) traffic of cars, bikes (motorcycles/scooters), autos and buses, 4 signal phases
(one approach green at a time), and an exclusive pedestrian phase inserted on demand. The signals
are run by a fixed timer or by an RL agent (a PPO neural network that every 5 s chooses to keep
the green or switch to the next phase).

Answer using ONLY the numbered sources. Rules:
1. Cite the source after each claim that uses it: [1] or [2][3].
2. Every number you write (counts, seconds, clock times, percentages, rates) must appear in a
   source as written there (rounding is fine). Do NOT compute new numbers: no differences, sums,
   averages or ratios of your own. Describe changes in words ("rose from 6 to 14").
3. If the sources do not answer the question, say "I don't know" and say what is missing. No
   outside knowledge for facts or numbers, no guessing.
4. Times in run sources are SIMULATED minutes:seconds from the start of that run: "3:00-4:00" is
   the fourth minute, not hours. Each run window covers one minute.
5. Guideline answers name the document (e.g. "the FHWA Signal Timing Manual") and cite it. US
   guidance (FHWA, MUTCD) and Indian practice (Kadiyali) can differ: say which source says what.
6. RL decisions: explain from what the agent saw (queues, backlogs, longest waits per approach),
   the signal state and its P(switch). It is a neural network: describe the situation it
   responded to; don't claim to know its inner reasons.
7. Be concise: 2-6 sentences or a short list. Plain text, simple markdown (bold, bullet lists)."""


@dataclass
class AnalystContext:
    """What the dashboard knows at the moment of the question."""
    run_ids: list[str]                      # the live run(s): main first, the AI twin while comparing
    run_dirs: list[Path] = field(default_factory=list)
    live_status: str = ""                   # a source text describing the dashboard right now
    settings: dict = field(default_factory=dict)
    limits: dict = field(default_factory=dict)
    capacity: str = ""
    previous_run_id: str | None = None
    now_s: float | None = None              # the live run's simulated time (for "the last minute")


class Analyst:
    def __init__(self, cfg, retriever, llm: GroqClient, kb=None, retrieval_mode: str = "hybrid_rerank"):
        """``retrieval_mode``: vector | bm25 | hybrid | hybrid_rerank (the dashboard always uses the
        last; Phase 13 answers the same questions with each to measure what every stage adds)."""
        self.cfg = cfg
        self.a = cfg.analyst
        self.retriever = retriever
        self.llm = llm
        self.kb = kb
        self.mode = retrieval_mode

    # ------------------------------------------------------------------ public
    def ask(self, question: str, history: list[dict], ctx: AnalystContext) -> dict:
        t0 = time.perf_counter()
        timing = {}
        try:
            route = self._route(question, history, ctx)
            timing["route_s"] = round(time.perf_counter() - t0, 2)
            if route["intent"] == "scenario":
                t1 = time.perf_counter()
                res = parse_scenario(self.llm, question, ctx.settings, ctx.limits, {"capacity": ctx.capacity})
                timing["scenario_s"] = round(time.perf_counter() - t1, 2)
                out = {"type": "scenario", **res, "intent": "scenario", "query": route["query"]}
            elif route["intent"] == "other":
                out = {"type": "answer", "intent": "other", "query": route["query"], "sources": [],
                       "answer": "I'm the traffic analyst for this junction. Ask me what happened in the run "
                                 "(\"why did the queue on N grow at 3:20?\"), why the AI switched, what the "
                                 "guidelines say (\"is a 120 s cycle too long?\"), or describe a scenario to "
                                 "set up (\"evening rush, 60% bikes, school crossing\").",
                       "grounding": {"unsupported": [], "bad_citations": [], "repaired": False}}
            else:
                out = self._answer(question, history, route, ctx, timing)
        except LLMError as exc:
            out = {"type": "error", "message": str(exc)}
        out["timing"] = {**timing, "total_s": round(time.perf_counter() - t0, 2)}
        self._log(question, out)
        return out

    # ------------------------------------------------------------------ steps
    def _route(self, question: str, history: list[dict], ctx: AnalystContext | None = None) -> dict:
        convo = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in history[-2 * self.a.history_turns:])
        user = (f"Conversation so far:\n{convo}\n\n" if convo else "") + f"Question: {question}"
        data, _ = self.llm.chat_json([{"role": "system", "content": SYSTEM_ROUTE},
                                      {"role": "user", "content": user}], temperature=0.0, max_tokens=300)
        intent = data.get("intent") if data.get("intent") in INTENTS else "mixed"
        query = str(data.get("query") or question).strip() or question
        ref = data.get("run_ref")
        ref = ref if isinstance(ref, str) and ref else None
        # Measured: the router read "what happened in the LAST MINUTE?" as "the previous run" and the
        # answer came from an older run. A switch away from the live run needs the user's own words.
        recent_user = " ".join([question] + [m["content"] for m in history[-2:] if m["role"] == "user"])
        if ref == "previous" and not _PREVIOUS_RUN.search(recent_user):
            ref = None
        # "the last minute" -> an explicit m:ss range, so the time-aware search can use it (live run only).
        if ctx is not None and ref is None and intent in ("run", "decision", "mixed"):
            rel = resolve_relative_time(question, ctx.now_s)
            if rel and rel not in query:
                query = f"{query} ({rel})"
        return {"intent": intent, "query": query, "run_ref": ref}

    def _target_runs(self, route: dict, ctx: AnalystContext) -> list[str]:
        ref = route.get("run_ref")
        known = set(self.kb.state["runs"]) if self.kb is not None else set()
        if ref == "previous" and ctx.previous_run_id:
            return [ctx.previous_run_id]
        if ref and ref in known:
            return [ref]
        return list(ctx.run_ids)

    def _retrieve(self, route: dict, ctx: AnalystContext) -> list[dict]:
        a, intent, q = self.a, route["intent"], route["query"]
        runs = self._target_runs(route, ctx)
        live = runs == list(ctx.run_ids)
        hits = []
        if intent in ("run", "decision", "mixed") and runs:
            f = {"run_id": runs}
            if intent == "decision":
                f["kind"] = ["rl_decisions", "run_window"]
            hits += self.retriever.search(q, k=a.k_run if intent != "mixed" else a.k_mixed, filters=f, mode=self.mode)
        if intent in ("guideline", "mixed"):
            hits += self.retriever.search(q, k=a.k_doc if intent == "guideline" else a.k_mixed,
                                          filters={"source_type": "doc"}, mode=self.mode)
        sources = [self._source(h.text, h.metadata, cid=h.id) for h in hits]
        if intent in ("run", "decision", "mixed") and runs:
            # The run's overview (settings, totals) and, for the live run, the dashboard right now.
            for rid in runs[:2]:
                cid = f"run:{rid}:summary"
                got = self.retriever.get(cid)
                if got and all(s["id"] != cid for s in sources):
                    sources.append(self._source(got[0], got[1], cid=cid))
            if live and ctx.live_status:
                sources.insert(0, {"id": "live", "citation": "live dashboard", "source_type": "live",
                                   "kind": "live_status", "text": ctx.live_status})
        for i, s in enumerate(sources, 1):
            s["n"] = i
        return sources

    def _source(self, text: str, meta: dict, cid: str | None = None) -> dict:
        words = text.split()
        if len(words) > self.a.max_source_words:  # keep the prompt inside the free-tier token budget
            text = " ".join(words[:self.a.max_source_words]) + " ..."
        keep = ("source_type", "kind", "citation", "run_id", "t_start", "t_end", "clock_start", "clock_end",
                "doc_id", "title", "cite", "page", "page_label", "section", "url", "year", "publisher")
        s = {k: meta[k] for k in keep if k in meta}
        s["text"] = text
        if cid:
            s["id"] = cid
        return s

    def _answer(self, question: str, history: list[dict], route: dict, ctx: AnalystContext, timing: dict) -> dict:
        t1 = time.perf_counter()
        sources = self._retrieve(route, ctx)
        timing["retrieve_s"] = round(time.perf_counter() - t1, 2)
        def label(s: dict) -> str:
            if s.get("source_type") == "run" and s.get("kind") != "run_summary":
                return f"{s['citation']}, simulated minutes:seconds"
            return s["citation"]
        block = "\n\n".join(f"[{s['n']}] ({label(s)})\n{s['text']}" for s in sources) or "(no sources found)"
        convo = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-2 * self.a.history_turns:])
        user = (f"Sources:\n{block}\n\n" + (f"Conversation so far:\n{convo}\n\n" if convo else "")
                + f"Question: {question}")
        messages = [{"role": "system", "content": SYSTEM_ANSWER}, {"role": "user", "content": user}]
        t2 = time.perf_counter()
        reply = self.llm.chat(messages, temperature=0.1, max_tokens=self.a.max_answer_tokens)
        answer = normalise_citations(reply.text.strip())
        texts = [s["text"] for s in sources]
        bad = unsupported_numbers(answer, texts, question)
        repaired = False
        if bad and sources:
            fix = (f"These numbers in your answer are not in the sources: {', '.join(bad)}. Rewrite the answer so "
                   "that every number appears in a source; describe anything else in words. Keep the citations.")
            reply2 = self.llm.chat(messages + [{"role": "assistant", "content": answer}, {"role": "user", "content": fix}],
                                   temperature=0.0, max_tokens=self.a.max_answer_tokens)
            answer2 = normalise_citations(reply2.text.strip())
            bad2 = unsupported_numbers(answer2, texts, question)
            if len(bad2) < len(bad):
                answer, bad, repaired, reply = answer2, bad2, True, reply2
        timing["generate_s"] = round(time.perf_counter() - t2, 2)
        return {"type": "answer", "intent": route["intent"], "query": route["query"], "answer": answer,
                "sources": sources, "model": reply.model,
                "grounding": {"unsupported": bad, "bad_citations": bad_citations(answer, len(sources)),
                              "repaired": repaired}}

    # ------------------------------------------------------------------ log
    def _log(self, question: str, out: dict) -> None:
        path = Path(self.a.log_path)
        if not path.is_absolute():
            from sim.config import PROJECT_ROOT
            path = PROJECT_ROOT / path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            rec = {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "question": question, **out}
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        except OSError:
            pass  # logging must never break an answer
