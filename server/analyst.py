"""The analyst behind the dashboard's chat: ``POST /api/chat``.

Chat is per browser, so it is a plain request/response endpoint, not the WebSocket. The heavy
parts (embedding + re-ranking models, ~9 s to load, the index and the LLM client) live in ONE
``AnalystEngine`` per server, shared by every session: they load on the first question, in a worker
thread, so the simulation stream never stalls, and questions are answered one at a time (a lock),
which is plenty for a demo and keeps the free-tier token budget predictable. ``AnalystService`` is
the per-session part: it knows which runs and settings "here" and "now" mean.

Before each question the live run(s) are brought up to date in the index: the run loggers are
flushed (on the event-loop thread, which owns them), then the changed run logs are re-indexed in
the worker thread. "What happened at 3:20?" then means 3:20 of the run on screen.

The LIVE STATUS source describes the dashboard at that moment (settings, cycle length, latest
queues, recent events), because "here" and "now" in a question refer to what the user is
watching, and the run log's manifest only holds the settings the run STARTED with.
"""

from __future__ import annotations

import asyncio
import logging

from rag.run_indexer import event_sentence
from sim.logger import list_runs
from sim.metrics import clock

log = logging.getLogger("traffic.analyst")


class AnalystEngine:
    """The shared, expensive half of the analyst: search models, knowledge base, LLM client."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._analyst = None
        self._kb = None
        self._lock = asyncio.Lock()

    def status(self) -> dict:
        from rag.llm import load_api_key
        key = bool(load_api_key())
        kb = (self.cfg.chroma_path() / "index_state.json").is_file()
        reason = None if key and kb else (
            "Add GROQ_API_KEY to .env and restart the server." if not key else
            "Build the knowledge base first: python -m rag.build")
        return {"available": key and kb, "model": self.cfg.analyst.model, "reason": reason,
                "loaded": self._analyst is not None}

    def _load(self):
        from rag.analyst import Analyst
        from rag.build import KnowledgeBase
        from rag.llm import GroqClient
        from rag.retrieval import Retriever

        kb = KnowledgeBase(self.cfg)
        retriever = Retriever.from_config(self.cfg, store=kb.store, embedder=kb.embedder)
        retriever.search("warm up", k=1)  # load both models now, not in the user's first answer
        a = self.cfg.analyst
        self._kb = kb
        self._analyst = Analyst(self.cfg, retriever, GroqClient(a.model, a.fallback_model), kb)

    def _ask_blocking(self, question: str, history: list[dict], ctx) -> dict:
        if self._analyst is None:
            self._load()
        for d in ctx.run_dirs:
            try:
                if self._kb.index_run(d):
                    self._analyst.retriever.invalidate()
            except Exception:  # an unreadable live log must not block questions about documents
                log.exception("could not index live run %s", d)
        return self._analyst.ask(question, history, ctx)

    async def ask(self, question: str, history: list[dict], make_context) -> dict:
        """``make_context()`` runs on the event-loop thread inside the lock (it flushes run logs)."""
        st = self.status()
        if not st["available"]:
            return {"type": "error", "message": st["reason"]}
        async with self._lock:
            ctx = make_context()
            try:
                return await asyncio.to_thread(self._ask_blocking, question, history, ctx)
            except Exception as exc:  # never a 500 with a stack trace in the chat
                log.exception("analyst failed")
                return {"type": "error", "message": f"The analyst hit an internal error ({type(exc).__name__}). "
                                                    "Details are in the server log."}

    def _forget_blocking(self, run_ids: list[str]) -> None:
        kb = self._kb
        if kb is None:
            return  # nothing was indexed by this process
        changed = False
        for rid in run_ids:
            if kb.state["runs"].pop(rid, None) is not None:
                kb.store.delete({"run_id": rid})
                changed = True
        if changed:
            kb._save_state()
            self._analyst.retriever.invalidate()

    async def forget_runs(self, run_ids: list[str]) -> None:
        """Drop runs from the index (a public-demo visitor's runs when their session closes), so
        the index and the in-memory BM25 don't grow with every visitor."""
        async with self._lock:
            await asyncio.to_thread(self._forget_blocking, run_ids)


class AnalystService:
    """The per-session half: what "this run", "here" and "now" mean for one dashboard session."""

    def __init__(self, session, engine: AnalystEngine | None = None):
        self.session = session
        self.cfg = session.base_cfg
        self.engine = engine or AnalystEngine(self.cfg)

    def status(self) -> dict:
        return {**self.engine.status(), "per_hour": self.cfg.server.chat_per_hour or None}

    # ------------------------------------------------------------------ context
    def _live_status(self) -> str:
        s = self.session
        runs = s.runs()
        sim = s.sim
        st = s.settings()
        sig = sim.cfg.signals
        n = len(sim.junction.phases)
        green = st["green_s"]
        cycle = n * (green + sig.amber_s + sig.all_red_s)
        ped = sim.cfg.pedestrians.walk_s + sim.ped_clear_s + sig.all_red_s
        vph = sim.cfg.demand.per_approach_vph
        lines = [f"Live dashboard at simulated time {clock(sim.time)} (run {runs[0].logger.run_id if runs[0].logger else 'not logged'})."]
        if s.twin:
            lines.append(f"Side-by-side comparison running: left = fixed timer (run {s.run.logger.run_id if s.run.logger else '-'}), "
                         f"right = AI (run {s.twin.logger.run_id if s.twin.logger else '-'}), same seed and settings.")
        else:
            lines.append(f"Signal control: {'the AI (RL agent)' if s.controller == 'rl' else 'the fixed timer'}.")
        lines.append(
            f"Settings: traffic volume {st['volume_vph']:g} veh/h per approach on average "
            f"({', '.join(f'{a} {v:g}' for a, v in sorted(vph.items()))} veh/h); pedestrians {st['ped_rate_ph']:g} per hour; "
            f"vehicle mix {', '.join(f'{t} {p:g}%' for t, p in st['vehicle_mix'].items())}; driver aggressiveness {st['aggressiveness']:g}.")
        lines.append(
            f"Fixed-timer plan: {green:g} s green per phase, {sig.amber_s:g} s amber, {sig.all_red_s:g} s all-red, "
            f"{n} phases, so a cycle of {cycle:g} s, plus a {ped:g} s pedestrian phase when a pedestrian has pressed "
            f"the button (at most once per cycle). Minimum green {sig.min_green_s:g} s."
            + (" The AI is driving now: it decides when each green ends, so the actual timing is the measured "
               "one below, not this plan (the plan is kept for when the AI is turned off)."
               if s.controller == "rl" and not s.twin else ""))
        for r in runs:
            row = r.collector.latest
            if not row:
                continue
            who = "" if len(runs) == 1 else (" (AI side)" if r is s.twin else " (fixed-timer side)")
            measured = self._measured_timing(r.sim)
            if measured:
                lines.append(f"Measured signal timing{who}: {measured}")
            arms = sorted(sim.junction.inbound)
            delay = row.get("mean_delay_window")
            lines.append(
                f"Now{who}: queued vehicles {', '.join(f'{a} {row[f'queue_{a}']}' for a in arms)}; "
                f"waiting to enter {row['backlog']}; vehicles in the junction area {row['in_network']}; "
                + (f"mean control delay (last 2 min) {delay} s, level of service {row['los']}; " if delay not in ("", None) else "")
                + f"throughput {row['throughput_vph']} veh/h; {row['peds_waiting']} pedestrians waiting.")
            recent = [event_sentence(e) for e in list(r.recent_events)[-6:]]
            recent = [x for x in recent if x]
            if recent:
                lines.append(f"Recent events{who}: " + " ".join(recent[-4:]))
        b = s.benchmark_brief()
        if b:
            sc = b["score"]
            lines.append(f"Benchmark capacity of this junction (fixed timer): {sc.get('mean_vph', 0):,.0f} veh/h "
                         f"in total.")
        return "\n".join(lines)

    @staticmethod
    def _measured_timing(sim, window_s: float = 300.0) -> str | None:
        """What the lights actually did over the last ``window_s``: mean green length and the real
        cycle (time between successive starts of the same phase's green), from the controller's
        logged phase changes. Measured, because under the AI (or with pedestrian phases) the cycle
        is not the plan's. Measured: asked "is the cycle too long here?" while the AI drove, the
        analyst judged the fixed-timer PLAN's 120 s, which was not what the junction was doing."""
        now = sim.time
        ev = [e for e in sim.signals.events if e.time >= now - window_s]
        greens, cycles, last_start = [], [], {}
        start = None
        for e in ev:
            if e.stage == "green":
                start = (e.phase, e.time)
                if e.phase in last_start:
                    cycles.append(e.time - last_start[e.phase])
                last_start[e.phase] = e.time
            elif e.stage == "amber" and start and start[0] == e.phase:
                greens.append(e.time - start[1])
                start = None
        peds = sum(1 for e in ev if e.stage == "walk")
        if not greens:
            return None
        txt = (f"over the last {window_s / 60:g} minutes, {len(greens)} greens averaging {sum(greens) / len(greens):.1f} s "
               f"(shortest {min(greens):.0f} s, longest {max(greens):.0f} s)")
        if cycles:
            txt += f"; actual cycle length {sum(cycles) / len(cycles):.0f} s on average"
        return txt + f"; {peds} pedestrian phase{'s' if peds != 1 else ''}."

    def _context(self):
        from rag.analyst import AnalystContext
        from server.schemas import GREEN_MAX_S, PED_RATE_MAX, VOLUME_MAX_VPH

        s = self.session
        runs = [r for r in s.runs() if r.logger]
        for r in runs:
            r.logger.flush()  # on the event-loop thread, which owns the files
        ids = [r.logger.run_id for r in runs]
        if s.private:
            # A public-demo visitor: only runs made in this session, never another visitor's.
            own = s.own_runs()
            allowed = set(own) | set(ids)
            prev = next((r for r in reversed(own) if r not in ids), None)
        else:
            allowed = None
            prev = next((x["run_id"] for x in list_runs(self.cfg.runs_path()) if x["run_id"] not in ids), None)
        st = s.settings()
        b = s.benchmark_brief()
        cap = (f"{b['score'].get('mean_vph', 0):,.0f} veh/h in total for the whole junction "
               f"(about {b['score'].get('mean_vph', 0) / 4:,.0f} veh/h per approach) with the fixed timer"
               if b else "about 450 veh/h per approach")
        return AnalystContext(
            run_ids=ids, run_dirs=[r.logger.dir for r in runs], live_status=self._live_status(),
            settings={**st, "controller": s.controller},
            limits={"volume_vph": [0, VOLUME_MAX_VPH], "green_s": [s.sim.cfg.signals.min_green_s, GREEN_MAX_S],
                    "ped_rate_ph": [0, PED_RATE_MAX]},
            capacity=cap, previous_run_id=prev, now_s=s.sim.time, allowed_runs=allowed)

    # ------------------------------------------------------------------ ask
    async def ask(self, question: str, history: list[dict]) -> dict:
        return await self.engine.ask(question, history, self._context)
