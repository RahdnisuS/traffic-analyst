"""Hybrid retrieval: BM25 + vector search, merged by reciprocal rank fusion, then re-ranked.

WHY two retrievers: they fail differently.
  * Vector search matches MEANING ("is the cycle too long?" finds "excessive cycle lengths
    increase delay"), but is vague about exact tokens: section numbers ("4D.05"), clock times
    ("3:20"), run ids and rare terms ("spillback") all look alike to it.
  * BM25 (classic keyword scoring: a word counts more the rarer it is across chunks, with
    diminishing returns for repeats and a penalty for long chunks) matches exactly those tokens,
    but knows no synonyms.
Each returns its top ``candidates``; RECIPROCAL RANK FUSION merges the two rankings:

    rrf(chunk) = sum over retrievers of 1 / (rrf_k + rank),   rank = 1, 2, 3, ...

RRF uses only RANKS, not scores. That matters because a BM25 score (0..~30) and a cosine
similarity (0..1) are on incomparable scales; ranks are always comparable. A chunk ranked well by
both retrievers beats one ranked first by only one; rrf_k = 60 (the value from the original
paper) keeps rank 1 from dominating rank 5.

Finally the cross-encoder (rag/rerank.py) re-scores the top ``rerank_pool`` fused chunks by
reading each one together with the question, and the best ``top_k`` are returned.

TIME-AWARE RUN SEARCH. A question about a run usually names a moment ("why did the AI switch at
1:35?", "what happened at minute 4?"). Measured on the first build: BM25 ranked the right window
first, but vector search barely registers "1:35" and the cross-encoder (trained on web search,
not log timestamps) promoted a similar-looking decision at 37:55. A timestamp is structured
data, so it is matched exactly instead of by similarity: ``time_ranges()`` reads times out of the
question, and in the hybrid modes the run chunks whose window covers them (``time_window_s``
either side) come FIRST, ordered among themselves by the re-ranker; everything else follows.
(The standard name is metadata filtering / "self-querying": turn part of the question into a
filter.)

Modes, so Phase 13 can measure what each stage adds: "vector", "bm25", "hybrid" (RRF, no
re-ranker) and "hybrid_rerank" (the default). Each hit records its rank at every stage, so the
analyst's answer can be traced back ("found by BM25 at rank 2, vectors rank 14").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import re

import numpy as np

from .chunking import tokenize
from .vectorstore import VectorStore, matches

MODES = ("vector", "bm25", "hybrid", "hybrid_rerank")


@dataclass
class Hit:
    id: str
    text: str
    metadata: dict
    score: float                      # the score of the final stage of the mode used
    ranks: dict = field(default_factory=dict)  # {"vector": r, "bm25": r, "rrf": r, "rerank": r} (1-based)

    @property
    def citation(self) -> str:
        return self.metadata.get("citation", self.id)


_CLOCK = re.compile(r"(?<![\d:.])(\d{1,3}):([0-5]\d)(?![\d:])")
_MINUTE = re.compile(r"\bminute\s+(\d{1,3})\b|\b(\d{1,3})(?:st|nd|rd|th)\s+minute\b|\b(\d{1,3})\s+minutes?\s+in\b", re.I)
_RANGE = re.compile(r"(?<![\d:.])(\d{1,3}:[0-5]\d)\s*(?:-|–|to|and|until)\s*(\d{1,3}:[0-5]\d)(?![\d:])", re.I)


def _secs(clock: str) -> float:
    m, s = clock.split(":")
    return int(m) * 60 + int(s)


def time_ranges(query: str, slack_s: float = 15.0) -> list[tuple[float, float]]:
    """Simulated-time ranges (seconds) a question refers to.
      "3:20"                 -> 3:05-3:35 (a moment, with ``slack_s`` either side)
      "between 3:00 and 5:00" / "3:00-5:00" -> 3:00-5:00
      "minute 4", "the 4th minute", "4 minutes in" -> 3:00-5:00 (people mean either "the
                             minute ending at 4:00" or "around 4:00"; both windows are kept)"""
    out, used = [], []
    for m in _RANGE.finditer(query):
        a, b = sorted((_secs(m.group(1)), _secs(m.group(2))))
        out.append((a, b))
        used.append(m.span())
    for m in _CLOCK.finditer(query):
        if any(lo <= m.start() < hi for lo, hi in used):
            continue
        t = int(m.group(1)) * 60 + int(m.group(2))
        out.append((max(0.0, t - slack_s), t + slack_s))
    for m in _MINUTE.finditer(query):
        n = int(next(g for g in m.groups() if g))
        out.append((max(0.0, (n - 1) * 60.0), (n + 1) * 60.0))
    return out


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion of several ranked id lists (best first). Ties keep first-seen order."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


class Retriever:
    def __init__(self, store: VectorStore, embedder, reranker=None, *, candidates: int = 30, rrf_k: int = 60,
                 rerank_pool: int = 30, top_k: int = 5, time_slack_s: float = 15.0):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.candidates, self.rrf_k, self.rerank_pool, self.top_k = candidates, rrf_k, rerank_pool, top_k
        self.time_slack_s = time_slack_s
        self._bm25 = None
        self._bm25_count = -1

    @classmethod
    def from_config(cls, cfg, store: VectorStore | None = None, embedder=None, reranker=None) -> "Retriever":
        from .embeddings import Embedder
        from .rerank import Reranker
        r = cfg.rag
        store = store or VectorStore(cfg.chroma_path(), r.collection)
        return cls(store, embedder or Embedder(r.embedding_model), reranker or Reranker(r.reranker_model),
                   candidates=r.candidates, rrf_k=r.rrf_k, rerank_pool=r.rerank_pool, top_k=r.top_k)

    # ------------------------------------------------------------------ BM25 index
    def _ensure_bm25(self) -> None:
        """BM25 has no database: it is built in memory from every chunk in the store, and rebuilt
        when the store has changed (a run was re-indexed). ~1,700 chunks take well under a second."""
        n = self.store.count()
        if self._bm25 is not None and n == self._bm25_count:
            return
        from rank_bm25 import BM25Okapi

        self.ids, self.texts, self.metas = self.store.all()
        self.index_of = {cid: i for i, cid in enumerate(self.ids)}
        tokens = [tokenize(t) for t in self.texts]
        self._bm25 = BM25Okapi(tokens) if tokens else None
        self._bm25_count = n

    def invalidate(self) -> None:
        self._bm25 = None

    def bm25_search(self, query: str, n: int, filters: dict | None = None) -> list[tuple[str, float]]:
        self._ensure_bm25()
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = np.argsort(-scores, kind="stable")
        out = []
        for i in order:
            if scores[i] <= 0:
                break  # no query word in common: not a match at all
            if matches(self.metas[i], filters):
                out.append((self.ids[i], float(scores[i])))
                if len(out) >= n:
                    break
        return out

    def time_search(self, query: str, filters: dict | None = None) -> list[str]:
        """Run-window and decision chunks covering a time the question names (see the module
        docstring), in index order. Empty if the question names no time."""
        ranges = time_ranges(query, self.time_slack_s)
        if not ranges:
            return []
        self._ensure_bm25()
        out = []
        for cid, meta in zip(self.ids, self.metas):
            if meta.get("source_type") != "run" or meta.get("kind") == "run_summary" or not matches(meta, filters):
                continue
            t0, t1 = float(meta.get("t_start", 0)), float(meta.get("t_end", 0))
            if any(t0 < hi and t1 > lo for lo, hi in ranges):  # the window overlaps the range
                out.append(cid)
        return out

    def vector_search(self, query: str, n: int, filters: dict | None = None) -> list[tuple[str, float]]:
        emb = self.embedder.encode([query])[0]
        return [(cid, 1.0 - dist) for cid, dist in self.store.query(emb, n, filters)]  # cosine similarity

    # ------------------------------------------------------------------ search
    def search(self, query: str, k: int | None = None, mode: str = "hybrid_rerank",
               filters: dict | None = None) -> list[Hit]:
        """The ``k`` best chunks for ``query``. ``filters``: metadata equality, e.g.
        {"source_type": "doc"} or {"run_id": "..."} (a list value means "any of")."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        k = k or self.top_k
        self._ensure_bm25()
        ranks: dict[str, dict] = {}
        vec = bm = []
        if mode in ("vector", "hybrid", "hybrid_rerank"):
            vec = self.vector_search(query, self.candidates, filters)
            for r, (cid, _) in enumerate(vec, 1):
                ranks.setdefault(cid, {})["vector"] = r
        if mode in ("bm25", "hybrid", "hybrid_rerank"):
            bm = self.bm25_search(query, self.candidates, filters)
            for r, (cid, _) in enumerate(bm, 1):
                ranks.setdefault(cid, {})["bm25"] = r

        if mode == "vector":
            ranked = vec
        elif mode == "bm25":
            ranked = bm
        else:
            timed = self.time_search(query, filters)
            ranked = rrf_fuse([[c for c, _ in vec], [c for c, _ in bm], timed], self.rrf_k)
            for r, (cid, _) in enumerate(ranked, 1):
                ranks.setdefault(cid, {})["rrf"] = r
            if timed:
                # The chunks at the time asked about come first (see the module docstring).
                tset = set(timed)
                for cid in timed:
                    ranks.setdefault(cid, {})["time"] = 1
                ranked = [x for x in ranked if x[0] in tset] + [x for x in ranked if x[0] not in tset]
            if mode == "hybrid_rerank" and self.reranker is not None and ranked:
                pool = ranked[:max(self.rerank_pool, len(timed))]
                scores = self.reranker.score(query, [self._text(cid) for cid, _ in pool])
                tset = set(timed)
                # Sort by re-ranker score, but never let a chunk outside the asked-about time
                # overtake one inside it.
                order = sorted(range(len(pool)), key=lambda i: (pool[i][0] not in tset, -scores[i]))
                ranked = [(pool[i][0], float(scores[i])) for i in order]
                for r, (cid, _) in enumerate(ranked, 1):
                    ranks[cid]["rerank"] = r
        return [Hit(cid, self._text(cid), self._meta(cid), score, ranks.get(cid, {})) for cid, score in ranked[:k]]

    def get(self, cid: str) -> tuple[str, dict] | None:
        """(text, metadata) of one chunk by id, or None."""
        self._ensure_bm25()
        i = self.index_of.get(cid)
        return None if i is None else (self.texts[i], self.metas[i])

    def _text(self, cid: str) -> str:
        return self.texts[self.index_of[cid]]

    def _meta(self, cid: str) -> dict:
        return self.metas[self.index_of[cid]]
