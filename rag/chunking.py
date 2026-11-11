"""Chunks: the unit the knowledge base stores, retrieves and cites.

Two kinds of source end up as the same thing, a ``Chunk`` (id + text + flat metadata):
  * a passage of a PDF (rag/ingest.py), cited as "<document>, p. <page>"
  * a summary of a simulation run, or of one time window of it (rag/run_indexer.py), cited as
    "run <run id> @ <m:ss>-<m:ss>"

WHY chunks of ~200 words for documents: the embedding model (all-MiniLM-L6-v2) reads at most
256 word pieces (~190 English words); text past that is silently ignored when the chunk is
embedded. Smaller chunks also make citations precise (a page, not a chapter). The 40-word
overlap keeps a sentence that straddles a boundary whole in at least one chunk.
WHY a chunk never crosses a page: every chunk then has ONE page number to cite.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)  # str/int/float/bool only (a ChromaDB rule)


def pack_words(paragraphs: list[str], max_words: int = 200, overlap: int = 40) -> list[str]:
    return [text for _, text in pack_word_spans(paragraphs, max_words, overlap)]


def pack_word_spans(paragraphs: list[str], max_words: int = 200, overlap: int = 40) -> list[tuple[int, str]]:
    """(index of the chunk's first word, chunk text). Pack paragraphs into chunks of at most ``max_words`` words. Paragraph breaks are kept where
    possible (a chunk ends at the end of a paragraph if the next one doesn't fit); a paragraph
    longer than ``max_words`` is split by words. Consecutive chunks share ``overlap`` words."""
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")
    words: list[str] = []
    breaks: set[int] = set()  # word indices where a paragraph ends
    for p in paragraphs:
        w = p.split()
        if w:
            words += w
            breaks.add(len(words))
    chunks, start = [], 0
    while start < len(words):
        end = min(start + max_words, len(words))
        if end < len(words):
            # Prefer to end at the last paragraph break in the second half of the window.
            cands = [b for b in breaks if start + max_words // 2 <= b <= end]
            if cands:
                end = max(cands)
        chunks.append((start, " ".join(words[start:end])))
        if end >= len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks


_TOKEN = re.compile(r"\d+:\d{2}|\d+(?:\.\d+)?|[a-z]+(?:'[a-z]+)?")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were "
    "will with which what when why how does did do can".split())


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens for BM25. Clock times ("3:20") and numbers ("120", "1.5") stay whole
    tokens: "what happened at 3:20" must match the window that says 3:20, and "120 s cycle" must
    match "120". Very common words are dropped (they match everything and rank nothing)."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "doc"
