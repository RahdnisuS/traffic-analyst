"""Build or update the knowledge base: ``python -m rag.build``.

Indexes every PDF under ``rag.docs_dir`` and every run log under ``logging.runs_dir`` into one
ChromaDB collection, with ``source_type`` = "doc" | "run" on every chunk.

INCREMENTAL: ``<chroma_dir>/index_state.json`` remembers what was indexed: each document's file
hash, and each run's fingerprint (size and modification time of its log files, so a run that was
still being written is re-indexed once it has grown). A rebuild only re-embeds what changed. If
the embedding model or the chunking settings change, everything is rebuilt (vectors from two
different models, or chunks cut two different ways, must not be mixed in one index).

The analyst (Phase 12) calls ``KnowledgeBase.index_run(run_dir)`` to bring the live run up to
date before answering a question about it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from sim.config import PROJECT_ROOT, Config, load_config

import hashlib

from .ingest import INGEST_VERSION, file_sha, find_pdfs, load_sources, pdf_chunks
from .run_indexer import run_chunks
from .vectorstore import VectorStore

STATE_FILE = "index_state.json"
RUN_FILES = ("manifest.json", "metrics.csv", "events.jsonl", "decisions.jsonl")


def run_fingerprint(run_dir: Path) -> str:
    parts = []
    for name in RUN_FILES:
        p = Path(run_dir) / name
        if p.is_file():
            st = p.stat()
            parts.append(f"{name}:{st.st_size}:{int(st.st_mtime)}")
    return "|".join(parts)


class KnowledgeBase:
    def __init__(self, cfg: Config, store: VectorStore | None = None, embedder=None):
        from .embeddings import Embedder
        self.cfg = cfg
        r = cfg.rag
        self.store = store or VectorStore(cfg.chroma_path(), r.collection)
        self.embedder = embedder or Embedder(r.embedding_model)
        self.state_path = cfg.chroma_path() / STATE_FILE
        self.signature = {"embedding_model": r.embedding_model, "chunk_words": r.chunk_words,
                          "chunk_overlap": r.chunk_overlap, "run_window_s": r.run_window_s, "format": 1}
        self.state = self._load_state()

    # ------------------------------------------------------------------ state
    def _load_state(self) -> dict:
        try:
            st = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            st = {}
        if st.get("signature") != self.signature:
            st = {"signature": self.signature, "docs": {}, "runs": {}}
            if self.store.count():
                self.store.reset()  # settings changed: the old vectors can't be mixed with new ones
        return st

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def _add(self, chunks) -> None:
        if chunks:
            self.store.add(chunks, self.embedder.encode([c.text for c in chunks]))

    # ------------------------------------------------------------------ documents
    def index_docs(self, force: bool = False, log=print) -> dict:
        r = self.cfg.rag
        pdfs = find_pdfs(self.cfg.docs_path())
        seen, stats = set(), {"indexed": 0, "unchanged": 0, "removed": 0, "failed": 0, "chunks": 0}
        for path in pdfs:
            key = str(path.relative_to(self.cfg.docs_path())).replace("\\", "/")
            seen.add(key)
            # The file, its catalogue entry (title/citation) and the ingestion code version: a change
            # to any of them changes the chunks, so it re-indexes the document.
            entry = json.dumps(load_sources().get(path.name.lower()), sort_keys=True, default=str)
            sha = f"{file_sha(path)}:{hashlib.sha256(entry.encode()).hexdigest()[:8]}:v{INGEST_VERSION}"
            if not force and self.state["docs"].get(key, {}).get("sha") == sha:
                stats["unchanged"] += 1
                continue
            t0 = time.perf_counter()
            try:
                chunks = pdf_chunks(path, r.chunk_words, r.chunk_overlap)
            except Exception as exc:  # a broken or encrypted PDF must not stop the whole build
                log(f"  ! {key}: could not read ({type(exc).__name__}: {exc}); skipped")
                stats["failed"] += 1
                continue
            if not chunks:
                log(f"  ! {key}: no extractable text (a scanned PDF needs OCR first); skipped")
                stats["failed"] += 1
                continue
            doc_id = chunks[0].metadata["doc_id"]
            old = self.state["docs"].get(key, {}).get("doc_id")
            for d in {doc_id, old} - {None}:
                self.store.delete({"doc_id": d})
            self._add(chunks)
            self.state["docs"][key] = {"sha": sha, "doc_id": doc_id, "chunks": len(chunks),
                                       "cite": chunks[0].metadata["cite"]}
            self._save_state()
            stats["indexed"] += 1
            stats["chunks"] += len(chunks)
            log(f"  + {key}: {len(chunks)} chunks ({chunks[0].metadata['cite']}) in {time.perf_counter() - t0:.0f} s")
        for key in list(self.state["docs"]):
            if key not in seen:  # the PDF was deleted (or renamed): its chunks go too...
                doc_id = self.state["docs"][key]["doc_id"]
                del self.state["docs"][key]
                # ...unless a file that IS still there has the same doc_id (a renamed file keeps its
                # catalogue id): deleting by id would wipe the chunks just indexed for it.
                if all(v["doc_id"] != doc_id for v in self.state["docs"].values()):
                    self.store.delete({"doc_id": doc_id})
                stats["removed"] += 1
        self._save_state()
        return stats

    # ------------------------------------------------------------------ runs
    def index_run(self, run_dir: Path, force: bool = False) -> int:
        """(Re-)index one run if its logs changed. Returns the number of chunks written (0 = unchanged)."""
        run_dir = Path(run_dir)
        rid = run_dir.name
        fp = run_fingerprint(run_dir)
        if not force and self.state["runs"].get(rid, {}).get("fingerprint") == fp:
            return 0
        chunks = run_chunks(run_dir, self.cfg.rag.run_window_s)
        self.store.delete({"run_id": rid})
        self._add(chunks)
        self.state["runs"][rid] = {"fingerprint": fp, "chunks": len(chunks)}
        self._save_state()
        return len(chunks)

    def index_runs(self, force: bool = False, log=print) -> dict:
        runs_dir = self.cfg.runs_path()
        dirs = sorted(p for p in runs_dir.iterdir() if (p / "manifest.json").is_file()) if runs_dir.is_dir() else []
        stats = {"indexed": 0, "unchanged": 0, "removed": 0, "failed": 0, "chunks": 0}
        for d in dirs:
            try:
                n = self.index_run(d, force)
            except Exception as exc:  # one corrupt run log must not stop the rest
                log(f"  ! {d.name}: {type(exc).__name__}: {exc}; skipped")
                stats["failed"] += 1
                continue
            if n:
                stats["indexed"] += 1
                stats["chunks"] += n
            else:
                stats["unchanged"] += 1
        names = {d.name for d in dirs}
        for rid in list(self.state["runs"]):
            if rid not in names:
                self.store.delete({"run_id": rid})
                del self.state["runs"][rid]
                stats["removed"] += 1
        self._save_state()
        return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Build or update the analyst's knowledge base (PDFs + run logs).")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    ap.add_argument("--rebuild", action="store_true", help="re-index everything")
    ap.add_argument("--docs-only", action="store_true")
    ap.add_argument("--runs-only", action="store_true")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = load_config(args.config)
    t0 = time.perf_counter()
    kb = KnowledgeBase(cfg)
    if args.rebuild:
        kb.store.reset()
        kb.state = {"signature": kb.signature, "docs": {}, "runs": {}}
    if not args.runs_only:
        print(f"documents in {cfg.docs_path()}:")
        s = kb.index_docs(force=args.rebuild)
        print(f"  {s['indexed']} indexed ({s['chunks']} chunks), {s['unchanged']} unchanged, "
              f"{s['removed']} removed, {s['failed']} failed")
    if not args.docs_only:
        print(f"runs in {cfg.runs_path()}:")
        s = kb.index_runs(force=args.rebuild)
        print(f"  {s['indexed']} indexed ({s['chunks']} chunks), {s['unchanged']} unchanged, "
              f"{s['removed']} removed, {s['failed']} failed")
    print(f"knowledge base: {kb.store.count()} chunks in {cfg.chroma_path()} [{time.perf_counter() - t0:.0f} s]")


if __name__ == "__main__":
    main()
