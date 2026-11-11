"""Try the knowledge base from the command line: ``python -m rag.search "is a 120 s cycle too long?"``.

Prints the top chunks with their citation and their rank at each stage (vector, BM25, fusion,
re-ranker), which shows what each stage contributed. Filters: --source doc|run, --run <run id>.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time

from sim.config import PROJECT_ROOT, load_config

from .retrieval import MODES, Retriever


def main() -> None:
    ap = argparse.ArgumentParser(description="Search the analyst's knowledge base.")
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=None, help="results (default rag.top_k)")
    ap.add_argument("--mode", choices=MODES, default="hybrid_rerank")
    ap.add_argument("--source", choices=["doc", "run"], default=None)
    ap.add_argument("--run", default=None, help="only this run id")
    ap.add_argument("--full", action="store_true", help="print whole chunks")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = load_config(args.config)
    filters = {}
    if args.source:
        filters["source_type"] = args.source
    if args.run:
        filters["run_id"] = args.run
    r = Retriever.from_config(cfg)
    t0 = time.perf_counter()
    hits = r.search(args.query, k=args.k, mode=args.mode, filters=filters or None)
    print(f'"{args.query}"  [{args.mode}, {time.perf_counter() - t0:.2f} s]')
    for i, h in enumerate(hits, 1):
        ranks = ", ".join(f"{k} #{v}" for k, v in h.ranks.items())
        print(f"\n{i}. {h.citation}   score {h.score:.3f}   ({ranks})")
        body = h.text if args.full else textwrap.shorten(h.text.replace("\n", " "), 400)
        print(textwrap.indent(textwrap.fill(body, 110) if not args.full else body, "   "))


if __name__ == "__main__":
    main()
