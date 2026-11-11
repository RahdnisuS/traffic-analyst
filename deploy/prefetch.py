"""Download the two search models into the image at build time (``python -m deploy.prefetch``), so
a visitor's first question doesn't wait for a ~150 MB download (and a restart needs no network)."""

from rag.embeddings import Embedder
from rag.rerank import Reranker
from sim.config import load_config

cfg = load_config()
Embedder(cfg.rag.embedding_model).encode(["warm up"])
Reranker(cfg.rag.reranker_model).score("warm up", ["warm up"])
print("search models cached")
