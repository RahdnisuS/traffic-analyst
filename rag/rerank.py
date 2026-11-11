"""Cross-encoder re-ranking: the last, most careful retrieval step.

A bi-encoder (rag/embeddings.py) embeds the query and each chunk SEPARATELY, so it can only
compare two summaries of meaning; that is what makes it fast enough to search everything. A
cross-encoder reads the query and one chunk TOGETHER, in one pass, with attention between their
words, and outputs one relevance score. It notices what the bi-encoder can't ("cycle length for
pedestrians" vs a passage about pedestrian clearance that never mentions cycle length), but it
costs one model run per (query, chunk) pair, so it only re-orders the ~30 candidates the cheap
retrievers found.

ms-marco-MiniLM-L-6-v2 is trained on MS MARCO, real Bing queries labelled with the passages that
answered them: exactly "which passage answers this question?".
"""

from __future__ import annotations

import numpy as np


class Reranker:
    def __init__(self, model_name: str, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device="cpu")
        return self._model

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0)
        return np.asarray(self.model.predict([(query, t) for t in texts], batch_size=self.batch_size,
                                             show_progress_bar=False), float)
