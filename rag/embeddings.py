"""Sentence embeddings (bi-encoder): text -> a 384-number vector whose direction encodes meaning.

all-MiniLM-L6-v2 (sentence-transformers): a 6-layer, 22M-parameter model distilled from a larger
one, trained on ~1 billion sentence pairs to put texts with similar meaning close together.
WHY this one: it's the brief's choice and the standard small model: ~90 MB, fast on a CPU (our
~1,700 chunks embed in about a minute), and good enough for retrieval when combined with BM25
and a re-ranker. Its limit: it reads at most 256 word pieces; the rest of a chunk is ignored, so
chunks are kept near 200 words with the key facts first.

Vectors are L2-normalised, so cosine similarity = dot product, and the store uses cosine
distance. The model is loaded lazily (first use), so importing this module is free.
"""

from __future__ import annotations

import numpy as np


class Embedder:
    def __init__(self, model_name: str, batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), np.float32)
        return np.asarray(self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                            convert_to_numpy=True, show_progress_bar=show_progress), np.float32)
