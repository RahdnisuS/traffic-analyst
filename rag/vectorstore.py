"""ChromaDB wrapper: persistent storage of chunks (text + metadata + embedding) and vector search.

WHY ChromaDB: an embedded database (a folder on disk, no server to run), with metadata filters
(source_type = doc | run, run_id, ...) applied inside the vector search. We compute embeddings
ourselves and pass them in, instead of letting Chroma call a model: the same Embedder then serves
indexing and querying (they must match), and tests can swap in a fake one.

Chunk ids are deterministic ("doc:<doc>:p<page>:c<n>", "run:<run>:w<seconds>"), so re-indexing a
document or a run first deletes everything carrying its doc_id / run_id, then adds the new
chunks: no duplicates, no orphans when a re-indexed run has fewer windows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .chunking import Chunk


def to_where(filters: dict | None) -> dict | None:
    """{"source_type": "doc"} -> Chroma's where; several keys -> $and; a list -> $in."""
    if not filters:
        return None
    parts = [{k: {"$in": list(v)} if isinstance(v, (list, tuple, set)) else v} for k, v in filters.items()]
    return parts[0] if len(parts) == 1 else {"$and": parts}


def matches(meta: dict, filters: dict | None) -> bool:
    """The same filter, evaluated in Python (for BM25, which has no database behind it)."""
    if not filters:
        return True
    for k, v in filters.items():
        if isinstance(v, (list, tuple, set)):
            if meta.get(k) not in v:
                return False
        elif meta.get(k) != v:
            return False
    return True


class VectorStore:
    def __init__(self, path: Path, collection: str):
        import chromadb
        from chromadb.config import Settings

        Path(path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})
        self.name = collection

    def count(self) -> int:
        return self.collection.count()

    def delete(self, filters: dict) -> None:
        self.collection.delete(where=to_where(filters))

    def reset(self) -> None:
        self.client.delete_collection(self.name)
        self.collection = self.client.get_or_create_collection(self.name, metadata={"hnsw:space": "cosine"})

    def add(self, chunks: list[Chunk], embeddings: np.ndarray, batch: int = 2000) -> None:
        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            self.collection.upsert(ids=[c.id for c in part], documents=[c.text for c in part],
                                   metadatas=[c.metadata for c in part],
                                   embeddings=embeddings[i:i + batch].tolist())

    def query(self, embedding: np.ndarray, n: int, filters: dict | None = None) -> list[tuple[str, float]]:
        """(id, cosine distance) of the ``n`` nearest chunks, nearest first."""
        n = min(n, self.count())
        if n <= 0:
            return []
        res = self.collection.query(query_embeddings=[embedding.tolist()], n_results=n, where=to_where(filters),
                                    include=["distances"])
        return list(zip(res["ids"][0], res["distances"][0]))

    def all(self) -> tuple[list[str], list[str], list[dict]]:
        """Every chunk: ids, texts, metadatas (for the BM25 index)."""
        res = self.collection.get(include=["documents", "metadatas"])
        return res["ids"], res["documents"], res["metadatas"]
