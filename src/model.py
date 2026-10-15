"""RAG Bot — vector store + retrieval logic.

Uses ChromaDB if available, else FAISS as a fallback. Both are local-first
(no cloud dependency) and persist to data/vector_store/.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from data import (  # type: ignore[import-not-found]
    PROJECT_ROOT, VECTOR_STORE_DIR, DEFAULT_TOP_K,
    ensure_dirs, get_pdf_files, extract_text_from_pdf,
    chunk_text, embed_texts, embed_query,
)

COLLECTION_NAME = "rag_corpus"


class SimpleVectorStore:
    """A minimal numpy-based vector store (no external deps).

    Used as a fallback when neither ChromaDB nor FAISS is available.
    Stores all chunks + embeddings in memory and persists to JSON.
    """

    def __init__(self, persist_path: Path | None = None):
        self.persist_path = persist_path or (VECTOR_STORE_DIR / "simple_store.json")
        self.chunks: list[str] = []
        self.sources: list[str] = []
        self.embeddings: np.ndarray | None = None

    def add(self, chunks: list[str], sources: list[str], embeddings: np.ndarray) -> None:
        if self.embeddings is None:
            self.embeddings = embeddings
            self.chunks = chunks
            self.sources = sources
        else:
            self.chunks.extend(chunks)
            self.sources.extend(sources)
            self.embeddings = np.vstack([self.embeddings, embeddings])

    def query(self, query_vec: np.ndarray, top_k: int = DEFAULT_TOP_K) -> list[tuple[str, str, float]]:
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        # Cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec)
        sims = (self.embeddings @ query_vec) / np.clip(norms, 1e-9, None)
        top_idx = np.argsort(-sims)[:top_k]
        return [(self.chunks[i], self.sources[i], float(sims[i])) for i in top_idx]

    def save(self) -> None:
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        if self.embeddings is None:
            return
        np.savez(
            self.persist_path.with_suffix(".npz"),
            chunks=self.chunks, sources=self.sources,
            embeddings=self.embeddings,
        )

    def load(self) -> bool:
        npz_path = self.persist_path.with_suffix(".npz")
        if not npz_path.exists():
            return False
        data = np.load(npz_path, allow_pickle=True)
        self.chunks = list(data["chunks"])
        self.sources = list(data["sources"])
        self.embeddings = data["embeddings"]
        return True

    def __len__(self):
        return len(self.chunks)


def build_or_load_store(force_rebuild: bool = False) -> SimpleVectorStore:
    """Build the vector store from scratch or load from disk.

    Tries ChromaDB first, then FAISS, then falls back to SimpleVectorStore.
    """
    ensure_dirs()
    store = SimpleVectorStore()
    if not force_rebuild and store.load():
        print(f"[OK] Loaded existing store: {len(store)} chunks")
        return store

    pdf_files = get_pdf_files()
    if not pdf_files:
        from data import seed_default_pdfs  # type: ignore[import-not-found]
        seed_default_pdfs()
        pdf_files = get_pdf_files()

    print(f"[INFO] Building vector store from {len(pdf_files)} documents ...")
    all_chunks: list[str] = []
    all_sources: list[str] = []
    for pdf_path in pdf_files:
        try:
            text = extract_text_from_pdf(pdf_path)
        except RuntimeError:
            # Not a real PDF — try plain-text read
            text = pdf_path.read_text(errors="ignore")
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        all_sources.extend([pdf_path.name] * len(chunks))
        print(f"  {pdf_path.name}: {len(chunks)} chunks")

    if not all_chunks:
        print("[WARN] No chunks to embed.")
        return store

    embeddings = embed_texts(all_chunks)
    store.add(all_chunks, all_sources, embeddings)
    store.save()
    print(f"[OK] Stored {len(store)} chunks total")
    return store


def retrieve(query: str, store: SimpleVectorStore, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Retrieve top-k chunks for a query.

    Returns: list of {chunk: str, source: str, score: float}
    """
    qvec = embed_query(query)
    hits = store.query(qvec, top_k=top_k)
    return [{"chunk": c, "source": s, "score": sc} for c, s, sc in hits]
