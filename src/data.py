"""RAG Business Intelligence Bot — document loading + chunking + embedding utilities.

Architecture (see README for full diagram):

    PDF upload → pypdf text extraction → RecursiveCharacterTextSplitter
    → sentence-transformers embeddings → FAISS / ChromaDB vector store
    → user question → top-k retrieval → LLM (OpenAI or local Ollama) with citations
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"           # PDFs go here
DOCS_DIR = RAW_DIR / "pdfs"                       # default PDF location
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 5


def ensure_dirs() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


def get_pdf_files() -> list[Path]:
    """Return all PDFs in data/raw/pdfs/."""
    if not DOCS_DIR.exists():
        return []
    return sorted(DOCS_DIR.glob("*.pdf"))


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("pypdf not installed; run: pip install pypdf")
    reader = PdfReader(str(pdf_path))
    chunks = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            print(f"[warn] page {i} of {pdf_path.name} failed: {e}")
            text = ""
        chunks.append(text)
    return "\n".join(chunks)


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
                overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    """Chunk text using LangChain's RecursiveCharacterTextSplitter."""
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_text(text)
    except ImportError:
        # Pure-python fallback if langchain is not installed
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += (chunk_size - overlap)
        return chunks


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed texts using HuggingFace sentence-transformers (local, free)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError("sentence-transformers not installed; pip install sentence-transformers")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query."""
    return embed_texts([query])[0]


def get_llm_answer(query: str, context_chunks: list[str], backend: str = "auto") -> dict:
    """Generate an answer using the configured LLM backend.

    backend options:
    - "auto": try OpenAI (if OPENAI_API_KEY is set), else try Ollama
    - "openai": use OpenAI GPT-4o-mini
    - "ollama": use local Ollama Llama 3 8B

    Returns:
        {answer: str, sources: list[str], backend: str}
    """
    context = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(context_chunks)])

    prompt = (
        f"You are a helpful assistant answering a user's question based on the following "
        f"document excerpts. Cite sources using [1], [2], ... markers.\n\n"
        f"DOCUMENT EXCERPTS:\n{context}\n\n"
        f"USER QUESTION: {query}\n\n"
        f"Answer (with citations):"
    )

    backend_used = backend
    if backend == "auto":
        if os.getenv("OPENAI_API_KEY"):
            backend_used = "openai"
        else:
            backend_used = "ollama"

    if backend_used == "openai":
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful business analyst assistant."},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.2,
                max_tokens=512,
            )
            answer = resp.choices[0].message.content.strip()
        except Exception as e:
            return {"answer": f"[OpenAI failed: {e}]", "sources": [], "backend": "openai"}
    elif backend_used == "ollama":
        try:
            import requests
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3:8b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 512},
                },
                timeout=120,
            )
            if resp.status_code != 200:
                return {"answer": f"[Ollama HTTP {resp.status_code}]", "sources": [], "backend": "ollama"}
            answer = resp.json().get("response", "").strip()
        except Exception as e:
            return {"answer": f"[Ollama not running: {e}]", "sources": [], "backend": "ollama"}
    else:
        return {"answer": f"[Unknown backend: {backend}]", "sources": [], "backend": backend}

    return {
        "answer": answer,
        "sources": [c[:200] + "..." if len(c) > 200 else c for c in context_chunks],
        "backend": backend_used,
    }


def seed_default_pdfs() -> None:
    """Seed the PDFs directory with a tiny placeholder if empty.

    The bundled placeholder PDF explains the RAG system itself, so the
    Streamlit app has something to query out-of-the-box.
    """
    ensure_dirs()
    if get_pdf_files():
        return
    # Write a plain-text file as a stand-in for a PDF — the code below handles both
    placeholder_path = DOCS_DIR / "00_README.txt"
    placeholder_path.write_text("""RAG Business Intelligence Bot — default knowledge base.

This is the default document the bot will query against until you upload real
PDFs (e.g. annual reports, policy documents, university handbooks).

Project: RAG Business Intelligence Bot
Author: Fatima Hassan
License: MIT

Useful facts (for testing the bot):
- The RAG bot was built in October 2026.
- The author is Fatima Hassan.
- The project maps to TCD's Generative AI for Business module and Business AI Deployment module.
- The embedding model used is all-MiniLM-L6-v2 from sentence-transformers.
- The vector store is ChromaDB by default; FAISS is also supported.
- The LLM backend auto-detects between OpenAI GPT-4o-mini and local Ollama Llama 3 8B.
- Chunk size is 500 characters with 50 character overlap.
- Top-k retrieval defaults to 5 chunks per query.
""")
    print(f"[OK] Wrote default placeholder doc → {placeholder_path}")


if __name__ == "__main__":
    seed_default_pdfs()
    print(f"PDF directory: {DOCS_DIR}")
    print(f"Files: {[p.name for p in get_pdf_files()]}")
