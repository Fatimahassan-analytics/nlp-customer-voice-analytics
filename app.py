"""Streamlit app — RAG Business Intelligence Bot.

Run locally:
    streamlit run app.py

The app lets you:
- Upload a PDF to add to the corpus
- Ask questions about the corpus
- See the answer + source citations
- Inspect the top-k retrieved chunks with similarity scores
"""
from __future__ import annotations
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from data import (  # noqa: E402
    ensure_dirs, get_pdf_files, extract_text_from_pdf, chunk_text,
    get_llm_answer, DOCS_DIR, seed_default_pdfs,
)
from model import build_or_load_store, retrieve  # noqa: E402


@st.cache_resource(show_spinner="Building vector store on first run ...")
def get_store():
    ensure_dirs()
    seed_default_pdfs()
    return build_or_load_store(force_rebuild=False)


def main():
    st.set_page_config(page_title="RAG Business Intelligence Bot", page_icon=":books:",
                       layout="wide")
    st.title("RAG Business Intelligence Bot")
    st.caption("Retrieval-Augmented Generation over a PDF document corpus, with source citations.")

    # Sidebar: corpus status + upload
    st.sidebar.header("Document corpus")
    pdf_files = get_pdf_files()
    st.sidebar.write(f"**{len(pdf_files)} document(s) in corpus:**")
    for p in pdf_files:
        st.sidebar.write(f"- `{p.name}`")

    uploaded = st.sidebar.file_uploader("Upload a PDF to add to the corpus", type=["pdf"])
    if uploaded is not None:
        new_path = DOCS_DIR / uploaded.name
        with open(new_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.sidebar.success(f"Added: {uploaded.name}")
        st.sidebar.info("Click 'Rebuild vector store' below to ingest the new document.")

    if st.sidebar.button("Rebuild vector store"):
        st.cache_resource.clear()
        st.rerun()

    # Backend info
    backend_status = "OpenAI GPT-4o-mini" if os.getenv("OPENAI_API_KEY") else "Local Ollama Llama 3 (requires `ollama serve`)"
    st.sidebar.write(f"**LLM backend:** {backend_status}")

    # Main area
    store = get_store()
    st.write(f"**Corpus size:** {len(store)} chunks")

    query = st.text_input("Ask a question about the documents:",
                         value="Who is the author of this project?",
                         placeholder="e.g. What modules does this project map to?")
    col1, col2 = st.columns([1, 3])
    with col1:
        top_k = st.slider("Top-k chunks", 1, 10, 5)
    with col2:
        backend = st.selectbox("LLM backend", ["auto", "openai", "ollama"], index=0)

    if st.button("Run query") and query.strip():
        with st.spinner("Retrieving relevant chunks..."):
            hits = retrieve(query, store, top_k=top_k)
            chunks = [h["chunk"] for h in hits]

        st.divider()
        st.subheader("Answer")
        with st.spinner("Generating answer with LLM..."):
            result = get_llm_answer(query, chunks, backend=backend)
        st.write(result["answer"])
        st.caption(f"Backend used: `{result['backend']}`")

        st.divider()
        st.subheader("Retrieved chunks (with citations)")
        rows = []
        for i, h in enumerate(hits):
            rows.append({
                "rank": i + 1,
                "source": h["source"],
                "score": f"{h['score']:.3f}",
                "chunk_preview": (h["chunk"][:200] + "...") if len(h["chunk"]) > 200 else h["chunk"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.divider()
    st.subheader("Sample queries to try")
    sample_queries = [
        "Who is the author of this project?",
        "What embedding model is used?",
        "What TCD modules does this project map to?",
        "What is the default chunk size?",
    ]
    for q in sample_queries:
        if st.button(q, key=f"sample_{q}"):
            st.session_state["query"] = q
            st.rerun()


if __name__ == "__main__":
    main()
