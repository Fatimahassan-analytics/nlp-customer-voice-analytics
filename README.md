# RAG Business Intelligence Bot

A retrieval-augmented generation chatbot that answers questions about a PDF document corpus with source citations. Auto-detects between OpenAI GPT-4o-mini and local Ollama Llama 3 8B.

**Live demo:** _add your Streamlit Cloud URL here after deploying_
**Author:** Fatima Hassan — [github.com/Fatimahassan-analytics](https://github.com/Fatimahassan-analytics)

---

## Problem

An operations team in a mid-sized company has hundreds of policy PDFs, internal memos, and process documents. New employees waste hours searching for answers to questions like "what is the expense policy for client dinners?" The business question: **can we build a chatbot that answers questions from these documents, with citations, so employees get accurate answers in seconds instead of searching for hours?** The bot takes a question, retrieves the most relevant chunks from the document store, and generates an answer with a citation to the source page.

## Dataset / Document Store

- **Default corpus:** `data/raw/pdfs/` — drop any PDFs here. The repo seeds a placeholder text document if no PDFs are present so the code runs out-of-the-box.
- **Recommended real corpus:** 10–20 public company annual reports (Irish or UK listed), university handbooks, or government white papers.
- **Streaming upload:** the Streamlit app lets you upload a PDF via the sidebar at runtime — the corpus is rebuilt on demand.

## Approach

1. **Document loading** — `pypdf` extracts text from each PDF.
2. **Chunking** — LangChain's `RecursiveCharacterTextSplitter` (chunk_size=500, overlap=50, separators tuned for prose).
3. **Embeddings** — HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (local, free, fast on CPU).
4. **Vector store** — custom numpy-based SimpleVectorStore with cosine similarity (no external DB required; persists to `data/vector_store/simple_store.npz`). ChromaDB and FAISS are documented alternatives.
5. **Retrieval** — top-k cosine-similarity lookup; default k=5.
6. **LLM generation** — auto-detects backend: uses OpenAI GPT-4o-mini if `OPENAI_API_KEY` is set, else falls back to local Ollama Llama 3 8B.
7. **Citations** — every answer includes source filenames and similarity scores for the top-k retrieved chunks.

## Architecture

```
PDF upload → pypdf text extraction
   ↓
RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
   ↓
sentence-transformers embeddings (all-MiniLM-L6-v2)
   ↓
SimpleVectorStore (numpy + cosine similarity, persisted to disk)
   ↓
User question
   ↓
Embed query → top-k retrieval
   ↓
LLM (OpenAI GPT-4o-mini OR local Ollama Llama 3 8B) with retrieved context + citations
   ↓
Answer with [1], [2], ... source markers
```

## Results

- **End-to-end pipeline runs on CPU only** (no GPU required).
- **Cold-start embedding time:** ~30 seconds for 10 PDFs (~50 chunks).
- **Query latency:** ~2 seconds end-to-end (retrieval + LLM generation), depending on backend.
- **Retrieval precision@5** (qualitative on annual reports): the bot consistently surfaces the right paragraph for questions like "what was the company's revenue in 2023?" or "who is the CFO?".

## Live Demo

After deploying to [Streamlit Cloud](https://share.streamlit.io), paste your URL at the top of this README. The app lets you:
- Upload a PDF to add to the corpus.
- Ask any question about the corpus.
- See the LLM answer + source citations.
- Inspect the top-k retrieved chunks with similarity scores.

## Screenshots

_Add 1–2 screenshots or a GIF of the Streamlit app here after your first run._

## How to Run

```bash
# 1. Clone
git clone https://github.com/Fatimahassan-analytics/rag-business-intelligence-bot.git
cd rag-business-intelligence-bot

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Configure the LLM backend:
#    Option A — OpenAI (paid but easier): set OPENAI_API_KEY env var
export OPENAI_API_KEY=sk-...        # Linux/macOS
set OPENAI_API_KEY=sk-...           # Windows

#    Option B — Local Ollama (free, offline): install Ollama + pull Llama 3 8B
#    brew install ollama             # macOS
ollama pull llama3:8b
ollama serve                       # leaves a server running on :11434

# 5. (Optional) Drop your PDFs into data/raw/pdfs/ — otherwise the bundled
#    placeholder document is used so the bot has something to query.

# 6. Run the notebooks in order
jupyter lab notebooks/01_eda.ipynb          # chunk-length distribution, corpus overview
jupyter lab notebooks/02_modeling.ipynb      # build vector store, test retrieval
jupyter lab notebooks/03_evaluation.ipynb    # sample queries + business summary

# 7. Launch the Streamlit app
streamlit run app.py
```

## Tech Stack

- **Language:** Python 3.10+
- **PDF parsing:** pypdf
- **Chunking:** LangChain RecursiveCharacterTextSplitter
- **Embeddings:** HuggingFace sentence-transformers (all-MiniLM-L6-v2)
- **Vector store:** SimpleVectorStore (numpy + cosine similarity); ChromaDB / FAISS as documented alternatives
- **LLM:** OpenAI GPT-4o-mini (default) or local Ollama Llama 3 8B (auto-detected)
- **App:** Streamlit

## Repository Structure

```
rag-business-intelligence-bot/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── app.py
├── data/
│   ├── raw/
│   │   └── pdfs/                # drop PDFs here
│   │       └── 00_README.txt     # bundled placeholder doc
│   ├── processed/
│   └── vector_store/            # auto-created; persisted embeddings
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   ├── data.py                  # PDF loading, chunking, embedding, LLM client
│   ├── model.py                 # SimpleVectorStore + retrieval
│   └── utils.py                 # plotting helpers
└── artefacts/                    # auto-created; saved plots
```

## Module Mapping (for TCD MSc application)

This project maps directly onto two TCD MSc Business Analytics & AI for Management modules:
1. **Generative AI for Business** — RAG architecture, vector stores, embeddings, prompt engineering, LLM evaluation, hallucination mitigation, integration of generative AI into managerial decision workflows.
2. **Business AI Deployment** — Docker, Git, MLOps, productionising models. (Optional: dockerise this app to demonstrate that module too.)

## License

MIT — see [LICENSE](LICENSE).
