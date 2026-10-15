RAG Business Intelligence Bot — default knowledge base.

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
