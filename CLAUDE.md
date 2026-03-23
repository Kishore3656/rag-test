# RAG Project

## Overview
Python RAG assistant using LangChain + Google Gemini + FAISS vector store, with a Streamlit UI.

## Run Commands
- `streamlit run app.py` — launch the web UI
- `python main.py` — run CLI mode
- `pip install -r requirements.txt` — install dependencies

## Environment
- Requires `GOOGLE_API_KEY` in `.env` (Google Generative AI key)
- Python 3.12–3.14 required
- FAISS index persisted to `./faiss_index/` (auto-created on first run)

## Key Files
- `rag_engine.py` — core RAG logic (PDF loading, FAISS, retrieval chain)
- `app.py` — Streamlit UI
- `main.py` — CLI entrypoint

## Models
- LLM: `gemini-2.0-flash` (temperature=0)
- Embeddings: `models/embedding-001`

## PDF Source
Place PDF at project root; update path in `rag_engine.py` / `main.py` as needed.
Current: `THINK-AND-GROW-RICH.pdf`
