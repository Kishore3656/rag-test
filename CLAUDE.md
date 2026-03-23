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
- LLM: `gemini-1.5-flash` (temperature=0) — gemini-2.0-flash hits free quota fast
- Embeddings: `all-MiniLM-L6-v2` via HuggingFace (local, no API quota)

## Gotchas
- Google embedding API quota exhausts quickly on free tier — use local HuggingFace embeddings instead
- `langchain-huggingface>=1.x` upgrades `langchain-core` to 1.x and breaks `langchain 0.3.x` — pin `langchain-huggingface==0.1.2` and `langchain-core==0.3.83`
- FAISS index saved per-PDF as `faiss_index_<filename>/` — all covered by `faiss_index*/` in `.gitignore`
- Run with: `.venv\Scripts\streamlit.exe run app.py` (PowerShell) or `.venv/Scripts/streamlit.exe run app.py` (bash)
- Venv setup: `uv venv --python 3.12 --clear && uv pip install -r requirements.txt`

## PDF Source
Place PDF at project root; update path in `rag_engine.py` / `main.py` as needed.
Current: `THINK-AND-GROW-RICH.pdf`
