# AI Doc Q&A (Local RAG with Re-Ranking + FLAN-T5)

A simple, local, fully open-source pipeline to ask questions about your PDFs / Word docs / TXT using:
- Chroma vector DB
- BERT-based sentence embeddings
- Cross-encoder re-ranking
- Google FLAN-T5 for generation
- FastAPI backend + minimal HTML/JS frontend

## Features
- Upload multiple files (PDF, DOCX, TXT, MD)
- Chunking with overlap, paragraph/sentence-aware
- Embedding + vector similarity search
- Re-ranking top candidates for better relevance
- System prompt to shape behavior (tone/policies)
- Persisted vector store and saved uploads
- CPU-only by default

## Architecture
1. **Ingestion**: `POST /upload`  
   - Extract text (PyMuPDF / python-docx / plain read)  
   - Chunk (~1000 chars, 200 overlap)  
   - Embed with `sentence-transformers/bert-base-nli-mean-tokens`  
   - Upsert into Chroma (persisted at `./storage/chroma`)

2. **Query**: `POST /ask`  
   - Vector similarity (top_k=8)  
   - Re-rank via `cross-encoder/ms-marco-MiniLM-L-6-v2` (top 5)  
   - Build prompt with system prompt + best contexts  
   - Generate with `google/flan-t5-base`  
   - Return answer + cited chunks

## Endpoints
- `GET /` : UI
- `POST /upload` : multipart form (files)
- `POST /ask` : `{ question, system_prompt?, top_k? }`
- `GET /download/{filename}` : fetch uploaded original

## Run
```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# open http://localhost:8000
