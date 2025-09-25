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
----------------------------------------------------------------------------------------------------------------------------------------

## End-to-end user flow for **Rag-DocQA project** ).
# 1) Page render (browser → server)

   1. **User opens the site**
      * **File:** `app/main.py`
      * **Endpoint:** `GET /`
      * **What happens:** Server reads and returns `frontend/index.html`. Static assets are served from `/static` (your `frontend/` folder).

   2. **Frontend bootstraps**
      * **Files:** `frontend/index.html`, `frontend/app.js`, `frontend/styles.css`
      * **What happens:** The page loads UI elements (file input, question box, system prompt, buttons).
      * **Session setup (optional but supported):** `app.js` can call `POST /session/new` to get a `session_id` and store it; otherwise default session `"default"` is used in the backend.
----------------------------------------------------------------------------------------------------------------------------------------
# 2) Uploading documents

   3. **User selects one or more files and clicks Upload**

      * **Frontend → Backend:** `POST /upload` (multipart form)
   
        * **File:** `app/main.py` → `upload()`
        * **Params:** `files[]`, and optionally `session_id` (Form).
        * **For each file:** reads bytes and calls `ingest_file(...)`.
   
   4. **Ingestion pipeline**
   
      * **File:** `app/ingest.py`
      * **Function:** `ingest_file(file_bytes, filename, session_id, meta=None)`
      * **Steps inside:**
   
        1. **Save to disk** in `data/uploads/` via `save_upload(...)`.
        2. **Extract text** via `extract_text(path)`:
   
           * **PDF:**
   
             * Try text layer: `parse_pdf_text_layer` (PyMuPDF `fitz`).
             * If empty, **OCR fallback**: `ocr_pdf_pages` (PyMuPDF page rasterization → `pytesseract`) and set `used_ocr=True`.
           * **DOCX:** `parse_docx` (python-docx).
           * **TXT/MD:** `parse_txt` (multi-encoding try, UTF-8 fallback).
        3. **Chunking** with overlap:
   
           * **Function:** `chunk_text(raw_text, max_chars=1800, overlap_chars=180)`
           * **Algorithm:** paragraph-first packing; if a paragraph is too large, sentence-level packing with a hard split for very long sentences; overlap tail to preserve context continuity.
        4. **Build document id** and per-chunk ids:
   
           * `document_id = _hash(filename | tag)`
           * Chunk ids are `_hash(document_id : chunk_idx : text_head)` for stability.
        5. **Upsert vectors into the vector DB (Chroma)**:
   
           * Create `Chunk` dataclasses with metadata:
   
             * `document_id`, `filename`, `chunk_index`, `session_id`
           * **Call:** `upsert_chunks(document_id, chunks)` → embeds and writes to Chroma.
   
   5. **Embedding + Vector DB write**
   
      * **File:** `app/rag.py`
      * **Function:** `upsert_chunks(...)`
      * **Embedding model:** `SentenceTransformer("sentence-transformers/all-mpnet-base-v2")` → **768-dim** embeddings
   
        * **Function:** `embed_texts(texts)` uses `_embedder.encode(..., normalize_embeddings=True)`
      * **Vector store:** **ChromaDB** (persistent), collection `docs` with cosine space
   
        * **Client:** `chromadb.PersistentClient(path=storage/chroma)`
        * **Collection:** `_client.get_or_create_collection(name="docs", metadata={"hnsw:space": "cosine"})`
      * **Operation:** `collection.upsert(ids, documents, metadatas, embeddings)`
   
        * This writes vectors and metadata scoped by `session_id` so later queries can be filtered to that session.
   
   6. **Upload response**
   
      * **File:** `app/main.py` → `upload()`
      * Returns a per-file status payload (ok/error, doc id, chunk count, OCR/encoding flags).
      * **Result:** As soon as the response returns, new chunks are already embedded and searchable.
   ----------------------------------------------------------------------------------------------------------------------------------------
# 3) Asking a question (Q\&A or summarization)

   7. **User enters a question and submits**

   * **Frontend → Backend:** `POST /ask`
   * **File:** `app/main.py` → `ask(payload)`
   * **Payload model:** `app/types.py` → `AskRequest`:

     * `question: str`
     * `system_prompt: str` (optional)
     * `top_k: int` (optional; for summaries use a higher number, e.g., 20)
     * `session_id: str` (optional; defaults to `"default"`)

   8. **RAG pipeline orchestration**

   * **File:** `app/rag.py`
   * **Function:** `ask(question, system_prompt, top_k, session_id)`
   * **Steps inside:**

        1. **Dense retrieval** (Vector DB search):

        * **Function:** `similarity_search(query, top_k, session_id)`
        * **What it does:**

          * Embeds the **query** with `all-mpnet-base-v2` (768-dim).
          * Calls Chroma `collection.query(query_embeddings=[q_emb], n_results=top_k, where={"session_id": <sid>})` to restrict results to this session’s documents.
          * Returns candidate passages with raw vector similarity scores and metadata.
        * **Algorithm:** cosine similarity in HNSW index (managed by Chroma).
        2. **Cross-encoder re-ranking**:

        * **Function:** `rerank(query, passages, top_k=5)`
        * **Model:** `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`
        * **What it does:**

          * Forms pairs `(query, passage_text)` for the retrieved candidates.
          * Scores each pair; higher scores = more relevant.
          * Sorts and keeps the best few (default 5).
        * **Why:** Re-ranking improves precision over raw vector similarity (helps “pick the right chunk”).
        3. **Prompt construction**:

        * **Function:** `build_prompt(system_prompt, question, contexts, max_ctx_chars=2000)`
        * **What it does:**

          * Concatenates the top re-ranked chunk texts into a bounded **context block** (character budget to fit generator input).
          * Prepends a **system prompt** (default: “Answer ONLY using the provided context; otherwise say you don’t know.”).
          * Appends the user question.
        4. **Answer generation (LLM)**:

        * **Function:** `generate_answer(system_prompt, question, contexts)`
        * **Model:** `pipeline("text2text-generation", model="google/flan-t5-base")`
        * **What it does:**

          * Generates the final answer strictly using the provided context block.
          * Default `max_new_tokens=384`.
        5. **Return answer + sources**:

        * Packs the generated answer plus the re-ranked source snippets (text, score, metadata such as `filename`, `document_id`, `chunk_index`) for UI display.

   9. **Response rendered in UI**

   * **Frontend:** `frontend/app.js`
   * Displays the model’s answer and the supporting chunks (doc name / chunk index / scores).
   * If you asked for a **summary**, the same pipeline runs; just use a higher `top_k` to bring more context, and a system prompt like “Summarize the following documents focusing on X.”
----------------------------------------------------------------------------------------------------------------------------------------
# 4) Session management (optional but supported)

   * **Create a new session**
   
     * `POST /session/new` → returns a `session_id` (UUID).
     * Use this in subsequent `/upload` and `/ask` calls so data and retrieval are **isolated per session**.
   
   * **Clear a session**
   
     * `POST /session/clear` with `session_id` (Form)
     * **File:** `app/main.py` → `clear_session(...)` → `rag_delete_session(sid)`
     * **File:** `app/rag.py` → `delete_session(session_id)` deletes all vectors whose metadata includes that `session_id`.
   
   This lets you reset the workspace without wiping the entire vector store.
----------------------------------------------------------------------------------------------------------------------------------------
# 5) Algorithms & models (summary)
   
   * **Chunking:** paragraph-first packing with sentence fallback; overlap to maintain continuity (`chunk_text` in `app/rag.py`).
   * **Embeddings:** `sentence-transformers/all-mpnet-base-v2` (**768-dim**), normalized vectors.
   * **Vector DB:** **ChromaDB** (persistent HNSW, cosine).
   * **Retrieval:** `collection.query` by query embedding with `where={"session_id": ...}` filtering.
   * **Re-ranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (pairwise scoring of (query, passage)).
   * **Generation:** `google/flan-t5-base` via `transformers` pipeline “text2text-generation”.
   * **Prompting:** strict grounding to provided context; system prompt is configurable via API/UI.
   * **Sessions:** metadata field `session_id` ensures tenant-like isolation.
----------------------------------------------------------------------------------------------------------------------------------------
   # 6) Typical control flow (numbered call chain)

         1. **Browser** → `GET /` → `index.html`
         2. **Browser** (optional) → `POST /session/new` → `{session_id}`
         3. **Browser** → `POST /upload` (files, session\_id)
         
            * `app/main.py: upload()`
            * `app/ingest.py: ingest_file()`
   
        * `extract_text()` → parser (pdf/docx/txt) + OCR fallback
        * `chunk_text()`
        * `upsert_chunks()` → **embed\_texts()** → Chroma `upsert`
         4. **Browser** → `POST /ask` (question, system\_prompt, top\_k, session\_id)

         * `app/main.py: ask()`
         * `app/rag.py: ask()`
      
           * `similarity_search()` → Chroma `query` (session filter)
           * `rerank()` → **CrossEncoder**
           * `build_prompt()`
           * `generate_answer()` → **FLAN-T5**
         * Response: `{ answer, sources[] }`
         5. **Browser** renders answer and cited chunks.
----------------------------------------------------------------------------------------------------------------------------------------
# 7) Notes for quality and correctness

   * If you want **longer summaries**, increase `top_k` in `/ask` so more chunks make it into the context, and optionally raise the `max_ctx_chars` budget in `build_prompt`.
   * If you upload the **same file name** again, `ingest_file` deletes prior vectors for the same `document_id` (scoped to session) before upserting fresh chunks—so retrieval reflects the latest upload.
   * OCR is only used when PDFs lack extractable text. If OCR is used, the upload response includes `used_ocr: true`.
----------------------------------------------------------------------------------------------------------------------------------------
