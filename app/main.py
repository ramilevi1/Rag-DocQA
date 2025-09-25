import os
import uuid
import hashlib
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, status, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .types import AskRequest, AskResponse
from .ingest import ingest_file, UPLOAD_DIR
from .rag import ask as rag_ask
from .rag import delete_session as rag_delete_session

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

app = FastAPI(title="AI Doc Q&A", version="1.1.0")

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # narrow in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend assets
app.mount("/static", StaticFiles(directory=os.path.join(ROOT, "frontend")), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(ROOT, "frontend", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

def _sha256(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

# -----------------------------
# Session Management
# -----------------------------
@app.post("/session/new")
def new_session():
    sid = str(uuid.uuid4())
    return {"session_id": sid}

@app.post("/session/clear")
def clear_session(session_id: Optional[str] = Form(default=None)):
    sid = session_id or "default"
    n = rag_delete_session(sid)
    return {"ok": True, "session_id": sid, "deleted": n}

# -----------------------------
# Upload files
# -----------------------------
@app.post("/upload")
async def upload(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(default=None),
):
    """
    Accept multiple files and immediately index their chunks for retrieval.
    Always 200 with per-file status for robustness.
    """
    sid = session_id or "default"
    results = []
    for uf in files:
        content = await uf.read()
        diag = {
            "sha256": _sha256(content or b""),
            "byte_length": len(content or b""),
            "first_32_bytes_hex": (content[:32] if content else b"").hex(),
            "session_id": sid,
        }
        try:
            info = ingest_file(content, uf.filename, session_id=sid)
            results.append({"filename": uf.filename, "status": "ok", "detail": info, "diag": diag})
        except Exception as e:
            results.append({"filename": uf.filename, "status": "error", "detail": str(e), "diag": diag})
    return JSONResponse({"results": results}, status_code=status.HTTP_200_OK)

# -----------------------------
# Upload raw text (diagnostic)
# -----------------------------
@app.post("/upload_text")
async def upload_text(
    text: str = Form(...),
    filename: str = Form(default="inline.txt"),
    session_id: Optional[str] = Form(default=None),
):
    sid = session_id or "default"
    content = text.encode("utf-8")
    try:
        info = ingest_file(content, filename, session_id=sid)
        return JSONResponse({"status": "ok", "detail": info})
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=400)

# -----------------------------
# Ask
# -----------------------------
@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest):
    """
    For summarization, ask: "Summarize the document(s) ..." and set top_k higher (e.g., 20)
    to pull more chunks into context.
    """
    answer, sources = rag_ask(
        payload.question,
        payload.system_prompt or "",
        max(1, payload.top_k),
        payload.session_id or "default"
    )
    return AskResponse(answer=answer, sources=sources)

# -----------------------------
# Download original
# -----------------------------
@app.get("/download/{filename}")
async def download(filename: str):
    path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(path):
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(path, filename=filename)
