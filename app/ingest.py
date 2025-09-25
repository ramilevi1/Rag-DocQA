import os
from typing import List, Dict, Optional, Tuple

import fitz  # PyMuPDF
import docx
from PIL import Image
import pytesseract

from .rag import chunk_text, Chunk, upsert_chunks, delete_document, _hash

UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTS = {".pdf", ".docx", ".txt", ".md"}

# If Tesseract isn't on PATH, you can set it explicitly:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def save_upload(file_bytes: bytes, filename: str) -> str:
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path

# -----------------------------
# Parsers
# -----------------------------
def parse_pdf_text_layer(path: str) -> str:
    parts: List[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            parts.append(page.get_text("text"))
    return "\n".join(parts).strip()

def ocr_pdf_pages(path: str, dpi: int = 220, lang: str = "eng") -> str:
    try:
        out_parts: List[str] = []
        with fitz.open(path) as doc:
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            for page in doc:
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img, lang=lang)
                if text:
                    out_parts.append(text)
        return "\n".join(out_parts).strip()
    except Exception:
        return ""

def parse_docx(path: str) -> str:
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs)

def parse_txt(path: str) -> Tuple[str, Optional[str]]:
    # try UTF-8, then fallbacks
    with open(path, "rb") as f:
        raw = f.read()
    tried = [
        "utf-8",
        "utf-8-sig",
        "utf-16-le", "utf-16-be",
        "utf-32-le", "utf-32-be",
        "cp1252", "latin-1",
    ]
    for enc in tried:
        try:
            txt = raw.decode(enc, errors="strict")
            if txt.strip():
                return txt, enc
        except Exception:
            continue
    # last resort
    return raw.decode("utf-8", errors="ignore"), None

def extract_text(path: str) -> Tuple[str, Dict]:
    ext = os.path.splitext(path)[1].lower()
    flags: Dict = {"used_ocr": False, "encoding": ""}

    if ext == ".pdf":
        txt = parse_pdf_text_layer(path)
        if not txt.strip():  # try OCR
            txt = ocr_pdf_pages(path, dpi=220, lang="eng")
            flags["used_ocr"] = True
        return txt, flags

    if ext == ".docx":
        return parse_docx(path), flags

    if ext in {".txt", ".md"}:
        txt, enc = parse_txt(path)
        flags["encoding"] = enc or ""
        return txt, flags

    raise ValueError(f"Unsupported extension: {ext}")

# -----------------------------
# Ingest
# -----------------------------
def ingest_file(file_bytes: bytes, filename: str, session_id: Optional[str] = None, meta: Optional[Dict] = None) -> Dict:
    """
    Saves the file, extracts text, chunks it, upserts to Chroma with embeddings.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        raise ValueError(f"File type {ext} not allowed. Allowed: {', '.join(sorted(ALLOWED_EXTS))}")

    path = save_upload(file_bytes, filename)
    raw_text, flags = extract_text(path)

    if not raw_text.strip():
        hint = ""
        if ext == ".pdf":
            hint = " The PDF appears to have no selectable text (likely scanned). Install Tesseract OCR and retry."
        if ext in {".txt", ".md"}:
            hint = " The text file seems empty or has unusual encoding. Try re-saving as UTF-8 and re-upload."
        raise ValueError("No text extracted." + hint)

    document_id = _hash(filename + "|" + (meta.get("tag", "") if meta else ""))

    # optional: delete previous vectors for same doc if re-uploaded
    delete_document(document_id, session_id=session_id)

    parts = chunk_text(raw_text, max_chars=1800, overlap_chars=180)
    prepared: List[Chunk] = []
    for i, txt in enumerate(parts):
        prepared.append(Chunk(
            id=_hash(f"{document_id}:{i}:{txt[:64]}"),
            document_id=document_id,
            text=txt,
            metadata={
                "document_id": document_id,
                "filename": filename,
                "chunk_index": i,
                "session_id": session_id or "default",
            }
        ))
    upsert_chunks(document_id, prepared)

    return {
        "document_id": document_id,
        "filename": os.path.basename(path),
        "chunks": len(prepared),
        "used_ocr": flags.get("used_ocr", False),
        "encoding": flags.get("encoding") or "",
        "session_id": session_id or "default",
    }
