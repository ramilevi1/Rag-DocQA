import os
import re
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import nltk
from nltk.tokenize import sent_tokenize

from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline

import chromadb
from chromadb.config import Settings

# -----------------------------
# Paths / Models
# -----------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STORAGE_DIR = os.path.join(ROOT_DIR, "storage", "chroma")
os.makedirs(STORAGE_DIR, exist_ok=True)

# IMPORTANT: use a 768-d model to match your existing collection dimensionality.
# Your error shows the collection is 768-d; MiniLM-L6-v2 is 384-d and caused the mismatch.
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"         # 768-d
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"           # cross-encoder re-ranker
GEN_MODEL_NAME = "google/flan-t5-base"                               # local, CPU-friendly

# -----------------------------
# Lazy NLTK setup (sentence split)
# -----------------------------
try:
    _ = sent_tokenize("Test.")
except LookupError:
    nltk.download("punkt", quiet=True)
    try:
        nltk.download("punkt_tab", quiet=True)  # newer NLTK
    except Exception:
        pass

# -----------------------------
# Instantiate models once
# -----------------------------
_embedder = SentenceTransformer(EMBED_MODEL_NAME)
_reranker = CrossEncoder(RERANK_MODEL_NAME)
_generator = pipeline("text2text-generation", model=GEN_MODEL_NAME)

# -----------------------------
# Vector DB (Chroma)
# -----------------------------
_client = chromadb.PersistentClient(
    path=STORAGE_DIR,
    settings=Settings(allow_reset=False, anonymized_telemetry=False),
)
# cosine space; persistent collection
_collection = _client.get_or_create_collection(
    name="docs",
    metadata={"hnsw:space": "cosine"}
)

# -----------------------------
# Data structures / helpers
# -----------------------------
@dataclass
class Chunk:
    id: str
    document_id: str
    text: str
    metadata: Dict

def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

def _normalize_ws(s: str) -> str:
    # keep paragraphs but normalize spaces
    lines = [ln.strip() for ln in s.splitlines()]
    paras, buf = [], []
    for ln in lines:
        if ln == "":
            if buf:
                paras.append(" ".join(buf))
                buf = []
        else:
            buf.append(ln)
    if buf:
        paras.append(" ".join(buf))
    return "\n\n".join(paras).strip()

def _split_sentences(p: str) -> List[str]:
    try:
        return sent_tokenize(p)
    except Exception:
        return re.split(r"(?<=[\.!\?])\s+", p)

def chunk_text(raw_text: str, max_chars: int = 1800, overlap_chars: int = 180) -> List[str]:
    """
    Robust, paragraph-first chunking with sentence fallback and character budget.
    """
    text = _normalize_ws(raw_text)
    if not text:
        return []

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for p in paragraphs:
        if len(p) <= max_chars:
            if len(buf) + len(p) + 2 <= max_chars:
                buf = (buf + "\n\n" + p).strip()
            else:
                # add overlap tail
                tail = buf[-overlap_chars:] if overlap_chars > 0 else ""
                flush()
                buf = (tail + "\n\n" + p).strip() if tail else p
        else:
            # paragraph too large → sentence pack
            sentences = _split_sentences(p)
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                if len(s) > max_chars:
                    # hard split long sentence
                    start = 0
                    while start < len(s):
                        end = min(start + max_chars, len(s))
                        piece = s[start:end]
                        if len(buf) + len(piece) + 1 <= max_chars:
                            buf = (buf + " " + piece).strip()
                        else:
                            tail = buf[-overlap_chars:] if overlap_chars > 0 else ""
                            flush()
                            buf = (tail + " " + piece).strip() if tail else piece
                        start = end
                else:
                    if len(buf) + len(s) + 1 <= max_chars:
                        buf = (buf + " " + s).strip()
                    else:
                        tail = buf[-overlap_chars:] if overlap_chars > 0 else ""
                        flush()
                        buf = (tail + " " + s).strip() if tail else s
    flush()
    return chunks

def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    embs = _embedder.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
    return embs.tolist()

# -----------------------------
# Chroma upsert / delete
# -----------------------------
def upsert_chunks(document_id: str, chunks: List[Chunk]) -> int:
    if not chunks:
        return 0
    # Stable, unique ids per chunk based on document_id + text hash
    ids = [f"{document_id}:{c.id}" for c in chunks]
    docs = [c.text for c in chunks]
    metas = [c.metadata for c in chunks]
    embs = embed_texts(docs)
    # Upsert into collection
    _collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    return len(chunks)

def delete_document(document_id: str, session_id: Optional[str] = None) -> int:
    # delete all ids that start with document_id
    where = {"document_id": document_id}
    if session_id:
        where["session_id"] = session_id
    try:
        _collection.delete(where=where)
        return 1
    except Exception:
        return 0

def delete_session(session_id: str) -> int:
    try:
        _collection.delete(where={"session_id": session_id})
        return 1
    except Exception:
        return 0

# -----------------------------
# Retrieval / Re-ranking
# -----------------------------
def similarity_search(query: str, top_k: int = 8, session_id: Optional[str] = None) -> List[Dict]:
    q_emb = embed_texts([query])[0]
    where = {"session_id": session_id} if session_id else {}
    res = _collection.query(query_embeddings=[q_emb], n_results=top_k, where=where)
    hits: List[Dict] = []
    if not res or not res.get("ids") or not res["ids"]:
        return hits
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0] if res.get("metadatas") else [{} for _ in ids]
    dists = res.get("distances", [[]])[0] if res.get("distances") else [0.0 for _ in ids]
    for i in range(len(ids)):
        hits.append({
            "id": ids[i],
            "text": docs[i],
            "score": float(dists[i]),
            "metadata": metas[i] or {}
        })
    return hits

def rerank(query: str, passages: List[Dict], top_k: int = 5) -> List[Dict]:
    if not passages:
        return []
    pairs = [(query, p["text"]) for p in passages]
    scores = _reranker.predict(pairs)
    for p, s in zip(passages, scores):
        p["rerank_score"] = float(s)
    passages.sort(key=lambda x: x["rerank_score"], reverse=True)
    return passages[:top_k]

# -----------------------------
# Prompting / Generation
# -----------------------------
def build_prompt(system_prompt: str, question: str, contexts: List[str], max_ctx_chars: int = 2000) -> str:
    # clip context aggressively for small models
    acc, out_ctx = 0, []
    for c in contexts:
        if acc + len(c) + 8 > max_ctx_chars:
            break
        out_ctx.append(c)
        acc += len(c) + 8
    ctx_block = "\n\n---\n\n".join(out_ctx)
    sys_prompt = system_prompt.strip() or (
        "You are a helpful assistant. Answer ONLY using the provided context. "
        "If the answer is not in the context, say you don't know."
    )
    return (
        f"{sys_prompt}\n\n"
        f"Context:\n{ctx_block}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )

def generate_answer(system_prompt: str, question: str, contexts: List[str]) -> str:
    prompt = build_prompt(system_prompt, question, contexts)
    out = _generator(prompt, max_new_tokens=384, do_sample=False)
    return out[0]["generated_text"].strip()

# -----------------------------
# Public API
# -----------------------------
def ask(question: str, system_prompt: str, top_k: int = 8, session_id: Optional[str] = None) -> Tuple[str, List[Dict]]:
    # 1) dense retrieval
    initial = similarity_search(question, top_k=max(top_k, 8), session_id=session_id)
    if not initial:
        return "I don't know. I could not retrieve any relevant context.", []

    # 2) re-rank with cross-encoder
    reranked = rerank(question, initial, top_k=min(5, len(initial)))
    contexts = [p["text"] for p in reranked]

    # 3) generate
    answer = generate_answer(system_prompt or "", question, contexts)

    # 4) sources (for UI)
    sources: List[Dict] = []
    for p in reranked:
        md = p.get("metadata", {}) or {}
        sources.append({
            "id": p.get("id"),
            "document_id": md.get("document_id"),
            "score": p.get("rerank_score", 0.0),
            "text": p.get("text"),
            "metadata": md
        })
    return answer, sources
