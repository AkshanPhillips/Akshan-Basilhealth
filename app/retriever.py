"""
RAG retrieval layer.

Embedding choice & justification (see design_notes.md Part A for the full version):
Groq does not serve an embeddings endpoint, so embedding has to happen locally. We use
sentence-transformers' all-MiniLM-L6-v2:
  - 384-dim, ~90MB download, runs comfortably on CPU (no GPU required) -- fits the stated
    constraint of running on a local CPU desktop.
  - Good quality-for-size tradeoff for short FAQ/policy passages (it's one of the most
    benchmarked small models for semantic search / STS tasks).
  - Fully local after the first download -- no per-query network cost or vendor lock-in for
    the retrieval step, which matters since Groq is already the network dependency for
    completions and STT.

Vector store choice: FAISS (flat, exact L2 search) over Chroma/pgvector etc. because:
  - With 10-20 short docs chunked into a few dozen vectors total, exact brute-force search
    (IndexFlatIP) is both fast enough (<1ms) and simpler to reason about than an ANN index
    or running a separate Chroma server process.
  - No server process to manage -- the index is a single file, which keeps local CPU setup
    trivial and keeps the "no fixed framework" ask honest (FAISS is a library, not a service).
  - If this had to scale to a large KB (thousands+ docs), the documented plan (see design
    notes) is to swap IndexFlatIP for IndexIVFFlat/HNSW or move to a managed vector DB --
    the retriever interface below is written so that swap doesn't touch calling code.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from app import config


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Simple word-count chunking with overlap. Good enough for short KB docs;
    for longer source docs a sentence-aware splitter would be worth the extra complexity."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text.strip()]
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]).strip())
        if end >= len(words):
            break
        start = end - overlap
    return chunks


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    source_path: str


class Retriever:
    def __init__(self):
        self._model = None  # lazy-loaded, see _ensure_model()
        self._index = None
        self._chunks: List[Chunk] = []

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(config.EMBED_MODEL_NAME)
        return self._model

    # ---------------- Build ----------------

    def build_from_kb(self, kb_dir: Path = config.KB_DIR) -> None:
        import faiss

        model = self._ensure_model()
        chunks: List[Chunk] = []
        for path in sorted(kb_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            for i, chunk_text in enumerate(_chunk_text(text, config.CHUNK_SIZE_WORDS, config.CHUNK_OVERLAP_WORDS)):
                chunks.append(Chunk(
                    doc_id=path.stem,
                    chunk_id=f"{path.stem}::{i}",
                    text=chunk_text,
                    source_path=str(path),
                ))

        texts = [c.text for c in chunks]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype="float32")

        index = faiss.IndexFlatIP(config.EMBED_DIM)  # cosine sim via normalized vectors + inner product
        index.add(embeddings)

        config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(config.INDEX_DIR / "kb.index"))
        meta_path = config.INDEX_DIR / "kb_meta.json"
        meta_path.write_text(json.dumps([c.__dict__ for c in chunks], indent=2), encoding="utf-8")

        self._index = index
        self._chunks = chunks

    # ---------------- Load ----------------

    def load(self) -> None:
        import faiss

        index_path = config.INDEX_DIR / "kb.index"
        meta_path = config.INDEX_DIR / "kb_meta.json"
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                "No FAISS index found. Run `python scripts/build_index.py` first."
            )
        self._index = faiss.read_index(str(index_path))
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        self._chunks = [Chunk(**r) for r in raw]

    # ---------------- Query ----------------

    def search(self, query: str, top_k: int = config.TOP_K_RETRIEVAL) -> List[dict]:
        if self._index is None:
            self.load()
        model = self._ensure_model()
        q_emb = model.encode([query], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype="float32")
        scores, idxs = self._index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            chunk = self._chunks[idx]
            results.append({
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "score": float(score),
            })
        return results


_singleton: Retriever | None = None


def get_retriever() -> Retriever:
    global _singleton
    if _singleton is None:
        _singleton = Retriever()
    return _singleton
