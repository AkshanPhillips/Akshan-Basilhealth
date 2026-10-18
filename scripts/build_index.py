"""
One-off / re-runnable script: embeds all docs in kb/ and writes a FAISS index to
data/faiss_index/. Run this once before starting the app, and again any time kb/ changes.

Usage:
    python scripts/build_index.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retriever import Retriever  # noqa: E402


def main():
    r = Retriever()
    print("Loading embedding model (first run downloads ~90MB, CPU only)...")
    r.build_from_kb()
    print(f"Indexed {len(r._chunks)} chunks from kb/. Index written to data/faiss_index/.")


if __name__ == "__main__":
    main()
