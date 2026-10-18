"""
Classifier response cache.

Why cache the classifier specifically (not the RAG/tool agents): support chat traffic is
heavy-tailed -- a small set of phrasings ("where's my order", "track my order", "wrong item
delivered") recur constantly across different users, and the router only needs the intent
label, which is stable for near-identical inputs. RAG/tool-agent responses are far more
likely to be unique per-user (they're grounded in a specific order_id/query), so caching
them has a much lower hit rate and higher staleness risk (e.g. caching an order_status
answer could literally serve a stale delivery status).

Cache key: normalized (lowercased, whitespace-collapsed) message text, hashed with sha256.
This is intentionally a *lexical* cache (exact-after-normalization match), not a semantic
one -- a semantic cache (embedding similarity) risks silently misrouting two different
intents that happen to be nearby in embedding space, and for a classifier call that costs
single-digit milliseconds on Groq, the complexity isn't worth it. See design_notes.md for
the scale-out version of this (shared Redis cache keyed the same way) for 10k concurrent
sessions.

Storage: SQLite file so the cache survives process restarts and hit-rate is measurable
across a whole local test run, without standing up Redis for a take-home assignment.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
import time
from dataclasses import dataclass

from app import config


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _key_for(text: str) -> str:
    return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0


class ClassifierCache:
    def __init__(self, db_path=config.CLASSIFIER_CACHE_PATH):
        self.db_path = db_path
        self.stats = CacheStats()
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS classifier_cache (
                   key TEXT PRIMARY KEY,
                   message TEXT,
                   intent TEXT,
                   created_at REAL
               )"""
        )
        self._conn.commit()

    def get(self, message: str) -> str | None:
        key = _key_for(message)
        row = self._conn.execute(
            "SELECT intent FROM classifier_cache WHERE key = ?", (key,)
        ).fetchone()
        if row:
            self.stats.hits += 1
            return row[0]
        self.stats.misses += 1
        return None

    def set(self, message: str, intent: str) -> None:
        key = _key_for(message)
        self._conn.execute(
            "INSERT OR REPLACE INTO classifier_cache (key, message, intent, created_at) VALUES (?, ?, ?, ?)",
            (key, message, intent, time.time()),
        )
        self._conn.commit()

    def report(self) -> dict:
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "total": self.stats.total,
            "hit_rate": round(self.stats.hit_rate, 4),
        }


_singleton: ClassifierCache | None = None


def get_cache() -> ClassifierCache:
    global _singleton
    if _singleton is None:
        _singleton = ClassifierCache()
    return _singleton
