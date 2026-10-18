"""
Latency instrumentation. The brief requires latency figures to be *instrumented*, not
estimated -- every timestamp here comes from time.perf_counter() calls wrapped directly
around the real operation (STT call, TTS first-chunk, etc.), not a guess or a hardcoded
"typical" number.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Timer:
    label: str
    start: float = field(default_factory=time.perf_counter)
    end: float | None = None

    def stop(self) -> float:
        self.end = time.perf_counter()
        return self.elapsed_ms

    @property
    def elapsed_ms(self) -> float:
        end = self.end if self.end is not None else time.perf_counter()
        return (end - self.start) * 1000.0


@contextmanager
def timed(label: str, sink: list | None = None):
    t = Timer(label)
    try:
        yield t
    finally:
        t.stop()
        if sink is not None:
            sink.append({"label": t.label, "ms": round(t.elapsed_ms, 2)})


class LatencyLog:
    """Accumulates per-turn latency events and can dump them as JSON for the demo report."""

    def __init__(self):
        self.turns: list[dict] = []

    def new_turn(self, turn_id: str) -> dict:
        turn = {"turn_id": turn_id, "events": [], "end_of_speech_ts": None, "first_audio_ts": None}
        self.turns.append(turn)
        return turn

    @staticmethod
    def mark_end_of_speech(turn: dict):
        turn["end_of_speech_ts"] = time.perf_counter()

    @staticmethod
    def mark_first_audio(turn: dict):
        turn["first_audio_ts"] = time.perf_counter()
        if turn["end_of_speech_ts"] is not None:
            turn["end_to_end_ms"] = round((turn["first_audio_ts"] - turn["end_of_speech_ts"]) * 1000.0, 2)

    def dump(self, path: Path):
        serializable = [
            {k: v for k, v in t.items() if k not in ("end_of_speech_ts", "first_audio_ts")}
            for t in self.turns
        ]
        path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    def summary(self) -> dict:
        e2e = [t["end_to_end_ms"] for t in self.turns if "end_to_end_ms" in t]
        if not e2e:
            return {"turns": len(self.turns), "end_to_end_ms": None}
        e2e_sorted = sorted(e2e)
        n = len(e2e_sorted)
        return {
            "turns_measured": n,
            "min_ms": e2e_sorted[0],
            "max_ms": e2e_sorted[-1],
            "mean_ms": round(sum(e2e_sorted) / n, 2),
            "p50_ms": e2e_sorted[n // 2],
        }
