"""
Speech-to-text via Groq's Whisper endpoint, with the provider-resilience behavior the
brief asks for: simulate the provider failing mid-call and show fallback/retry without a
hung or crashed turn.

Resilience strategy (documented, not just implemented silently):
  - 1 retry with a short backoff for transient errors (timeout, connection reset) -- most
    real STT provider blips are transient.
  - If retries are exhausted, we do NOT hang the turn. We return a StructuredError the
    voice coordinator can act on immediately: play a short pre-recorded/TTS apology
    ("Sorry, I'm having trouble hearing you -- could you repeat that?") and re-open the
    mic, rather than leaving the call in limbo.
  - We deliberately do NOT fail over to a different STT provider for this assignment (no
    second STT vendor is in scope / no fixed framework requirement to add one) -- see
    Part C's answer on "Groq degrades mid-launch" for the documented production plan for
    that, which is a bigger architectural decision than this demo needs to prove.
"""
from __future__ import annotations

import io
import time

from app import config
from app.groq_client import get_client


class STTError(Exception):
    def __init__(self, message: str, recoverable: bool = True):
        super().__init__(message)
        self.recoverable = recoverable


def transcribe(wav_bytes: bytes, filename: str = "audio.wav", max_retries: int = 1, simulate_failure: bool = False) -> dict:
    """Returns {"text": str, "latency_ms": float, "attempts": int}.
    simulate_failure=True forces every attempt to raise, to exercise the fallback path
    deterministically in the demo harness without needing to actually kill the network."""
    client = get_client()
    last_err = None
    t0 = time.perf_counter()

    for attempt in range(1, max_retries + 2):
        try:
            if simulate_failure:
                raise ConnectionError("Simulated STT provider failure (network/key invalidated)")
            resp = client.audio.transcriptions.create(
                file=(filename, io.BytesIO(wav_bytes)),
                model=config.GROQ_WHISPER_MODEL,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {"text": resp.text, "latency_ms": round(latency_ms, 2), "attempts": attempt}
        except Exception as e:  # noqa: BLE001 -- network boundary, intentionally broad
            last_err = e
            if attempt <= max_retries:
                time.sleep(0.2 * attempt)  # short linear backoff
                continue
            break

    latency_ms = (time.perf_counter() - t0) * 1000.0
    raise STTError(
        f"STT failed after {max_retries + 1} attempts ({last_err}); turn degraded gracefully, not hung.",
        recoverable=True,
    )
