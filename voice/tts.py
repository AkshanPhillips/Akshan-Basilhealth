"""
Text-to-speech with a primary + fallback provider, which doubles as the TTS half of the
provider-resilience demo.

Primary: edge-tts (Microsoft Edge's free, cloud-based neural TTS). Chosen over gTTS for
noticeably more natural prosody at zero cost, and over a locally-hosted neural TTS model
(e.g. Coqui) because a multi-hundred-MB model download/CPU-inference cost isn't justified
for a support-line voice when a free cloud option sounds better and is faster to set up --
consistent with the "local CPU desktop" constraint (no heavy local model needed here).

Fallback: pyttsx3, which wraps the OS's own offline TTS engine (SAPI5 on Windows, NSSpeech
on macOS, espeak on Linux). Chosen specifically BECAUSE it has zero network dependency --
it's the right fallback for exactly the failure mode we're simulating (edge-tts's cloud
endpoint being unreachable). Voice quality is noticeably more robotic, which is an honest,
visible signal in the demo that the system has degraded to its fallback, rather than
silently sounding identical.

Playback cancellation (for barge-in): generate_and_save() always produces the audio into a
file BEFORE playback starts, so playback is just "play this file" -- which can be killed
cleanly via sounddevice's stop(), with no risk of a half-decoded audio chunk or a TTS
generation thread needing to be interrupted mid-synthesis.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

from app import config


class TTSError(Exception):
    pass


async def _edge_tts_synthesize(text: str, out_path: Path, voice: str = config.TTS_VOICE_EDGE):
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_path))


def _pyttsx3_synthesize(text: str, out_path: Path):
    import sys
    if sys.platform == "darwin":
        import subprocess
        aiff_path = out_path.with_suffix(".aiff")
        subprocess.run(["/usr/bin/say", "-o", str(aiff_path), text], check=True)
        subprocess.run(["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16", str(aiff_path), str(out_path)], check=True)
        if aiff_path.exists():
            aiff_path.unlink()
        return

    import pyttsx3
    engine = pyttsx3.init()
    engine.save_to_file(text, str(out_path))
    engine.runAndWait()


def synthesize(text: str, out_path: Path, simulate_primary_failure: bool = False) -> dict:
    """Returns {"provider": "edge-tts"|"pyttsx3", "latency_ms": float, "path": str}.
    Falls back to pyttsx3 if edge-tts raises OR if simulate_primary_failure=True (used by
    the demo harness to deterministically exercise the fallback path)."""
    t0 = time.perf_counter()

    if not simulate_primary_failure:
        try:
            asyncio.run(_edge_tts_synthesize(text, out_path))
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {"provider": "edge-tts", "latency_ms": round(latency_ms, 2), "path": str(out_path)}
        except Exception:  # noqa: BLE001 -- network boundary, fall through to offline fallback
            pass  # fall through to fallback below

    try:
        _pyttsx3_synthesize(text, out_path)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {"provider": "pyttsx3", "latency_ms": round(latency_ms, 2), "path": str(out_path), "degraded": True}
    except Exception as e:
        raise TTSError(f"Both edge-tts and pyttsx3 failed: {e}")
