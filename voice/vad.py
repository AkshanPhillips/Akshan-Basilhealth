"""
Thin wrapper around webrtcvad, framed for streaming use by the barge-in coordinator.

webrtcvad classifies fixed-size frames (10/20/30ms of 16kHz 16-bit mono PCM) as voiced or
not. It's a lightweight, CPU-trivial, pre-deep-learning VAD -- chosen deliberately over a
neural VAD (e.g. Silero) for this assignment because:
  - Zero model download, pure C extension, runs instantly on any CPU -- fits the "local
    CPU desktop" constraint with no setup friction.
  - For the barge-in task we don't need SOTA accuracy on noisy far-field audio; we need a
    fast, deterministic voiced/unvoiced signal per 30ms frame that a simple state machine
    can threshold against sustained duration (see barge_in.py). A heavier VAD would add
    latency (bad, since this whole pipeline is latency-sensitive) for marginal gain at this
    scale.
  - Documented limitation (see decisions_log.md / red-team findings): webrtcvad is an
    energy+spectral heuristic, not a learned speech/non-speech classifier -- loud
    non-speech noise bursts (a door slam, a dog bark) can still register as "voiced" for
    a frame or two. That's exactly why barge-in.py requires SUSTAINED voiced frames
    (300ms+) before treating it as a real interruption, rather than triggering on a single
    voiced frame.
"""
from __future__ import annotations

import webrtcvad

from app import config


class FrameVAD:
    def __init__(self, aggressiveness: int = config.VAD_AGGRESSIVENESS, sample_rate: int = config.SAMPLE_RATE):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_bytes = int(sample_rate * (config.VAD_FRAME_MS / 1000.0) * 2)  # 16-bit mono

    def iter_frames(self, pcm_bytes: bytes):
        """Yield fixed-size frames from a raw PCM buffer, dropping a trailing partial frame."""
        n = len(pcm_bytes)
        for start in range(0, n - self.frame_bytes + 1, self.frame_bytes):
            yield pcm_bytes[start:start + self.frame_bytes]

    def is_speech(self, frame: bytes) -> bool:
        if len(frame) != self.frame_bytes:
            return False
        return self.vad.is_speech(frame, self.sample_rate)

    def classify_stream(self, pcm_bytes: bytes) -> list[bool]:
        """Returns a list of per-frame voiced/unvoiced booleans for an entire buffer."""
        return [self.is_speech(f) for f in self.iter_frames(pcm_bytes)]
