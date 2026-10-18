"""
Small helpers for reading/writing WAV audio consistently at the pipeline's working format
(16kHz, 16-bit, mono -- what both webrtcvad and Groq Whisper expect).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from app import config


def load_wav_as_pcm16(path: Path) -> bytes:
    """Load a WAV file, resample/convert to 16kHz mono 16-bit if needed, return raw PCM bytes."""
    data, sr = sf.read(str(path), dtype="int16", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1).astype("int16")
    if sr != config.SAMPLE_RATE:
        # Simple linear resample -- fine for speech-grade demo audio; a production system
        # would use a proper polyphase resampler (e.g. scipy.signal.resample_poly).
        duration = len(data) / sr
        target_len = int(duration * config.SAMPLE_RATE)
        x_old = np.linspace(0, 1, num=len(data))
        x_new = np.linspace(0, 1, num=target_len)
        data = np.interp(x_new, x_old, data).astype("int16")
    return data.tobytes()


def save_pcm16_as_wav(pcm_bytes: bytes, path: Path, sample_rate: int = config.SAMPLE_RATE):
    data = np.frombuffer(pcm_bytes, dtype="int16")
    sf.write(str(path), data, sample_rate, subtype="PCM_16")


def record_from_mic(seconds: float, sample_rate: int = config.SAMPLE_RATE) -> bytes:
    """Blocking mic capture -- used by the live demo harness. Requires sounddevice and a
    working input device, which is why this is separate from the offline-replay path."""
    import sounddevice as sd
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    return audio.tobytes()


def play_wav_file(path: Path, blocking: bool = False):
    import sounddevice as sd
    data, sr = sf.read(str(path), dtype="float32")
    sd.play(data, sr)
    if blocking:
        sd.wait()


def stop_playback():
    import sounddevice as sd
    sd.stop()
