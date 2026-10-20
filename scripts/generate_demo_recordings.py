"""
Generates the 4 reference scenario clips for demo/recordings/:
  1. silence.wav - 3s of near-zero ambient tone
  2. noise_burst.wav - 3s of ambient with a short 80ms noise burst
  3. real_interruption.wav - speech saying "Wait, wrong order"
  4. adversarial_false_start.wav - speech saying "Um... wait, that is the wrong order"
"""
import subprocess
import sys
from pathlib import Path
import random
import struct
import subprocess
import wave
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = ROOT_DIR / "demo" / "recordings"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

SR = 16000

def write_wav(path: Path, samples: list[int]):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        raw = struct.pack("<" + "h" * len(samples), *samples)
        wf.writeframes(raw)

def make_silence():
    # 3 seconds of near silence
    samples = [int(random.gauss(0, 5)) for _ in range(3 * SR)]
    write_wav(RECORDINGS_DIR / "silence.wav", samples)
    print("Created silence.wav")

def make_noise_burst():
    # 3 seconds with an 80ms burst at 1.0s (under 240ms threshold)
    samples = [int(random.gauss(0, 5)) for _ in range(3 * SR)]
    start = int(1.0 * SR)
    burst_len = int(0.08 * SR)
    for i in range(burst_len):
        samples[start + i] = max(-32768, min(32767, int(random.gauss(0, 3000))))
    write_wav(RECORDINGS_DIR / "noise_burst.wav", samples)
    print("Created noise_burst.wav")

def make_speech_clips():
    for name, text in [
        ("real_interruption.wav", "Wait, wrong order."),
        ("adversarial_false_start.wav", "Um... wait, that is the wrong order.")
    ]:
        out_wav = RECORDINGS_DIR / name
        aiff_path = RECORDINGS_DIR / f"_{name}.aiff"
        try:
            subprocess.run(["/usr/bin/say", "-o", str(aiff_path), text], check=True)
            subprocess.run(["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16", "-c", "1", "-r", "16000", str(aiff_path), str(out_wav)], check=True)
            if aiff_path.exists():
                aiff_path.unlink()
            print(f"Created {name}")
        except Exception as e:
            print(f"Failed to generate {name}: {e}")

if __name__ == "__main__":
    make_silence()
    make_noise_burst()
    make_speech_clips()
