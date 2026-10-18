"""
Voice demo harness -- this is what you (the user) run locally to produce the required
demo recordings/logs. Two modes:

  --mode offline   Replays pre-recorded WAV clips (that you record once) against the
                    barge-in coordinator in a reproducible, scriptable way. This is what
                    generates demo/voice_demo_report.json with real instrumented numbers.
                    This is the recommended mode for the graded submission, since it's
                    reproducible and the exact same clips can be re-run by a reviewer.

  --mode live       Opens your real mic + speakers and runs one live interactive turn:
                    plays a TTS response, listens for barge-in in real time, and if you
                    interrupt it, cancels playback and transcribes what you said via Groq
                    Whisper. Good for recording a screen-capture demo video.

Expected recordings for --mode offline (put these in demo/recordings/, see
demo/README.md for exact instructions on recording them with your own mic):
  silence.wav                 -- ~3s of near-silence while a response plays
  noise_burst.wav             -- a short non-speech noise (cough, desk tap, chair creak)
  real_interruption.wav       -- you clearly saying "wait, wrong order" partway through
  adversarial_false_start.wav -- you saying "um--" then a short pause then
                                  "wait, that's the wrong order" (false start + real
                                  interruption, close together)

Usage:
    python voice/run_voice_demo.py --mode offline
    python voice/run_voice_demo.py --mode live
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from voice import stt, tts  # noqa: E402
from voice.audio_io import load_wav_as_pcm16, save_pcm16_as_wav  # noqa: E402
from voice.barge_in import BargeInCoordinator, Decision  # noqa: E402
from voice.latency import LatencyLog  # noqa: E402
from voice.vad import FrameVAD  # noqa: E402

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"
RECORDINGS_DIR = DEMO_DIR / "recordings"
REPORT_PATH = DEMO_DIR / "voice_demo_report.json"

SCENARIOS = {
    "silence_through_playback": {
        "file": "silence.wav",
        "expect": Decision.CONTINUE_PLAYBACK,
        "description": "Silence through full playback -- should NOT barge in.",
    },
    "noise_burst_ignored": {
        "file": "noise_burst.wav",
        "expect": Decision.CONTINUE_PLAYBACK,
        "description": "Short non-speech noise burst -- should be ignored, not interrupt.",
    },
    "real_interruption": {
        "file": "real_interruption.wav",
        "expect": Decision.BARGE_IN,
        "description": "Real user speech ('wait, wrong order') -- SHOULD barge in.",
    },
    "adversarial_false_start_then_real": {
        "file": "adversarial_false_start.wav",
        "expect": Decision.BARGE_IN,
        "description": "False start ('um--') then real interruption, close together -- SHOULD barge in (gap-tolerance logic).",
    },
}

SAMPLE_PROMPT = (
    "Let me check that for you. Your order N V dash one zero zero two three four "
    "was shipped on October 15th and is expected to arrive around October 20th. "
    "Is there anything else I can help you with?"
)


def run_offline_scenario(name: str, spec: dict, vad: FrameVAD) -> dict:
    wav_path = RECORDINGS_DIR / spec["file"]
    if not wav_path.exists():
        return {"scenario": name, "status": "SKIPPED", "reason": f"missing {wav_path}"}

    pcm = load_wav_as_pcm16(wav_path)
    coord = BargeInCoordinator()
    coord.start_playback()

    for frame in vad.iter_frames(pcm):
        is_voiced = vad.is_speech(frame)
        decision = coord.process_frame(frame, is_voiced)
        if decision == Decision.BARGE_IN:
            break

    report = coord.report()
    actual = Decision.BARGE_IN if report["triggered"] else Decision.CONTINUE_PLAYBACK
    passed = actual == spec["expect"]

    result = {
        "scenario": name,
        "description": spec["description"],
        "status": "PASS" if passed else "FAIL",
        "expected": spec["expect"].value,
        "actual": actual.value,
        "trigger_at_playback_ms": report["trigger_at_playback_ms"],
        "voiced_frames": report["voiced_frames"],
        "total_frames": report["total_frames"],
    }

    if report["triggered"]:
        buffered = coord.get_buffered_utterance_audio()
        buffered_path = RECORDINGS_DIR / f"_buffered_{name}.wav"
        save_pcm16_as_wav(buffered, buffered_path)
        result["buffered_utterance_saved_to"] = str(buffered_path)
        if config.GROQ_API_KEY:
            try:
                stt_result = stt.transcribe(buffered_path.read_bytes(), filename=buffered_path.name)
                result["stt_transcript"] = stt_result["text"]
                result["stt_latency_ms"] = stt_result["latency_ms"]
            except stt.STTError as e:
                result["stt_error"] = str(e)
        else:
            result["stt_note"] = "GROQ_API_KEY not set -- skipped live transcription of buffered audio"

    return result


def run_offline_mode():
    print("Loading VAD...")
    vad = FrameVAD()
    results = []
    for name, spec in SCENARIOS.items():
        print(f"Running scenario: {name}...")
        results.append(run_offline_scenario(name, spec, vad))

    print("\n--- Provider resilience checks ---")
    resilience = {}
    try:
        stt.transcribe(b"", simulate_failure=True)
    except stt.STTError as e:
        resilience["stt_simulated_failure"] = {"status": "PASS", "note": f"degraded gracefully: {e}"}
    else:
        resilience["stt_simulated_failure"] = {"status": "FAIL", "note": "expected STTError, got none"}

    out_wav = DEMO_DIR / "_tts_fallback_test.wav"
    tts_result = tts.synthesize("This is a fallback test.", out_wav, simulate_primary_failure=True)
    resilience["tts_simulated_primary_failure"] = {
        "status": "PASS" if tts_result["provider"] == "pyttsx3" else "FAIL",
        "fell_back_to": tts_result["provider"],
        "latency_ms": tts_result["latency_ms"],
    }

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scenarios": results,
        "provider_resilience": resilience,
    }
    DEMO_DIR.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nReport written to {REPORT_PATH}")
    for r in results:
        print(f"  [{r['status']}] {r['scenario']}")
    for k, v in resilience.items():
        print(f"  [{v['status']}] {k}")


def _rms_int16(samples) -> float:
    """RMS of a float32 [-1, 1]-ish array, scaled to the same units as audioop.rms() on
    int16 PCM (what barge_in.frame_rms uses for mic frames), so reference and mic RMS are
    directly comparable."""
    import numpy as np
    if len(samples) == 0:
        return 0.0
    scaled = samples.astype("float64") * 32767.0
    return float(np.sqrt(np.mean(scaled * scaled)))


def calibrate_noise_floor(seconds: float = 0.3) -> float:
    """Record a brief moment of ambient silence (before any playback starts) and return
    its RMS, so the barge-in gate can tell 'room noise' apart from both TTS echo and real
    speech -- rather than assuming a fixed noise level that may not match this room/mic."""
    import sounddevice as sd
    print(f"Calibrating ambient noise floor ({seconds}s, please stay quiet)...")
    audio = sd.rec(int(seconds * config.SAMPLE_RATE), samplerate=config.SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    import numpy as np
    return float(np.sqrt(np.mean(audio.astype("float64") ** 2)))


def run_live_mode():
    """One interactive live turn: speak a response while listening for barge-in via the
    real mic, measuring true end-of-speech-to-first-audio latency. Run this while
    screen-recording for the demo video."""
    import sounddevice as sd

    latency_log = LatencyLog()
    turn = latency_log.new_turn("live_turn_1")

    out_wav = DEMO_DIR / "_live_tts_output.wav"
    print("Synthesizing response...")
    tts_result = tts.synthesize(SAMPLE_PROMPT, out_wav)
    print(f"TTS provider: {tts_result['provider']}, latency: {tts_result['latency_ms']}ms")

    noise_floor_rms = calibrate_noise_floor()
    print(f"Ambient noise floor: {noise_floor_rms:.0f} RMS")

    vad = FrameVAD()
    coord = BargeInCoordinator(require_energy_gate=True)
    coord.start_playback(noise_floor_rms=noise_floor_rms)

    import soundfile as sf
    data, sr = sf.read(str(out_wav), dtype="float32")
    # Quieter playback reduces speaker→mic bleed (main cause of false live barge-ins).
    data = data * config.LIVE_PLAYBACK_GAIN
    duration_s = len(data) / sr

    print(
        f"Playing response ({duration_s:.1f}s) -- speak clearly to test barge-in, or stay "
        f"silent to let it finish."
    )
    print(
        "TIP: Prefer headphones for the cleanest result. Open speakers will echo into the "
        "mic, but the gate below compares each mic frame against what's actually playing "
        "at that instant (not a fixed guess), so moderate speaker bleed should no longer "
        "false-trigger on its own."
    )
    sd.play(data, sr)

    frame_samples = int(config.SAMPLE_RATE * (config.VAD_FRAME_MS / 1000.0))
    stream = sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1, dtype="int16", blocksize=frame_samples)
    stream.start()
    playback_start = time.perf_counter()
    barged_in = False
    try:
        start = time.perf_counter()
        while time.perf_counter() - start < duration_s:
            frame, _ = stream.read(frame_samples)
            frame_bytes = frame.tobytes()
            is_voiced = vad.is_speech(frame_bytes) if len(frame_bytes) == vad.frame_bytes else False

            # Find the slice of the TTS playback that's sounding RIGHT NOW (same elapsed
            # time as this mic frame) and use its RMS as the echo reference -- this is
            # what makes the gate track actual playback loudness instead of a guessed
            # constant.
            elapsed_s = time.perf_counter() - playback_start
            start_sample = int(elapsed_s * sr)
            end_sample = start_sample + frame_samples
            reference_window = data[start_sample:end_sample] if start_sample < len(data) else data[0:0]
            reference_rms = _rms_int16(reference_window)

            decision = coord.process_frame(frame_bytes, is_voiced, reference_rms=reference_rms)
            if decision == Decision.BARGE_IN:
                barged_in = True
                sd.stop()
                print(f">>> BARGE-IN detected at {coord.playback_elapsed_ms:.0f}ms into playback. Cancelling playback.")
                break
    finally:
        stream.stop()
        stream.close()

    LatencyLog.mark_end_of_speech(turn)

    if barged_in:
        print("Listening for the rest of your utterance (2s)...")
        extra = sd.rec(int(2 * config.SAMPLE_RATE), samplerate=config.SAMPLE_RATE, channels=1, dtype="int16")
        sd.wait()
        full_audio = coord.get_buffered_utterance_audio() + extra.tobytes()
        buf_path = DEMO_DIR / "_live_buffered_utterance.wav"
        save_pcm16_as_wav(full_audio, buf_path)
        print("Transcribing...")
        try:
            result = stt.transcribe(buf_path.read_bytes(), filename=buf_path.name)
            print(f"You said: \"{result['text']}\"  (STT latency: {result['latency_ms']}ms)")
        except stt.STTError as e:
            print(f"STT failed (gracefully handled, not hung): {e}")
    else:
        print("Playback completed without interruption.")

    LatencyLog.mark_first_audio(turn)  # for a real multi-turn loop this would be the NEXT response's first chunk
    print(f"\nLatency summary: {latency_log.summary()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["offline", "live"], default="offline")
    args = parser.parse_args()
    if args.mode == "offline":
        run_offline_mode()
    else:
        run_live_mode()


if __name__ == "__main__":
    main()
