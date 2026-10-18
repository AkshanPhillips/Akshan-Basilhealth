"""
Interactive Voice Support Agent (Part B + Part A integration).
Framed as a live phone-support line for NovaVolt Electronics:

Pipeline:
  User mic input (VAD start/end-of-speech detection)
    -> Groq Whisper STT
    -> Orchestrator (Input Guardrail -> Router/Cache -> Tools/RAG -> Output Guardrail)
    -> Edge-TTS / pyttsx3 fallback
    -> Speaker playback with live Barge-In monitoring

Usage:
  python voice/voice_pipeline.py
  python main.py --voice
"""
from __future__ import annotations

import collections
import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import sounddevice as sd
import soundfile as sf

from app import config
from app.orchestrator import handle_message
from voice import stt, tts
from voice.audio_io import save_pcm16_as_wav
from voice.barge_in import BargeInCoordinator, Decision
from voice.latency import LatencyLog
from voice.vad import FrameVAD

DEMO_DIR = ROOT_DIR / "demo"
DEMO_DIR.mkdir(exist_ok=True)
TEMP_INPUT_WAV = DEMO_DIR / "_voice_pipeline_input.wav"
TEMP_OUTPUT_WAV = DEMO_DIR / "_voice_pipeline_output.wav"

GREETING_TEXT = (
    "Thank you for calling NovaVolt Support. I can help with order tracking, "
    "product questions, shipping, or returns. How can I help you today?"
)


def _rms_int16(samples: np.ndarray) -> float:
    if len(samples) == 0:
        return 0.0
    scaled = samples.astype("float64") * 32767.0
    return float(np.sqrt(np.mean(scaled * scaled)))


def calibrate_ambient_noise(seconds: float = 0.5) -> float:
    """Samples ambient room noise for calibration."""
    print("🎧 Calibrating microphone noise floor (0.5s, please remain quiet)...", end="", flush=True)
    audio = sd.rec(int(seconds * config.SAMPLE_RATE), samplerate=config.SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    rms = float(np.sqrt(np.mean(audio.astype("float64") ** 2)))
    print(f" done (noise floor: {rms:.0f} RMS).")
    return rms


def listen_for_utterance(
    stream: sd.InputStream,
    vad: FrameVAD,
    noise_floor_rms: float,
    initial_frames: list[bytes] | None = None,
    silence_timeout_ms: int = 900,
    max_duration_s: float = 20.0,
) -> tuple[bytes, float | None]:
    """
    Listens to the microphone stream until the user finishes an utterance.
    Uses VAD to detect speech start and silence/end-of-speech (EOU).
    Returns (pcm_bytes, end_of_speech_timestamp).
    """
    frame_samples = int(config.SAMPLE_RATE * (config.VAD_FRAME_MS / 1000.0))
    pre_speech_buffer: collections.deque[bytes] = collections.deque(maxlen=10)  # ~300ms pre-roll
    speech_frames: list[bytes] = []

    if initial_frames:
        # Barge-in occurred: user speech is already in progress!
        speech_started = True
        speech_frames.extend(initial_frames)
        last_voiced_time = time.perf_counter()
        consecutive_silence_ms = 0
        print("\n🎙️  [Interruption captured, continuing to listen...]")
    else:
        speech_started = False
        last_voiced_time = None
        consecutive_silence_ms = 0
        print("\n🎙️  Listening... (speak clearly)")

    start_wait = time.perf_counter()

    while time.perf_counter() - start_wait < max_duration_s:
        frame, _ = stream.read(frame_samples)
        frame_bytes = frame.tobytes()
        is_voiced = vad.is_speech(frame_bytes) if len(frame_bytes) == vad.frame_bytes else False

        # Additional basic energy floor check
        frame_amp = float(np.sqrt(np.mean(frame.astype("float64") ** 2)))
        energy_ok = frame_amp >= max(config.BARGE_IN_MIN_RMS * 0.7, noise_floor_rms * 1.5)
        voiced_speech = is_voiced and energy_ok

        if not speech_started:
            pre_speech_buffer.append(frame_bytes)
            if voiced_speech:
                speech_started = True
                speech_frames.extend(pre_speech_buffer)
                speech_frames.append(frame_bytes)
                last_voiced_time = time.perf_counter()
                print("    (Speech detected...)", end="", flush=True)
        else:
            speech_frames.append(frame_bytes)
            if voiced_speech:
                consecutive_silence_ms = 0
                last_voiced_time = time.perf_counter()
            else:
                consecutive_silence_ms += config.VAD_FRAME_MS
                if consecutive_silence_ms >= silence_timeout_ms:
                    # User stopped speaking: End-of-Utterance reached
                    print(" done.")
                    break

    end_of_speech_ts = last_voiced_time or time.perf_counter()
    return b"".join(speech_frames), end_of_speech_ts


def play_with_barge_in(
    wav_path: Path,
    stream: sd.InputStream,
    vad: FrameVAD,
    noise_floor_rms: float,
) -> tuple[bool, list[bytes]]:
    """
    Plays TTS audio while actively monitoring the microphone for barge-in interruptions.
    Returns:
      (interrupted: bool, buffered_speech_frames: list[bytes])
    """
    coord = BargeInCoordinator(require_energy_gate=True)
    coord.start_playback(noise_floor_rms=noise_floor_rms)

    data, sr = sf.read(str(wav_path), dtype="float32")
    data = data * config.LIVE_PLAYBACK_GAIN
    duration_s = len(data) / sr

    sd.play(data, sr)

    frame_samples = int(config.SAMPLE_RATE * (config.VAD_FRAME_MS / 1000.0))
    playback_start = time.perf_counter()
    interrupted = False
    buffered_frames: list[bytes] = []

    try:
        while time.perf_counter() - playback_start < duration_s:
            frame, _ = stream.read(frame_samples)
            frame_bytes = frame.tobytes()
            is_voiced = vad.is_speech(frame_bytes) if len(frame_bytes) == vad.frame_bytes else False

            elapsed_s = time.perf_counter() - playback_start
            start_sample = int(elapsed_s * sr)
            end_sample = start_sample + frame_samples
            reference_window = data[start_sample:end_sample] if start_sample < len(data) else data[0:0]
            reference_rms = _rms_int16(reference_window)

            decision = coord.process_frame(frame_bytes, is_voiced, reference_rms=reference_rms)
            if decision == Decision.BARGE_IN:
                interrupted = True
                sd.stop()
                print(f"\n⚡ [BARGE-IN DETECTED at {coord.playback_elapsed_ms:.0f}ms] Interrupting playback...")
                buffered_frames = coord.buffered_frames[:]
                break
    except Exception as e:
        sd.stop()
        print(f"Playback error: {e}")

    if not interrupted:
        sd.wait()
        coord.finish_playback_normally()

    return interrupted, buffered_frames


def run_voice_call():
    """
    Runs the full interactive voice support line turn-taking loop.
    """
    print("\n========================================================")
    print("📞  NovaVolt Interactive Voice Support Line (Connected)")
    print("========================================================")
    print("TIP: Wear headphones for best results to prevent speaker echo.")
    print("Type Ctrl+C or say 'goodbye' / 'hang up' at any time to exit.\n")

    vad = FrameVAD()
    noise_floor = calibrate_ambient_noise()
    latency_log = LatencyLog()

    frame_samples = int(config.SAMPLE_RATE * (config.VAD_FRAME_MS / 1000.0))
    stream = sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1, dtype="int16", blocksize=frame_samples)
    stream.start()

    conversation_history: list[dict] = []
    turn_idx = 1
    next_speech_buffer: list[bytes] = []

    try:
        # Step 1: Initial Greeting
        print(f"\n🤖 Assistant: \"{GREETING_TEXT}\"")
        tts_greeting = tts.synthesize(GREETING_TEXT, TEMP_OUTPUT_WAV)
        print(f"   [TTS: {tts_greeting['provider']} | {tts_greeting['latency_ms']:.1f}ms]")

        interrupted, barge_frames = play_with_barge_in(TEMP_OUTPUT_WAV, stream, vad, noise_floor)
        if interrupted:
            next_speech_buffer = barge_frames

        # Main conversational loop
        while True:
            turn_name = f"turn_{turn_idx}"
            turn_metric = latency_log.new_turn(turn_name)

            # Step 2: Listen for User Speech
            pcm_audio, end_of_speech_ts = listen_for_utterance(
                stream, vad, noise_floor, initial_frames=next_speech_buffer
            )
            next_speech_buffer = []

            if len(pcm_audio) < int(config.SAMPLE_RATE * 0.4 * 2):
                # Less than 0.4 seconds of audio captured
                print("   [No clear speech detected, listening again...]")
                continue

            LatencyLog.mark_end_of_speech(turn_metric)
            turn_metric["end_of_speech_ts"] = end_of_speech_ts

            # Step 3: Transcribe Speech (STT)
            save_pcm16_as_wav(pcm_audio, TEMP_INPUT_WAV)
            print("⏳ Transcribing audio via Groq Whisper...")
            try:
                stt_res = stt.transcribe(TEMP_INPUT_WAV.read_bytes(), filename=TEMP_INPUT_WAV.name)
                user_text = stt_res["text"].strip()
                print(f"👤 Caller: \"{user_text}\" (STT latency: {stt_res['latency_ms']:.1f}ms)")
            except stt.STTError as e:
                print(f"⚠️ STT issue: {e}")
                user_text = ""

            if not user_text:
                apology = "I'm sorry, I couldn't hear that clearly. Could you please repeat?"
                tts_res = tts.synthesize(apology, TEMP_OUTPUT_WAV)
                play_with_barge_in(TEMP_OUTPUT_WAV, stream, vad, noise_floor)
                continue

            # Check for exit words
            normalized = user_text.lower().rstrip(".,!?")
            if any(term in normalized for term in ("bye", "goodbye", "hang up", "exit call", "quit")):
                farewell = "Thank you for calling NovaVolt support. Have a great day!"
                print(f"\n🤖 Assistant: \"{farewell}\"")
                tts.synthesize(farewell, TEMP_OUTPUT_WAV)
                data, sr = sf.read(str(TEMP_OUTPUT_WAV), dtype="float32")
                sd.play(data * config.LIVE_PLAYBACK_GAIN, sr)
                sd.wait()
                print("\n📞 [CALL ENDED BY CALLER]")
                break

            # Step 4: Process via Part A Orchestrator (Guardrails -> Router/Cache -> Agents)
            t_orch_start = time.perf_counter()
            response = handle_message(user_text, conversation_context=conversation_history)
            orch_ms = (time.perf_counter() - t_orch_start) * 1000.0

            print(f"🤖 Assistant: \"{response.reply}\"")
            print(f"   [Intent: {response.intent} | Cache Hit: {response.cache_hit} | Guardrail: {response.blocked_at or 'PASS'}]")
            if response.tool_calls:
                print(f"   [Tool Calls: {[t['name'] for t in response.tool_calls]}]")

            conversation_history.append({"role": "user", "content": user_text})
            conversation_history.append({"role": "assistant", "content": response.reply})

            # Step 5: Synthesize Bot Response (TTS)
            tts_res = tts.synthesize(response.reply, TEMP_OUTPUT_WAV)

            # Step 6: Mark End-to-End Latency and Play Audio with Barge-in
            LatencyLog.mark_first_audio(turn_metric)
            e2e_ms = turn_metric.get("end_to_end_ms", 0.0)

            print(
                f"⏱️  Latency: E2E (End-of-Speech -> First Audio) = {e2e_ms:.1f}ms "
                f"[STT: {stt_res['latency_ms']:.1f}ms | Agent: {orch_ms:.1f}ms | TTS: {tts_res['latency_ms']:.1f}ms ({tts_res['provider']})]"
            )

            # Step 7: Play response with active barge-in
            interrupted, barge_frames = play_with_barge_in(TEMP_OUTPUT_WAV, stream, vad, noise_floor)
            if interrupted:
                next_speech_buffer = barge_frames

            turn_idx += 1

    except KeyboardInterrupt:
        print("\n\n📞 [CALL DISCONNECTED BY USER]")
    finally:
        stream.stop()
        stream.close()
        print(f"\nSession summary: {latency_log.summary()}")


if __name__ == "__main__":
    run_voice_call()
