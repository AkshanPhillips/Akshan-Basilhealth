"""
Central config. All values can be overridden via environment variables / .env.
Nothing here requires a GPU -- embedding model and FAISS index both run on CPU.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
KB_DIR = ROOT_DIR / "kb"
INDEX_DIR = ROOT_DIR / "data" / "faiss_index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# --- Groq ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# Pick a current Groq-hosted instruct model with function-calling support.
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b")
GROQ_ROUTER_MODEL = os.getenv("GROQ_ROUTER_MODEL", "openai/gpt-oss-20b")  # cheap/fast for classification
GROQ_WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")

# --- Embeddings ---
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = 384  # all-MiniLM-L6-v2 output size
CHUNK_SIZE_WORDS = 120
CHUNK_OVERLAP_WORDS = 20
TOP_K_RETRIEVAL = 3

# --- Router / cache ---
ROUTER_INTENTS = ["order_status", "product_question", "greeting", "escalate_to_human"]
CLASSIFIER_CACHE_PATH = ROOT_DIR / "data" / "classifier_cache.sqlite3"

# --- Voice ---
TTS_VOICE_EDGE = os.getenv("TTS_VOICE_EDGE", "en-IN-NeerjaNeural")
VAD_AGGRESSIVENESS = 2  # webrtcvad 0-3, higher = more aggressive filtering of non-speech
VAD_FRAME_MS = 30
BARGE_IN_MIN_SPEECH_MS = 450  # sustained voiced audio needed before we treat it as a real interruption
# Ignore voiced frames quieter than this RMS (int16). This is a FLOOR, not the real gate --
# see BARGE_IN_ECHO_MARGIN below. A static RMS floor alone isn't reliable: how loud echo
# is depends on speaker volume/mic sensitivity/room, which changes run to run, so a fixed
# number like 1200 can be cleared easily by moderate speaker bleed. Kept as a cheap first
# check (and as the fallback when no reference signal is available, e.g. in unit tests).
BARGE_IN_MIN_RMS = int(os.getenv("BARGE_IN_MIN_RMS", "1200"))
# The real echo gate: while TTS is playing, we know exactly what SHOULD be causing any
# mic echo -- the audio we're playing right now. Each mic frame's RMS is compared against
# the RMS of the TTS reference signal at that same point in playback, scaled by this
# margin. Echo is always *proportional* to what's playing (louder TTS -> louder echo);
# real user speech is not tied to the TTS waveform at all, so a frame only counts as real
# speech if its RMS clears (reference_rms * BARGE_IN_ECHO_MARGIN). This adapts automatically
# to playback volume instead of needing a hand-tuned constant per machine/room.
BARGE_IN_ECHO_MARGIN = float(os.getenv("BARGE_IN_ECHO_MARGIN", "2.2"))
# Ambient-room noise floor, measured fresh before each live playback (see
# run_voice_demo.py's calibrate_noise_floor()) -- covers HVAC hum, fan noise, etc. that
# isn't TTS echo but also isn't the user talking. Frames must also clear
# (noise_floor_rms * this margin).
BARGE_IN_NOISE_FLOOR_MARGIN = float(os.getenv("BARGE_IN_NOISE_FLOOR_MARGIN", "3.0"))
BARGE_IN_GRACE_PERIOD_MS = int(os.getenv("BARGE_IN_GRACE_PERIOD_MS", "600"))
# Live-mode playback gain (0-1). Quieter speakers = less mic bleed / false barge-in.
# Still worth keeping low -- it reduces absolute echo level, which helps the noise-floor
# check -- but the echo_margin gate above is what actually makes this reliable regardless
# of this value.
LIVE_PLAYBACK_GAIN = float(os.getenv("LIVE_PLAYBACK_GAIN", "0.4"))
SAMPLE_RATE = 16000
