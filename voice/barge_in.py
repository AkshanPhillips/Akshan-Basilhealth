"""
Barge-in coordinator: the state machine that decides, frame by frame while the assistant
is "speaking" (TTS playback), whether incoming mic audio is real user speech that should
interrupt playback, or noise that should be ignored.

States:
    IDLE      -- nothing playing, nothing listening (between turns)
    PLAYING   -- TTS audio is being played out; mic is monitored for barge-in
    LISTENING -- playback has stopped (normally or via barge-in); capturing the user's turn

Core decision rule while PLAYING: a SINGLE voiced VAD frame is NOT enough to interrupt
(webrtcvad frame-level voiced/unvoiced is noisy -- see vad.py). We require SUSTAINED voiced
audio: config.BARGE_IN_MIN_SPEECH_MS of cumulative voiced time, tolerating short gaps so a
natural speech cadence ("wait... wrong order," with a tiny pause) isn't treated as two
separate sub-threshold bursts that each get discarded.

Additionally, a frame only counts as voiced for barge-in if its PCM RMS clears a gate.
That gate is NOT just the static config.BARGE_IN_MIN_RMS floor -- a single fixed RMS
number can't reliably tell real speech from speaker echo, because how loud the echo is
depends on speaker volume / mic sensitivity / room, which varies run to run (confirmed in
testing: a static 1200 floor was cleared by ordinary pyttsx3/`say` playback bleeding
through open speakers, causing the system to "hear" and transcribe its own TTS output).

The gate is instead REFERENCE-AWARE when a reference_rms is supplied per frame (see
run_voice_demo.py's run_live_mode, which computes the RMS of the TTS audio actually
playing at that exact moment and passes it in): a frame only counts as voiced if its RMS
clears max(BARGE_IN_MIN_RMS, noise_floor_rms * BARGE_IN_NOISE_FLOOR_MARGIN,
reference_rms * BARGE_IN_ECHO_MARGIN). Echo is always proportional to what's currently
playing; real user speech is not tied to the TTS waveform at all, so this adapts
automatically to playback volume instead of needing a hand-tuned constant. When no
reference_rms is supplied (reference_rms=0.0, e.g. offline scenario replay or unit tests
feeding synthetic bytes), the gate falls back to the static BARGE_IN_MIN_RMS floor only,
unchanged from before. This is still a soft substitute for real AEC, not a full fix -- a
headset or physically separated mic remains the most reliable setup for live demos.

No double-handling: when a barge-in triggers, the audio frames already captured from the
moment sustained speech *started* (not from the trigger moment) are kept and handed to STT
as the beginning of the user's utterance -- the user is never asked to repeat the
beginning of what they just said because the system was "still catching up."

Known limitation: this is still not real AEC (acoustic echo cancellation) -- it's a
proportional energy heuristic, not a signal-level echo canceller. It will not help if the
user speaks AT THE SAME TIME as echo is present and both happen to be at a similar
relative loudness to the reference (true double-talk), and it assumes the reference
signal (the exact samples being played) is available to compare against, which is only
true in-process the way this project's live-mode harness does it -- a hardware audio path
with its own buffering/delay would need the reference time-aligned to the mic signal
first. Documented in decisions_log / red-team.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np

from app import config

GRACE_PERIOD_MS = config.BARGE_IN_GRACE_PERIOD_MS
GAP_TOLERANCE_MS = 400         # a voiced->unvoiced gap shorter than this doesn't reset the run
MIN_SPEECH_MS = config.BARGE_IN_MIN_SPEECH_MS
MIN_RMS = config.BARGE_IN_MIN_RMS
ECHO_MARGIN = config.BARGE_IN_ECHO_MARGIN
NOISE_FLOOR_MARGIN = config.BARGE_IN_NOISE_FLOOR_MARGIN


class Decision(enum.Enum):
    CONTINUE_PLAYBACK = "continue_playback"
    NOISE_IGNORED = "noise_ignored"
    BARGE_IN = "barge_in"


class PipelineState(enum.Enum):
    IDLE = "idle"
    PLAYING = "playing"
    LISTENING = "listening"


@dataclass
class FrameLog:
    t_ms: float
    is_voiced: bool
    decision: Decision
    voiced_run_ms: float


def frame_rms(frame_bytes: bytes) -> float:
    """int16 PCM RMS energy for one VAD frame. Returns 0 for empty/odd-length buffers.

    Implemented with numpy rather than the stdlib `audioop` module: audioop was removed
    entirely in Python 3.13 (PEP 594), so importing it unconditionally would break this
    module on newer interpreters even though it worked fine wherever it was last tested on
    an older Python. numpy is already a hard dependency of this project (retriever.py,
    audio_io.py), so this has no new cost and removes an interpreter-version landmine."""
    if len(frame_bytes) < 2:
        return 0.0
    # Need an even number of bytes for 16-bit samples.
    even = frame_bytes if len(frame_bytes) % 2 == 0 else frame_bytes[:-1]
    if not even:
        return 0.0
    samples = np.frombuffer(even, dtype="<i2").astype("float64")
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples * samples)))


@dataclass
class BargeInCoordinator:
    frame_ms: int = config.VAD_FRAME_MS
    state: PipelineState = PipelineState.IDLE
    playback_elapsed_ms: float = 0.0
    voiced_run_ms: float = 0.0
    gap_ms: float = 0.0
    run_start_ms: float | None = None  # when the current voiced run began, for buffering
    buffered_frames: list[bytes] = field(default_factory=list)
    frame_log: list[FrameLog] = field(default_factory=list)
    triggered: bool = False
    # When False, RMS gate is skipped (unit tests feed dummy frame bytes).
    require_energy_gate: bool = True
    # Ambient noise floor measured fresh before this playback started (0.0 if not
    # calibrated, e.g. offline scenario replay) -- see run_voice_demo.py calibrate_noise_floor().
    noise_floor_rms: float = 0.0

    def start_playback(self, noise_floor_rms: float = 0.0):
        self.state = PipelineState.PLAYING
        self.playback_elapsed_ms = 0.0
        self.voiced_run_ms = 0.0
        self.gap_ms = 0.0
        self.run_start_ms = None
        self.buffered_frames = []
        self.frame_log = []
        self.triggered = False
        self.noise_floor_rms = noise_floor_rms

    def _effective_voiced(self, frame_bytes: bytes, is_voiced: bool, reference_rms: float = 0.0) -> bool:
        if not is_voiced:
            return False
        if not self.require_energy_gate:
            return True
        threshold = max(
            MIN_RMS,
            self.noise_floor_rms * NOISE_FLOOR_MARGIN,
            reference_rms * ECHO_MARGIN,
        )
        return frame_rms(frame_bytes) >= threshold

    def process_frame(self, frame_bytes: bytes, is_voiced: bool, reference_rms: float = 0.0) -> Decision:
        """Call once per VAD frame while PLAYING. Returns the decision for this frame.
        reference_rms (optional): RMS of the TTS audio actually playing at this instant --
        when supplied, the energy gate scales with it (see ECHO_MARGIN docs above) instead
        of relying solely on the static MIN_RMS floor. Omit it (default 0.0) for offline
        replay / synthetic test frames, where there's no simultaneous playback to reference."""
        if self.state != PipelineState.PLAYING:
            return Decision.CONTINUE_PLAYBACK  # no-op outside PLAYING

        self.playback_elapsed_ms += self.frame_ms
        in_grace = self.playback_elapsed_ms <= GRACE_PERIOD_MS
        voiced = self._effective_voiced(frame_bytes, is_voiced, reference_rms)

        if voiced and not in_grace:
            if self.run_start_ms is None:
                self.run_start_ms = self.playback_elapsed_ms
            self.voiced_run_ms += self.frame_ms
            self.gap_ms = 0.0
            self.buffered_frames.append(frame_bytes)
        elif self.run_start_ms is not None:
            # unvoiced frame, but we're mid-run -- check gap tolerance before resetting
            self.gap_ms += self.frame_ms
            self.buffered_frames.append(frame_bytes)  # keep buffering through small gaps
            if self.gap_ms > GAP_TOLERANCE_MS:
                # settled silence -- the prior burst was noise, not the start of real speech
                self.voiced_run_ms = 0.0
                self.run_start_ms = None
                self.gap_ms = 0.0
                self.buffered_frames = []
                decision = Decision.NOISE_IGNORED
                self.frame_log.append(FrameLog(self.playback_elapsed_ms, voiced, decision, self.voiced_run_ms))
                return decision
        else:
            decision = Decision.CONTINUE_PLAYBACK
            self.frame_log.append(FrameLog(self.playback_elapsed_ms, voiced, decision, self.voiced_run_ms))
            return decision

        if self.voiced_run_ms >= MIN_SPEECH_MS:
            self.triggered = True
            self.state = PipelineState.LISTENING
            decision = Decision.BARGE_IN
        else:
            decision = Decision.CONTINUE_PLAYBACK

        self.frame_log.append(FrameLog(self.playback_elapsed_ms, voiced, decision, self.voiced_run_ms))
        return decision

    def finish_playback_normally(self):
        """Call when TTS playback completes without a barge-in."""
        if self.state == PipelineState.PLAYING:
            self.state = PipelineState.LISTENING

    def get_buffered_utterance_audio(self) -> bytes:
        """Audio captured from the moment sustained speech began -- handed to STT as the
        start of the user's utterance so they don't have to repeat themselves."""
        return b"".join(self.buffered_frames)

    def report(self) -> dict:
        return {
            "triggered": self.triggered,
            "trigger_at_playback_ms": self.run_start_ms if self.triggered else None,
            "total_frames": len(self.frame_log),
            "voiced_frames": sum(1 for f in self.frame_log if f.is_voiced),
            "final_voiced_run_ms": self.voiced_run_ms,
            "decisions": [f.decision.value for f in self.frame_log],
        }
