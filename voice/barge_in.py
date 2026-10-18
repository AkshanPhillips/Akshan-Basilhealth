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

Additionally, a frame only counts as voiced for barge-in if its PCM RMS is >=
config.BARGE_IN_MIN_RMS. That filters quiet laptop-speaker echo that webrtcvad still
labels as "speech" (the live-demo failure mode where STT transcribed the bot's own
"Let me..." from the speakers). This is a soft substitute for real AEC, not a full fix --
a headset or physically separated mic is still the right setup for demos.

No double-handling: when a barge-in triggers, the audio frames already captured from the
moment sustained speech *started* (not from the trigger moment) are kept and handed to STT
as the beginning of the user's utterance -- the user is never asked to repeat the
beginning of what they just said because the system was "still catching up."

Known limitation: no real acoustic echo cancellation. Loud open-speaker setups can still
false-trigger if echo RMS exceeds BARGE_IN_MIN_RMS. Documented in decisions_log / red-team.
"""
from __future__ import annotations

import audioop
import enum
from dataclasses import dataclass, field

from app import config

GRACE_PERIOD_MS = config.BARGE_IN_GRACE_PERIOD_MS
GAP_TOLERANCE_MS = 400         # a voiced->unvoiced gap shorter than this doesn't reset the run
MIN_SPEECH_MS = config.BARGE_IN_MIN_SPEECH_MS
MIN_RMS = config.BARGE_IN_MIN_RMS


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


def frame_rms(frame_bytes: bytes) -> int:
    """int16 PCM RMS energy for one VAD frame. Returns 0 for empty/odd-length buffers."""
    if len(frame_bytes) < 2:
        return 0
    # audioop requires even length for width=2
    even = frame_bytes if len(frame_bytes) % 2 == 0 else frame_bytes[:-1]
    if not even:
        return 0
    return audioop.rms(even, 2)


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

    def start_playback(self):
        self.state = PipelineState.PLAYING
        self.playback_elapsed_ms = 0.0
        self.voiced_run_ms = 0.0
        self.gap_ms = 0.0
        self.run_start_ms = None
        self.buffered_frames = []
        self.frame_log = []
        self.triggered = False

    def _effective_voiced(self, frame_bytes: bytes, is_voiced: bool) -> bool:
        if not is_voiced:
            return False
        if not self.require_energy_gate:
            return True
        return frame_rms(frame_bytes) >= MIN_RMS

    def process_frame(self, frame_bytes: bytes, is_voiced: bool) -> Decision:
        """Call once per VAD frame while PLAYING. Returns the decision for this frame."""
        if self.state != PipelineState.PLAYING:
            return Decision.CONTINUE_PLAYBACK  # no-op outside PLAYING

        self.playback_elapsed_ms += self.frame_ms
        in_grace = self.playback_elapsed_ms <= GRACE_PERIOD_MS
        voiced = self._effective_voiced(frame_bytes, is_voiced)

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
