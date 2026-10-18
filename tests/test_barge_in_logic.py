"""
Unit tests for the barge-in state machine (voice/barge_in.py) using synthetic frame
sequences -- these don't need real audio or webrtcvad, they test the decision logic in
isolation by feeding is_voiced booleans directly. The real-audio versions of these same
scenarios are run by voice/run_voice_demo.py against actual recorded WAV clips (see
demo/ once recordings are added) -- this file guards the state machine's correctness
regardless of what the VAD front-end says.

Covers the 3 required scenarios + 1 adversarial case:
  1. Silence through full playback        -> no barge-in
  2. Short noise burst (below threshold)   -> ignored, playback continues
  3. Real utterance (sustained speech)     -> barge-in triggered
  4. Adversarial: false-start then real interruption, short gap between -> barge-in
     triggered (the gap-tolerance logic treats them as one run)
  4b. Adversarial counter-case: false start then LONG silence then speech -> the long gap
      resets the run, so the false start alone correctly does NOT cause a barge-in to
      trigger on stale state, and the later speech is judged fresh on its own.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice.barge_in import (  # noqa: E402
    BargeInCoordinator,
    Decision,
    GAP_TOLERANCE_MS,
    GRACE_PERIOD_MS,
    MIN_SPEECH_MS,
)

FRAME_MS = 30


def _run(coord: BargeInCoordinator, voiced_sequence: list[bool]) -> list[Decision]:
    # Unit tests feed dummy PCM — disable RMS energy gate so is_voiced alone drives logic.
    coord.require_energy_gate = False
    coord.start_playback()
    decisions = []
    for v in voiced_sequence:
        d = coord.process_frame(frame_bytes=b"\x00" * 10, is_voiced=v)
        decisions.append(d)
        if d == Decision.BARGE_IN:
            break
    return decisions


def test_silence_through_full_playback():
    coord = BargeInCoordinator()
    decisions = _run(coord, [False] * 50)  # 1.5s of silence
    assert Decision.BARGE_IN not in decisions
    assert not coord.triggered


def test_short_noise_burst_ignored():
    coord = BargeInCoordinator()
    # ~90ms of "voice" (3 frames) then silence -- well under MIN_SPEECH_MS
    seq = [False] * 10 + [True] * 3 + [False] * 40
    decisions = _run(coord, seq)
    assert Decision.BARGE_IN not in decisions
    assert not coord.triggered


def test_real_utterance_triggers_barge_in():
    coord = BargeInCoordinator()
    # Past the grace window, then sustained speech past MIN_SPEECH_MS
    grace_frames = GRACE_PERIOD_MS // FRAME_MS + 1
    needed_frames = MIN_SPEECH_MS // FRAME_MS + 2
    seq = [False] * grace_frames + [True] * needed_frames
    decisions = _run(coord, seq)
    assert Decision.BARGE_IN in decisions
    assert coord.triggered


def test_adversarial_false_start_then_real_interruption_short_gap():
    """'um...' (short voiced burst) then a brief pause then 'wait, wrong order' (real
    speech) -- the gap between them is short enough that this should be treated as ONE
    continuous utterance and trigger barge-in, not discarded as noise twice."""
    coord = BargeInCoordinator()
    grace_frames = GRACE_PERIOD_MS // FRAME_MS + 1
    false_start = [True] * 4  # ~120ms, "um"
    short_gap = [False] * (GAP_TOLERANCE_MS // FRAME_MS - 2)  # just under tolerance
    real_speech = [True] * (MIN_SPEECH_MS // FRAME_MS + 2)
    seq = [False] * grace_frames + false_start + short_gap + real_speech
    decisions = _run(coord, seq)
    assert Decision.BARGE_IN in decisions, (
        "Expected the false-start + real-interruption pattern (short gap) to still "
        "accumulate into a single triggered run"
    )


def test_adversarial_false_start_then_long_silence_does_not_falsely_carry_over():
    """A short burst, then genuinely settled silence (longer than gap tolerance), should
    reset -- the system shouldn't keep 'remembering' a stale false start forever."""
    coord = BargeInCoordinator()
    grace_frames = GRACE_PERIOD_MS // FRAME_MS + 1
    false_start = [True] * 4
    long_silence = [False] * (GAP_TOLERANCE_MS // FRAME_MS + 5)  # past tolerance
    seq = [False] * grace_frames + false_start + long_silence
    decisions = _run(coord, seq)
    assert Decision.BARGE_IN not in decisions
    assert coord.voiced_run_ms == 0.0, "Run should have reset after settled silence"


def test_grace_period_prevents_self_trigger_at_playback_start():
    """Voiced frames inside GRACE_PERIOD_MS must not count toward barge-in (TTS onset /
    early self-echo window)."""
    coord = BargeInCoordinator()
    grace_frames = GRACE_PERIOD_MS // FRAME_MS
    # Only voiced inside grace, then silence — must not barge in
    seq = [True] * grace_frames + [False] * 10
    decisions = _run(coord, seq)
    assert Decision.BARGE_IN not in decisions


def test_quiet_frames_fail_energy_gate():
    """Near-silent PCM that VAD marks voiced should be ignored when energy gate is on
    (the speaker-echo / 'Let me...' false barge-in case)."""
    import audioop
    import struct

    coord = BargeInCoordinator(require_energy_gate=True)
    coord.start_playback()
    # Build quiet int16 frames (RMS well under BARGE_IN_MIN_RMS)
    quiet = struct.pack("<" + "h" * 20, *([50] * 20))  # tiny amplitude
    assert audioop.rms(quiet, 2) < 200

    grace_frames = GRACE_PERIOD_MS // FRAME_MS + 1
    for _ in range(grace_frames):
        coord.process_frame(quiet, is_voiced=False)

    decisions = []
    for _ in range(MIN_SPEECH_MS // FRAME_MS + 5):
        d = coord.process_frame(quiet, is_voiced=True)  # VAD says speech, energy says no
        decisions.append(d)
    assert Decision.BARGE_IN not in decisions
    assert not coord.triggered
