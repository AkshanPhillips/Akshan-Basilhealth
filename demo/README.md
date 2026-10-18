# Voice Demo — Recording Instructions

This is the part you need to actually run on your own machine with your own mic — I can't
record real audio from this sandboxed session. Here's exactly what to do.

## 1. Install voice deps locally

```
pip install -r requirements.txt
```

`webrtcvad` needs a C compiler to build on some systems — if it fails on Windows, install
"Microsoft C++ Build Tools" first, or use WSL.

## 2. Record the 4 scenario clips (offline/reproducible mode — recommended)

Record these with any tool (Voice Recorder on Windows, QuickTime/`afrecord` on Mac,
`arecord` on Linux, or even your phone exported as WAV) and save them into
`demo/recordings/` with these exact filenames. Keep each clip roughly 3–5 seconds,
16-bit mono if your recorder lets you choose, any sample rate is fine (the harness
resamples to 16kHz automatically).

| Filename | What to record |
|---|---|
| `silence.wav` | Just stay quiet — record ~3s of near-silence (room tone) |
| `noise_burst.wav` | A short *non-speech* sound — a cough, a desk tap, a chair creak, clapping once |
| `real_interruption.wav` | Clearly say: **"wait, wrong order"** partway through the clip |
| `adversarial_false_start.wav` | Say **"um—"**, pause briefly (under half a second), then say **"wait, that's the wrong order"** — a false start followed closely by the real interruption |

## 3. Run the offline harness

```
python voice/run_voice_demo.py --mode offline
```

This runs each clip through the real VAD + barge-in coordinator, checks the decision
against what's expected, simulates an STT provider failure and a TTS primary-provider
failure, and writes `demo/voice_demo_report.json` with real instrumented numbers
(frame counts, trigger timing, latency). If you've set `GROQ_API_KEY` in `.env`, it will
also transcribe the buffered audio from any clip that triggered a barge-in, using the
*actual audio captured from the moment speech started* — not a re-recording — to show the
no-double-handling behavior in `voice/barge_in.py` working end to end.

Commit `demo/recordings/*.wav` and `demo/voice_demo_report.json` as your voice demo
evidence (the assignment asks for recordings/logs/transcripts — the report.json plus the
raw clips covers both).

## 4. (Optional but recommended) Record a screen capture in live mode

```
python voice/run_voice_demo.py --mode live
```

This plays a real TTS response through your speakers and listens on your real mic in real
time — speak over it partway through to see the barge-in cancel playback live, or stay
quiet to let it finish. Screen-record this (OBS, QuickTime screen recording, Windows
Game Bar) for the "screen-recorded demo" option the brief mentions, and save it as
`demo/live_demo.mp4` (or similar) in this folder.

## 5. Add your own entries to decisions_log.md / red-team notes

Once you've actually run this, note in `decisions_log.md` anything that surprised you —
e.g. if your mic setup causes more false barge-in triggers than expected (this is the
known echo/self-trigger limitation documented in `voice/barge_in.py` — a headset mic
will behave much better than open speakers+mic), write that down honestly. That's exactly
the kind of "say so explicitly" the brief rewards.
