# Decisions Log

Timestamped as work actually happened. This is kept separate from design_notes.md on
purpose — design_notes.md is the polished "here's the architecture" writeup; this is the
messier "here's what I actually tried, in order, including dead ends" record the brief
asks for.

---

**2026-10-18 00:50 IST** — Picked the KB domain: an invented electronics brand ("NovaVolt
Electronics") rather than a real company. Reasoning: using a real brand would mean either
inventing fake policy numbers for a real company (confusing / arguably misleading) or
scraping real docs (more setup time than the assignment's 10-20-doc scope justifies).
Invented brand + 16 docs lets every number in the KB (10-day return window, 2200W plug
rating, etc.) be deliberately chosen to create interesting guardrail/tool-agent edge cases.

**2026-10-18 00:55 IST** — Embedding model: went with sentence-transformers
`all-MiniLM-L6-v2` over larger options (e.g. `all-mpnet-base-v2`) specifically because the
user will run this on a CPU-only desktop. MiniLM is ~5x smaller and noticeably faster to
embed/encode on CPU, and for a 16-doc KB the retrieval quality difference between MiniLM
and a larger model is not going to be the bottleneck — the FAQ passages are short and
mostly lexically distinct from each other.

**2026-10-18 01:05 IST** — Vector store: FAISS flat index over Chroma. Tried sketching out
a Chroma-based version first; abandoned it because Chroma wants to run as a client/server
or persist a full DB directory, which is overkill for ~60-80 chunks and adds a dependency
surface (and a thing that can fail to start) for no retrieval-quality benefit at this
scale. FAISS IndexFlatIP + a JSON metadata sidecar file is simpler to reason about and
easier for a reviewer to audit (it's literally "read the JSON file").

**2026-10-18 01:15 IST** — Router/guardrail model: used the smaller, faster
`llama-3.1-8b-instant` for classification and both guardrail checks, and reserved the
larger `llama-3.3-70b-versatile` for the RAG/tool agents that actually generate the
customer-facing answer. Classification and SAFE/UNSAFE/OK/VIOLATION checks are closed-set
decisions that don't need the larger model's generation quality, and keeping them on the
cheap/fast model matters for the "10k concurrent sessions" latency/cost story in the
design notes.

**2026-10-18 01:25 IST** — Guardrail design tension (explicitly called out in the brief):
spent time deciding how the output guardrail (no unconfirmed refund amount/date) should
interact with the tool-calling agent (which legitimately needs to state real numbers from
a tool call). First instinct was a pure keyword blocklist on the output ("never say a
currency symbol") — rejected immediately, since that would also block the tool agent's
legitimate, tool-confirmed answers. Settled on passing `tool_confirmed_facts` (raw tool
output, untouched by the LLM) alongside the draft reply into the output guardrail's prompt,
so the guardrail can tell "confirmed by a tool this turn" apart from "the model just said
it." Full writeup in design_notes.md.

**2026-10-18 01:35 IST** — Wrote tests/test_no_bypass.py as a *structural* check (AST/regex
scan of the source) rather than only an example-based test. Reasoning: a suite of example
inputs ("try 10 jailbreak phrases, assert they're all blocked") can never prove the
negative the brief is asking for ("no path exists") — it only shows the paths I thought to
test are covered. A structural test that fails if a second call site to rag_agent.run/
tool_agent.run is ever added is a stronger (and honestly, easier) guarantee.

**2026-10-18 01:45 IST** — Live voice demo was false-triggering barge-in at ~450ms and STT
returned `"Let me."` / `"Let me..."` — i.e. the bot transcribed its own TTS from the
speakers, not the user. Confirmed the known no-AEC limitation is the live failure mode,
not a VAD bug. Mitigations shipped (still not real AEC): (1) longer grace window
(600ms), (2) RMS energy gate so quiet speaker bleed doesn't count as barge-in speech,
(3) quieter live playback gain (0.4), (4) headset tip printed in live mode. Intentionally
did *not* try to implement WebRTC AEC in this take-home — scope/time, and offline WAV
scenarios remain the reproducible graded path.

**2026-10-18 01:46 IST** — `"hey"` was routing to escalate_to_human because the classifier
prompt treated anything that didn't clearly fit order/product as escalate (fail-closed),
and greetings have no order/product content. Fixed by adding a `greeting` intent with a
local regex pre-check *before* the classifier cache (so a stale cached escalate on "hey"
can't stick), plus a canned welcome reply in the orchestrator. Also tightened the LLM
prompt so greetings aren't escalate and vague product asks prefer product_question.

**2026-10-18 16:25 IST** — Integrated full conversational voice support line (`voice/voice_pipeline.py` & `main.py --voice`):
Previously, `voice/run_voice_demo.py` only ran single-turn barge-in tests against canned prompts.
Built the complete interactive phone-support pipeline as required by Part B of the brief:
(1) Bot answers with spoken greeting, (2) VAD speech listener with rolling pre-roll buffer detects
start of utterance and silence-based End-of-Utterance (EOU), (3) Groq Whisper transcribes speech,
(4) Part A multi-agent orchestrator handles the turn (guardrails, cached routing, tool/RAG agents),
(5) Edge-TTS (with pyttsx3 fallback) synthesizes speech, (6) Output plays with active real-time
barge-in monitoring, retaining any interruption speech frames for the next turn ("no double handling"),
(7) Instrumented E2E latency ($t_{first\_audio} - t_{eos}$) is measured and reported on each turn.

---
*(more entries will be added as LangGraph migration, red-team cases, and eval work proceed)*
