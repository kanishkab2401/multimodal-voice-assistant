# Real-Time Voice Assistant — Phase 1

A streaming voice assistant pipeline: browser mic → ASR (Groq Whisper) →
LLM (Groq Llama) → TTS (ElevenLabs) → browser playback, all over one
WebSocket connection.

## How it works

```
Browser (mic)                Backend (FastAPI)
  │  raw PCM16 audio            │
  ├─────────────────────────────▶  VAD segments into utterances
  │                              │
  │                              ├─▶ ASR (Groq Whisper)      transcript.final
  │                              ├─▶ LLM (Groq, streaming)   llm.token (per token)
  │                              ├─▶ TTS (ElevenLabs, per    tts.sentence_start/end
  │                              │   sentence, streamed)
  │  JSON events + MP3 bytes    │
  ◀─────────────────────────────┤
  plays audio, renders log      │
```

Every event on the wire follows one shape: `{"type": "...", "ts": ..., ...}`.
That consistency is what makes Phase 2 (latency dashboard) straightforward —
every timing signal already flows through `send_event()` in `main.py`.

## Setup

1. **Get free API keys**
   - Groq (ASR + LLM): https://console.groq.com/keys — free tier, very fast inference
   - ElevenLabs (TTS): https://elevenlabs.io — free tier, ~10k characters/month

2. **Backend**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env      # then fill in your keys
   uvicorn main:app --reload --port 8000
   ```

3. **Frontend**
   Just open `frontend/index.html` directly in a browser (Chrome/Edge
   recommended for mic permissions). It connects to `ws://localhost:8000/ws`.

4. **Try it**
   Click the mic button, say something, then pause for ~0.7s. You'll see
   the transcript appear, the assistant's response stream in token by
   token, and hear the reply spoken back.

## What's implemented (Phase 1)

- End-to-end streaming: audio in → transcript → LLM tokens → TTS audio out
- Sentence-level TTS streaming (doesn't wait for the full LLM response —
  starts speaking sentence 1 while sentence 2 is still generating)
- Simple energy-based VAD to detect when you've stopped talking
- A consistent event schema (`asr.*`, `llm.*`, `tts.*`, `turn.complete`, `error`)
- A debug console UI showing the live transcript and event log

## What's NOT implemented yet (by design)

This is deliberately Phase 1 — plumbing over optimization:

- **No true incremental ASR** — Groq's Whisper API transcribes complete
  utterances, not word-by-word partials. Swap `services/asr.py` for
  Deepgram's streaming WebSocket API to get that; nothing else changes.
- **No latency dashboard yet** — timestamps exist in every event
  (`ts` field, plus `latency_ms` on key events), but there's no
  visualization yet. That's Phase 2.
- **No resilience** — if Groq or ElevenLabs times out or errors, you'll
  see an `error` event in the log and the turn just stops. No retries,
  no fallback TTS, no timeout enforcement. That's Phase 3.
- **No replay mode** — nothing is persisted yet, so you can't feed a
  recorded session back through the pipeline. Also Phase 3.

## Next steps (Phase 2 preview)

Every event already carries a `ts` timestamp and several carry
`latency_ms`. The next step is to:
1. Persist per-turn timing (`asr.started` → `transcript.final` →
   `llm.first_token` → `tts.first_byte` → `turn.complete`) to a log file
   or SQLite table, keyed by a `turn_id`.
2. Build a small dashboard (a stacked bar chart per turn: ASR time / LLM
   TTFT / TTS TTFB / overhead) so you can point at one number and say
   exactly where it went.

Once you've got Phase 1 running end-to-end, let's move on to that.
