# Mumble ↔ OpenClaw Bridge — Implementation Plan

## Goal

Build a standalone Python service that connects to a Mumble server as a bot client, enabling two-way voice communication between human users on Mumble and an OpenClaw AI assistant. Humans speak into Mumble; the bridge transcribes speech via OpenAI Whisper, sends text to OpenClaw, receives the response, synthesizes audio via OpenAI/ElevenLabs TTS, and plays it back into the Mumble channel.

## Architecture

```
┌─────────────┐       audio (PCM)        ┌──────────────────┐
│  Mumble      │ ◄──────────────────────► │  Bridge Service  │
│  Server      │                          │                  │
│  (murmur)    │                          │  ┌────────────┐  │
└─────────────┘                          │  │ MumbleClient│  │
                                          │  └─────┬──────┘  │
                                          │        │         │
                                          │  ┌─────▼──────┐  │
                                          │  │ AudioBuffer │  │
                                          │  │ + VAD       │  │
                                          │  └─────┬──────┘  │
                                          │        │         │
                                   ┌──────┼────────┼─────────┼──────┐
                                   │      │  ┌─────▼──────┐  │      │
                                   │ STT  │  │ Whisper API│  │ TTS  │
                                   │      │  └─────┬──────┘  │      │
                                   │      │        │         │      │
                                   │      │  ┌─────▼──────┐  │      │
                                   │      │  │ OpenClaw   │  │      │
                                   │      │  │ Webhook    │  │      │
                                   │      │  └─────┬──────┘  │      │
                                   │      │        │         │      │
                                   │      │  ┌─────▼──────┐  │      │
                                   │      │  │ TTS API    │  │      │
                                   │      │  └────────────┘  │      │
                                   └──────┼──────────────────┼──────┘
                                          └──────────────────┘
                                              Cloud APIs
```

## Components

### 1. Mumble Client (`mumble_client.py`)

**Library:** `pymumble3` (actively maintained fork of pymumble)

**Responsibilities:**
- Connect to Mumble server with bot credentials
- Join a configured channel
- Receive PCM audio callbacks per-user
- Transmit synthesized PCM audio back to the channel

**Key details:**
- pymumble delivers audio as 16-bit PCM, 48kHz, mono — per user
- Outbound audio must also be 16-bit PCM, 48kHz, mono
- The bot appears as a named user in Mumble (e.g., "OpenClaw")

### 2. Voice Activity Detection + Audio Buffering (`vad.py`) — Phase 2 (always-on follow-up)

**Not needed for Phase 1.** With PTT, Mumble only delivers audio while a user is transmitting, so utterance boundaries are defined by PTT press/release. VAD will be needed later for always-on mode.

**Library:** `webrtcvad` (Google's WebRTC VAD, lightweight, no GPU)

**Configuration (for always-on mode):**
- `vad_aggressiveness`: 1-3 (3 = most aggressive filtering, good for noisy environments)
- `silence_timeout_ms`: 800 (how long silence before utterance is considered complete)
- `min_utterance_ms`: 300 (discard very short sounds)

### 3. Speech-to-Text (`stt.py`)

**API:** OpenAI Whisper (`/v1/audio/transcriptions`)

**Responsibilities:**
- Accept a complete utterance (PCM audio bytes)
- Convert to WAV or FLAC format in-memory (Whisper API requires file upload)
- Call Whisper API, return transcription text
- Include speaker identification from Mumble username for context

**Key details:**
- Whisper API accepts up to 25MB files
- Typical utterance at 48kHz mono 16-bit: ~1-10 seconds = 96KB-960KB (well within limits)
- Downsample to 16kHz before sending (Whisper's native rate, reduces upload size 3x)
- Language hint can be configured

### 4. OpenClaw Integration (`openclaw_client.py`)

**Integration point:** OpenAI-compatible Chat Completions API (NOT webhooks)

**Why not webhooks:** OpenClaw's `/hooks/agent` endpoint is fire-and-forget — it returns HTTP 202 and delivers responses to messaging channels asynchronously. It does not stream the response back to the caller. This makes it unsuitable for our bridge.

**Why the Chat Completions API:** OpenClaw exposes an OpenAI-compatible endpoint at `POST /v1/chat/completions` on the Gateway port (default 18789). With `stream: true`, it returns SSE (Server-Sent Events) — exactly what we need for streaming TTS with barge-in.

**Prerequisites — OpenClaw configuration changes:**

```json
{
  "gateway": {
    "http": {
      "endpoints": {
        "chatCompletions": { "enabled": true }
      }
    }
  }
}
```

This endpoint is disabled by default. Enable it with:
```bash
openclaw config set gateway.http.endpoints.chatCompletions.enabled true
```

**Responsibilities:**
- Send transcribed text to the Chat Completions endpoint
- Consume SSE stream token-by-token
- Yield tokens to the sentence splitter for streaming TTS
- Support cancellation (close the HTTP connection on barge-in)
- Map Mumble usernames into the shared session via the `user` field

**Shared session via `user` field:**
The OpenAI-compatible API derives a stable session key from the `user` field. We use a fixed value (e.g., `"mumble-room"`) so all requests share one conversation context. Speaker attribution is embedded in the message content.

**Request format (streaming):**
```
POST http://<gateway-host>:18789/v1/chat/completions
Authorization: Bearer <gateway-token>
Content-Type: application/json

{
  "model": "openclaw",
  "stream": true,
  "user": "mumble-room",
  "messages": [
    {"role": "user", "content": "[john]: What's the weather like today?"}
  ]
}
```

**SSE response format:**
```
data: {"choices":[{"delta":{"content":"The"}}]}
data: {"choices":[{"delta":{"content":" weather"}}]}
data: {"choices":[{"delta":{"content":" today"}}]}
...
data: [DONE]
```

**Authentication:**
Uses Gateway auth. When `gateway.auth.mode="token"`, send `Authorization: Bearer <OPENCLAW_GATEWAY_TOKEN>`.

**Cancellation on barge-in:**
When the barge-in controller fires, the client closes the SSE connection. This is clean — the server sees a dropped connection and the agent run may continue in the background but we stop consuming tokens. The `httpx` async client supports this via task cancellation.

**Error handling:**
- Gateway returns standard HTTP errors (401 auth failure, 500 server error)
- SSE stream may disconnect mid-response (network issue) — retry or report
- Agent timeout — configurable server-side via OpenClaw config

### 5. Text-to-Speech (`tts.py`)

**API:** OpenAI TTS (`/v1/audio/speech`) or ElevenLabs

**Responsibilities:**
- Accept response text from OpenClaw
- Call TTS API, receive audio bytes
- Convert to 48kHz mono 16-bit PCM (Mumble's required format)
- Return PCM buffer for playback

**Configuration:**
- `tts_provider`: "openai" | "elevenlabs"
- `voice`: voice ID (e.g., "nova" for OpenAI, or an ElevenLabs voice ID)
- `speed`: playback speed multiplier

**Key details:**
- OpenAI TTS returns MP3/OPUS/AAC/FLAC — decode with `pydub` or `ffmpeg`
- ElevenLabs can return PCM directly, which is more efficient
- Resample to 48kHz for Mumble playback
- Must support cancellation: abort mid-stream when barge-in occurs
- No response length limits — the bot may read long content; streaming handles this naturally

### 6. Main Bridge Orchestrator (`bridge.py`)

**Responsibilities:**
- Initialize all components
- Wire the pipeline: audio in → VAD → STT → OpenClaw → TTS → audio out
- Handle concurrent speakers (queue per user, process sequentially or in parallel)
- Manage graceful shutdown
- Log all interactions

**Concurrency model:**
- Main thread: Mumble client event loop
- Per-utterance: async task (STT → OpenClaw → TTS stream → playback)
- Use `asyncio` with thread bridge for pymumble callbacks
- Playback queue with cancellation support for barge-in
- PTT state change triggers barge-in controller transitions

## Configuration (`config.yaml`)

```yaml
mumble:
  host: "192.168.1.100"       # Mumble server LAN IP
  port: 64738
  username: "OpenClaw"
  password: ""                 # Server password if set
  channel: "General"           # Channel to join
  certfile: null               # Optional client certificate

stt:
  provider: "openai"
  openai_api_key: "${OPENAI_API_KEY}"
  language: "en"               # Hint for Whisper
  model: "whisper-1"

openclaw:
  gateway_url: "http://127.0.0.1:18789"  # OpenClaw Gateway
  gateway_token: "${OPENCLAW_GATEWAY_TOKEN}"
  agent_id: "main"                         # Target agent
  session_user: "mumble-room"              # Shared session key
  timeout_seconds: 60

tts:
  provider: "openai"           # "openai" or "elevenlabs"
  openai_api_key: "${OPENAI_API_KEY}"
  voice: "nova"
  speed: 1.0
  # elevenlabs_api_key: "${ELEVENLABS_API_KEY}"
  # elevenlabs_voice_id: "pMsXgVXv3BLzUgSXRplE"

bridge:
  max_concurrent_utterances: 2
  log_level: "INFO"
```

### Required OpenClaw Configuration

Add to `~/.openclaw/openclaw.json` (or via `openclaw config set`):

```json
{
  "gateway": {
    "http": {
      "endpoints": {
        "chatCompletions": { "enabled": true }
      }
    },
    "auth": {
      "mode": "token",
      "token": "your-gateway-token"
    }
  }
}
```

Restart the Gateway after changing config: `openclaw gateway restart`

## File Structure

```
mumble-openclaw-bridge/
├── config.yaml
├── requirements.txt
├── bridge.py              # Main entry point + orchestrator
├── mumble_client.py       # Mumble connection + audio I/O
├── vad.py                 # Voice activity detection + buffering
├── stt.py                 # Whisper API client
├── tts.py                 # TTS API client (OpenAI / ElevenLabs)
├── openclaw_client.py     # Webhook client for OpenClaw
├── audio_utils.py         # PCM conversion, resampling helpers
├── tests/
│   ├── test_vad.py
│   ├── test_stt.py
│   ├── test_tts.py
│   ├── test_openclaw_client.py
│   ├── test_audio_utils.py
│   ├── test_bridge_integration.py
│   └── fixtures/
│       ├── speech_sample.wav
│       └── silence_sample.wav
└── README.md
```

## Dependencies (`requirements.txt`)

```
pymumble3>=1.0
webrtcvad>=2.0.10
openai>=1.0
httpx>=0.25
pydub>=0.25
numpy>=1.24
scipy>=1.11
PyYAML>=6.0
```

System dependencies: `ffmpeg` (for pydub audio conversion), `libopus-dev` (for pymumble)

## Implementation Order

| Phase | What | Why |
|-------|------|-----|
| 1 | `audio_utils.py` + `vad.py` | Foundation — can unit test with fixture WAVs, no network needed |
| 2 | `stt.py` | Can test standalone with a WAV file → text |
| 3 | `tts.py` | Can test standalone with text → WAV file, verify audio quality |
| 4 | `openclaw_client.py` | Can test with a mock HTTP server or against real OpenClaw |
| 5 | `mumble_client.py` | Connects to Mumble, verifies join/leave/audio callbacks |
| 6 | `bridge.py` | Wire everything together, end-to-end |

Each phase is independently testable before moving to the next.

## Testing Approach

### Unit Tests (no network, no Mumble server)

| Test | What it verifies |
|------|-----------------|
| `test_vad.py` | Given a WAV with speech+silence, VAD correctly segments into utterances. Given pure silence, no utterance emitted. Given speech shorter than `min_utterance_ms`, discarded. |
| `test_audio_utils.py` | PCM 48kHz→16kHz downsampling preserves audio. PCM→WAV encoding produces valid WAV headers. WAV/MP3→PCM 48kHz upsampling for Mumble playback. |
| `test_stt.py` | Mock the OpenAI API; verify correct request format (file upload, model, language). Verify error handling on API timeout/failure. |
| `test_tts.py` | Mock the TTS API; verify correct request params. Verify MP3→PCM conversion pipeline. Verify error handling. |
| `test_openclaw_client.py` | Mock HTTP; verify webhook payload format (message, sender, session_id). Verify bearer token is sent. Verify timeout handling. |

### Integration Tests (require running services)

| Test | What it verifies | Services needed |
|------|-----------------|-----------------|
| STT round-trip | Record a known phrase → Whisper API → verify transcription matches | OpenAI API key |
| TTS round-trip | Known text → TTS API → play back, verify audible and correct | OpenAI/ElevenLabs API key |
| OpenClaw round-trip | Send test message via webhook → verify response received | Running OpenClaw instance |
| Mumble join/leave | Bot connects, joins channel, other clients see it | Running Mumble server |
| End-to-end | Speak into Mumble → bot transcribes → OpenClaw responds → TTS plays in channel | All services |

### Manual Testing Checklist

- [ ] Bot appears in Mumble channel with correct name
- [ ] Speaking triggers transcription (check logs for STT output)
- [ ] OpenClaw receives message and responds (check OpenClaw logs)
- [ ] TTS audio plays back in Mumble channel clearly
- [ ] Multiple users speaking — bot handles them without crosstalk
- [ ] Long OpenClaw response — audio doesn't cut off
- [ ] Network interruption — bot reconnects to Mumble gracefully
- [ ] Silence — bot doesn't send empty transcriptions to OpenClaw

## Latency Budget (Target: < 3 seconds end-to-end)

| Step | Expected latency |
|------|-----------------|
| VAD silence timeout | ~800ms (configurable) |
| Audio upload to Whisper | ~100-200ms |
| Whisper transcription | ~500-1000ms |
| OpenClaw webhook round-trip | ~500-2000ms (depends on LLM) |
| TTS generation | ~300-500ms |
| Audio playback start | ~50ms |
| **Total** | **~2.2-4.5s** |

The LLM response time is the biggest variable. For short responses this should be well under 3 seconds; for longer responses, consider streaming TTS (generate audio as text tokens arrive).

## Design Decisions (Confirmed)

1. **Push-to-talk first.** Mumble's PTT gating means we only receive audio when a user is actively transmitting — no VAD needed for Phase 1. Always-on with VAD is a follow-up.

2. **Global shared session.** All Mumble users share one OpenClaw conversation context. Messages include the speaker's Mumble username for attribution, but `session_id` is a single fixed value (e.g., `"mumble-room"`).

3. **No response length limits.** The bot may be asked to read long content aloud. Streaming TTS handles this naturally — audio starts playing before the full response is generated.

4. **Streaming TTS with barge-in.** This is Phase 1, not deferred.

## Streaming + Barge-in Architecture

Barge-in means: when any user begins transmitting (PTT) while the bot is playing TTS audio, the bot immediately stops playback and begins capturing the new utterance. This requires careful concurrency design.

### Pipeline (streaming mode)

```
User speaks (PTT) ──► Mumble delivers PCM ──► Buffer until PTT release
                                                      │
                                                      ▼
                                              Encode WAV, send to Whisper
                                                      │
                                                      ▼
                                              POST text to OpenClaw webhook
                                                      │
                                                      ▼
                                              OpenClaw streams response tokens
                                                      │
                                                      ▼
                                              Sentence splitter accumulates tokens
                                                      │
                                              ┌───────┴───────┐
                                              ▼               ▼
                                         Sentence 1      Sentence 2 ...
                                              │               │
                                              ▼               ▼
                                         TTS API call    TTS API call
                                              │               │
                                              ▼               ▼
                                         PCM chunk       PCM chunk
                                              │               │
                                              ▼               ▼
                                    ┌─── Playback Queue (ordered) ───┐
                                    │  chunk1 → chunk2 → chunk3 ...  │
                                    └────────────┬───────────────────┘
                                                 │
                                                 ▼
                                          Mumble audio out
                                                 │
                                    ◄── BARGE-IN INTERRUPT ──►
                                    Any user PTT detected:
                                      1. Cancel pending TTS API calls
                                      2. Drain playback queue
                                      3. Stop current audio transmission
                                      4. Begin capturing new utterance
```

### Sentence Splitter (`sentence_splitter.py`)

Streaming from OpenClaw delivers tokens/chunks of text. We can't TTS individual tokens (too fragmented) or wait for the full response (defeats streaming). The sentence splitter accumulates text and emits complete sentences for TTS.

**Rules:**
- Emit on sentence-ending punctuation (`. ! ? :` followed by space or EOF)
- Emit on newline boundaries
- Flush remaining text when the OpenClaw stream closes
- Minimum emit length: ~20 characters (avoid TTS calls for tiny fragments like "OK.")
- Maximum buffer before forced emit: ~500 characters (safety valve for run-on text)

### Barge-in Controller (`barge_in.py`)

**State machine:**

```
         ┌──────────┐
         │  IDLE    │ ◄──────────────────────────────┐
         └────┬─────┘                                │
              │ user PTT starts                      │
              ▼                                      │
         ┌──────────┐                                │
         │ LISTENING│                                │
         └────┬─────┘                                │
              │ user PTT ends                        │
              ▼                                      │
         ┌──────────┐                                │
         │PROCESSING│  (STT → OpenClaw → TTS stream) │
         └────┬─────┘                                │
              │ first TTS chunk ready                │
              ▼                                      │
         ┌──────────┐                                │
         │ SPEAKING │ ──── user PTT starts ──────────┤
         └────┬─────┘     (BARGE-IN: cancel + LISTENING)
              │ playback queue empty                  │
              └──────────────────────────────────────┘
```

**Barge-in behavior:**
- When state is `SPEAKING` and any user's PTT activates:
  - Set a cancellation flag (Python `asyncio.Event` or `threading.Event`)
  - All pending TTS API calls check this flag and abort
  - Playback loop checks this flag between chunks and stops
  - Transition to `LISTENING` for the new utterance
- The cancellation must be fast — no finishing "just one more sentence"

### Concurrency Model (revised for streaming)

```
Main thread:       pymumble event loop (audio callbacks, PTT state)
                        │
Async event loop:  asyncio running in a dedicated thread
  │
  ├── Task: STT pipeline (per utterance)
  ├── Task: OpenClaw streaming reader
  ├── Task: Sentence splitter → TTS dispatcher
  ├── Task pool: TTS API calls (up to 2 concurrent, for pre-buffering)
  └── Task: Playback feeder (PCM chunks → pymumble)
                        │
Shared state:      cancellation_event (Event)
                   playback_queue (asyncio.Queue)
                   bridge_state (enum: IDLE/LISTENING/PROCESSING/SPEAKING)
```

### OpenClaw Chat Completions — Streaming via SSE (Confirmed)

The OpenAI-compatible endpoint at `/v1/chat/completions` with `stream: true` returns SSE. The bridge consumes this stream token-by-token, feeding tokens into the sentence splitter. This is confirmed supported — no spike needed.

**Cancellation on barge-in:** The bridge closes the HTTP SSE connection. The `httpx` async client supports this cleanly via `asyncio` task cancellation. Pending tokens after the connection close are discarded.

## Updated File Structure

```
mumble-openclaw-bridge/
├── config.yaml
├── requirements.txt
├── bridge.py                # Main entry point + orchestrator
├── mumble_client.py         # Mumble connection + audio I/O + PTT state
├── vad.py                   # Voice activity detection (Phase 2: always-on)
├── stt.py                   # Whisper API client
├── tts.py                   # TTS API client (OpenAI / ElevenLabs)
├── openclaw_client.py       # Webhook/WS client for OpenClaw (streaming)
├── audio_utils.py           # PCM conversion, resampling helpers
├── sentence_splitter.py     # Token stream → sentence chunks for TTS
├── barge_in.py              # Barge-in state machine + cancellation
├── tests/
│   ├── test_vad.py
│   ├── test_stt.py
│   ├── test_tts.py
│   ├── test_openclaw_client.py
│   ├── test_audio_utils.py
│   ├── test_sentence_splitter.py
│   ├── test_barge_in.py
│   ├── test_bridge_integration.py
│   └── fixtures/
│       ├── speech_sample.wav
│       └── silence_sample.wav
└── README.md
```

## Updated Implementation Order

| Phase | What | Why |
|-------|------|-----|
| 1 | `audio_utils.py` | Foundation — PCM conversion, resampling. Unit testable with fixture WAVs. |
| 2 | `sentence_splitter.py` | Pure logic, no I/O. Critical for streaming TTS. Easy to unit test. |
| 3 | `barge_in.py` | State machine, pure logic + events. Unit testable with simulated transitions. |
| 4 | `stt.py` | Whisper API client. Test with fixture WAV → text. |
| 5 | `tts.py` | TTS API client. Test with text → WAV. Must handle cancellation mid-request. |
| 6 | `openclaw_client.py` | SSE streaming client for `/v1/chat/completions`. Test against running Gateway. |
| 7 | `mumble_client.py` | Connects to Mumble, PTT state detection, audio send/receive. |
| 8 | `bridge.py` | Wire everything. End-to-end integration. |

Each phase is independently testable before moving to the next. No integration risk spikes needed — SSE streaming is confirmed supported by the OpenAI-compatible endpoint.

## Updated Testing Approach

### Unit Tests (no network)

| Test | What it verifies |
|------|-----------------|
| `test_audio_utils.py` | 48kHz↔16kHz resampling roundtrip. PCM↔WAV encoding. MP3→PCM decoding. |
| `test_sentence_splitter.py` | Token-by-token feed produces correct sentence boundaries. Handles `. ! ? :` and newlines. Respects min/max length. Flush on stream end. Edge cases: URLs, abbreviations (Dr., etc.), numbered lists. |
| `test_barge_in.py` | State transitions: IDLE→LISTENING→PROCESSING→SPEAKING→IDLE. Barge-in: SPEAKING + PTT → cancellation event fires, state → LISTENING. Rapid barge-in: multiple interrupts don't corrupt state. |
| `test_stt.py` | Mock OpenAI API; verify WAV upload format, model param, language hint. Error/timeout handling. |
| `test_tts.py` | Mock TTS API; verify request params. Verify cancellation: mid-request abort returns promptly. MP3→PCM conversion. |
| `test_openclaw_client.py` | Mock HTTP SSE stream; verify request format (model, stream:true, user field). Verify token-by-token consumption from SSE events. Verify cancellation stops reading stream and closes connection. Verify auth header sent. |

### Integration Tests

| Test | Services needed | What it verifies |
|------|----------------|-----------------|
| STT round-trip | OpenAI API | Known WAV → Whisper → expected text |
| TTS round-trip | OpenAI/ElevenLabs API | Known text → TTS → audible correct audio |
| TTS cancellation | OpenAI/ElevenLabs API | Start TTS, cancel mid-stream, verify no hung connections |
| OpenClaw streaming | Running OpenClaw Gateway | Send message via Chat Completions API, consume SSE token stream |
| Mumble connectivity | Running Mumble server | Bot joins, appears in channel, sends/receives audio |
| End-to-end | All | Speak → transcribe → OpenClaw → stream TTS → Mumble playback |
| Barge-in E2E | All | Bot speaking → user PTT → audio stops within 200ms → new utterance captured |

### Manual Testing Checklist

- [ ] Bot appears in Mumble channel as "OpenClaw"
- [ ] PTT speech triggers transcription (verify in logs)
- [ ] OpenClaw receives message with speaker attribution
- [ ] TTS audio begins playing before full response is generated (streaming)
- [ ] Barge-in: pressing PTT while bot is speaking stops playback immediately
- [ ] Barge-in: the interrupted response is discarded, new utterance is processed
- [ ] Long response (e.g., "read me this document"): streams smoothly without gaps
- [ ] Multiple users: speaker names correctly attributed in shared session
- [ ] Mumble disconnect/reconnect: bot recovers gracefully
- [ ] API failure (Whisper/TTS/OpenClaw down): bot logs error, doesn't crash

## Updated Latency Budget (Streaming)

| Step | Latency | Notes |
|------|---------|-------|
| PTT release → audio ready | ~50ms | Mumble buffering |
| WAV encode + upload to Whisper | ~100-200ms | |
| Whisper transcription | ~500-1000ms | |
| OpenClaw webhook → first token | ~300-800ms | LLM time-to-first-token |
| Sentence accumulation | ~200-500ms | Wait for sentence boundary |
| TTS first sentence | ~300-500ms | |
| **Time to first audio** | **~1.5-3.0s** | |

Subsequent sentences overlap: while sentence N plays, sentence N+1 is already being TTS'd. Perceived latency after the first sentence is near-zero.

## Next Steps

After plan approval:
1. Scaffold project structure and `requirements.txt`
2. Implement phases 1-3 (pure logic, fully unit testable)
3. Proceed through remaining phases, testing each before moving on
4. End-to-end integration test with all services running
