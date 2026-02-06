# Mumble ↔ OpenClaw Voice Bridge

A standalone Python service that connects a Mumble voice channel to an OpenClaw
AI assistant, enabling push-to-talk voice conversations with streaming TTS
responses and barge-in support.

## Architecture

```
Mumble Users ──PTT──► [Mumble Server] ──audio──► [Bridge] ──WAV──► Whisper STT
                                                     │
                                                     ▼
                                              [OpenClaw Gateway]
                                              /v1/chat/completions
                                              (SSE streaming)
                                                     │
                                                     ▼
                                              [Sentence Splitter]
                                                     │
                                                     ▼
                                              [TTS API] ──PCM──► [Mumble Server]
                                                                       │
                                                              ◄── Audio plays ──►
```

## Features

- **Push-to-talk**: Leverages Mumble's native PTT — no custom VAD needed
- **Streaming TTS**: Audio starts playing before the full response is generated
- **Barge-in**: Press PTT while the bot is speaking to interrupt immediately
- **Shared session**: All Mumble users share one OpenClaw conversation context
- **Speaker attribution**: Each user's name is included in messages to OpenClaw

## Prerequisites

- Python 3.11+
- Mumble server (murmur/mumble-server) running on your LAN
- OpenClaw Gateway with Chat Completions endpoint enabled
- OpenAI API key (for Whisper STT and TTS)
- `ffmpeg` installed (for audio format conversion)
- `libopus-dev` (for pymumble)

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

System packages (Ubuntu/Debian):
```bash
sudo apt install ffmpeg libopus-dev
```

### 2. Enable OpenClaw Chat Completions API

```bash
openclaw config set gateway.http.endpoints.chatCompletions.enabled true
openclaw gateway restart
```

### 3. Configure

Copy `config.yaml` and edit:

```yaml
mumble:
  host: "192.168.1.100"  # Your Mumble server IP
  channel: "General"

openclaw:
  gateway_url: "http://127.0.0.1:18789"
  gateway_token: "your-token-here"

tts:
  voice: "nova"  # OpenAI voice
```

Set environment variables:
```bash
export OPENAI_API_KEY="sk-..."
export OPENCLAW_GATEWAY_TOKEN="your-token"
```

### 4. Run

```bash
python bridge.py config.yaml
```

## Usage

1. Connect to the Mumble server with any Mumble client
2. Join the configured channel
3. The "OpenClaw" bot will appear in the channel
4. Press PTT to speak — the bot transcribes and responds
5. Press PTT while the bot is speaking to interrupt (barge-in)

## Testing

```bash
# Unit tests (no network required)
python -m pytest tests/ -v

# Or with unittest
python -m unittest discover tests/ -v
```
