"""Mumble ↔ OpenClaw voice bridge.

Main orchestrator that wires together:
  Mumble client → STT → OpenClaw → Sentence splitter → TTS → Mumble playback

Supports streaming TTS with barge-in.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import ssl

# Monkey-patch ssl.wrap_socket for pymumble on Python 3.12+
if not hasattr(ssl, "wrap_socket"):
    def wrap_socket(sock, keyfile=None, certfile=None,
                    server_side=False, cert_reqs=ssl.CERT_NONE,
                    ssl_version=ssl.PROTOCOL_TLS, ca_certs=None,
                    do_handshake_on_connect=True,
                    suppress_ragged_eofs=True,
                    ciphers=None):
        context = ssl.SSLContext(ssl_version)
        if certfile:
            context.load_cert_chain(certfile, keyfile)
        if ca_certs:
            context.load_verify_locations(ca_certs)
        if ciphers:
            context.set_ciphers(ciphers)
        context.verify_mode = cert_reqs
        return context.wrap_socket(
            sock, server_side=server_side,
            do_handshake_on_connect=do_handshake_on_connect,
            suppress_ragged_eofs=suppress_ragged_eofs
        )
    ssl.wrap_socket = wrap_socket

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from audio_utils import pcm_duration_ms, MUMBLE_SAMPLE_RATE
from barge_in import BargeInController, BridgeState
from mumble_client import MumbleClient, MumbleConfig
from openclaw_client import OpenClawClient
from sentence_splitter import SentenceSplitter
from stt import STTClient
from tts import TTSClient

logger = logging.getLogger(__name__)

# Minimum audio duration to bother transcribing (ms)
MIN_UTTERANCE_MS = 300


@dataclass
class Bridge:
    """Main voice bridge orchestrator.

    Connects to Mumble and OpenClaw, handles the full voice pipeline
    with streaming TTS and barge-in support.
    """

    config_path: str = "config.yaml"

    # Components (initialized in start())
    _mumble: MumbleClient = field(default=None, init=False, repr=False)
    _stt: STTClient = field(default=None, init=False, repr=False)
    _tts: TTSClient = field(default=None, init=False, repr=False)
    _openclaw: OpenClawClient = field(default=None, init=False, repr=False)
    _controller: BargeInController = field(default=None, init=False, repr=False)
    _loop: asyncio.AbstractEventLoop = field(default=None, init=False, repr=False)
    _config: dict = field(default_factory=dict, init=False, repr=False)

    def _load_config(self) -> dict:
        """Load and resolve config from YAML file."""
        path = Path(self.config_path)
        if not path.exists():
            logger.warning("Config file %s not found, using defaults", path)
            return {}

        with open(path) as f:
            config = yaml.safe_load(f) or {}

        # Resolve environment variable references like ${VAR_NAME}
        return self._resolve_env_vars(config)

    def _resolve_env_vars(self, obj):
        """Recursively resolve ${ENV_VAR} references in config values."""
        if isinstance(obj, str):
            if obj.startswith("${") and obj.endswith("}"):
                var = obj[2:-1]
                return os.environ.get(var, obj)
            return obj
        if isinstance(obj, dict):
            return {k: self._resolve_env_vars(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._resolve_env_vars(v) for v in obj]
        return obj

    def _init_components(self) -> None:
        """Initialize all bridge components from config."""
        cfg = self._config

        # Mumble client
        mcfg = cfg.get("mumble", {})
        self._mumble = MumbleClient(
            config=MumbleConfig(
                host=mcfg.get("host", "127.0.0.1"),
                port=mcfg.get("port", 64738),
                username=mcfg.get("username", "OpenClaw"),
                password=mcfg.get("password", ""),
                channel=mcfg.get("channel", "General"),
                certfile=mcfg.get("certfile"),
            ),
            on_ptt_start=self._handle_ptt_start,
            on_ptt_end=self._handle_ptt_end,
        )

        # STT
        stt_cfg = cfg.get("stt", {})
        provider = stt_cfg.get("provider", "elevenlabs")
        
        # Determine API key based on provider
        if provider == "openai":
            api_key = stt_cfg.get("openai_api_key", "")
        else:
            api_key = stt_cfg.get("elevenlabs_api_key", "")
            
        self._stt = STTClient(
            api_key=api_key,
            model=stt_cfg.get("model", "scribe_v1"),
            language_code=stt_cfg.get("language", "en"),
            provider=provider,
        )

        # TTS
        tts_cfg = cfg.get("tts", {})
        self._tts = TTSClient(
            provider=tts_cfg.get("provider", "openai"),
            api_key=tts_cfg.get("openai_api_key", "")
            if tts_cfg.get("provider", "openai") == "openai"
            else tts_cfg.get("elevenlabs_api_key", ""),
            voice=tts_cfg.get("voice", "nova"),
            speed=tts_cfg.get("speed", 1.0),
            elevenlabs_voice_id=tts_cfg.get("elevenlabs_voice_id", ""),
        )

        # OpenClaw
        oc_cfg = cfg.get("openclaw", {})
        self._openclaw = OpenClawClient(
            gateway_url=oc_cfg.get("gateway_url", "http://127.0.0.1:18789"),
            gateway_token=oc_cfg.get("gateway_token", ""),
            agent_id=oc_cfg.get("agent_id", "main"),
            session_user=oc_cfg.get("session_user", "mumble-room"),
            timeout_seconds=oc_cfg.get("timeout_seconds", 60),
        )

        # Barge-in controller
        self._controller = BargeInController(
            on_state_change=self._log_state_change,
        )

    def start(self) -> None:
        """Start the bridge: load config, connect to Mumble, run event loop."""
        self._config = self._load_config()

        # Configure logging
        log_level = self._config.get("bridge", {}).get("log_level", "INFO")
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )

        logger.info("Starting Mumble ↔ OpenClaw voice bridge")
        self._init_components()

        # Connect to Mumble
        self._mumble.connect()
        logger.info("Connected to Mumble. Listening for PTT...")

        # Run asyncio event loop
        self._loop = asyncio.new_event_loop()

        # Handle shutdown signals
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda s, f: self.stop())

        try:
            self._loop.run_forever()
        finally:
            self._loop.close()

    def stop(self) -> None:
        """Stop the bridge gracefully."""
        logger.info("Stopping bridge...")
        self._mumble.disconnect()
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ── Mumble callbacks (called from pymumble thread) ──

    def _handle_ptt_start(self, username: str) -> None:
        """Called when a user starts PTT (from pymumble thread)."""
        barged = self._controller.on_ptt_start(username)
        if barged:
            logger.info("Barge-in: %s interrupted playback", username)

    def _handle_ptt_end(self, username: str, pcm_data: bytes) -> None:
        """Called when a user releases PTT with their audio (from pymumble thread)."""
        duration = pcm_duration_ms(pcm_data, MUMBLE_SAMPLE_RATE)
        if duration < MIN_UTTERANCE_MS:
            logger.debug("Discarding short utterance from %s (%dms)", username, duration)
            self._controller.on_error()  # Reset to IDLE
            return

        self._controller.on_ptt_end(username)
        self._controller.prepare_new_response()

        # Schedule the async pipeline on the event loop
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._process_utterance(username, pcm_data), self._loop
            )

    # ── Async pipeline ──

    async def _process_utterance(self, username: str, pcm_data: bytes) -> None:
        """Full pipeline: STT → OpenClaw (streaming) → TTS (streaming) → Mumble playback."""
        cancel = self._controller.cancellation_event

        try:
            # Step 1: Speech-to-text
            text = await self._stt.transcribe(pcm_data, user=username)
            if not text:
                logger.debug("No transcription for %s, returning to IDLE", username)
                self._controller.on_error()
                return

            if cancel.is_set():
                logger.debug("Cancelled after STT")
                return

            # Step 2: Stream response from OpenClaw → sentence split → TTS → play
            splitter = SentenceSplitter()
            first_audio = True

            async for token in self._openclaw.send_streaming(
                text, speaker=username, cancellation_event=cancel
            ):
                if cancel.is_set():
                    logger.debug("Cancelled during OpenClaw stream")
                    return

                sentences = splitter.feed(token)
                for sentence in sentences:
                    await self._speak_sentence(sentence, cancel, first_audio)
                    first_audio = False
                    if cancel.is_set():
                        return

            # Flush remaining text
            for sentence in splitter.flush():
                if cancel.is_set():
                    return
                await self._speak_sentence(sentence, cancel, first_audio)
                first_audio = False

            # Playback complete
            self._controller.on_playback_complete()

        except Exception as e:
            logger.exception("Error in utterance pipeline for %s", username)
            # Try to inform the user if possible via TTS (if TTS is working)
            # but don't crash the bridge.
            self._controller.on_error()
        finally:
            # Ensure we don't leave the controller in a hung state
            if self._controller.state != BridgeState.IDLE:
                 self._controller.on_error()

    async def _speak_sentence(
        self,
        sentence: str,
        cancel: asyncio.Event,
        is_first: bool,
    ) -> None:
        """TTS a single sentence and play it into Mumble."""
        if cancel.is_set():
            return

        pcm = await self._tts.synthesize(sentence, cancellation_event=cancel)
        if not pcm or cancel.is_set():
            return

        if is_first:
            self._controller.on_response_start()

        # Send audio to Mumble
        self._mumble.send_audio(pcm)
        
        # Give the audio thread a tiny bit of breathing room between sentences
        # to prevent potential scheduling bottlenecks in pymumble
        await asyncio.sleep(0.05)

    # ── Logging ──

    @staticmethod
    def _log_state_change(old: BridgeState, new: BridgeState) -> None:
        logger.info("Bridge state: %s → %s", old.value, new.value)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    bridge = Bridge(config_path=config_path)
    bridge.start()


if __name__ == "__main__":
    main()
