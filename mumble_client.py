"""Mumble client wrapper for voice bridge.

Connects to a Mumble server, joins a channel, tracks PTT state per user,
buffers incoming audio per-user, and provides audio transmission.
"""

from __future__ import annotations

import logging
import threading
import time
import ssl

# Monkey-patch ssl.wrap_socket for Python 3.12+ compatibility (pymumble fix)
if not hasattr(ssl, 'wrap_socket'):
    def wrap_socket(sock, keyfile=None, certfile=None, server_side=False, cert_reqs=ssl.CERT_NONE, ssl_version=ssl.PROTOCOL_TLS, ca_certs=None, do_handshake_on_connect=True, suppress_ragged_eofs=True, ciphers=None):
        context = ssl.SSLContext(ssl_version)
        if certfile:
            context.load_cert_chain(certfile, keyfile)
        if ca_certs:
            context.load_verify_locations(ca_certs)
        if cert_reqs:
            context.verify_mode = cert_reqs
        if ciphers:
            context.set_ciphers(ciphers)
        return context.wrap_socket(sock, server_side=server_side, do_handshake_on_connect=do_handshake_on_connect, suppress_ragged_eofs=suppress_ragged_eofs)
    ssl.wrap_socket = wrap_socket

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)

# Audio chunk size for Mumble playback (20ms at 48kHz mono 16-bit)
MUMBLE_FRAME_SIZE = 48000 * 2 * 20 // 1000  # 1920 bytes = 960 samples

# Opus profile: "voip" for lower latency speech, "audio" for higher quality general audio
OPUS_PROFILE = "voip"


@dataclass
class MumbleConfig:
    """Mumble connection configuration."""

    host: str = "127.0.0.1"
    port: int = 64738
    username: str = "OpenClaw"
    password: str = ""
    channel: str = "General"
    certfile: str | None = None


@dataclass
class MumbleClient:
    """Wrapper around pymumble for the voice bridge.

    Tracks PTT state by detecting when users start/stop transmitting audio.
    Buffers audio per-user for the duration of their PTT transmission.

    Attributes:
        config: Connection configuration.
        on_ptt_start: Callback(username) when a user begins transmitting.
        on_ptt_end: Callback(username, pcm_bytes) when a user stops transmitting.
    """

    config: MumbleConfig = field(default_factory=MumbleConfig)
    on_ptt_start: Callable[[str], None] | None = None
    on_ptt_end: Callable[[str, bytes], None] | None = None

    # Internal state
    _mumble: object = field(default=None, init=False, repr=False)
    _user_buffers: dict[str, bytearray] = field(
        default_factory=lambda: defaultdict(bytearray), init=False, repr=False
    )
    _user_last_audio_time: dict[str, float] = field(
        default_factory=dict, init=False, repr=False
    )
    _user_is_talking: dict[str, bool] = field(
        default_factory=lambda: defaultdict(bool), init=False, repr=False
    )
    _connected: bool = field(default=False, init=False)
    _silence_checker: threading.Thread | None = field(default=None, init=False, repr=False)
    _running: bool = field(default=False, init=False)

    # How long after last audio frame to consider PTT released (ms)
    _silence_timeout_ms: int = 200

    def connect(self) -> None:
        """Connect to the Mumble server and join the configured channel."""
        try:
            import pymumble_py3 as pymumble
        except ImportError:
            import pymumble3 as pymumble

        # Apply timing and bandwidth fixes
        try:
            from mumble_timing_fix import apply_all_patches
            apply_all_patches()
        except ImportError:
            logger.warning("mumble_timing_fix.py not found, applying basic fixes")
            # Fallback to basic fixes if the fix file is missing
            def patched_set_bandwidth(self_so):
                pass
            pymumble.soundoutput.SoundOutput._set_bandwidth = patched_set_bandwidth
        
        # Override the default Opus profile to "voip" for lower latency
        # Default is "audio" which has higher latency but better music quality
        pymumble.constants.PYMUMBLE_AUDIO_TYPE_OPUS_PROFILE = OPUS_PROFILE
        logger.info("Using Opus profile: %s", OPUS_PROFILE)

        logger.info(
            "Connecting to Mumble at %s:%d as %s",
            self.config.host,
            self.config.port,
            self.config.username,
        )

        self._mumble = pymumble.Mumble(
            self.config.host,
            self.config.username,
            port=self.config.port,
            password=self.config.password,
            certfile=self.config.certfile,
            reconnect=True,
        )

        self._mumble.set_receive_sound(True)
        self._mumble.callbacks.set_callback(
            pymumble.constants.PYMUMBLE_CLBK_SOUNDRECEIVED,
            self._on_sound_received,
        )

        self._mumble.start()
        self._mumble.is_ready()

        # Join channel
        try:
            channel = self._mumble.channels.find_by_name(self.config.channel)
            if channel:
                channel.move_in()
                logger.info("Joined channel: %s", self.config.channel)
        except Exception:
            logger.warning("Channel '%s' not found, staying in root", self.config.channel)

        self._connected = True
        self._running = True

        # Start silence detection thread
        self._silence_checker = threading.Thread(
            target=self._check_silence_loop, daemon=True
        )
        self._silence_checker.start()

    def disconnect(self) -> None:
        """Disconnect from the Mumble server."""
        self._running = False
        if self._mumble:
            self._mumble.stop()
        self._connected = False
        logger.info("Disconnected from Mumble")

    def send_audio(self, pcm_data: bytes) -> None:
        """Send PCM audio to the Mumble channel.

        Args:
            pcm_data: Raw PCM audio (48kHz, 16-bit, mono).
        """
        if not self._connected or not self._mumble:
            logger.warning("Cannot send audio: not connected")
            return

        # pymumble expects audio in 20ms frames
        offset = 0
        while offset < len(pcm_data):
            chunk = pcm_data[offset : offset + MUMBLE_FRAME_SIZE]
            if len(chunk) < MUMBLE_FRAME_SIZE:
                # Pad last frame with silence
                chunk = chunk + b"\x00" * (MUMBLE_FRAME_SIZE - len(chunk))
            self._mumble.sound_output.add_sound(chunk)
            offset += MUMBLE_FRAME_SIZE

    def is_connected(self) -> bool:
        """Check if connected to the Mumble server."""
        return self._connected

    def _on_sound_received(self, user, soundchunk) -> None:
        """pymumble callback: audio received from a user."""
        username = user["name"]
        now = time.monotonic()

        # Track that this user is transmitting
        was_talking = self._user_is_talking.get(username, False)
        self._user_is_talking[username] = True
        self._user_last_audio_time[username] = now

        if not was_talking:
            logger.debug("PTT start: %s", username)
            if self.on_ptt_start:
                self.on_ptt_start(username)

        # Append audio to per-user buffer
        self._user_buffers[username].extend(soundchunk.pcm)

    def _check_silence_loop(self) -> None:
        """Background thread: detect when users stop transmitting."""
        while self._running:
            now = time.monotonic()
            timeout_s = self._silence_timeout_ms / 1000.0

            for username in list(self._user_is_talking.keys()):
                if not self._user_is_talking.get(username, False):
                    continue

                last_time = self._user_last_audio_time.get(username, 0)
                if now - last_time > timeout_s:
                    # User stopped transmitting
                    self._user_is_talking[username] = False
                    audio_data = bytes(self._user_buffers[username])
                    self._user_buffers[username] = bytearray()

                    logger.debug(
                        "PTT end: %s (%d bytes)", username, len(audio_data)
                    )
                    if self.on_ptt_end and audio_data:
                        self.on_ptt_end(username, audio_data)

            time.sleep(0.05)  # 50ms polling interval
