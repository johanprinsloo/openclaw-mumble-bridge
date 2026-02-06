"""Text-to-speech client with cancellation support.

Supports OpenAI TTS and ElevenLabs. Returns audio in Mumble-ready
PCM format (48kHz, 16-bit, mono). Checks a cancellation event between
operations to support barge-in.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from audio_utils import audio_to_mumble_pcm

logger = logging.getLogger(__name__)


@dataclass
class TTSClient:
    """TTS client that converts text to Mumble-ready PCM audio.

    Attributes:
        provider: "openai" or "elevenlabs".
        api_key: API key for the provider.
        voice: Voice identifier.
        speed: Playback speed (OpenAI only).
        elevenlabs_voice_id: ElevenLabs voice ID (if provider is elevenlabs).
    """

    provider: str = "openai"
    api_key: str = ""
    voice: str = "nova"
    speed: float = 1.0
    elevenlabs_voice_id: str = ""

    async def synthesize(
        self,
        text: str,
        cancellation_event: asyncio.Event | None = None,
    ) -> bytes | None:
        """Convert text to Mumble-ready PCM audio.

        Args:
            text: Text to synthesize.
            cancellation_event: If set, abort and return None.

        Returns:
            PCM bytes (48kHz, 16-bit, mono) or None if cancelled/failed.
        """
        if cancellation_event and cancellation_event.is_set():
            logger.debug("TTS cancelled before start")
            return None

        if self.provider == "openai":
            return await self._synthesize_openai(text, cancellation_event)
        elif self.provider == "elevenlabs":
            return await self._synthesize_elevenlabs(text, cancellation_event)
        else:
            logger.error("Unknown TTS provider: %s", self.provider)
            return None

    async def _synthesize_openai(
        self, text: str, cancellation_event: asyncio.Event | None
    ) -> bytes | None:
        """Synthesize via OpenAI TTS API."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/audio/speech",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "tts-1",
                        "input": text,
                        "voice": self.voice,
                        "speed": self.speed,
                        "response_format": "mp3",
                    },
                )
                response.raise_for_status()

                if cancellation_event and cancellation_event.is_set():
                    logger.debug("TTS cancelled after API response")
                    return None

                mp3_data = response.content
                return audio_to_mumble_pcm(mp3_data, "mp3")

        except httpx.HTTPStatusError as e:
            logger.error("OpenAI TTS error: %s %s", e.response.status_code, e.response.text)
            return None
        except httpx.RequestError as e:
            logger.error("OpenAI TTS request failed: %s", e)
            return None

    async def _synthesize_elevenlabs(
        self, text: str, cancellation_event: asyncio.Event | None
    ) -> bytes | None:
        """Synthesize via ElevenLabs TTS API."""
        import httpx

        voice_id = self.elevenlabs_voice_id or self.voice
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                    headers={
                        "xi-api-key": self.api_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "text": text,
                        "model_id": "eleven_multilingual_v2",
                        "output_format": "mp3_44100_128",
                    },
                )
                response.raise_for_status()

                if cancellation_event and cancellation_event.is_set():
                    logger.debug("TTS cancelled after API response")
                    return None

                mp3_data = response.content
                return audio_to_mumble_pcm(mp3_data, "mp3")

        except httpx.HTTPStatusError as e:
            logger.error("ElevenLabs TTS error: %s %s", e.response.status_code, e.response.text)
            return None
        except httpx.RequestError as e:
            logger.error("ElevenLabs TTS request failed: %s", e)
            return None
