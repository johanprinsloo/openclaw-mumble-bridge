"""Speech-to-text client using ElevenLabs Scribe API.

Accepts PCM audio from Mumble, converts to WAV, and sends to ElevenLabs
for transcription.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from audio_utils import mumble_to_whisper

logger = logging.getLogger(__name__)


@dataclass
class STTClient:
    """ElevenLabs Scribe STT client.

    Attributes:
        api_key: ElevenLabs API key.
        model: ElevenLabs Scribe model ID (default "scribe_v1").
        language_code: Optional ISO 639-1 language code (e.g., "en").
    """

    api_key: str
    model: str = "scribe_v1"
    language_code: str | None = None

    async def transcribe(self, pcm_data: bytes, user: str = "unknown") -> str | None:
        """Transcribe Mumble PCM audio to text using ElevenLabs Scribe.

        Args:
            pcm_data: Raw PCM from Mumble (48kHz, 16-bit, mono).
            user: Mumble username for logging.

        Returns:
            Transcribed text, or None if transcription failed or was empty.
        """
        import httpx

        # ElevenLabs Scribe also works well with 16kHz WAV, so reuse existing utility
        wav_data = mumble_to_whisper(pcm_data)
        logger.debug("Transcribing %d bytes of WAV from %s via ElevenLabs", len(wav_data), user)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                files = {"file": ("audio.wav", wav_data, "audio/wav")}
                data = {"model_id": self.model}
                if self.language_code:
                    data["language_code"] = self.language_code

                response = await client.post(
                    "https://api.elevenlabs.io/v1/speech-to-text",
                    headers={"xi-api-key": self.api_key},
                    files=files,
                    data=data,
                )
                response.raise_for_status()
                result = response.json()
                text = result.get("text", "").strip()

                if not text:
                    logger.debug("Empty transcription from %s", user)
                    return None

                logger.info("STT [%s]: %s", user, text)
                return text

        except httpx.HTTPStatusError as e:
            logger.error("ElevenLabs STT API error: %s %s", e.response.status_code, e.response.text)
            return None
        except httpx.RequestError as e:
            logger.error("ElevenLabs STT API request failed: %s", e)
            return None
