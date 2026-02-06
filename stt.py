"""Speech-to-text client using OpenAI Whisper API.

Accepts PCM audio from Mumble, converts to WAV, and sends to Whisper
for transcription.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from audio_utils import mumble_to_whisper

logger = logging.getLogger(__name__)


@dataclass
class STTClient:
    """OpenAI Whisper STT client.

    Attributes:
        api_key: OpenAI API key.
        model: Whisper model name (default "whisper-1").
        language: Language hint for transcription.
    """

    api_key: str
    model: str = "whisper-1"
    language: str = "en"

    async def transcribe(self, pcm_data: bytes, user: str = "unknown") -> str | None:
        """Transcribe Mumble PCM audio to text.

        Args:
            pcm_data: Raw PCM from Mumble (48kHz, 16-bit, mono).
            user: Mumble username for logging.

        Returns:
            Transcribed text, or None if transcription failed or was empty.
        """
        import httpx

        wav_data = mumble_to_whisper(pcm_data)
        logger.debug("Transcribing %d bytes of WAV from %s", len(wav_data), user)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files={"file": ("audio.wav", wav_data, "audio/wav")},
                    data={"model": self.model, "language": self.language},
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
            logger.error("Whisper API error: %s %s", e.response.status_code, e.response.text)
            return None
        except httpx.RequestError as e:
            logger.error("Whisper API request failed: %s", e)
            return None
