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
    provider: str = "elevenlabs"  # "elevenlabs" or "local_whisper"

    async def transcribe(self, pcm_data: bytes, user: str = "unknown") -> str | None:
        """Transcribe Mumble PCM audio to text.

        Args:
            pcm_data: Raw PCM from Mumble (48kHz, 16-bit, mono).
            user: Mumble username for logging.

        Returns:
            Transcribed text, or None if transcription failed or was empty.
        """
        import httpx
        from audio_utils import mumble_to_whisper

        # Convert to WAV (16kHz)
        wav_data = mumble_to_whisper(pcm_data)
        
        if self.provider == "local_whisper":
            return await self._transcribe_local(wav_data, user)

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

    async def _transcribe_local(self, wav_data: bytes, user: str) -> str | None:
        """Transcribe using local openai-whisper (requires 'whisper' installed)."""
        import tempfile
        import subprocess
        import os
        
        logger.debug("Transcribing %d bytes from %s via Local Whisper", len(wav_data), user)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_data)
            tmp_path = f.name
            
        try:
            # Run whisper CLI: whisper audio.wav --model base.en --output_format txt
            # Assumes 'whisper' is in PATH or venv
            cmd = [
                "whisper",
                tmp_path,
                "--model", "base.en",  # fast and decent
                "--output_format", "txt",
                "--output_dir", os.path.dirname(tmp_path),
                "--fp16", "False" # CPU fallback if needed
            ]
            
            # Run subprocess in thread to avoid blocking asyncio loop
            import asyncio
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                logger.error("Local Whisper failed: %s", stderr.decode())
                return None
                
            # Read output file
            txt_path = tmp_path + ".txt"
            if os.path.exists(txt_path):
                with open(txt_path, "r") as f:
                    text = f.read().strip()
                os.unlink(txt_path)
                logger.info("STT (Local) [%s]: %s", user, text)
                return text
            else:
                logger.error("Local Whisper produced no output file")
                return None
                
        except Exception as e:
            logger.error("Local STT error: %s", e)
            return None
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
