"""Audio utilities for PCM conversion and resampling.

Mumble delivers and expects 16-bit signed PCM at 48kHz mono.
Whisper API expects 16kHz audio in WAV/FLAC format.
TTS APIs return MP3/OPUS/PCM that must be converted to 48kHz for Mumble.
"""

import io
import struct
import wave

import numpy as np
from scipy.signal import resample_poly
from math import gcd


# Mumble audio format constants
MUMBLE_SAMPLE_RATE = 48000
MUMBLE_SAMPLE_WIDTH = 2  # 16-bit
MUMBLE_CHANNELS = 1

# Whisper optimal sample rate
WHISPER_SAMPLE_RATE = 16000


def pcm_to_wav(pcm_data: bytes, sample_rate: int, sample_width: int = 2, channels: int = 1) -> bytes:
    """Convert raw PCM bytes to WAV format in memory.

    Args:
        pcm_data: Raw PCM audio bytes (signed 16-bit little-endian).
        sample_rate: Sample rate in Hz.
        sample_width: Bytes per sample (default 2 for 16-bit).
        channels: Number of audio channels (default 1 for mono).

    Returns:
        WAV file bytes.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


def wav_to_pcm(wav_data: bytes) -> tuple[bytes, int, int, int]:
    """Extract raw PCM data from a WAV file.

    Args:
        wav_data: WAV file bytes.

    Returns:
        Tuple of (pcm_bytes, sample_rate, sample_width, channels).
    """
    buf = io.BytesIO(wav_data)
    with wave.open(buf, "rb") as wf:
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, wf.getframerate(), wf.getsampwidth(), wf.getnchannels()


def resample_pcm(pcm_data: bytes, from_rate: int, to_rate: int, sample_width: int = 2) -> bytes:
    """Resample PCM audio from one sample rate to another.

    Uses polyphase resampling for quality. Input and output are 16-bit signed PCM.

    Args:
        pcm_data: Raw PCM bytes (signed 16-bit little-endian).
        from_rate: Source sample rate in Hz.
        to_rate: Target sample rate in Hz.
        sample_width: Bytes per sample (must be 2).

    Returns:
        Resampled PCM bytes.
    """
    if from_rate == to_rate:
        return pcm_data

    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM supported, got sample_width={sample_width}")

    # Convert bytes to numpy array of int16
    samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float64)

    # Compute up/down factors using GCD for integer ratio
    divisor = gcd(from_rate, to_rate)
    up = to_rate // divisor
    down = from_rate // divisor

    # Resample
    resampled = resample_poly(samples, up, down)

    # Clip and convert back to int16
    resampled = np.clip(resampled, -32768, 32767).astype(np.int16)
    return resampled.tobytes()


def mumble_to_whisper(pcm_data: bytes) -> bytes:
    """Convert Mumble PCM (48kHz) to Whisper-ready WAV (16kHz).

    Args:
        pcm_data: Raw PCM from Mumble (48kHz, 16-bit, mono).

    Returns:
        WAV file bytes at 16kHz, ready for Whisper API upload.
    """
    resampled = resample_pcm(pcm_data, MUMBLE_SAMPLE_RATE, WHISPER_SAMPLE_RATE)
    return pcm_to_wav(resampled, WHISPER_SAMPLE_RATE)


def audio_to_mumble_pcm(audio_data: bytes, source_format: str, source_sample_rate: int | None = None) -> bytes:
    """Convert audio from various formats to Mumble-ready PCM (48kHz, 16-bit, mono).

    Args:
        audio_data: Audio bytes in the specified format.
        source_format: One of "pcm", "wav", "mp3", "opus".
        source_sample_rate: Required for "pcm" format. Ignored for self-describing formats.

    Returns:
        PCM bytes at 48kHz, 16-bit, mono.
    """
    if source_format == "pcm":
        if source_sample_rate is None:
            raise ValueError("source_sample_rate required for raw PCM input")
        return resample_pcm(audio_data, source_sample_rate, MUMBLE_SAMPLE_RATE)

    if source_format == "wav":
        pcm, rate, width, channels = wav_to_pcm(audio_data)
        if channels > 1:
            pcm = _stereo_to_mono(pcm, channels, width)
        if width != 2:
            raise ValueError(f"Only 16-bit WAV supported, got {width * 8}-bit")
        return resample_pcm(pcm, rate, MUMBLE_SAMPLE_RATE)

    if source_format in ("mp3", "opus", "ogg", "flac", "aac"):
        # Use ffmpeg via subprocess directly for robust resampling
        import subprocess
        
        try:
            process = subprocess.Popen(
                [
                    "/opt/homebrew/bin/ffmpeg",
                    "-i", "pipe:0",
                    "-f", "s16le",
                    "-ac", "1",
                    "-ar", str(MUMBLE_SAMPLE_RATE),
                    "pipe:1"
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            out, _ = process.communicate(input=audio_data)
            if process.returncode != 0:
                # Fallback to pydub if ffmpeg fails (e.g. not in path)
                raise RuntimeError("ffmpeg failed")
            return out
        except Exception:
            # Fallback to pydub
            from pydub import AudioSegment
            seg = AudioSegment.from_file(io.BytesIO(audio_data), format=source_format)
            seg = seg.set_channels(1).set_sample_width(2).set_frame_rate(MUMBLE_SAMPLE_RATE)
            return seg.raw_data

    raise ValueError(f"Unsupported source format: {source_format}")


def pcm_duration_ms(pcm_data: bytes, sample_rate: int, sample_width: int = 2) -> int:
    """Calculate duration of PCM audio in milliseconds.

    Args:
        pcm_data: Raw PCM bytes.
        sample_rate: Sample rate in Hz.
        sample_width: Bytes per sample.

    Returns:
        Duration in milliseconds.
    """
    num_samples = len(pcm_data) // sample_width
    return int((num_samples / sample_rate) * 1000)


def _stereo_to_mono(pcm_data: bytes, channels: int, sample_width: int) -> bytes:
    """Mix multi-channel PCM down to mono by averaging channels."""
    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM supported for stereo-to-mono")

    samples = np.frombuffer(pcm_data, dtype=np.int16)
    # Reshape to (num_frames, channels) and average
    samples = samples.reshape(-1, channels)
    mono = samples.mean(axis=1).astype(np.int16)
    return mono.tobytes()
