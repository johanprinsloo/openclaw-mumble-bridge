"""Tests for audio utility functions.

Tests PCM/WAV conversion, resampling, and format conversion utilities.
Uses synthetic audio data to avoid external dependencies.
"""

import struct
import wave
import io
import math

import pytest
import numpy as np

from audio_utils import (
    pcm_to_wav,
    wav_to_pcm,
    resample_pcm,
    mumble_to_whisper,
    audio_to_mumble_pcm,
    pcm_duration_ms,
    MUMBLE_SAMPLE_RATE,
    WHISPER_SAMPLE_RATE,
)


def generate_sine_wave_pcm(
    freq_hz: float,
    duration_ms: int,
    sample_rate: int,
    amplitude: int = 16000,
) -> bytes:
    """Generate a sine wave as 16-bit signed PCM.
    
    Args:
        freq_hz: Frequency of the sine wave in Hz.
        duration_ms: Duration in milliseconds.
        sample_rate: Sample rate in Hz.
        amplitude: Peak amplitude (0-32767).
    
    Returns:
        Raw PCM bytes (signed 16-bit little-endian).
    """
    num_samples = int(sample_rate * duration_ms / 1000)
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        value = int(amplitude * math.sin(2 * math.pi * freq_hz * t))
        samples.append(value)
    return struct.pack(f"<{len(samples)}h", *samples)


def generate_silence_pcm(duration_ms: int, sample_rate: int) -> bytes:
    """Generate silence as 16-bit signed PCM."""
    num_samples = int(sample_rate * duration_ms / 1000)
    return b"\x00\x00" * num_samples


class TestPcmToWav:
    """Test PCM to WAV conversion."""

    def test_creates_valid_wav(self):
        pcm = generate_sine_wave_pcm(440, 100, 48000)
        wav_data = pcm_to_wav(pcm, sample_rate=48000)
        
        # Should be valid WAV (starts with RIFF header)
        assert wav_data[:4] == b"RIFF"
        assert b"WAVE" in wav_data[:12]

    def test_preserves_sample_rate(self):
        pcm = generate_sine_wave_pcm(440, 100, 48000)
        wav_data = pcm_to_wav(pcm, sample_rate=48000)
        
        with wave.open(io.BytesIO(wav_data), "rb") as wf:
            assert wf.getframerate() == 48000

    def test_preserves_channels(self):
        pcm = generate_sine_wave_pcm(440, 100, 48000)
        wav_data = pcm_to_wav(pcm, sample_rate=48000, channels=1)
        
        with wave.open(io.BytesIO(wav_data), "rb") as wf:
            assert wf.getnchannels() == 1

    def test_preserves_sample_width(self):
        pcm = generate_sine_wave_pcm(440, 100, 48000)
        wav_data = pcm_to_wav(pcm, sample_rate=48000, sample_width=2)
        
        with wave.open(io.BytesIO(wav_data), "rb") as wf:
            assert wf.getsampwidth() == 2

    def test_different_sample_rates(self):
        for rate in [8000, 16000, 22050, 44100, 48000]:
            pcm = generate_sine_wave_pcm(440, 100, rate)
            wav_data = pcm_to_wav(pcm, sample_rate=rate)
            
            with wave.open(io.BytesIO(wav_data), "rb") as wf:
                assert wf.getframerate() == rate


class TestWavToPcm:
    """Test WAV to PCM extraction."""

    def test_extracts_pcm_data(self):
        original_pcm = generate_sine_wave_pcm(440, 100, 48000)
        wav_data = pcm_to_wav(original_pcm, sample_rate=48000)
        
        extracted_pcm, rate, width, channels = wav_to_pcm(wav_data)
        
        assert extracted_pcm == original_pcm
        assert rate == 48000
        assert width == 2
        assert channels == 1

    def test_roundtrip_preserves_data(self):
        """PCM → WAV → PCM should be lossless."""
        original = generate_sine_wave_pcm(1000, 50, 16000)
        wav_data = pcm_to_wav(original, sample_rate=16000)
        recovered, _, _, _ = wav_to_pcm(wav_data)
        
        assert recovered == original


class TestResamplePcm:
    """Test PCM resampling."""

    def test_same_rate_returns_unchanged(self):
        pcm = generate_sine_wave_pcm(440, 100, 48000)
        result = resample_pcm(pcm, from_rate=48000, to_rate=48000)
        
        assert result == pcm

    def test_downsample_48k_to_16k(self):
        pcm_48k = generate_sine_wave_pcm(440, 100, 48000)
        pcm_16k = resample_pcm(pcm_48k, from_rate=48000, to_rate=16000)
        
        # 48k→16k is 3:1 ratio, so output should be ~1/3 the size
        expected_ratio = 16000 / 48000
        actual_ratio = len(pcm_16k) / len(pcm_48k)
        
        assert abs(actual_ratio - expected_ratio) < 0.01

    def test_upsample_16k_to_48k(self):
        pcm_16k = generate_sine_wave_pcm(440, 100, 16000)
        pcm_48k = resample_pcm(pcm_16k, from_rate=16000, to_rate=48000)
        
        # 16k→48k is 1:3 ratio, so output should be ~3x the size
        expected_ratio = 48000 / 16000
        actual_ratio = len(pcm_48k) / len(pcm_16k)
        
        assert abs(actual_ratio - expected_ratio) < 0.01

    def test_output_is_valid_int16(self):
        """Resampled output should be valid 16-bit signed integers."""
        pcm = generate_sine_wave_pcm(440, 100, 48000, amplitude=30000)
        result = resample_pcm(pcm, from_rate=48000, to_rate=16000)
        
        # Parse as int16 - should not raise
        samples = np.frombuffer(result, dtype=np.int16)
        
        # All values should be in valid range
        assert samples.min() >= -32768
        assert samples.max() <= 32767

    def test_rejects_non_16bit(self):
        """Should reject non-16-bit PCM."""
        pcm = b"\x00" * 100  # Pretend 8-bit
        
        with pytest.raises(ValueError, match="16-bit"):
            resample_pcm(pcm, from_rate=48000, to_rate=16000, sample_width=1)

    def test_arbitrary_rates(self):
        """Test resampling between non-standard rates."""
        pcm = generate_sine_wave_pcm(440, 100, 44100)
        result = resample_pcm(pcm, from_rate=44100, to_rate=22050)
        
        expected_ratio = 22050 / 44100
        actual_ratio = len(result) / len(pcm)
        
        assert abs(actual_ratio - expected_ratio) < 0.01


class TestMumbleToWhisper:
    """Test the Mumble → Whisper conversion helper."""

    def test_converts_48k_to_16k_wav(self):
        pcm_48k = generate_sine_wave_pcm(440, 100, 48000)
        wav_16k = mumble_to_whisper(pcm_48k)
        
        # Should be valid WAV
        assert wav_16k[:4] == b"RIFF"
        
        # Should be 16kHz
        with wave.open(io.BytesIO(wav_16k), "rb") as wf:
            assert wf.getframerate() == WHISPER_SAMPLE_RATE
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2

    def test_output_duration_preserved(self):
        """Duration should be preserved after conversion."""
        duration_ms = 500
        pcm_48k = generate_sine_wave_pcm(440, duration_ms, 48000)
        wav_16k = mumble_to_whisper(pcm_48k)
        
        with wave.open(io.BytesIO(wav_16k), "rb") as wf:
            actual_duration_ms = (wf.getnframes() / wf.getframerate()) * 1000
            
            # Allow 1ms tolerance for rounding
            assert abs(actual_duration_ms - duration_ms) < 2


class TestAudioToMumblePcm:
    """Test conversion from various formats to Mumble PCM."""

    def test_pcm_conversion_requires_sample_rate(self):
        pcm = generate_sine_wave_pcm(440, 100, 16000)
        
        with pytest.raises(ValueError, match="sample_rate required"):
            audio_to_mumble_pcm(pcm, source_format="pcm")

    def test_pcm_conversion_with_sample_rate(self):
        pcm_16k = generate_sine_wave_pcm(440, 100, 16000)
        pcm_48k = audio_to_mumble_pcm(
            pcm_16k, source_format="pcm", source_sample_rate=16000
        )
        
        # Should be upsampled to 48kHz
        expected_ratio = MUMBLE_SAMPLE_RATE / 16000
        actual_ratio = len(pcm_48k) / len(pcm_16k)
        
        assert abs(actual_ratio - expected_ratio) < 0.01

    def test_wav_conversion(self):
        pcm_16k = generate_sine_wave_pcm(440, 100, 16000)
        wav_16k = pcm_to_wav(pcm_16k, sample_rate=16000)
        
        pcm_48k = audio_to_mumble_pcm(wav_16k, source_format="wav")
        
        # Should be upsampled to 48kHz
        expected_samples = int(len(pcm_16k) / 2 * (48000 / 16000))
        actual_samples = len(pcm_48k) / 2
        
        assert abs(actual_samples - expected_samples) < 10

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            audio_to_mumble_pcm(b"data", source_format="unknown")


class TestPcmDurationMs:
    """Test PCM duration calculation."""

    def test_calculates_correct_duration(self):
        # 1 second at 48kHz, 16-bit = 96000 bytes
        pcm = generate_sine_wave_pcm(440, 1000, 48000)
        duration = pcm_duration_ms(pcm, sample_rate=48000)
        
        assert duration == 1000

    def test_short_duration(self):
        pcm = generate_sine_wave_pcm(440, 50, 48000)
        duration = pcm_duration_ms(pcm, sample_rate=48000)
        
        assert duration == 50

    def test_different_sample_rates(self):
        for rate in [8000, 16000, 44100, 48000]:
            pcm = generate_sine_wave_pcm(440, 100, rate)
            duration = pcm_duration_ms(pcm, sample_rate=rate)
            
            assert abs(duration - 100) < 2  # Allow small rounding error

    def test_empty_pcm(self):
        duration = pcm_duration_ms(b"", sample_rate=48000)
        assert duration == 0

    def test_silence(self):
        pcm = generate_silence_pcm(200, 48000)
        duration = pcm_duration_ms(pcm, sample_rate=48000)
        
        assert duration == 200


class TestConstants:
    """Test that audio constants are correct."""

    def test_mumble_sample_rate(self):
        assert MUMBLE_SAMPLE_RATE == 48000

    def test_whisper_sample_rate(self):
        assert WHISPER_SAMPLE_RATE == 16000
