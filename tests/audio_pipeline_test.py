#!/usr/bin/env python3
"""
Audio Pipeline Test

Tests the audio processing pipeline without Mumble:
1. Generate TTS audio
2. Convert to Mumble PCM format (48kHz, 16-bit, mono)
3. Simulate Opus encode/decode cycle
4. Convert back and transcribe with STT
5. Compare transcription to original

This tests the audio_utils.py conversion quality independently.

Usage:
    python tests/audio_pipeline_test.py [--text "custom text"]
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
import wave

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from audio_utils import (
    audio_to_mumble_pcm,
    pcm_to_wav,
    resample_pcm,
    MUMBLE_SAMPLE_RATE,
    WHISPER_SAMPLE_RATE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("audio_pipeline")

DEFAULT_TEST_TEXT = "The quick brown fox jumps over the lazy dog. Testing audio quality one two three."


@dataclass
class TestConfig:
    elevenlabs_api_key: str
    tts_voice_id: str
    stt_model: str = "scribe_v1"


def load_config() -> TestConfig:
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return TestConfig(
        elevenlabs_api_key=cfg["tts"]["elevenlabs_api_key"],
        tts_voice_id=cfg["tts"].get("elevenlabs_voice_id", "pFZP5JQG7iQjIQuC4Bku"),
        stt_model=cfg["stt"].get("model", "scribe_v1"),
    )


async def generate_tts(text: str, config: TestConfig) -> bytes:
    """Generate TTS audio using ElevenLabs, return raw MP3."""
    import httpx
    
    logger.info(f"Generating TTS for: {text[:50]}...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{config.tts_voice_id}",
            headers={
                "xi-api-key": config.elevenlabs_api_key,
                "Content-Type": "application/json",
            },
            json={
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "output_format": "mp3_44100_128",
            },
        )
        response.raise_for_status()
        return response.content


async def transcribe_audio(wav_data: bytes, config: TestConfig) -> str | None:
    """Transcribe WAV audio using ElevenLabs Scribe."""
    import httpx
    
    logger.info(f"Transcribing {len(wav_data)} bytes of audio...")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        files = {"file": ("audio.wav", wav_data, "audio/wav")}
        data = {"model_id": config.stt_model}
        
        response = await client.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": config.elevenlabs_api_key},
            files=files,
            data=data,
        )
        response.raise_for_status()
        result = response.json()
        return result.get("text", "").strip()


def simulate_opus_cycle(pcm_48k: bytes, profile: str = "voip") -> bytes:
    """Simulate Opus encode/decode cycle.
    
    Args:
        pcm_48k: Raw PCM at 48kHz mono 16-bit
        profile: Opus profile - "voip" (lower latency) or "audio" (higher quality)
    """
    try:
        import opuslib
        
        # Create encoder/decoder at 48kHz mono
        encoder = opuslib.Encoder(48000, 1, profile)
        decoder = opuslib.Decoder(48000, 1)
        
        # Note: Setting bitrate via encoder.bitrate is broken in this opuslib version
        # The default bitrate is used (around 64kbps for voice)
        
        # Process in 20ms frames (960 samples = 1920 bytes)
        frame_size = 960
        frame_bytes = frame_size * 2
        
        output = bytearray()
        offset = 0
        encoded_sizes = []
        
        while offset < len(pcm_48k):
            frame = pcm_48k[offset:offset + frame_bytes]
            if len(frame) < frame_bytes:
                frame = frame + b'\x00' * (frame_bytes - len(frame))
            
            # Encode and decode
            encoded = encoder.encode(frame, frame_size)
            encoded_sizes.append(len(encoded))
            decoded = decoder.decode(encoded, frame_size)
            output.extend(decoded)
            
            offset += frame_bytes
        
        avg_encoded = sum(encoded_sizes) / len(encoded_sizes) if encoded_sizes else 0
        estimated_bitrate = avg_encoded * 8 * 50 / 1000  # 50 packets/sec at 20ms, in kbps
        logger.info(f"Opus: {len(encoded_sizes)} frames, avg encoded size: {avg_encoded:.1f} bytes (~{estimated_bitrate:.1f} kbps)")
        
        return bytes(output)
    
    except ImportError:
        logger.warning("opuslib not available, skipping Opus simulation")
        return pcm_48k


def calculate_similarity(original: str, transcribed: str) -> float:
    import re
    def normalize(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r'[^\w\s]', '', s)
        return ' '.join(s.split())
    return SequenceMatcher(None, normalize(original), normalize(transcribed)).ratio()


async def run_pipeline_test(test_text: str | None = None, skip_opus: bool = False) -> dict:
    """Run the audio pipeline test."""
    config = load_config()
    
    if test_text is None:
        test_text = DEFAULT_TEST_TEXT
    
    results = {
        "original_text": test_text,
        "transcribed_text": "",
        "similarity": 0.0,
        "stages": {},
    }
    
    try:
        # Stage 1: TTS
        logger.info("=== Stage 1: TTS Generation ===")
        mp3_data = await generate_tts(test_text, config)
        results["stages"]["tts_mp3_bytes"] = len(mp3_data)
        logger.info(f"TTS MP3: {len(mp3_data)} bytes")
        
        # Stage 2: Convert to Mumble PCM
        logger.info("=== Stage 2: Convert to Mumble PCM (48kHz) ===")
        pcm_48k = audio_to_mumble_pcm(mp3_data, "mp3")
        duration_ms = len(pcm_48k) / (MUMBLE_SAMPLE_RATE * 2) * 1000
        results["stages"]["pcm_48k_bytes"] = len(pcm_48k)
        results["stages"]["duration_ms"] = duration_ms
        logger.info(f"PCM 48kHz: {len(pcm_48k)} bytes ({duration_ms:.1f}ms)")
        
        # Stage 3: Simulate Opus encode/decode
        if not skip_opus:
            logger.info("=== Stage 3: Opus Encode/Decode Simulation ===")
            pcm_after_opus = simulate_opus_cycle(pcm_48k)
            results["stages"]["pcm_after_opus_bytes"] = len(pcm_after_opus)
            logger.info(f"PCM after Opus: {len(pcm_after_opus)} bytes")
        else:
            pcm_after_opus = pcm_48k
            logger.info("=== Stage 3: SKIPPED (--skip-opus) ===")
        
        # Stage 4: Resample to 16kHz for STT
        logger.info("=== Stage 4: Resample to 16kHz for STT ===")
        pcm_16k = resample_pcm(pcm_after_opus, MUMBLE_SAMPLE_RATE, WHISPER_SAMPLE_RATE)
        wav_16k = pcm_to_wav(pcm_16k, WHISPER_SAMPLE_RATE)
        results["stages"]["pcm_16k_bytes"] = len(pcm_16k)
        results["stages"]["wav_16k_bytes"] = len(wav_16k)
        logger.info(f"PCM 16kHz: {len(pcm_16k)} bytes, WAV: {len(wav_16k)} bytes")
        
        # Save intermediate files for inspection
        Path("/tmp/pipeline_test_48k.pcm").write_bytes(pcm_48k)
        Path("/tmp/pipeline_test_16k.wav").write_bytes(wav_16k)
        logger.info("Saved: /tmp/pipeline_test_48k.pcm, /tmp/pipeline_test_16k.wav")
        
        # Stage 5: STT Transcription
        logger.info("=== Stage 5: STT Transcription ===")
        transcription = await transcribe_audio(wav_16k, config)
        results["transcribed_text"] = transcription or ""
        
        if transcription:
            results["similarity"] = calculate_similarity(test_text, transcription)
        
        return results
        
    except Exception as e:
        logger.exception("Pipeline test failed")
        results["error"] = str(e)
        return results


def print_results(results: dict):
    print("\n" + "=" * 60)
    print("AUDIO PIPELINE TEST RESULTS")
    print("=" * 60)
    
    print(f"\n📝 Original Text:")
    print(f"   {results['original_text']}")
    
    print(f"\n🎤 Transcribed Text:")
    print(f"   {results['transcribed_text'] or '(empty)'}")
    
    print(f"\n📊 Statistics:")
    print(f"   Similarity Score: {results['similarity']:.1%}")
    
    if "stages" in results:
        stages = results["stages"]
        print(f"\n🔧 Pipeline Stages:")
        print(f"   TTS MP3:        {stages.get('tts_mp3_bytes', 0):,} bytes")
        print(f"   PCM 48kHz:      {stages.get('pcm_48k_bytes', 0):,} bytes ({stages.get('duration_ms', 0):.1f}ms)")
        print(f"   After Opus:     {stages.get('pcm_after_opus_bytes', 0):,} bytes")
        print(f"   PCM 16kHz:      {stages.get('pcm_16k_bytes', 0):,} bytes")
        print(f"   WAV 16kHz:      {stages.get('wav_16k_bytes', 0):,} bytes")
    
    if results.get("error"):
        print(f"\n⚠️  Error: {results['error']}")
    
    print("\n🎯 Quality Assessment:")
    if results["similarity"] >= 0.95:
        print("   ✅ EXCELLENT - Pipeline produces perfect audio!")
    elif results["similarity"] >= 0.85:
        print("   ✅ GOOD - Pipeline quality is acceptable")
    elif results["similarity"] >= 0.70:
        print("   ⚠️  FAIR - Some quality loss in pipeline")
    elif results["similarity"] >= 0.50:
        print("   ❌ POOR - Significant quality problems in pipeline")
    else:
        print("   ❌ FAILED - Pipeline produces unintelligible audio")
    
    print("=" * 60 + "\n")


async def main():
    parser = argparse.ArgumentParser(description="Test audio processing pipeline")
    parser.add_argument("--text", "-t", help="Custom text to test with")
    parser.add_argument("--skip-opus", action="store_true", help="Skip Opus encode/decode simulation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    results = await run_pipeline_test(
        test_text=args.text,
        skip_opus=args.skip_opus,
    )
    
    print_results(results)
    
    if results["similarity"] < 0.70:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
