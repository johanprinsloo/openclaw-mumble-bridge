#!/usr/bin/env python3
"""
Audio Loopback Test Harness

Tests Mumble audio quality by:
1. Spawning Client A ("Speaker") and Client B ("Listener")
2. Client A sends TTS audio (or reference WAV) to the channel
3. Client B records the stream from the channel
4. Client B transcribes via ElevenLabs Scribe STT
5. Prints transcription vs original text similarity score

Usage:
    python tests/audio_loopback.py [--text "custom text"] [--wav path/to/file.wav]
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import os
import sys
import threading
import time
import wave
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import yaml

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from audio_utils import audio_to_mumble_pcm, pcm_to_wav, MUMBLE_SAMPLE_RATE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("audio_loopback")

# Default test text
DEFAULT_TEST_TEXT = "The quick brown fox jumps over the lazy dog. Testing audio quality one two three."


@dataclass
class TestConfig:
    """Test configuration loaded from config.yaml."""
    mumble_host: str
    mumble_port: int
    mumble_channel: str
    elevenlabs_api_key: str
    tts_voice_id: str
    stt_model: str = "scribe_v1"


def load_config() -> TestConfig:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    return TestConfig(
        mumble_host=cfg["mumble"]["host"],
        mumble_port=cfg["mumble"]["port"],
        mumble_channel=cfg["mumble"].get("channel", "Root"),
        elevenlabs_api_key=cfg["tts"]["elevenlabs_api_key"],
        tts_voice_id=cfg["tts"].get("elevenlabs_voice_id", "pFZP5JQG7iQjIQuC4Bku"),
        stt_model=cfg["stt"].get("model", "scribe_v1"),
    )


def create_mumble_client(username: str, config: TestConfig):
    """Create a pymumble client with proper SSL patching."""
    import ssl
    
    # Monkey-patch ssl.wrap_socket for Python 3.12+ compatibility
    if not hasattr(ssl, 'wrap_socket'):
        def wrap_socket(sock, keyfile=None, certfile=None, server_side=False, 
                       cert_reqs=ssl.CERT_NONE, ssl_version=ssl.PROTOCOL_TLS, 
                       ca_certs=None, do_handshake_on_connect=True, 
                       suppress_ragged_eofs=True, ciphers=None):
            context = ssl.SSLContext(ssl_version)
            if certfile:
                context.load_cert_chain(certfile, keyfile)
            if ca_certs:
                context.load_verify_locations(ca_certs)
            if cert_reqs:
                context.verify_mode = cert_reqs
            if ciphers:
                context.set_ciphers(ciphers)
            return context.wrap_socket(sock, server_side=server_side, 
                                       do_handshake_on_connect=do_handshake_on_connect, 
                                       suppress_ragged_eofs=suppress_ragged_eofs)
        ssl.wrap_socket = wrap_socket
    
    try:
        import pymumble_py3 as pymumble
    except ImportError:
        import pymumble3 as pymumble
    
    # Monkey-patch SoundOutput._set_bandwidth to fix Opus error
    # This matches the fix in mumble_client.py
    if hasattr(pymumble, 'soundoutput'):
        def patched_set_bandwidth(self_so):
            if not getattr(self_so, 'encoder', None):
                return
            # Force 32kbps which is safe for Opus voice or just do nothing
            # Note: opuslib.api.ctl.set_bitrate often fails with 'invalid argument'
            # We will ignore it as the default bitrate is usually fine
            pass
        
        pymumble.soundoutput.SoundOutput._set_bandwidth = patched_set_bandwidth

    client = pymumble.Mumble(
        config.mumble_host,
        username,
        port=config.mumble_port,
        reconnect=False,
    )
    return client


async def generate_tts(text: str, config: TestConfig) -> bytes:
    """Generate TTS audio using ElevenLabs and return Mumble-ready PCM."""
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
        mp3_data = response.content
    
    logger.info(f"TTS response: {len(mp3_data)} bytes MP3")
    
    # Convert to Mumble PCM
    pcm_data = audio_to_mumble_pcm(mp3_data, "mp3")
    duration_ms = len(pcm_data) / (MUMBLE_SAMPLE_RATE * 2) * 1000
    logger.info(f"Converted to PCM: {len(pcm_data)} bytes ({duration_ms:.1f}ms)")
    
    return pcm_data


def load_wav_as_pcm(wav_path: str) -> bytes:
    """Load a WAV file and convert to Mumble-ready PCM."""
    with open(wav_path, "rb") as f:
        wav_data = f.read()
    return audio_to_mumble_pcm(wav_data, "wav")


async def transcribe_audio(pcm_data: bytes, config: TestConfig) -> str | None:
    """Transcribe PCM audio using ElevenLabs Scribe."""
    import httpx
    
    # Convert to WAV at 16kHz for STT
    from audio_utils import mumble_to_whisper
    wav_data = mumble_to_whisper(pcm_data)
    
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


def calculate_similarity(original: str, transcribed: str) -> float:
    """Calculate similarity ratio between original and transcribed text."""
    # Normalize: lowercase, strip whitespace, remove punctuation
    import re
    def normalize(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r'[^\w\s]', '', s)
        return ' '.join(s.split())
    
    orig_norm = normalize(original)
    trans_norm = normalize(transcribed)
    
    return SequenceMatcher(None, orig_norm, trans_norm).ratio()


class AudioRecorder:
    """Records audio from a Mumble client."""
    
    def __init__(self):
        self.buffer = bytearray()
        self.recording = False
        self.lock = threading.Lock()
    
    def start(self):
        with self.lock:
            self.buffer = bytearray()
            self.recording = True
    
    def stop(self) -> bytes:
        with self.lock:
            self.recording = False
            return bytes(self.buffer)
    
    def on_sound(self, user, sound_chunk):
        with self.lock:
            if self.recording:
                self.buffer.extend(sound_chunk.pcm)


async def run_loopback_test(
    test_text: str | None = None,
    wav_path: str | None = None,
    config: TestConfig | None = None,
) -> dict:
    """
    Run the full loopback test.
    
    Returns dict with:
        - original_text: The text sent
        - transcribed_text: What STT heard
        - similarity: 0.0 - 1.0 score
        - pcm_bytes_sent: Amount of audio sent
        - pcm_bytes_received: Amount of audio received
    """
    if config is None:
        config = load_config()
    
    if test_text is None:
        test_text = DEFAULT_TEST_TEXT
    
    results = {
        "original_text": test_text,
        "transcribed_text": "",
        "similarity": 0.0,
        "pcm_bytes_sent": 0,
        "pcm_bytes_received": 0,
        "error": None,
    }
    
    try:
        import pymumble_py3 as pymumble
    except ImportError:
        import pymumble3 as pymumble
    
    # Create clients
    logger.info("Creating Mumble clients...")
    speaker = create_mumble_client("TestSpeaker", config)
    listener = create_mumble_client("TestListener", config)
    
    recorder = AudioRecorder()
    
    try:
        # Connect listener first (to receive audio)
        logger.info("Connecting listener...")
        listener.set_receive_sound(True)
        listener.callbacks.set_callback(
            pymumble.constants.PYMUMBLE_CLBK_SOUNDRECEIVED,
            recorder.on_sound,
        )
        listener.start()
        listener.is_ready()
        
        # Join channel
        try:
            channel = listener.channels.find_by_name(config.mumble_channel)
            if channel:
                channel.move_in()
        except Exception:
            pass
        
        # Connect speaker
        logger.info("Connecting speaker...")
        speaker.start()
        speaker.is_ready()
        
        # Join same channel
        try:
            channel = speaker.channels.find_by_name(config.mumble_channel)
            if channel:
                channel.move_in()
        except Exception:
            pass
        
        # Give time for clients to settle
        await asyncio.sleep(0.5)
        
        # Generate or load audio
        if wav_path:
            logger.info(f"Loading WAV from {wav_path}...")
            pcm_data = load_wav_as_pcm(wav_path)
        else:
            pcm_data = await generate_tts(test_text, config)
        
        results["pcm_bytes_sent"] = len(pcm_data)
        
        # Start recording
        logger.info("Starting recording...")
        recorder.start()
        
        # Send audio
        logger.info("Sending audio...")
        speaker.sound_output.add_sound(pcm_data)
        
        # Wait for audio to finish playing + buffer
        audio_duration_s = len(pcm_data) / (MUMBLE_SAMPLE_RATE * 2)
        # Increase wait buffer significantly to account for buffering/latency
        wait_time = audio_duration_s * 1.5 + 3.0
        logger.info(f"Waiting {wait_time:.1f}s for audio playback...")
        await asyncio.sleep(wait_time)
        
        # Stop recording
        recorded_pcm = recorder.stop()
        results["pcm_bytes_received"] = len(recorded_pcm)
        logger.info(f"Recorded {len(recorded_pcm)} bytes")
        
        if len(recorded_pcm) < 1000:
            results["error"] = "No audio received (listener may not hear speaker in same channel)"
            logger.warning(results["error"])
            return results
        
        # Transcribe
        transcription = await transcribe_audio(recorded_pcm, config)
        results["transcribed_text"] = transcription or ""
        
        if transcription:
            results["similarity"] = calculate_similarity(test_text, transcription)
        
        return results
        
    finally:
        # Cleanup
        logger.info("Disconnecting clients...")
        try:
            speaker.stop()
        except Exception:
            pass
        try:
            listener.stop()
        except Exception:
            pass


def print_results(results: dict):
    """Print test results in a nice format."""
    print("\n" + "=" * 60)
    print("AUDIO LOOPBACK TEST RESULTS")
    print("=" * 60)
    
    print(f"\n📝 Original Text:")
    print(f"   {results['original_text']}")
    
    print(f"\n🎤 Transcribed Text:")
    print(f"   {results['transcribed_text'] or '(empty)'}")
    
    print(f"\n📊 Statistics:")
    print(f"   Similarity Score: {results['similarity']:.1%}")
    print(f"   PCM Sent:     {results['pcm_bytes_sent']:,} bytes")
    print(f"   PCM Received: {results['pcm_bytes_received']:,} bytes")
    
    if results.get("error"):
        print(f"\n⚠️  Error: {results['error']}")
    
    # Quality assessment
    print("\n🎯 Quality Assessment:")
    if results["similarity"] >= 0.95:
        print("   ✅ EXCELLENT - Audio is crystal clear!")
    elif results["similarity"] >= 0.85:
        print("   ✅ GOOD - Audio quality is acceptable")
    elif results["similarity"] >= 0.70:
        print("   ⚠️  FAIR - Some quality issues detected")
    elif results["similarity"] >= 0.50:
        print("   ❌ POOR - Significant quality problems")
    else:
        print("   ❌ FAILED - Audio is unintelligible")
    
    print("=" * 60 + "\n")


async def main():
    parser = argparse.ArgumentParser(description="Test Mumble audio quality")
    parser.add_argument("--text", "-t", help="Custom text to test with")
    parser.add_argument("--wav", "-w", help="Path to WAV file to use instead of TTS")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    results = await run_loopback_test(
        test_text=args.text,
        wav_path=args.wav,
    )
    
    print_results(results)
    
    # Exit with error code if quality is poor
    if results["similarity"] < 0.70:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
