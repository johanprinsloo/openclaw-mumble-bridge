#!/usr/bin/env python3
"""
Audio Latency Test with Timing Fix Applied

Tests the timing fix that corrects the ~2x slowdown in pymumble.
"""

from __future__ import annotations

import asyncio
import logging
import ssl
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

# Apply timing fix BEFORE importing pymumble
from mumble_timing_fix import apply_all_patches
apply_all_patches()

from audio_utils import audio_to_mumble_pcm, MUMBLE_SAMPLE_RATE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("audio_latency_test_fixed")


@dataclass
class TestConfig:
    mumble_host: str
    mumble_port: int
    mumble_channel: str
    elevenlabs_api_key: str
    tts_voice_id: str


def load_config() -> TestConfig:
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    return TestConfig(
        mumble_host=cfg["mumble"]["host"],
        mumble_port=cfg["mumble"]["port"],
        mumble_channel=cfg["mumble"].get("channel", "Root"),
        elevenlabs_api_key=cfg["tts"]["elevenlabs_api_key"],
        tts_voice_id=cfg["tts"].get("elevenlabs_voice_id", "pFZP5JQG7iQjIQuC4Bku"),
    )


def patch_ssl():
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


def create_mumble_client(username: str, config: TestConfig):
    patch_ssl()
    
    try:
        import pymumble_py3 as pymumble
    except ImportError:
        import pymumble3 as pymumble
    
    # Set voip profile for lower latency
    pymumble.constants.PYMUMBLE_AUDIO_TYPE_OPUS_PROFILE = "voip"
    
    client = pymumble.Mumble(
        config.mumble_host,
        username,
        port=config.mumble_port,
        reconnect=False,
    )
    return client


@dataclass
class TimingRecorder:
    first_packet_time: Optional[float] = None
    last_packet_time: Optional[float] = None
    packet_times: list = field(default_factory=list)
    total_bytes: int = 0
    recording: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def start(self):
        with self.lock:
            self.first_packet_time = None
            self.last_packet_time = None
            self.packet_times = []
            self.total_bytes = 0
            self.recording = True
    
    def stop(self) -> dict:
        with self.lock:
            self.recording = False
            return {
                "first_packet_time": self.first_packet_time,
                "last_packet_time": self.last_packet_time,
                "packet_times": list(self.packet_times),
                "total_bytes": self.total_bytes,
            }
    
    def on_sound(self, user, sound_chunk):
        now = time.perf_counter()
        with self.lock:
            if not self.recording:
                return
            if self.first_packet_time is None:
                self.first_packet_time = now
            self.last_packet_time = now
            self.packet_times.append(now)
            self.total_bytes += len(sound_chunk.pcm)


async def generate_tts_audio(text: str, config: TestConfig) -> bytes:
    import httpx
    
    logger.info(f"Generating TTS: {text[:30]}...")
    
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
    
    pcm_data = audio_to_mumble_pcm(mp3_data, "mp3")
    return pcm_data


async def run_test():
    test_config = load_config()
    
    try:
        import pymumble_py3 as pymumble
    except ImportError:
        import pymumble3 as pymumble
    
    logger.info("Creating Mumble clients with timing fix applied...")
    speaker = create_mumble_client("FixedSpeaker", test_config)
    listener = create_mumble_client("FixedListener", test_config)
    
    recorder = TimingRecorder()
    
    try:
        # Connect listener
        logger.info("Connecting listener...")
        listener.set_receive_sound(True)
        listener.callbacks.set_callback(
            pymumble.constants.PYMUMBLE_CLBK_SOUNDRECEIVED,
            recorder.on_sound,
        )
        listener.start()
        listener.is_ready()
        
        try:
            channel = listener.channels.find_by_name(test_config.mumble_channel)
            if channel:
                channel.move_in()
        except Exception:
            pass
        
        # Connect speaker
        logger.info("Connecting speaker...")
        speaker.start()
        speaker.is_ready()
        
        try:
            channel = speaker.channels.find_by_name(test_config.mumble_channel)
            if channel:
                channel.move_in()
        except Exception:
            pass
        
        await asyncio.sleep(0.5)
        
        # Generate TTS audio
        pcm_data = await generate_tts_audio(
            "Testing one two three four five. The quick brown fox jumps over the lazy dog.",
            test_config
        )
        
        expected_duration = len(pcm_data) / (MUMBLE_SAMPLE_RATE * 2)
        logger.info(f"Audio duration: {expected_duration:.2f}s, {len(pcm_data):,} bytes")
        
        # Start recording and send
        recorder.start()
        send_start_time = time.perf_counter()
        
        speaker.sound_output.add_sound(pcm_data)
        
        # Wait for playback
        wait_time = expected_duration * 1.5 + 2.0
        logger.info(f"Waiting {wait_time:.1f}s...")
        await asyncio.sleep(wait_time)
        
        # Gather results
        timing_data = recorder.stop()
        
        if timing_data["first_packet_time"] is None:
            print("\n❌ No audio received!")
            return
        
        # Calculate metrics
        first_packet_latency = timing_data["first_packet_time"] - send_start_time
        actual_duration = timing_data["last_packet_time"] - timing_data["first_packet_time"]
        drift_percent = ((actual_duration - expected_duration) / expected_duration) * 100
        
        # Calculate jitter
        if len(timing_data["packet_times"]) > 1:
            inter_packet_times = []
            for i in range(1, len(timing_data["packet_times"])):
                gap = timing_data["packet_times"][i] - timing_data["packet_times"][i-1]
                inter_packet_times.append(gap * 1000)
            
            mean_inter_packet_ms = statistics.mean(inter_packet_times)
            jitter_ms = statistics.stdev(inter_packet_times) if len(inter_packet_times) > 1 else 0
            max_gap_ms = max(inter_packet_times)
        else:
            mean_inter_packet_ms = jitter_ms = max_gap_ms = 0
        
        # Print results
        print("\n" + "="*60)
        print("📊 TIMING FIX TEST RESULTS")
        print("="*60)
        
        print(f"\n⏱️  Latency:")
        print(f"   First Packet: {first_packet_latency*1000:.1f} ms")
        
        print(f"\n📐 Duration Drift:")
        print(f"   Expected: {expected_duration:.3f}s")
        print(f"   Actual:   {actual_duration:.3f}s")
        drift_icon = "✅" if abs(drift_percent) < 5 else "⚠️" if abs(drift_percent) < 15 else "❌"
        print(f"   Drift:    {drift_percent:+.1f}% {drift_icon}")
        
        print(f"\n🔀 Jitter:")
        print(f"   Mean Inter-Packet: {mean_inter_packet_ms:.2f} ms")
        print(f"   Jitter (stddev):   {jitter_ms:.2f} ms")
        print(f"   Max Gap:           {max_gap_ms:.2f} ms")
        
        print(f"\n📦 Data:")
        print(f"   Sent:     {len(pcm_data):,} bytes")
        print(f"   Received: {timing_data['total_bytes']:,} bytes")
        print(f"   Packets:  {len(timing_data['packet_times'])}")
        
        print("\n" + "="*60)
        
        # Overall assessment
        if abs(drift_percent) < 5 and first_packet_latency < 0.3:
            print("✅ TIMING FIX SUCCESSFUL!")
        elif abs(drift_percent) < 15:
            print("⚠️  Improved but some drift remains")
        else:
            print("❌ Fix did not resolve the timing issue")
        
        print("="*60 + "\n")
        
    finally:
        logger.info("Disconnecting...")
        try:
            speaker.stop()
        except Exception:
            pass
        try:
            listener.stop()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(run_test())
