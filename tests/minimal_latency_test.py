#!/usr/bin/env python3
"""
FINAL MINIMAL LATENCY TEST: 10ms Frames, Baseline Profile.

Combines the stable connection method with the timing fix and the lowest frame size (10ms).
"""

from __future__ import annotations

import asyncio
import logging
import ssl
import sys
import threading
import time
import statistics
from dataclasses import dataclass, field
from pathlib import Path

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
logger = logging.getLogger("final_minimal_test")

# --- Configuration Overrides ---
TEST_DURATION = 3.0
FRAME_SIZE_S = 0.010  # 10ms (Task 2: Test lower frames)
BANDWIDTH = 64000     # Decent bitrate
# OPUS_PROFILE is left to server default ("audio") for connection stability
# --- End Configuration ---


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
        # Standard SSL patch from previous attempts
        def wrap_socket(sock, keyfile=None, certfile=None, server_side=False, 
                       cert_reqs=ssl.CERT_NONE, ssl_version=ssl.PROTOCOL_TLS, 
                       ca_certs=None, do_handshake_on_connect=True, 
                       suppress_ragged_eofs=True, ciphers=None):
            context = ssl.SSLContext(ssl_version)
            if certfile: context.load_cert_chain(certfile, keyfile)
            if ca_certs: context.load_verify_locations(ca_certs)
            if cert_reqs: context.verify_mode = cert_reqs
            if ciphers: context.set_ciphers(ciphers)
            return context.wrap_socket(sock, server_side=server_side, 
                                       do_handshake_on_connect=do_handshake_on_connect, 
                                       suppress_ragged_eofs=suppress_ragged_eofs)
        ssl.wrap_socket = wrap_socket


def create_mumble_client_stable(username: str, config: TestConfig):
    """Create client using stable method, overriding frame size/bandwidth."""
    patch_ssl()
    
    try:
        import pymumble_py3 as pymumble
    except ImportError:
        import pymumble3 as pymumble
        
    # Override constants BEFORE client creation
    pymumble.constants.PYMUMBLE_AUDIO_PER_PACKET = FRAME_SIZE_S
    pymumble.constants.PYMUMBLE_BANDWIDTH = BANDWIDTH
    
    # Apply the safe bandwidth patch explicitly (must be done on soundoutput)
    def safe_set_bandwidth(self):
        if not self.encoder: return
        try:
            overhead_per_packet = 20 + (3 * int(self.audio_per_packet / self.encoder_framesize))
            if self.mumble_object.udp_active: overhead_per_packet += 12
            else: overhead_per_packet += 26
            overhead_per_second = int(overhead_per_packet * 8 / self.audio_per_packet)
            bitrate = self.bandwidth - overhead_per_second
            bitrate = max(6000, min(510000, bitrate))
            self.encoder.bitrate = bitrate
        except Exception:
            pass
    
    pymumble.soundoutput.SoundOutput._set_bandwidth = safe_set_bandwidth
    
    # Direct, stable instantiation
    client = pymumble.Mumble(
        config.mumble_host,
        username,
        port=config.mumble_port,
        reconnect=False,
    )
    return client


@dataclass
class TimingRecorder:
    # ... (TimingRecorder implementation remains the same) ...
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
            headers={"xi-api-key": config.elevenlabs_api_key, "Content-Type": "application/json"},
            json={"text": text, "model_id": "eleven_multilingual_v2", "output_format": "mp3_44100_128"},
        )
        response.raise_for_status()
        pcm_data = audio_to_mumble_pcm(response.content, "mp3")
    return pcm_data


async def run_test():
    test_config = load_config()
    
    try:
        import pymumble_py3 as pymumble
    except ImportError:
        import pymumble3 as pymumble
    
    logger.info(f"Testing Config: {FRAME_SIZE_S*1000:.0f}ms Frames, {BANDWIDTH/1000:.0f}kbps, Server Default Profile")
    
    speaker = create_mumble_client_stable("MinSpeaker", test_config)
    listener = create_mumble_client_stable("MinListener", test_config)
    
    recorder = TimingRecorder()
    
    try:
        logger.info("Connecting listener...")
        listener.set_receive_sound(True)
        listener.callbacks.set_callback(pymumble.constants.PYMUMBLE_CLBK_SOUNDRECEIVED, recorder.on_sound)
        listener.start()
        listener.is_ready()
        
        try:
            channel = listener.channels.find_by_name(test_config.mumble_channel)
            if channel: channel.move_in()
        except: pass
        
        logger.info("Connecting speaker...")
        speaker.start()
        speaker.is_ready()
        
        try:
            channel = speaker.channels.find_by_name(test_config.mumble_channel)
            if channel: channel.move_in()
        except: pass
        
        await asyncio.sleep(0.5)
        
        pcm_data = await generate_tts_audio(
            "Test 10 milliseconds frame latency check. One two three four.",
            test_config
        )
        
        expected_duration = len(pcm_data) / (MUMBLE_SAMPLE_RATE * 2)
        logger.info(f"Audio duration: {expected_duration:.2f}s, {len(pcm_data):,} bytes")
        
        recorder.start()
        send_start_time = time.perf_counter()
        
        speaker.sound_output.add_sound(pcm_data)
        
        wait_time = expected_duration * 1.5 + 1.0 
        logger.info(f"Waiting {wait_time:.1f}s...")
        await asyncio.sleep(wait_time)
        
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
        print(f"📊 FINAL 10ms TEST RESULTS (Stable Connection)")
        print("="*60)
        
        print(f"\n⏱️  Latency Metrics:")
        print(f"   First Packet Latency: {first_packet_latency*1000:.1f} ms")
        
        print(f"\n📐 Duration Drift:")
        print(f"   Expected: {expected_duration:.3f}s")
        print(f"   Actual:   {actual_duration:.3f}s")
        drift_icon = "✅" if abs(drift_percent) < 5 else "⚠️" if abs(drift_percent) < 15 else "❌"
        print(f"   Drift:    {drift_percent:+.1f}% {drift_icon}")
        
        print(f"\n🔀 Jitter Metrics:")
        print(f"   Mean Inter-Packet: {mean_inter_packet_ms:.2f} ms")
        print(f"   Jitter (stddev):   {jitter_ms:.2f} ms")
        print(f"   Max Gap:           {max_gap_ms:.2f} ms")
        
        print(f"\n📦 Data:")
        print(f"   Sent:     {len(pcm_data):,} bytes")
        print(f"   Received: {timing_data['total_bytes']:,} bytes")
        print(f"   Packets:  {len(timing_data['packet_times'])}")
        
        print("\n" + "="*60)
        
        # Overall assessment
        if abs(drift_percent) < 5 and first_packet_latency < 0.15:
            print("🏆 OPTIMAL CONFIG FOUND!")
        elif abs(drift_percent) < 15:
            print("✅ Good performance achieved.")
        else:
            print("⚠️  Timing issue remains.")
        
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
