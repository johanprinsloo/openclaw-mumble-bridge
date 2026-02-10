#!/usr/bin/env python3
"""
Audio Latency & Drift Test Harness

Measures:
1. Latency: Time from first packet sent to first packet received
2. Duration Drift: Does 5.0s of sent audio take 5.5s to receive?
3. Jitter: Variance in inter-packet arrival times

Tests different configurations:
- Frame sizes: 10ms vs 20ms
- Bitrates: 32kbps, 64kbps, 96kbps
- Opus profiles: voip vs audio

Usage:
    python tests/audio_latency_test.py [--config CONFIG_NAME]
"""

from __future__ import annotations

import argparse
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

from audio_utils import audio_to_mumble_pcm, MUMBLE_SAMPLE_RATE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("audio_latency_test")


@dataclass
class TestConfig:
    """Test configuration loaded from config.yaml."""
    mumble_host: str
    mumble_port: int
    mumble_channel: str
    elevenlabs_api_key: str
    tts_voice_id: str


@dataclass
class AudioConfig:
    """Audio encoding configuration to test."""
    name: str
    audio_per_packet: float  # seconds (0.010 = 10ms, 0.020 = 20ms)
    bandwidth: int  # bits per second
    opus_profile: str  # "voip" or "audio"


# Configurations to test
TEST_CONFIGS = [
    AudioConfig("baseline_voip_20ms_50k", 0.020, 50000, "voip"),
    AudioConfig("voip_10ms_50k", 0.010, 50000, "voip"),
    AudioConfig("voip_20ms_64k", 0.020, 64000, "voip"),
    AudioConfig("voip_20ms_96k", 0.020, 96000, "voip"),
    AudioConfig("voip_10ms_64k", 0.010, 64000, "voip"),
    AudioConfig("audio_20ms_64k", 0.020, 64000, "audio"),
]


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
    )


def patch_ssl():
    """Monkey-patch ssl.wrap_socket for Python 3.12+ compatibility."""
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


def create_mumble_client(username: str, config: TestConfig, audio_cfg: AudioConfig):
    """Create a pymumble client with specific audio configuration."""
    patch_ssl()
    
    try:
        import pymumble_py3 as pymumble
    except ImportError:
        import pymumble3 as pymumble
    
    # Patch _set_bandwidth to handle high bitrates without errors
    def patched_set_bandwidth(self_so):
        if not getattr(self_so, 'encoder', None):
            return
        try:
            overhead_per_packet = 20 + (3 * int(self_so.audio_per_packet / self_so.encoder_framesize))
            if self_so.mumble_object.udp_active:
                overhead_per_packet += 12
            else:
                overhead_per_packet += 26
            overhead_per_second = int(overhead_per_packet * 8 / self_so.audio_per_packet)
            bitrate = self_so.bandwidth - overhead_per_second
            # Clamp to Opus limits (6000 - 510000)
            bitrate = max(6000, min(510000, bitrate))
            self_so.encoder.bitrate = bitrate
            logger.debug(f"Set encoder bitrate to {bitrate}")
        except Exception as e:
            logger.debug(f"Bitrate setting failed (using default): {e}")
    
    pymumble.soundoutput.SoundOutput._set_bandwidth = patched_set_bandwidth
    
    # Set opus profile via constants
    pymumble.constants.PYMUMBLE_AUDIO_TYPE_OPUS_PROFILE = audio_cfg.opus_profile
    pymumble.constants.PYMUMBLE_AUDIO_PER_PACKET = audio_cfg.audio_per_packet
    pymumble.constants.PYMUMBLE_BANDWIDTH = audio_cfg.bandwidth
    
    client = pymumble.Mumble(
        config.mumble_host,
        username,
        port=config.mumble_port,
        reconnect=False,
    )
    return client


@dataclass
class TimingRecorder:
    """Records timing information for received audio packets."""
    
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


@dataclass
class LatencyResults:
    """Results from a latency test run."""
    config_name: str
    
    # Latency metrics (seconds)
    first_packet_latency: float = 0.0
    
    # Duration drift
    expected_duration: float = 0.0
    actual_duration: float = 0.0
    drift_percent: float = 0.0
    
    # Jitter metrics
    mean_inter_packet_ms: float = 0.0
    jitter_ms: float = 0.0  # stddev of inter-packet times
    max_gap_ms: float = 0.0
    
    # Data metrics
    bytes_sent: int = 0
    bytes_received: int = 0
    packet_count: int = 0
    
    error: Optional[str] = None


async def generate_test_tone(duration_s: float = 5.0) -> bytes:
    """Generate a simple tone for testing (440Hz sine wave)."""
    import numpy as np
    
    # Generate 440Hz sine wave
    t = np.linspace(0, duration_s, int(MUMBLE_SAMPLE_RATE * duration_s), dtype=np.float32)
    tone = np.sin(2 * np.pi * 440 * t) * 0.5  # 440Hz at 50% volume
    
    # Convert to 16-bit PCM
    pcm = (tone * 32767).astype(np.int16).tobytes()
    return pcm


async def generate_tts_audio(text: str, config: TestConfig) -> bytes:
    """Generate TTS audio using ElevenLabs."""
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


async def run_latency_test(
    audio_cfg: AudioConfig,
    test_config: Optional[TestConfig] = None,
    use_tts: bool = False,
    test_duration: float = 5.0,
) -> LatencyResults:
    """
    Run a latency test with the specified audio configuration.
    """
    if test_config is None:
        test_config = load_config()
    
    results = LatencyResults(config_name=audio_cfg.name)
    
    try:
        import pymumble_py3 as pymumble
    except ImportError:
        import pymumble3 as pymumble
    
    # Create clients with specific config
    logger.info(f"Testing config: {audio_cfg.name}")
    speaker = create_mumble_client("LatencySpeaker", test_config, audio_cfg)
    listener = create_mumble_client("LatencyListener", test_config, audio_cfg)
    
    recorder = TimingRecorder()
    
    try:
        # Connect listener first
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
        
        await asyncio.sleep(0.5)  # Let clients settle
        
        # Generate audio
        if use_tts:
            pcm_data = await generate_tts_audio(
                "Testing one two three four five. The quick brown fox jumps.",
                test_config
            )
        else:
            pcm_data = await generate_test_tone(test_duration)
        
        results.bytes_sent = len(pcm_data)
        results.expected_duration = len(pcm_data) / (MUMBLE_SAMPLE_RATE * 2)
        
        logger.info(f"Audio duration: {results.expected_duration:.2f}s, {results.bytes_sent:,} bytes")
        
        # Start recording and capture send time
        recorder.start()
        send_start_time = time.perf_counter()
        
        # Send audio
        speaker.sound_output.add_sound(pcm_data)
        
        # Wait for playback to complete + buffer
        wait_time = results.expected_duration * 1.5 + 2.0
        logger.info(f"Waiting {wait_time:.1f}s for audio...")
        await asyncio.sleep(wait_time)
        
        # Stop recording and gather results
        timing_data = recorder.stop()
        
        results.bytes_received = timing_data["total_bytes"]
        results.packet_count = len(timing_data["packet_times"])
        
        if timing_data["first_packet_time"] is None:
            results.error = "No audio received"
            return results
        
        # Calculate latency (time from send start to first packet received)
        results.first_packet_latency = timing_data["first_packet_time"] - send_start_time
        
        # Calculate duration drift
        results.actual_duration = timing_data["last_packet_time"] - timing_data["first_packet_time"]
        if results.expected_duration > 0:
            results.drift_percent = ((results.actual_duration - results.expected_duration) / results.expected_duration) * 100
        
        # Calculate jitter
        if len(timing_data["packet_times"]) > 1:
            inter_packet_times = []
            for i in range(1, len(timing_data["packet_times"])):
                gap = timing_data["packet_times"][i] - timing_data["packet_times"][i-1]
                inter_packet_times.append(gap * 1000)  # Convert to ms
            
            results.mean_inter_packet_ms = statistics.mean(inter_packet_times)
            results.jitter_ms = statistics.stdev(inter_packet_times) if len(inter_packet_times) > 1 else 0
            results.max_gap_ms = max(inter_packet_times)
        
        return results
        
    except Exception as e:
        results.error = str(e)
        logger.exception("Test failed")
        return results
        
    finally:
        logger.info("Disconnecting clients...")
        try:
            speaker.stop()
        except Exception:
            pass
        try:
            listener.stop()
        except Exception:
            pass


def print_results(results: LatencyResults):
    """Print test results in a nice format."""
    print(f"\n{'='*60}")
    print(f"📊 {results.config_name}")
    print(f"{'='*60}")
    
    if results.error:
        print(f"❌ Error: {results.error}")
        return
    
    print(f"\n⏱️  Latency Metrics:")
    print(f"   First Packet Latency: {results.first_packet_latency*1000:.1f} ms")
    
    print(f"\n📐 Duration Drift:")
    print(f"   Expected: {results.expected_duration:.3f}s")
    print(f"   Actual:   {results.actual_duration:.3f}s")
    drift_icon = "✅" if abs(results.drift_percent) < 5 else "⚠️" if abs(results.drift_percent) < 10 else "❌"
    print(f"   Drift:    {results.drift_percent:+.1f}% {drift_icon}")
    
    print(f"\n🔀 Jitter Metrics:")
    print(f"   Mean Inter-Packet: {results.mean_inter_packet_ms:.2f} ms")
    print(f"   Jitter (stddev):   {results.jitter_ms:.2f} ms")
    print(f"   Max Gap:           {results.max_gap_ms:.2f} ms")
    
    print(f"\n📦 Data:")
    print(f"   Sent:     {results.bytes_sent:,} bytes")
    print(f"   Received: {results.bytes_received:,} bytes")
    print(f"   Packets:  {results.packet_count}")
    
    # Quality assessment
    print(f"\n🎯 Assessment:")
    issues = []
    if results.first_packet_latency > 0.5:
        issues.append(f"High latency ({results.first_packet_latency*1000:.0f}ms)")
    if abs(results.drift_percent) > 5:
        issues.append(f"Duration drift ({results.drift_percent:+.1f}%)")
    if results.jitter_ms > 10:
        issues.append(f"High jitter ({results.jitter_ms:.1f}ms)")
    if results.max_gap_ms > 100:
        issues.append(f"Large gaps ({results.max_gap_ms:.0f}ms)")
    
    if not issues:
        print("   ✅ All metrics within acceptable range!")
    else:
        for issue in issues:
            print(f"   ⚠️  {issue}")


def print_comparison_table(all_results: list[LatencyResults]):
    """Print a comparison table of all results."""
    print("\n" + "="*100)
    print("📊 COMPARISON TABLE")
    print("="*100)
    print(f"{'Config':<25} {'Latency':<12} {'Drift':<10} {'Jitter':<10} {'Max Gap':<10} {'Status'}")
    print("-"*100)
    
    for r in all_results:
        if r.error:
            print(f"{r.config_name:<25} {'ERROR':<12}")
            continue
        
        latency = f"{r.first_packet_latency*1000:.0f}ms"
        drift = f"{r.drift_percent:+.1f}%"
        jitter = f"{r.jitter_ms:.1f}ms"
        max_gap = f"{r.max_gap_ms:.0f}ms"
        
        # Determine status
        issues = 0
        if r.first_packet_latency > 0.3:
            issues += 1
        if abs(r.drift_percent) > 5:
            issues += 1
        if r.jitter_ms > 10:
            issues += 1
        
        status = "✅" if issues == 0 else "⚠️" if issues == 1 else "❌"
        
        print(f"{r.config_name:<25} {latency:<12} {drift:<10} {jitter:<10} {max_gap:<10} {status}")
    
    print("="*100)


async def run_single_config(config_name: str):
    """Run test for a single configuration."""
    test_config = load_config()
    
    for cfg in TEST_CONFIGS:
        if cfg.name == config_name:
            results = await run_latency_test(cfg, test_config, use_tts=True)
            print_results(results)
            return results
    
    print(f"Config '{config_name}' not found. Available configs:")
    for cfg in TEST_CONFIGS:
        print(f"  - {cfg.name}")
    return None


async def run_all_configs():
    """Run tests for all configurations."""
    test_config = load_config()
    all_results = []
    
    for cfg in TEST_CONFIGS:
        results = await run_latency_test(cfg, test_config, use_tts=True)
        print_results(results)
        all_results.append(results)
        
        # Wait between tests to let server reset
        await asyncio.sleep(2.0)
    
    print_comparison_table(all_results)
    
    # Find best config
    valid_results = [r for r in all_results if r.error is None]
    if valid_results:
        # Score: lower latency + lower drift + lower jitter is better
        def score(r):
            return r.first_packet_latency + abs(r.drift_percent)/100 + r.jitter_ms/1000
        
        best = min(valid_results, key=score)
        print(f"\n🏆 Best Configuration: {best.config_name}")
        print(f"   Latency: {best.first_packet_latency*1000:.0f}ms, Drift: {best.drift_percent:+.1f}%, Jitter: {best.jitter_ms:.1f}ms")
    
    return all_results


async def main():
    parser = argparse.ArgumentParser(description="Test Mumble audio latency & drift")
    parser.add_argument("--config", "-c", help="Run single config by name")
    parser.add_argument("--list", "-l", action="store_true", help="List available configs")
    parser.add_argument("--all", "-a", action="store_true", help="Run all configs")
    args = parser.parse_args()
    
    if args.list:
        print("Available configurations:")
        for cfg in TEST_CONFIGS:
            print(f"  {cfg.name}: {cfg.audio_per_packet*1000:.0f}ms frames, {cfg.bandwidth/1000:.0f}kbps, {cfg.opus_profile}")
        return
    
    if args.config:
        await run_single_config(args.config)
    elif args.all:
        await run_all_configs()
    else:
        # Default: run baseline only
        await run_single_config("baseline_voip_20ms_50k")


if __name__ == "__main__":
    asyncio.run(main())
