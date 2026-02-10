#!/usr/bin/env python3
"""
Mumble Timing Diagnostic Test

Investigates the pacing of audio packets in pymumble to diagnose
"slow motion" or "choppy" audio issues.

This script:
1. Connects to Mumble
2. Monitors the actual timing of packet transmission
3. Reports statistics on pacing accuracy

Usage:
    python tests/mumble_timing_test.py
"""

from __future__ import annotations

import logging
import os
import ssl
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("mumble_timing")


def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def patch_ssl():
    """Patch ssl.wrap_socket for Python 3.12+ compatibility."""
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


def create_instrumented_soundoutput():
    """Create a wrapper around SoundOutput that logs timing information."""
    try:
        import pymumble_py3 as pymumble
    except ImportError:
        import pymumble3 as pymumble
    
    original_send_audio = pymumble.soundoutput.SoundOutput.send_audio
    
    packet_times = []
    
    def instrumented_send_audio(self):
        """Wrapper that logs packet transmission times."""
        before_len = len(self.pcm)
        start_time = time.time()
        
        original_send_audio(self)
        
        after_len = len(self.pcm)
        end_time = time.time()
        
        packets_sent = before_len - after_len
        if packets_sent > 0:
            packet_times.append({
                'time': end_time,
                'packets': packets_sent,
                'buffer_remaining': after_len,
                'sequence': self.sequence,
            })
            logger.debug(f"Sent {packets_sent} packets, buffer: {after_len}, seq: {self.sequence}")
    
    pymumble.soundoutput.SoundOutput.send_audio = instrumented_send_audio
    
    return packet_times


def main():
    patch_ssl()
    
    try:
        import pymumble_py3 as pymumble
    except ImportError:
        import pymumble3 as pymumble
    
    # Load config
    cfg = load_config()
    mcfg = cfg.get("mumble", {})
    
    logger.info("=== Mumble Timing Diagnostic ===")
    logger.info(f"Connecting to {mcfg['host']}:{mcfg['port']}...")
    
    # Create instrumented client
    packet_times = create_instrumented_soundoutput()
    
    client = pymumble.Mumble(
        mcfg['host'],
        "TimingTest",
        port=mcfg['port'],
        reconnect=False,
    )
    
    try:
        client.start()
        client.is_ready()
        
        # Join channel
        try:
            channel = client.channels.find_by_name(mcfg.get('channel', 'Root'))
            if channel:
                channel.move_in()
        except Exception:
            pass
        
        logger.info("Connected! Checking audio settings...")
        
        # Report current settings
        so = client.sound_output
        logger.info(f"Audio per packet: {so.audio_per_packet * 1000:.1f}ms")
        logger.info(f"Encoder framesize: {so.encoder_framesize}")
        logger.info(f"Bandwidth: {so.bandwidth}")
        
        if so.encoder:
            logger.info(f"Encoder application: {so.encoder.application}")
        
        # Generate test audio (sine wave)
        logger.info("\nGenerating 3 seconds of test audio...")
        import struct
        import math
        
        sample_rate = 48000
        duration_s = 3.0
        freq = 440  # A4 note
        
        samples = []
        for i in range(int(sample_rate * duration_s)):
            sample = int(16000 * math.sin(2 * math.pi * freq * i / sample_rate))
            samples.append(sample)
        
        pcm_data = struct.pack('<' + 'h' * len(samples), *samples)
        
        # Send audio
        logger.info(f"Sending {len(pcm_data)} bytes ({duration_s}s of audio)...")
        
        frame_size = 1920  # 20ms at 48kHz mono
        start_send = time.time()
        offset = 0
        frames_added = 0
        
        while offset < len(pcm_data):
            chunk = pcm_data[offset:offset + frame_size]
            if len(chunk) < frame_size:
                chunk = chunk + b'\x00' * (frame_size - len(chunk))
            so.add_sound(chunk)
            frames_added += 1
            offset += frame_size
        
        add_time = time.time() - start_send
        logger.info(f"Added {frames_added} frames in {add_time*1000:.1f}ms")
        
        # Wait for playback to complete
        logger.info("Waiting for playback...")
        wait_start = time.time()
        
        while so.get_buffer_size() > 0:
            time.sleep(0.01)
            if time.time() - wait_start > duration_s + 5:
                logger.warning("Timeout waiting for buffer to drain!")
                break
        
        total_time = time.time() - start_send
        
        # Analyze timing
        logger.info("\n=== Timing Analysis ===")
        logger.info(f"Expected duration: {duration_s:.1f}s")
        logger.info(f"Actual duration: {total_time:.2f}s")
        logger.info(f"Ratio: {total_time / duration_s:.2f}x")
        
        if packet_times:
            intervals = []
            for i in range(1, len(packet_times)):
                interval = packet_times[i]['time'] - packet_times[i-1]['time']
                intervals.append(interval)
            
            if intervals:
                avg_interval = sum(intervals) / len(intervals)
                min_interval = min(intervals)
                max_interval = max(intervals)
                
                logger.info(f"\nPacket Intervals:")
                logger.info(f"  Count: {len(intervals)}")
                logger.info(f"  Average: {avg_interval*1000:.2f}ms (expected: 20ms)")
                logger.info(f"  Min: {min_interval*1000:.2f}ms")
                logger.info(f"  Max: {max_interval*1000:.2f}ms")
                
                if avg_interval > 0.025:
                    logger.warning("⚠️  Average interval > 25ms - packets being sent too slowly!")
                elif avg_interval < 0.015:
                    logger.warning("⚠️  Average interval < 15ms - packets may be bunched up!")
                else:
                    logger.info("✅ Packet pacing looks normal")
        
        if total_time > duration_s * 1.1:
            logger.warning("⚠️  Audio played SLOWER than expected - this causes 'slow motion' effect!")
        elif total_time < duration_s * 0.9:
            logger.warning("⚠️  Audio played FASTER than expected!")
        else:
            logger.info("✅ Overall timing looks correct!")
    
    except Exception as e:
        logger.error(f"Failed to connect: {e}")
        logger.info("\nEnsure the Mumble server is running on the configured host/port.")
        return 1
    
    finally:
        try:
            client.stop()
        except Exception:
            pass
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
