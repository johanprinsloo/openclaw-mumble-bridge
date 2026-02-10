#!/usr/bin/env python3
"""Baseline debug test WITHOUT timing patches."""

from __future__ import annotations

import asyncio
import logging
import ssl
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

# DON'T apply timing fix - test baseline
from audio_utils import MUMBLE_SAMPLE_RATE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("debug_baseline")


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


def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


@dataclass
class DebugRecorder:
    total_bytes: int = 0
    packet_count: int = 0
    users_seen: dict = field(default_factory=dict)
    sequences: list = field(default_factory=list)
    recording: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def start(self):
        with self.lock:
            self.total_bytes = 0
            self.packet_count = 0
            self.users_seen = {}
            self.sequences = []
            self.recording = True
    
    def stop(self):
        with self.lock:
            self.recording = False
            return {
                "total_bytes": self.total_bytes,
                "packet_count": self.packet_count,
                "users_seen": dict(self.users_seen),
                "sequences": list(self.sequences[-50:]),
            }
    
    def on_sound(self, user, sound_chunk):
        with self.lock:
            if not self.recording:
                return
            
            username = user.get("name", "unknown") if isinstance(user, dict) else str(user)
            
            self.total_bytes += len(sound_chunk.pcm)
            self.packet_count += 1
            
            if username not in self.users_seen:
                self.users_seen[username] = {"bytes": 0, "packets": 0}
            self.users_seen[username]["bytes"] += len(sound_chunk.pcm)
            self.users_seen[username]["packets"] += 1
            
            self.sequences.append(sound_chunk.sequence)


async def run_test():
    patch_ssl()
    cfg = load_config()
    
    try:
        import pymumble_py3 as pymumble
    except ImportError:
        import pymumble3 as pymumble
    
    # Apply ONLY the bandwidth patch (needed to avoid crash)
    def safe_set_bandwidth(self):
        pass  # Skip bitrate setting
    pymumble.soundoutput.SoundOutput._set_bandwidth = safe_set_bandwidth
    
    print("📍 Running BASELINE test (no timing patches)")
    
    logger.info("Creating clients...")
    
    speaker = pymumble.Mumble(
        cfg["mumble"]["host"],
        "BaselineSpeaker",
        port=cfg["mumble"]["port"],
        reconnect=False,
    )
    
    listener = pymumble.Mumble(
        cfg["mumble"]["host"],
        "BaselineListener",
        port=cfg["mumble"]["port"],
        reconnect=False,
    )
    
    recorder = DebugRecorder()
    
    try:
        logger.info("Connecting listener...")
        listener.set_receive_sound(True)
        listener.callbacks.set_callback(
            pymumble.constants.PYMUMBLE_CLBK_SOUNDRECEIVED,
            recorder.on_sound,
        )
        listener.start()
        listener.is_ready()
        
        try:
            channel = listener.channels.find_by_name(cfg["mumble"].get("channel", "Root"))
            if channel:
                channel.move_in()
        except:
            pass
        
        logger.info("Connecting speaker...")
        speaker.start()
        speaker.is_ready()
        
        try:
            channel = speaker.channels.find_by_name(cfg["mumble"].get("channel", "Root"))
            if channel:
                channel.move_in()
        except:
            pass
        
        await asyncio.sleep(0.5)
        
        # Generate test tone
        import numpy as np
        duration = 3.0
        t = np.linspace(0, duration, int(MUMBLE_SAMPLE_RATE * duration), dtype=np.float32)
        tone = np.sin(2 * np.pi * 440 * t) * 0.3
        pcm_data = (tone * 32767).astype(np.int16).tobytes()
        
        expected_bytes = len(pcm_data)
        expected_packets = len(pcm_data) // 1920
        
        print(f"\n📤 Sending {expected_bytes:,} bytes ({duration}s, ~{expected_packets} packets)...")
        
        recorder.start()
        send_time = time.perf_counter()
        
        speaker.sound_output.add_sound(pcm_data)
        
        # Wait longer since baseline has 2x slowdown
        await asyncio.sleep(duration * 3 + 1.0)
        
        results = recorder.stop()
        elapsed = time.perf_counter() - send_time
        
        print(f"\n📊 BASELINE Results:")
        print(f"   Expected bytes:  {expected_bytes:,}")
        print(f"   Received bytes:  {results['total_bytes']:,}")
        print(f"   Ratio:           {results['total_bytes']/expected_bytes:.2f}x")
        print(f"   Expected pkts:   ~{expected_packets}")
        print(f"   Received pkts:   {results['packet_count']}")
        print(f"   Elapsed:         {elapsed:.2f}s")
        
        print(f"\n👥 Users seen:")
        for user, data in results["users_seen"].items():
            print(f"   {user}: {data['bytes']:,} bytes, {data['packets']} packets")
        
        if results["sequences"]:
            seqs = results["sequences"]
            if len(seqs) > 1:
                gaps = [seqs[i+1] - seqs[i] for i in range(len(seqs)-1)]
                avg_gap = sum(gaps) / len(gaps)
                print(f"\n🔢 Avg sequence gap: {avg_gap:.1f} (should be ~2 for 20ms packets)")
            print(f"   Last sequences: {seqs[-20:]}")
        
    finally:
        logger.info("Disconnecting...")
        try:
            speaker.stop()
        except:
            pass
        try:
            listener.stop()
        except:
            pass


if __name__ == "__main__":
    asyncio.run(run_test())
