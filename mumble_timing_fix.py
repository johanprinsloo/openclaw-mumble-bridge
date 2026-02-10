"""
Pymumble timing fix module.

The original pymumble soundoutput.send_audio() has a timing bug that causes
audio to play back at approximately 50% speed. The bug is:

1. After sending a packet, sequence_last_time is set to:
   sequence_start_time + (sequence * SEQUENCE_DURATION)
   
2. sequence increments by audio_per_packet / SEQUENCE_DURATION (e.g., 2 for 20ms packets)

3. The loop condition checks: sequence_last_time + audio_per_packet <= time()

4. This means: (start + seq*0.01) + 0.02 must be <= time()

5. After packet 1 (seq=2): need start + 0.02 + 0.02 <= time, i.e., 40ms!

6. The next packet waits 40ms instead of 20ms, causing 2x slowdown.

FIX: Use real-time tracking instead of sequence-based timing.
"""

import struct
import threading
from time import time
import socket

import opuslib


def patch_pymumble_timing():
    """
    Apply the timing fix to pymumble's SoundOutput class.
    Call this BEFORE creating any Mumble clients.
    """
    try:
        import pymumble_py3 as pymumble
    except ImportError:
        import pymumble3 as pymumble
    
    from pymumble_py3.constants import (
        PYMUMBLE_SAMPLERATE,
        PYMUMBLE_SEQUENCE_DURATION,
        PYMUMBLE_SEQUENCE_RESET_INTERVAL,
        PYMUMBLE_AUDIO_TYPE_OPUS,
        PYMUMBLE_MSG_TYPES_UDPTUNNEL,
    )
    from pymumble_py3.tools import VarInt
    
    def fixed_send_audio(self):
        """
        Fixed version of send_audio that uses proper real-time pacing.
        
        The original bug: sequence_last_time was set to (start + seq * 0.01), which
        advances by 20ms per 20ms packet. The check then adds ANOTHER audio_per_packet,
        causing 40ms delays between 20ms packets.
        
        This fix uses a simpler approach: track the "next scheduled send time" and
        send when that time arrives.
        """
        if not self.encoder or len(self.pcm) == 0:
            return ()
        
        samples = int(self.encoder_framesize * PYMUMBLE_SAMPLERATE * 2 * self.channels)
        current_time = time()
        
        # Initialize or reset timing if we've been idle
        if self.sequence_last_time == 0 or \
           self.sequence_last_time + PYMUMBLE_SEQUENCE_RESET_INTERVAL <= current_time:
            self.sequence = 0
            self.sequence_start_time = current_time
            self.sequence_last_time = current_time  # Next send is NOW
        
        # Send packets that are due (scheduled time has passed)
        while len(self.pcm) > 0 and self.sequence_last_time <= current_time:
            payload = bytearray()
            audio_encoded = 0
            
            while len(self.pcm) > 0 and audio_encoded < self.audio_per_packet:
                self.lock.acquire()
                to_encode = self.pcm.pop(0)
                self.lock.release()
                
                if len(to_encode) != samples:
                    to_encode += b'\x00' * (samples - len(to_encode))
                
                try:
                    encoded = self.encoder.encode(to_encode, len(to_encode) // (2 * self.channels))
                except opuslib.exceptions.OpusError:
                    encoded = b''
                
                audio_encoded += self.encoder_framesize
                
                if self.codec_type == PYMUMBLE_AUDIO_TYPE_OPUS:
                    frameheader = VarInt(len(encoded)).encode()
                else:
                    frameheader = len(encoded)
                    if audio_encoded < self.audio_per_packet and len(self.pcm) > 0:
                        frameheader += (1 << 7)
                    frameheader = struct.pack('!B', frameheader)
                
                payload += frameheader + encoded
            
            header = self.codec_type << 5
            sequence = VarInt(self.sequence).encode()
            
            udppacket = struct.pack('!B', header | self.target) + sequence + payload
            if self.mumble_object.positional:
                udppacket += struct.pack("fff", *self.mumble_object.positional)
            
            self.Log.debug("audio packet to send: sequence:{sequence}, type:{type}, length:{len}".format(
                sequence=self.sequence,
                type=self.codec_type,
                len=len(udppacket)
            ))
            
            tcppacket = struct.pack("!HL", PYMUMBLE_MSG_TYPES_UDPTUNNEL, len(udppacket)) + udppacket
            
            while len(tcppacket) > 0:
                sent = self.mumble_object.control_socket.send(tcppacket)
                if sent < 0:
                    raise socket.error("Server socket error")
                tcppacket = tcppacket[sent:]
            
            # Advance sequence and schedule next send
            self.sequence += int(self.audio_per_packet / PYMUMBLE_SEQUENCE_DURATION)
            self.sequence_last_time += self.audio_per_packet
            
            current_time = time()
    
    # Apply the patch
    pymumble.soundoutput.SoundOutput.send_audio = fixed_send_audio
    print("[mumble_timing_fix] Applied timing fix to pymumble SoundOutput.send_audio")


def patch_bandwidth_safe():
    """
    Apply a safe bandwidth patch that won't crash on recent opuslib versions.
    """
    try:
        import pymumble_py3 as pymumble
    except ImportError:
        import pymumble3 as pymumble
    
    def safe_set_bandwidth(self):
        """Safely set encoder bitrate, catching exceptions."""
        if not self.encoder:
            return
        try:
            overhead_per_packet = 20 + (3 * int(self.audio_per_packet / self.encoder_framesize))
            if self.mumble_object.udp_active:
                overhead_per_packet += 12
            else:
                overhead_per_packet += 26
            overhead_per_second = int(overhead_per_packet * 8 / self.audio_per_packet)
            bitrate = self.bandwidth - overhead_per_second
            bitrate = max(6000, min(510000, bitrate))  # Clamp to Opus limits
            self.encoder.bitrate = bitrate
        except Exception:
            pass  # Use encoder defaults
    
    pymumble.soundoutput.SoundOutput._set_bandwidth = safe_set_bandwidth
    print("[mumble_timing_fix] Applied safe bandwidth patch")


def apply_all_patches():
    """Apply all pymumble patches."""
    patch_pymumble_timing()
    patch_bandwidth_safe()


if __name__ == "__main__":
    # Test that patches can be applied
    apply_all_patches()
    print("All patches applied successfully!")
