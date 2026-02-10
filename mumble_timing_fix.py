import logging
import time
import struct
import socket
import opuslib
from pymumble_py3.constants import *
from pymumble_py3.tools import VarInt

logger = logging.getLogger("mumble_timing_fix")

def apply_all_patches():
    try:
        import pymumble_py3 as pymumble
    except ImportError:
        import pymumble3 as pymumble

    logger.info("Applying pymumble timing and bandwidth patches...")

    # 1. Fix Opus Bitrate Setter Crash
    def patched_set_bandwidth(self_so):
        # Do nothing to avoid 'invalid argument' error on encoder.bitrate setter
        pass
    
    pymumble.soundoutput.SoundOutput._set_bandwidth = patched_set_bandwidth

    # 2. Fix Timing Drift in send_audio
    # Original implementation uses time.time() which can drift or cause gaps.
    # We replace it with a version that tracks 'audio time' strictly.
    
    def patched_send_audio(self):
        """Send the available audio to the server, with fixed timing logic."""
        if not self.encoder or len(self.pcm) == 0:
            return

        samples = int(self.encoder_framesize * PYMUMBLE_SAMPLERATE * 2 * self.channels)

        current_time = time.time()

        # Initialize timing if needed or if reset required
        if self.sequence_last_time == 0 or (current_time - self.sequence_last_time > 5.0):
            self.sequence = 0
            self.sequence_start_time = current_time
            self.sequence_last_time = current_time - self.audio_per_packet

        # Send loop
        while len(self.pcm) > 0:
            # Pacing check: Don't send if we are too far ahead of real time
            # Allow 60ms jitter buffer (3 frames)
            if self.sequence_last_time + self.audio_per_packet > current_time + 0.06:
                break

            # Update sequence time based on AUDIO duration (no drift)
            self.sequence_last_time += self.audio_per_packet
            
            # Increment sequence number
            # 20ms packet = 2 * 10ms steps (assuming SEQUENCE_DURATION is 10ms)
            step = int(self.audio_per_packet / PYMUMBLE_SEQUENCE_DURATION)
            self.sequence += step

            payload = bytearray()
            audio_encoded = 0

            # Construct packet (same as original)
            while len(self.pcm) > 0 and audio_encoded < self.audio_per_packet:
                self.lock.acquire()
                if len(self.pcm) > 0:
                    to_encode = self.pcm.pop(0)
                else:
                    to_encode = None
                self.lock.release()
                
                if to_encode is None:
                    break

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
                udppacket += struct.pack("fff", self.mumble_object.positional[0], self.mumble_object.positional[1], self.mumble_object.positional[2])

            tcppacket = struct.pack("!HL", PYMUMBLE_MSG_TYPES_UDPTUNNEL, len(udppacket)) + udppacket

            try:
                sent = self.mumble_object.control_socket.send(tcppacket)
                if sent < 0:
                    logger.error("Socket error sending audio")
            except Exception as e:
                logger.error(f"Send failed: {e}")
                return

    # Apply the patch
    pymumble.soundoutput.SoundOutput.send_audio = patched_send_audio
