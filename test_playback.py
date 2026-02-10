import sys
import time
import os

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mumble_client import MumbleClient, MumbleConfig
from audio_utils import audio_to_mumble_pcm

def main():
    config = MumbleConfig(
        host="192.168.108.212",
        port=64738,
        username="QualityTest",
        password="",
        channel="Root"
    )
    client = MumbleClient(config=config)
    print("Connecting...")
    client.connect()
    
    while not client.is_connected():
        time.sleep(0.1)
    
    print("Connected. Preparing audio...")
    
    # Generate audio if missing (using verify_stt's file if available)
    if not os.path.exists("test_input.wav"):
        print("test_input.wav not found. Please run verify_stt.py first or provide a wav.")
        return

    with open("test_input.wav", "rb") as f:
        wav_data = f.read()
        
    pcm = audio_to_mumble_pcm(wav_data, "wav")
    
    print(f"Sending {len(pcm)} bytes of PCM...")
    client.send_audio(pcm)
    
    # Keep alive while playing
    duration = len(pcm) / (48000 * 2)
    print(f"Playing for {duration:.2f}s...")
    time.sleep(duration + 2)
    
    client.disconnect()
    print("Done.")

if __name__ == "__main__":
    main()
