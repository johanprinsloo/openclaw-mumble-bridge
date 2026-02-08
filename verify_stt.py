import asyncio
import os
import sys

# Add current directory to path so we can import stt
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stt import STTClient

# Use the key from your context
API_KEY = "sk_d32f07264340ae40027cde22db9b4bc9fa290ac2b026fefb"

async def test_stt():
    print("Initializing STT Client...")
    client = STTClient(api_key=API_KEY, model="scribe_v1_base")
    
    # Read raw PCM file
    pcm_path = "/Users/kleo/Documents/code/openclaw-mumble-bridge/test_input.pcm"
    print(f"Reading PCM from {pcm_path}...")
    with open(pcm_path, "rb") as f:
        pcm_data = f.read()
    
    print(f"Read {len(pcm_data)} bytes. Transcribing...")
    
    try:
        # The transcribe method in stt.py is async
        # We need to call the method on the instance
        text = await client.transcribe(pcm_data, user="TestScript")
        if text:
            print(f"\nSUCCESS! Transcription:\n----------------------\n{text}\n----------------------")
        else:
            print("\nFAILURE: No text returned.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_stt())
