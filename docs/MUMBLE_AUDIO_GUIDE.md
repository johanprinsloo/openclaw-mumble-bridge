# Mumble Audio Configuration Guide

This document covers the audio specifications for Mumble/pymumble and optimal settings for the OpenClaw voice bridge.

## Mumble Audio Specifications

### Sample Rate & Format
- **Sample Rate:** 48,000 Hz (48 kHz) - required by Mumble/Opus
- **Bit Depth:** 16-bit signed little-endian PCM
- **Channels:** Mono (1 channel)
- **Bytes per sample:** 2 bytes

### Frame Sizes
Opus supports frame sizes: **2.5ms, 5ms, 10ms, 20ms, 40ms, 60ms**

Mumble standard configurations:
| Frame Size | Samples per Frame | Bytes per Frame | Packets/Second |
|------------|-------------------|-----------------|----------------|
| 10ms       | 480               | 960             | 100            |
| 20ms       | 960               | 1,920           | 50             |
| 40ms       | 1,920             | 3,840           | 25             |
| 60ms       | 2,880             | 5,760           | ~17            |

**Calculation:** `bytes = sample_rate × channels × (bit_depth/8) × (frame_ms/1000)`
- 20ms frame: `48000 × 1 × 2 × 0.020 = 1,920 bytes`

### Bitrates
Opus voice codec typical ranges:
| Quality     | Bitrate (kbps) | Use Case              |
|-------------|----------------|-----------------------|
| Narrowband  | 6-12           | Low bandwidth voice   |
| Wideband    | 16-24          | Standard voice        |
| Fullband    | 32-64          | High quality voice    |
| Music       | 64-128         | Music streaming       |

**pymumble defaults:**
- `PYMUMBLE_BANDWIDTH = 50,000` (50 kbps total, including overhead)
- Protocol overhead per packet: ~28-46 bytes (IP+UDP or IP+TCP+tunnel)

### Opus Encoder Profiles
- **"voip"** - Optimized for speech, includes features like FEC (Forward Error Correction)
- **"audio"** - General audio, better for music
- **pymumble default:** `"audio"` (set via `PYMUMBLE_AUDIO_TYPE_OPUS_PROFILE`)

---

## pymumble Implementation Details

### Key Constants (`pymumble_py3/constants.py`)
```python
PYMUMBLE_SAMPLERATE = 48000           # Hz
PYMUMBLE_AUDIO_PER_PACKET = 0.020     # 20ms per packet
PYMUMBLE_BANDWIDTH = 50000            # 50 kbps total
PYMUMBLE_LOOP_RATE = 0.01             # 10ms main loop interval
PYMUMBLE_SEQUENCE_DURATION = 0.010    # 10ms sequence unit
```

### SoundOutput Behavior
The `SoundOutput` class in pymumble:

1. **Buffering:** Audio is queued via `add_sound(pcm)` into `self.pcm[]` list
2. **Frame Assembly:** Splits PCM into encoder frame chunks
3. **Timing:** `send_audio()` paces output based on real-time sequence tracking
4. **Encoding:** Uses `opuslib` to encode each frame

**Critical timing logic in `send_audio()`:**
```python
while len(self.pcm) > 0 and self.sequence_last_time + self.audio_per_packet <= time():
    # Send audio only when enough time has passed
```

### Audio Flow
```
TTS API → MP3/WAV → ffmpeg (resample to 48kHz) → PCM → add_sound() → 
    → encode Opus → packet with sequence number → TCP/UDP → Mumble server
```

---

## Known Issues & Solutions

### Issue: "Choppy / Slow Motion" Audio

**Symptoms:** Audio sounds slowed down or has gaps

**Potential Causes:**

1. **Bitrate Misconfiguration**
   - Setting bitrate directly bypasses overhead calculation
   - Fix: Use `set_bandwidth()` instead of setting `encoder.bitrate` directly

2. **Incorrect Sample Rate in Source Audio**
   - If source audio isn't truly 48kHz, playback will be wrong speed
   - Fix: Verify ffmpeg output is exactly 48kHz

3. **Frame Size Mismatch**
   - Feeding partial or oversized frames causes encoder issues
   - Fix: Always pad frames to exact `MUMBLE_FRAME_SIZE`

4. **Buffer Overrun**
   - Adding audio faster than real-time causes buffering issues
   - Fix: Add audio at approximately real-time rate, or clear buffer before new audio

5. **Opus Profile**
   - "audio" profile may introduce more latency than "voip"
   - Try: Switch to "voip" profile for lower latency

### Debugging Tips

1. **Check audio duration matches expectation:**
   ```python
   duration_ms = len(pcm_data) / (48000 * 2) * 1000
   print(f"Audio duration: {duration_ms:.1f}ms")
   ```

2. **Verify sample rate with ffprobe:**
   ```bash
   ffprobe -i input.mp3 -show_entries stream=sample_rate -of csv=p=0
   ```

3. **Monitor buffer size:**
   ```python
   buffer_sec = mumble.sound_output.get_buffer_size()
   print(f"Buffer: {buffer_sec:.3f}s")
   ```

---

## Optimal Settings (Golden Configuration)

Based on testing, these settings provide the best audio quality for voice:

### mumble_client.py Settings
```python
# Frame size for Mumble playback (20ms at 48kHz mono 16-bit)
MUMBLE_FRAME_SIZE = 1920  # bytes = 960 samples

# Opus profile: "voip" for speech (lower latency), "audio" for music
OPUS_PROFILE = "voip"
```

### audio_utils.py (ffmpeg flags)
```bash
ffmpeg -i pipe:0 -f s16le -ac 1 -ar 48000 pipe:1
```

### Key Findings

1. **Opus Profile**: Use `"voip"` instead of `"audio"` for speech
   - "voip" has lower latency and better speech optimization
   - "audio" is designed for music with higher latency

2. **Bitrate**: Let the encoder use defaults (~50kbps)
   - The opuslib `encoder.bitrate` setter is broken in recent versions
   - Default bitrate works well for voice quality
   - Don't try to set bitrate explicitly

3. **Frame Size**: 20ms (1920 bytes) is the sweet spot
   - Balances latency vs. overhead
   - Matches pymumble defaults

4. **Audio Pipeline**: The conversion chain is solid
   - TTS → MP3 → ffmpeg resample to 48kHz → Opus → Mumble
   - STT: Mumble → resample to 16kHz → ElevenLabs Scribe
   - Both directions tested at 100% transcription accuracy

### Testing

Run the audio pipeline test to verify conversion quality:
```bash
python tests/audio_pipeline_test.py
```

Expected output: `Similarity Score: 100.0%`

For full loopback testing (requires running Mumble server):
```bash
python tests/audio_loopback.py
```

For timing diagnostics:
```bash
python tests/mumble_timing_test.py
```

---

## Troubleshooting

### "Slow Motion" Audio
If audio sounds slowed down:

1. **Check sample rate consistency**:
   - Ensure all audio is properly converted to 48kHz before sending
   - Run `python tests/audio_pipeline_test.py` to verify

2. **Check Opus profile**:
   - Use `"voip"` profile, not `"audio"`
   - Edit `OPUS_PROFILE` in `mumble_client.py`

3. **Check network latency**:
   - Run `python tests/mumble_timing_test.py` to analyze packet timing

### "Choppy" Audio
If audio has gaps or stutters:

1. **Buffer underrun**:
   - Audio is being consumed faster than produced
   - Check TTS generation speed

2. **Network issues**:
   - TCP/UDP congestion causing packet delays
   - Check ping to Mumble server

3. **Frame size mismatch**:
   - Ensure `MUMBLE_FRAME_SIZE = 1920` bytes

### No Audio Received
1. Check that both clients are in the same channel
2. Verify the speaker isn't muted
3. Check server logs for errors

---

## References

- [Opus Codec Documentation](https://opus-codec.org/docs/)
- [Mumble Protocol Wiki](https://wiki.mumble.info/wiki/Protocol)
- [pymumble GitHub](https://github.com/azlux/pymumble)
- [Mumble FAQ - Audio Settings](https://wiki.mumble.info/wiki/FAQ/English)
