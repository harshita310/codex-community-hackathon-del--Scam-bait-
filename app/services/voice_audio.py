import wave


BIAS = 0x84
CLIP = 32635


def _ulaw_byte_to_pcm16(value: int) -> int:
    value = (~value) & 0xFF
    sample = ((value & 0x0F) << 3) + BIAS
    sample <<= (value & 0x70) >> 4
    return BIAS - sample if value & 0x80 else sample - BIAS


def _pcm16_to_ulaw_byte(sample: int) -> int:
    sign = 0x80 if sample < 0 else 0
    sample = min(abs(sample), CLIP) + BIAS

    exponent = 7
    mask = 0x4000
    while exponent > 0 and not (sample & mask):
        mask >>= 1
        exponent -= 1

    mantissa = (sample >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa)) & 0xFF


def _bytes_to_pcm16(frames: bytes, sample_width: int) -> list[int]:
    if sample_width == 1:
        return [(value - 128) << 8 for value in frames]

    if sample_width == 2:
        return [
            int.from_bytes(frames[index:index + 2], "little", signed=True)
            for index in range(0, len(frames) - 1, 2)
        ]

    if sample_width == 4:
        return [
            int.from_bytes(frames[index:index + 4], "little", signed=True) >> 16
            for index in range(0, len(frames) - 3, 4)
        ]

    raise ValueError(f"Unsupported WAV sample width: {sample_width}")


def _resample_nearest(samples: list[int], source_rate: int, target_rate: int) -> list[int]:
    if source_rate == target_rate or not samples:
        return samples

    target_length = max(1, round(len(samples) * target_rate / source_rate))
    return [
        samples[min(len(samples) - 1, round(index * source_rate / target_rate))]
        for index in range(target_length)
    ]


def twilio_mulaw_to_wav(audio_chunk: bytes, output_path: str) -> str:
    """Convert Twilio Media Streams mulaw/8k audio into PCM WAV for STT."""
    pcm_audio = b"".join(
        _ulaw_byte_to_pcm16(value).to_bytes(2, "little", signed=True)
        for value in audio_chunk
    )

    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(pcm_audio)

    return output_path


def wav_to_twilio_mulaw(input_path: str) -> bytes:
    """Convert a WAV TTS file into raw mulaw/8k bytes for Twilio."""
    with wave.open(input_path, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    samples = _bytes_to_pcm16(frames, sample_width)
    if channels > 1:
        samples = [
            round(sum(samples[index:index + channels]) / channels)
            for index in range(0, len(samples), channels)
        ]

    samples = _resample_nearest(samples, frame_rate, 8000)
    return bytes(_pcm16_to_ulaw_byte(sample) for sample in samples)
