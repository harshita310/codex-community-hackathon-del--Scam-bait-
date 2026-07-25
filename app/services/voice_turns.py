from app.services.voice_audio import is_probable_speech


class VoiceTurnBuffer:
    """Collect caller audio until a real silence boundary is reached."""

    def __init__(
        self,
        *,
        min_speech_bytes: int = 8000,
        silence_bytes: int = 6400,
        max_speech_bytes: int = 48000,
    ):
        self.min_speech_bytes = min_speech_bytes
        self.silence_bytes = silence_bytes
        self.max_speech_bytes = max_speech_bytes
        self._speech = bytearray()
        self._silence_bytes = 0

    def add(self, audio_chunk: bytes) -> bytes | None:
        if not audio_chunk:
            return None

        if is_probable_speech(audio_chunk):
            self._speech.extend(audio_chunk)
            self._silence_bytes = 0
            if len(self._speech) >= self.max_speech_bytes:
                return self._pop_if_enough()
            return None

        if not self._speech:
            return None

        self._silence_bytes += len(audio_chunk)
        if self._silence_bytes >= self.silence_bytes:
            return self._pop_if_enough()

        return None

    def flush(self) -> bytes | None:
        return self._pop_if_enough()

    def clear(self):
        self._speech.clear()
        self._silence_bytes = 0

    def _pop_if_enough(self) -> bytes | None:
        speech = bytes(self._speech)
        self.clear()
        if len(speech) < self.min_speech_bytes:
            return None
        return speech
