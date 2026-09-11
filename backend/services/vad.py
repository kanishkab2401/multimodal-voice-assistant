import numpy as np

from config import SILENCE_MS, ENERGY_THRESHOLD, CHUNK_MS


class SimpleVAD:
    """
    Energy-based voice activity detector.

    This is intentionally simple for Phase 1 — it just tracks mean absolute
    amplitude per chunk and declares an utterance "done" once we've seen
    speech followed by enough silence. Good enough to get the pipeline
    flowing end-to-end. Swap in webrtcvad or Silero VAD later for more
    robust segmentation (handles background noise much better).
    """

    def __init__(self):
        self.silence_chunks_needed = max(1, int(SILENCE_MS / CHUNK_MS))
        self.energy_threshold = ENERGY_THRESHOLD
        self.silence_count = 0
        self.has_speech = False

    def process_chunk(self, pcm_bytes: bytes) -> bool:
        """Feed one chunk of int16 PCM audio. Returns True if this chunk
        marks the end of an utterance (speech followed by enough silence)."""
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        if len(samples) == 0:
            return False

        energy = np.abs(samples).mean()

        if energy > self.energy_threshold:
            self.has_speech = True
            self.silence_count = 0
        elif self.has_speech:
            self.silence_count += 1

        if self.has_speech and self.silence_count >= self.silence_chunks_needed:
            self.has_speech = False
            self.silence_count = 0
            return True

        return False

    def reset(self):
        self.has_speech = False
        self.silence_count = 0
