import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class SentenceAggregator:
    """
    Buffers streamed LLM tokens and releases complete sentences as soon as
    they're detected, so TTS can start on sentence 1 while the LLM is still
    generating sentence 2. This is what gets total latency down — without
    it you'd wait for the entire LLM response before any audio starts.
    """

    def __init__(self):
        self.buffer = ""

    def feed(self, token: str) -> list[str]:
        self.buffer += token
        parts = _SENTENCE_END.split(self.buffer)
        if len(parts) > 1:
            complete, self.buffer = parts[:-1], parts[-1]
            return [p for p in complete if p.strip()]
        return []

    def flush(self) -> list[str]:
        remainder = self.buffer.strip()
        self.buffer = ""
        return [remainder] if remainder else []
