import io
import wave

from groq import Groq

from config import GROQ_API_KEY, ASR_MODEL, SAMPLE_RATE

_client = Groq(api_key=GROQ_API_KEY)


def _pcm16_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def transcribe(pcm_bytes: bytes) -> str:
    """
    Synchronous call (Groq's SDK is sync under the hood) — run this via
    loop.run_in_executor from async code so it doesn't block the event loop.

    Note: this transcribes one complete utterance at a time (buffered by the
    VAD), not word-by-word partials. For true incremental streaming ASR with
    partial hypotheses, swap this module for Deepgram's streaming websocket
    API — the rest of the pipeline (event schema, orchestrator) doesn't need
    to change, only this file.
    """
    if len(pcm_bytes) < 1000:
        return ""

    wav_bytes = _pcm16_to_wav_bytes(pcm_bytes)
    result = _client.audio.transcriptions.create(
        file=("utterance.wav", wav_bytes),
        model=ASR_MODEL,
        response_format="text",
        language="en",  # force English — stops Whisper from mis-guessing
                         # the language on short/ambiguous audio and
                         # transcribing (or hallucinating) in the wrong one
    )
    return str(result).strip()
