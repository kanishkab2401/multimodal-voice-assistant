import httpx

from config import ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID

_BASE_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"


async def stream_speech(text: str):
    """
    Yields raw MP3 audio bytes as they stream back from ElevenLabs.

    Free-tier note: ElevenLabs' free tier gives ~10k characters/month, which
    is plenty for development but will run out with heavy testing. If you
    hit the cap, switch this module to a local TTS engine (Piper or Kokoro)
    with the same generator signature — nothing else in the pipeline needs
    to change. This swap is also exactly what Phase 3's fallback logic does
    automatically when ElevenLabs errors or times out.
    """
    if not text.strip():
        return

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": text,
        "model_id": "eleven_turbo_v2_5",
        "voice_settings": {"stability": 0.4, "similarity_boost": 0.8},
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream("POST", _BASE_URL, json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise RuntimeError(f"ElevenLabs TTS error {resp.status_code}: {body[:200]}")
            async for chunk in resp.aiter_bytes():
                if chunk:
                    yield chunk
