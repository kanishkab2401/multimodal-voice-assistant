import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
# Default voice: "Rachel" (a stock ElevenLabs voice available on the free tier)
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

# Tavily: dedicated web search API for feeding real, current results into the
# LLM ourselves, instead of relying on a model's own built-in (opaque, less
# reliable) search decision. Free tier: 1,000 searches/month, no card needed.
# Get a key at https://app.tavily.com
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

ASR_MODEL = os.getenv("ASR_MODEL", "whisper-large-v3-turbo")
# Plain gpt-oss-120b, NOT the "compound" search-enabled variant — we do web
# search ourselves (see services/search.py) for reliability and control,
# and use this stable model purely for understanding + writing the reply.
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

# VAD tuning
SAMPLE_RATE = 16000
SILENCE_MS = 700          # how long a pause must last to consider the utterance finished
ENERGY_THRESHOLD = 500    # mean absolute amplitude threshold for "speech present" (int16 scale)
CHUNK_MS = 100            # size of audio chunks the frontend sends, used to convert SILENCE_MS -> chunk count
