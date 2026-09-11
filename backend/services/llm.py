from groq import AsyncGroq

from config import GROQ_API_KEY, LLM_MODEL

_client = AsyncGroq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = (
    "You are a helpful, concise voice assistant. Keep responses short "
    "(1-3 sentences) and conversational, since they will be read aloud. "
    "Avoid lists, markdown, or anything that doesn't make sense spoken."
)


async def stream_response(user_text: str, history: list, search_context: str = ""):
    """Yields response text token-by-token as it streams from the model.

    If search_context is provided (see services/search.py), it's included
    alongside the user's question so the model can ground its answer in
    real, current information instead of only its training knowledge.
    """
    user_content = user_text
    if search_context:
        user_content = (
            f"Current web search results that may help answer this:\n"
            f"{search_context}\n\n"
            f"Using the above if relevant, answer: {user_text}"
        )

    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": user_content}]
    )

    stream = await _client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
