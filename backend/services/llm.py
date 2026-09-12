import json

from groq import AsyncGroq

from config import GROQ_API_KEY, LLM_MODEL
from services.search import web_search

_client = AsyncGroq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = (
    "You are a helpful, concise voice assistant. Keep responses short "
    "(1-3 sentences) and conversational, since they will be read aloud. "
    "Avoid lists, markdown, or anything that doesn't make sense spoken. "
    "Use the web_search tool only when you genuinely need current, recent, "
    "or time-sensitive information you're not confident about — not for "
    "stable general knowledge, definitions, or historical facts you "
    "already know well."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current, real-time, or recent "
                "information — news, events, specific dates, scores, "
                "weather, prices, or anything that may have changed or "
                "been announced after your training data. Do not use this "
                "for stable historical facts, definitions, or general "
                "knowledge you already know well."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A focused search query for what you need to find out.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

# Safety cap: the model could, in principle, keep asking to search again and
# again. This limits it to a few rounds so one turn can never hang or loop
# forever — a small, deliberate piece of resilience baked in from the start.
MAX_TOOL_ROUNDS = 3


async def stream_response(user_text: str, history: list):
    """
    Async generator yielding event dicts as the reply is produced:
      {"type": "search_started", "query": <str>}
      {"type": "search_done", "found": <bool>}
      {"type": "token", "text": <str>}

    The model decides for itself, on EVERY round, whether it still needs to
    search before giving a final answer — not just once. It keeps that
    option available until it either answers directly or hits the round
    cap, at which point it's forced to answer with whatever it has.
    """
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": user_text}]
    )

    for round_num in range(MAX_TOOL_ROUNDS):
        is_last_round = round_num == MAX_TOOL_ROUNDS - 1

        stream = await _client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=None if is_last_round else TOOLS,
            tool_choice=None if is_last_round else "auto",
            stream=True,
        )

        tool_calls = {}
        saw_tool_call = False
        got_content = False

        async for chunk in stream:
            delta = chunk.choices[0].delta

            if getattr(delta, "tool_calls", None):
                saw_tool_call = True
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls:
                        tool_calls[idx] = {"id": None, "name": "", "arguments": ""}
                    if tc.id:
                        tool_calls[idx]["id"] = tc.id
                    if tc.function and tc.function.name:
                        tool_calls[idx]["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        tool_calls[idx]["arguments"] += tc.function.arguments

            if delta.content:
                got_content = True
                yield {"type": "token", "text": delta.content}

        if not saw_tool_call:
            return  # model gave a final answer this round — done

        if got_content:
            return

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in tool_calls.values()
            ],
        })

        for tc in tool_calls.values():
            try:
                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            query = args.get("query", user_text)

            yield {"type": "search_started", "query": query}
            try:
                result = await web_search(query)
            except Exception:
                result = ""
            yield {"type": "search_done", "found": bool(result)}

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result or "No results found.",
            })
