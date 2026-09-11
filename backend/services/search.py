import httpx

from config import TAVILY_API_KEY

_TAVILY_URL = "https://api.tavily.com/search"


async def web_search(query: str, max_results: int = 3) -> str:
    """
    Returns a short block of plain text summarizing current web results for
    the query, ready to drop directly into the LLM's prompt as context.

    Deliberately fails soft: if Tavily errors, times out, or no key is
    configured, this returns "" instead of raising. The caller should treat
    an empty string as "no extra context available" and continue the turn
    normally — a broken search should never be allowed to block or crash
    the whole conversation.
    """
    if not TAVILY_API_KEY:
        return ""

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": True,
    }

    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.post(_TAVILY_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        return ""

    parts = []
    if data.get("answer"):
        parts.append(f"Quick answer: {data['answer']}")
    for r in data.get("results", [])[:max_results]:
        title = r.get("title", "")
        content = (r.get("content", "") or "")[:300]
        if title or content:
            parts.append(f"- {title}: {content}")

    return "\n".join(parts)
