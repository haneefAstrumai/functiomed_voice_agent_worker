"""
booking/rag_client.py
=====================
Retrieval-only client for the backend RAG system.
Called by the agent's on_user_turn_completed() hook.

No LLM here — just FAISS + BM25 retrieval.
The agent's own LLM (OpenAI) generates the final answer.
"""

import logging
import os
import httpx

log = logging.getLogger(__name__)

RAG_BACKEND_URL = os.getenv("RAG_BACKEND_URL", "http://localhost:8000")
RETRIEVE_TIMEOUT = 4.0   # seconds — keep low for voice latency
TOP_K = 5                # chunks to inject into context

# Voice agents must keep context small, otherwise the LLM tends to produce
# long answers which can exceed TTS input limits.
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "1400"))
MAX_CHUNK_CHARS = int(os.getenv("RAG_MAX_CHUNK_CHARS", "350"))


async def retrieve_context(query: str, top_k: int = TOP_K) -> str:
    """
    Call backend /retrieve and return a formatted string
    ready to inject into the agent's chat context.

    Returns empty string if backend is unreachable (agent falls back
    to its own knowledge gracefully).
    """
    try:
        async with httpx.AsyncClient(timeout=RETRIEVE_TIMEOUT) as client:
            resp = await client.post(
                f"{RAG_BACKEND_URL}/retrieve",
                json={"query": query, "k": top_k},
            )
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])
        if not results:
            return ""

        # Format chunks for injection into chat context
        lines = ["Relevant clinic knowledge:"]
        total = 0
        for r in results:
            source = r.get("page_name") or "clinic docs"
            content = (r.get("content", "") or "").strip()
            if content:
                if len(content) > MAX_CHUNK_CHARS:
                    content = content[: MAX_CHUNK_CHARS - 1].rstrip() + "…"
                snippet = f"[{source}] {content}"
                # Stop before exceeding global cap
                if total + len(snippet) > MAX_CONTEXT_CHARS:
                    break
                lines.append(snippet)
                total += len(snippet)

        return "\n\n".join(lines)

    except httpx.TimeoutException:
        log.warning("RAG retrieval timed out for query: %s", query[:60])
        return ""
    except Exception as e:
        log.error("Cannot reach RAG backend at %s: %s", RAG_BACKEND_URL, e)
        return ""


async def health_check() -> bool:
    """Returns True if backend is reachable."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{RAG_BACKEND_URL}/")
            return resp.status_code == 200
    except Exception as e:
        log.warning("RAG health check failed: %s", e)
        return False