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
import time
import asyncio

log = logging.getLogger(__name__)

RAG_BACKEND_URL  = os.getenv("RAG_BACKEND_URL", "http://localhost:8000")
RETRIEVE_TIMEOUT = float(os.getenv("RAG_RETRIEVE_TIMEOUT", "6.0"))   # increased from 4.0
TOP_K            = int(os.getenv("RAG_TOP_K", "5"))

# Increased limits — 1400 chars was too small for useful answers.
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "3000"))
MAX_CHUNK_CHARS   = int(os.getenv("RAG_MAX_CHUNK_CHARS",   "600"))

_rag_cache: dict[str, dict[str, object]] = {}
_RAG_CACHE_TTL_S = float(os.getenv("RAG_CACHE_TTL_S", "60"))
_CACHE_LOCK = asyncio.Lock()

async def retrieve_context(query: str, top_k: int = TOP_K) -> str:
    """
    Call backend /retrieve and return a formatted string
    ready to inject into the agent's chat context.

    Returns empty string if backend is unreachable (agent falls back
    to its own knowledge gracefully).
    """
    try:
        cache_key = f"{query}:{top_k}"
        async with _CACHE_LOCK:
            entry = _rag_cache.get(cache_key)
            if entry and time.monotonic() < float(entry.get("expires_at") or 0.0):
                log.info("[RAG] Cache hit for query: %s", query[:60])
                return str(entry.get("value", ""))

        t0_req = time.perf_counter()
        async with httpx.AsyncClient(timeout=RETRIEVE_TIMEOUT) as client:
            resp = await client.post(
                f"{RAG_BACKEND_URL}/retrieve",
                json={"query": query, "k": top_k},
            )
            resp.raise_for_status()
            data = resp.json()
            
        dt_req_ms = (time.perf_counter() - t0_req) * 1000.0
        log.info("[TIMER][rag] httpx retrieve backend call completed in %.1fms", dt_req_ms)

        results = data.get("results", [])
        log.info("[RAG] /retrieve returned %d results for query: %s", len(results), query[:80])

        if not results:
            log.warning("[RAG] No results from backend for query: %s", query[:80])
            return ""

        # Format chunks for injection into chat context.
        # The header explicitly instructs the LLM to use this content —
        # without it the LLM tends to ignore injected context.
        lines = [
            "CLINIC KNOWLEDGE BASE — use the following facts to answer the patient's question. "
            "Do NOT contradict or ignore this information:"
        ]
        total = 0
        for r in results:
            source  = r.get("page_name") or "clinic docs"
            content = (r.get("content", "") or "").strip()
            score   = r.get("score", 0)
            if not content:
                continue
            if len(content) > MAX_CHUNK_CHARS:
                content = content[:MAX_CHUNK_CHARS - 1].rstrip() + "…"
            snippet = f"[{source}] {content}"
            if total + len(snippet) > MAX_CONTEXT_CHARS:
                log.info("[RAG] Context cap reached after %d chars", total)
                break
            lines.append(snippet)
            total += len(snippet)
            log.debug("[RAG] chunk score=%.4f source=%s", score, source)

        if len(lines) == 1:
            # Only the header, no real chunks made it in
            return ""

        context = "\n\n".join(lines)
        log.info("[RAG] Injecting %d chars across %d chunks", total, len(lines) - 1)
        return context

    except httpx.TimeoutException:
        log.warning("[RAG] Retrieval timed out (%.1fs) for query: %s", RETRIEVE_TIMEOUT, query[:60])
        return ""
    except Exception as e:
        log.error("[RAG] Cannot reach RAG backend at %s: %s", RAG_BACKEND_URL, e)
        return ""


async def health_check() -> bool:
    """Returns True if backend is reachable."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{RAG_BACKEND_URL}/")
            return resp.status_code == 200
    except Exception as e:
        log.warning("[RAG] Health check failed: %s", e)
        return False