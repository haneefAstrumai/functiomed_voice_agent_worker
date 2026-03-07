"""
Agent Service — booking/rag_client.py
=======================================
All HTTP calls from the Agent to the RAG backend.

The RAG backend is the single source of truth for:
  - Clinic knowledge (services, doctors, answers)
  - Saved bookings

The Agent calls these functions from its @function_tools.
"""

import logging
import os
import re

import httpx
from dotenv import load_dotenv

load_dotenv()

logger   = logging.getLogger(__name__)
RAG_URL  = os.getenv("RAG_BACKEND_URL", "http://localhost:8000")
TIMEOUT  = 20.0


# ── Knowledge queries (RAG /ask) ──────────────────────────────

async def ask_rag(question: str, top_n: int = 8, voice: bool = True) -> str:
    """Ask the RAG backend a question. Returns answer text."""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.post(
                f"{RAG_URL}/ask",
                json={"question": question, "top_n": top_n, "voice": voice},
            )
            r.raise_for_status()
            return r.json().get("answer", "")
    except httpx.ConnectError:
        logger.error("Cannot reach RAG backend at %s", RAG_URL)
        return "I'm having trouble connecting to our knowledge base right now."
    except Exception as e:
        logger.error("ask_rag failed: %s", e)
        return "I couldn't retrieve that information right now."


async def get_available_services() -> list[str]:
    """
    Ask RAG for all services functiomed offers.
    Returns a clean list like ["Physiotherapy", "Massage", "Osteopathy"].
    """
    answer = await ask_rag(
        "List all the medical services and treatments offered at Functiomed. "
        "Return only the service names, one per line, no descriptions.",
        top_n=10, voice=False,
    )
    return _parse_list(answer)


async def get_doctors_for_service(service_name: str) -> list[str]:
    """
    Ask RAG who the practitioners are for a service.
    Returns a clean list of names.
    """
    answer = await ask_rag(
        f"Who are the doctors, therapists, or practitioners at Functiomed "
        f"for {service_name}? List only their full names, one per line.",
        top_n=10, voice=False,
    )
    return _parse_list(answer)


async def validate_service(user_input: str, available: list[str]) -> str | None:
    """Match user's spoken service to available list. Returns canonical name or None."""
    low = user_input.lower().strip()
    for svc in available:
        if low in svc.lower() or svc.lower() in low:
            return svc
    if available:
        answer = await ask_rag(
            f"The user said '{user_input}'. Available services: {', '.join(available)}. "
            f"Which service did they mean? Reply with ONLY the exact name, or 'none'.",
            top_n=5, voice=False,
        )
        cleaned = answer.strip().strip(".")
        for svc in available:
            if svc.lower() in cleaned.lower():
                return svc
    return None


async def validate_doctor(user_input: str, available: list[str]) -> str | None:
    """Match user's spoken doctor name to available list. Returns canonical name or None."""
    low = user_input.lower()
    for doc in available:
        parts = doc.lower().replace("dr.", "").replace("msc", "").split()
        if any(p in low for p in parts if len(p) > 2):
            return doc
    if available:
        answer = await ask_rag(
            f"The user said '{user_input}'. Available practitioners: {', '.join(available)}. "
            f"Which one did they mean? Reply with ONLY the exact name, or 'none'.",
            top_n=5, voice=False,
        )
        cleaned = answer.strip().strip(".")
        for doc in available:
            if any(p.lower() in cleaned.lower() for p in doc.split() if len(p) > 2):
                return doc
    return None


# ── Booking save (RAG /bookings) ──────────────────────────────

async def save_booking_to_rag(
    service_name: str,
    doctor_name: str,
    patient_name: str,
    language: str = "en",
    session_summary: str = None,
) -> dict:
    """
    POST /bookings on the RAG backend to save the confirmed appointment.
    Returns the booking record including confirmation_number.
    Raises on failure so the agent can handle the error gracefully.
    """
    payload = {
        "service_name":    service_name,
        "doctor_name":     doctor_name,
        "patient_name":    patient_name,
        "language":        language,
        "session_summary": session_summary,
    }
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.post(f"{RAG_URL}/bookings/", json=payload)
        r.raise_for_status()
        return r.json()["booking"]


# ── Health check ──────────────────────────────────────────────

async def check_rag_health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{RAG_URL}/")
            return r.status_code == 200
    except Exception:
        return False


# ── Internal helpers ──────────────────────────────────────────

def _parse_list(text: str) -> list[str]:
    """Parse bullet/numbered/comma list from LLM free-text output."""
    if not text or len(text.strip()) < 3:
        return []
    items = []
    for line in text.splitlines():
        clean = re.sub(r"^[\s\-•*\d\.]+", "", line).strip().strip(".,;:")
        if 2 < len(clean) < 80:
            items.append(clean)
    if items:
        return items
    return [p.strip() for p in text.split(",") if 2 < len(p.strip()) < 80]