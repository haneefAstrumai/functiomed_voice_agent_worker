"""
Agent Service — booking/rag_client.py
=======================================
HTTP calls from the Agent to the RAG backend.

Architecture after DB upgrade:
  ┌─────────────────────────────────────────────────────┐
  │  CLINIC DATA (services / doctors / slots)           │
  │  → GET /clinic/services                             │
  │  → GET /clinic/doctors?service=...                  │
  │  → GET /clinic/slots?service=...&doctor=...         │
  │  Source: SQLite DB  (fast, structured, reliable)    │
  ├─────────────────────────────────────────────────────┤
  │  KNOWLEDGE BASE (clinic info, prices, hours, etc.)  │
  │  → POST /ask                                        │
  │  Source: RAG (FAISS + BM25 + LLM)                   │
  ├─────────────────────────────────────────────────────┤
  │  BOOKINGS (save confirmed appointments)             │
  │  → POST /bookings/                                  │
  │  Source: SQLite DB                                  │
  └─────────────────────────────────────────────────────┘
"""

import logging
import os
import re

import httpx
from dotenv import load_dotenv

load_dotenv()

logger  = logging.getLogger(__name__)
RAG_URL = os.getenv("RAG_BACKEND_URL", "http://localhost:8000")
TIMEOUT = 20.0


# ─────────────────────────────────────────────────────────────
# Clinic DB endpoints  (replaces RAG for structured data)
# ─────────────────────────────────────────────────────────────

async def get_available_services() -> list[str]:
    """
    Fetch all active service names from the SQLite database.
    Returns e.g. ["Osteopathy", "IV Therapy", "Dry Needling", ...]
    """
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.get(f"{RAG_URL}/clinic/services")
            r.raise_for_status()
            data = r.json()
            services = data.get("names", [])
            logger.info("get_available_services from DB: %r", services)
            return services
    except httpx.ConnectError:
        logger.error("Cannot reach backend at %s", RAG_URL)
        return []
    except Exception as e:
        logger.error("get_available_services failed: %s", e)
        return []


async def get_doctors_for_service(service_name: str) -> list[str]:
    """
    Fetch doctor names for a service from the SQLite database.
    Returns e.g. ["Dr. Anna Müller", "Lisa Schneider"]
    """
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.get(
                f"{RAG_URL}/clinic/doctors",
                params={"service": service_name},
            )
            r.raise_for_status()
            data = r.json()
            doctors = data.get("names", [])
            logger.info("get_doctors_for_service(%r) from DB: %r", service_name, doctors)
            return doctors
    except Exception as e:
        logger.error("get_doctors_for_service failed: %s", e)
        return []


async def get_slots_for_doctor(
    service_name: str,
    doctor_name: str,
) -> tuple[list[dict], list[str]]:
    """
    Fetch available appointment slots from the SQLite database.

    Returns:
        (slots, spoken_options)
        slots          — list of dicts with id, slot_date, slot_time, etc.
        spoken_options — list of human-readable strings for voice e.g.
                         ["Option 1: Monday March 10th at 9 AM", ...]
    """
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.get(
                f"{RAG_URL}/clinic/slots",
                params={"service": service_name, "doctor": doctor_name},
            )
            r.raise_for_status()
            data = r.json()
            slots          = data.get("slots", [])
            spoken_options = data.get("spoken_options", [])
            logger.info(
                "get_slots_for_doctor(%r, %r): %d slots",
                service_name, doctor_name, len(slots),
            )
            return slots, spoken_options
    except Exception as e:
        logger.error("get_slots_for_doctor failed: %s", e)
        return [], []


def validate_slot_choice(user_said: str, slots: list[dict], spoken_options: list[str]) -> dict | None:
    """
    Match user's spoken slot choice to available slots.

    Accepts:
      - "option 1" / "the first one" / "number 2"
      - partial date/time mention: "Monday", "9 AM", "March 15"

    Returns the matched slot dict or None.
    """
    if not slots:
        return None

    low = user_said.lower().strip()

    # ── Ordinal / number matching ─────────────────────────────
    ordinal_map = {
        "first": 0, "one": 0, "1": 0,
        "second": 1, "two": 1, "2": 1,
        "third": 2, "three": 2, "3": 2,
        "fourth": 3, "four": 3, "4": 3,
        "fifth": 4, "five": 4, "5": 4,
        "sixth": 5, "six": 5, "6": 5,
    }
    for word, idx in ordinal_map.items():
        if word in low and idx < len(slots):
            logger.info("validate_slot_choice: ordinal match → index %d", idx)
            return slots[idx]

    # ── "option N" matching ───────────────────────────────────
    m = re.search(r"option\s*(\d+)", low)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(slots):
            logger.info("validate_slot_choice: option match → index %d", idx)
            return slots[idx]

    # ── Partial date/time matching ────────────────────────────
    days_of_week = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    months = ["january", "february", "march", "april", "may", "june",
              "july", "august", "september", "october", "november", "december"]

    for i, opt in enumerate(spoken_options):
        opt_low = opt.lower()
        # Check if any day/month word in user_said matches this option
        for day in days_of_week:
            if day in low and day in opt_low:
                logger.info("validate_slot_choice: day match '%s' → index %d", day, i)
                return slots[i]
        for month in months:
            if month in low and month in opt_low:
                logger.info("validate_slot_choice: month match '%s' → index %d", month, i)
                return slots[i]

    logger.warning("validate_slot_choice: no match for %r", user_said)
    return None


# ─────────────────────────────────────────────────────────────
# Fuzzy name matching  (no longer uses RAG for this)
# ─────────────────────────────────────────────────────────────

def validate_service_local(user_input: str, available: list[str]) -> str | None:
    """Match user's spoken service to available list (local, no RAG call)."""
    low = user_input.lower().strip()
    # Exact / substring match
    for svc in available:
        if low in svc.lower() or svc.lower() in low:
            return svc
    # Word overlap match
    user_words = set(re.findall(r"\b\w{3,}\b", low))
    for svc in available:
        svc_words = set(re.findall(r"\b\w{3,}\b", svc.lower()))
        if user_words & svc_words:
            return svc
    return None


def validate_doctor_local(user_input: str, available: list[str]) -> str | None:
    """Match user's spoken doctor name to available list (local, no RAG call)."""
    low = user_input.lower()
    for doc in available:
        parts = re.findall(r"\b\w{3,}\b", doc.lower().replace("dr.", ""))
        if any(p in low for p in parts):
            return doc
    return None


# Keep async wrappers for compatibility with main.py
async def validate_service(user_input: str, available: list[str]) -> str | None:
    result = validate_service_local(user_input, available)
    if result:
        logger.info("validate_service: match → %r", result)
    else:
        logger.warning("validate_service: no match for %r in %r", user_input, available)
    return result


async def validate_doctor(user_input: str, available: list[str]) -> str | None:
    result = validate_doctor_local(user_input, available)
    if result:
        logger.info("validate_doctor: match → %r", result)
    else:
        logger.warning("validate_doctor: no match for %r in %r", user_input, available)
    return result


# ─────────────────────────────────────────────────────────────
# RAG knowledge base  (unchanged — used only for clinic Q&A)
# ─────────────────────────────────────────────────────────────

async def ask_rag(question: str, top_n: int = 8, voice: bool = True) -> str:
    """Ask the RAG backend a question. Returns answer text."""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.post(
                f"{RAG_URL}/ask",
                json={"question": question, "top_n": top_n, "voice": voice},
            )
            r.raise_for_status()
            data = r.json()
            logger.debug("ask_rag raw response: %r", data)
            return data.get("answer", "")
    except httpx.ConnectError:
        logger.error("Cannot reach RAG backend at %s", RAG_URL)
        return "I'm having trouble connecting to our knowledge base right now."
    except Exception as e:
        logger.error("ask_rag failed: %s", e)
        return "I couldn't retrieve that information right now."


# ─────────────────────────────────────────────────────────────
# Booking save
# ─────────────────────────────────────────────────────────────

async def save_booking_to_rag(
    service_name: str,
    doctor_name: str,
    patient_name: str,
    slot_id: str | None = None,
    language: str = "en",
    session_summary: str | None = None,
) -> dict:
    """
    POST /bookings/ to save confirmed appointment.
    Returns booking record with confirmation_number.
    """
    payload = {
        "service_name":    service_name,
        "doctor_name":     doctor_name,
        "patient_name":    patient_name,
        "slot_id":         slot_id,
        "language":        language,
        "session_summary": session_summary,
    }
    logger.info("save_booking_to_rag payload: %r", payload)
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.post(f"{RAG_URL}/bookings/", json=payload)
        r.raise_for_status()
        data = r.json()
        logger.info("save_booking_to_rag response: %r", data)
        return data["booking"]


# ─────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────

async def check_rag_health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{RAG_URL}/")
            reachable = r.status_code == 200
            logger.info("RAG health: %s (status=%s)", "OK" if reachable else "FAIL", r.status_code)
            return reachable
    except Exception as e:
        logger.warning("RAG health check failed: %s", e)
        return False