"""
booking/booking_client.py
=========================
HTTP client for backend booking + clinic endpoints.
Unchanged from original — booking logic stays the same.
"""

import logging
import os
import httpx

log = logging.getLogger(__name__)

RAG_BACKEND_URL = os.getenv("RAG_BACKEND_URL", "http://localhost:8000")
TIMEOUT = 8.0


async def get_services() -> list[dict]:
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(f"{RAG_BACKEND_URL}/clinic/services")
            resp.raise_for_status()
            data = resp.json()
            return [s for s in data.get("services", data) if s.get("active", 1)]
    except Exception as e:
        log.error("get_services failed: %s", e)
        return []


async def get_doctors_for_service(service_name: str) -> list[dict]:
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(
                f"{RAG_BACKEND_URL}/clinic/doctors",
                params={"service": service_name},
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("doctors", data)
    except Exception as e:
        log.error("get_doctors_for_service failed: %s", e)
        return []


async def get_slots(service_name: str, doctor_name: str) -> list[dict]:
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(
                f"{RAG_BACKEND_URL}/clinic/slots",
                params={
                    "service_name": service_name,
                    "doctor_name": doctor_name,
                    "available_only": True,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("slots", data)
    except Exception as e:
        log.error("get_slots failed: %s", e)
        return []


async def save_appointment(
    service_name: str,
    doctor_name: str,
    patient_name: str,
    slot_id: str | None = None,
    language: str = "en",
) -> dict:
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(
                f"{RAG_BACKEND_URL}/bookings/",
                json={
                    "service_name": service_name,
                    "doctor_name":  doctor_name,
                    "patient_name": patient_name,
                    "slot_id":      slot_id,
                    "language":     language,
                },
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        log.error("save_appointment failed: %s", e)
        return {}