"""
booking/booking_client.py
=========================
HTTP client for backend booking + clinic endpoints.
Unchanged from original — booking logic stays the same.
"""

import asyncio
import logging
import os
import time

import httpx

log = logging.getLogger(__name__)

RAG_BACKEND_URL = os.getenv("RAG_BACKEND_URL", "http://localhost:8000")
TIMEOUT = 8.0

_HTTP_CLIENT: httpx.AsyncClient | None = None
_HTTP_CLIENT_LOCK = asyncio.Lock()
_HTTP_LIMITS = httpx.Limits(max_connections=20, max_keepalive_connections=10)

# Small TTL caches to reduce repeated HTTP calls during the booking flow.
# Keep slots cache TTL low to avoid stale availability.
_CACHE_TTL_SERVICES_S = float(os.getenv("BOOKING_CACHE_SERVICES_TTL_S", "300"))
_CACHE_TTL_DOCTORS_S = float(os.getenv("BOOKING_CACHE_DOCTORS_TTL_S", "300"))
_CACHE_TTL_SLOTS_S = float(os.getenv("BOOKING_CACHE_SLOTS_TTL_S", "10"))

_services_cache: dict[str, object] = {"value": None, "expires_at": 0.0}
_doctors_cache: dict[str, dict[str, object]] = {}
_slots_cache: dict[tuple[str, str], dict[str, object]] = {}
_CACHE_LOCK = asyncio.Lock()


async def _get_http_client() -> httpx.AsyncClient:
    """Return a pooled reusable AsyncClient for booking HTTP calls."""
    global _HTTP_CLIENT
    if _HTTP_CLIENT is not None:
        return _HTTP_CLIENT
    async with _HTTP_CLIENT_LOCK:
        if _HTTP_CLIENT is not None:
            return _HTTP_CLIENT
        _HTTP_CLIENT = httpx.AsyncClient(timeout=TIMEOUT, limits=_HTTP_LIMITS)
        return _HTTP_CLIENT


def _norm_name(s: str) -> str:
    return (s or "").strip().lower()


def _cache_get(cache_entry: dict[str, object]) -> object | None:
    expires_at = float(cache_entry.get("expires_at") or 0.0)
    if time.monotonic() < expires_at:
        return cache_entry.get("value")
    return None


def _cache_set(cache_entry: dict[str, object], value: object, ttl_s: float) -> None:
    cache_entry["value"] = value
    cache_entry["expires_at"] = time.monotonic() + ttl_s


async def get_services() -> list[dict]:
    try:
        async with _CACHE_LOCK:
            cached = _cache_get(_services_cache)
        if cached is not None:
            log.info("[BOOKING_CLIENT] services cache hit")
            return cached  # type: ignore[return-value]

        t0 = time.perf_counter()
        client = await _get_http_client()
        resp = await client.get(f"{RAG_BACKEND_URL}/clinic/services")
        resp.raise_for_status()
        data = resp.json()
        services = [s for s in data.get("services", data) if s.get("active", 1)]
        dt_ms = (time.perf_counter() - t0) * 1000.0

        async with _CACHE_LOCK:
            _cache_set(_services_cache, services, _CACHE_TTL_SERVICES_S)

        log.info("[BOOKING_CLIENT] services fetched in %.1fms (cached for %.0fs)", dt_ms, _CACHE_TTL_SERVICES_S)
        return services
    except Exception as e:
        log.error("get_services failed: %s", e)
        return []


async def get_doctors_for_service(service_name: str) -> list[dict]:
    try:
        key = _norm_name(service_name)
        async with _CACHE_LOCK:
            entry = _doctors_cache.get(key)
            cached = _cache_get(entry) if entry else None
        if cached is not None:
            log.info("[BOOKING_CLIENT] doctors cache hit service=%r", service_name)
            return cached  # type: ignore[return-value]

        t0 = time.perf_counter()
        client = await _get_http_client()
        resp = await client.get(
            f"{RAG_BACKEND_URL}/clinic/doctors",
            params={"service": service_name},
        )
        resp.raise_for_status()
        data = resp.json()
        doctors = data.get("doctors", data)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        async with _CACHE_LOCK:
            _doctors_cache[key] = {"value": doctors, "expires_at": time.monotonic() + _CACHE_TTL_DOCTORS_S}

        log.info("[BOOKING_CLIENT] doctors fetched in %.1fms service=%r (cached for %.0fs)", dt_ms, service_name, _CACHE_TTL_DOCTORS_S)
        return doctors
    except Exception as e:
        log.error("get_doctors_for_service failed: %s", e)
        return []


async def get_slots(service_name: str, doctor_name: str) -> list[dict]:
    try:
        key = (_norm_name(service_name), _norm_name(doctor_name))
        async with _CACHE_LOCK:
            entry = _slots_cache.get(key)
            cached = _cache_get(entry) if entry else None
        if cached is not None:
            log.info("[BOOKING_CLIENT] slots cache hit service=%r doctor=%r", service_name, doctor_name)
            return cached  # type: ignore[return-value]

        t0 = time.perf_counter()
        client = await _get_http_client()
        resp = await client.get(
            f"{RAG_BACKEND_URL}/clinic/slots",
            params={
                # Backend expects these exact query param names for the agent path.
                "service": service_name,
                "doctor": doctor_name,
                "available_only": True,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        slots = data.get("slots", data)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        # Don't cache "no slots" for long; availability may change quickly.
        cached_this_call = False
        async with _CACHE_LOCK:
            if slots:
                _slots_cache[key] = {
                    "value": slots,
                    "expires_at": time.monotonic() + _CACHE_TTL_SLOTS_S,
                }
                cached_this_call = True

        if cached_this_call:
            log.info(
                "[BOOKING_CLIENT] slots fetched in %.1fms service=%r doctor=%r (cached for %.0fs)",
                dt_ms,
                service_name,
                doctor_name,
                _CACHE_TTL_SLOTS_S,
            )
        else:
            log.info(
                "[BOOKING_CLIENT] slots fetched in %.1fms service=%r doctor=%r (not cached: 0 slots)",
                dt_ms,
                service_name,
                doctor_name,
            )
        return slots
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
        t0 = time.perf_counter()
        client = await _get_http_client()
        resp = await client.post(
            f"{RAG_BACKEND_URL}/bookings/",
            json={
                "service_name": service_name,
                "doctor_name": doctor_name,
                "patient_name": patient_name,
                "slot_id": slot_id,
                "language": language,
            },
        )
        resp.raise_for_status()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        log.info("[BOOKING_CLIENT] save_appointment completed in %.1fms", dt_ms)
        return resp.json()
    except Exception as e:
        log.error("save_appointment failed: %s", e)
        return {}