"""
Agent Service — booking/session.py
=====================================
Booking state per LiveKit room + DataChannel message builder.

Every time the booking state changes, the agent sends a
DataChannel message to the frontend so the React UI can
update its booking progress stepper in real time.

DataChannel message format (JSON string):
{
  "type":    "booking_update",
  "step":    "collect_service" | "collect_doctor" | "collect_name" | "confirm" | "done" | "cancelled",
  "data": {
    "service":  "Physiotherapy" | null,
    "doctor":   "Dr. Müller"    | null,
    "name":     "Hans Schmidt"  | null,
    "confirmation": "FM-2025-AB12CD" | null
  },
  "available": ["option1", "option2"]   // shown as chips in UI
}
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BookingStep(str, Enum):
    IDLE            = "idle"
    COLLECT_SERVICE = "collect_service"
    COLLECT_DOCTOR  = "collect_doctor"
    COLLECT_NAME    = "collect_name"
    CONFIRM         = "confirm"
    DONE            = "done"
    CANCELLED       = "cancelled"


@dataclass
class BookingSession:
    step: BookingStep = BookingStep.IDLE

    available_services: list[str] = field(default_factory=list)
    available_doctors:  list[str] = field(default_factory=list)

    service_name:  Optional[str] = None
    doctor_name:   Optional[str] = None
    patient_name:  Optional[str] = None
    confirmation_number: Optional[str] = None

    language: str = "en"
    retries:  int = 0
    MAX_RETRIES: int = 2

    def reset(self):
        self.step               = BookingStep.IDLE
        self.available_services = []
        self.available_doctors  = []
        self.service_name       = None
        self.doctor_name        = None
        self.patient_name       = None
        self.confirmation_number = None
        self.retries            = 0

    def next_retry(self) -> bool:
        self.retries += 1
        return self.retries > self.MAX_RETRIES

    def reset_retries(self):
        self.retries = 0

    def is_complete(self) -> bool:
        return all([self.service_name, self.doctor_name, self.patient_name])

    def summary(self) -> str:
        """Voice-friendly confirmation summary."""
        if self.language == "de":
            return (
                f"Ich buche {self.service_name} bei {self.doctor_name} "
                f"für {self.patient_name}. Ist das korrekt?"
            )
        return (
            f"Let me confirm: {self.service_name} with {self.doctor_name} "
            f"for {self.patient_name}. Shall I confirm this booking?"
        )

    # ── DataChannel payload builder ───────────────────────────

    def to_datachannel_msg(self, available: list[str] = None) -> bytes:
        """
        Build a JSON bytes message to send over LiveKit DataChannel.
        The React frontend listens for these and updates the UI.

        Usage:
            await ctx.room.local_participant.publish_data(
                session.to_datachannel_msg(available=session.available_services),
                reliable=True,
            )
        """
        payload = {
            "type": "booking_update",
            "step": self.step.value,
            "data": {
                "service":      self.service_name,
                "doctor":       self.doctor_name,
                "name":         self.patient_name,
                "confirmation": self.confirmation_number,
            },
            "available": available or [],
        }
        return json.dumps(payload).encode("utf-8")


# ── Global session store ──────────────────────────────────────

_sessions: dict[str, BookingSession] = {}


def get_session(room_name: str) -> BookingSession:
    if room_name not in _sessions:
        _sessions[room_name] = BookingSession()
    return _sessions[room_name]


def clear_session(room_name: str):
    _sessions.pop(room_name, None)