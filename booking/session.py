"""
Agent Service — booking/session.py
=====================================
Booking state per LiveKit room + DataChannel message builder.

DataChannel message format (JSON string):
{
  "type": "booking_update",
  "step": "collect_service" | "collect_doctor" | "collect_slot"
          | "collect_name" | "confirm" | "done" | "cancelled",
  "data": {
    "service":      "IV Therapy"               | null,
    "doctor":       "Dr. Stefan Koch"          | null,
    "slot":         "Wednesday March 12 at 9 AM" | null,
    "slot_date":    "2025-03-12"               | null,
    "slot_time":    "09:00"                    | null,
    "name":         "Hans Schmidt"             | null,
    "confirmation": "FM-2025-AB12CD"           | null
  },
  "available": ["option1", "option2"]
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
    COLLECT_SLOT    = "collect_slot"
    COLLECT_NAME    = "collect_name"
    CONFIRM         = "confirm"
    DONE            = "done"
    CANCELLED       = "cancelled"


@dataclass
class BookingSession:
    step: BookingStep = BookingStep.IDLE

    available_services:   list[str]  = field(default_factory=list)
    available_doctors:    list[str]  = field(default_factory=list)
    available_slots:      list[dict] = field(default_factory=list)   # full slot dicts from DB
    available_slot_labels: list[str] = field(default_factory=list)  # FIX: spoken labels stored here

    service_name: Optional[str] = None
    doctor_name:  Optional[str] = None
    patient_name: Optional[str] = None

    # Slot fields
    slot_id:    Optional[str] = None   # UUID from DB slots table
    slot_date:  Optional[str] = None   # "2025-03-12"
    slot_time:  Optional[str] = None   # "09:00"
    slot_label: Optional[str] = None   # "Wednesday March 12th at 9 AM"

    confirmation_number: Optional[str] = None

    language: str = "en"
    retries:  int = 0
    MAX_RETRIES: int = 3

    def reset(self):
        self.step                  = BookingStep.IDLE
        self.available_services    = []
        self.available_doctors     = []
        self.available_slots       = []
        self.available_slot_labels = []
        self.service_name          = None
        self.doctor_name           = None
        self.patient_name          = None
        self.slot_id               = None
        self.slot_date             = None
        self.slot_time             = None
        self.slot_label            = None
        self.confirmation_number   = None
        self.retries               = 0

    def next_retry(self) -> bool:
        self.retries += 1
        return self.retries > self.MAX_RETRIES

    def reset_retries(self):
        self.retries = 0

    def is_complete(self) -> bool:
        return all([self.service_name, self.doctor_name, self.patient_name])

    def summary(self) -> str:
        """Voice-friendly confirmation summary."""
        slot_part = f" on {self.slot_label}" if self.slot_label else ""
        if self.language == "de":
            return (
                f"Ich buche {self.service_name} bei {self.doctor_name}"
                f"{slot_part} für {self.patient_name}. Ist das korrekt?"
            )
        return (
            f"Let me confirm: {self.service_name} with {self.doctor_name}"
            f"{slot_part} for {self.patient_name}. Shall I confirm this booking?"
        )

    def to_datachannel_msg(self, available: list[str] = None) -> bytes:
        """Build JSON bytes for LiveKit DataChannel → React frontend."""
        payload = {
            "type": "booking_update",
            "step": self.step.value,
            "data": {
                "service":      self.service_name,
                "doctor":       self.doctor_name,
                "slot":         self.slot_label,
                "slot_date":    self.slot_date,
                "slot_time":    self.slot_time,
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
#