"""
This is my
Agent Service — main.py
=========================
LiveKit voice agent for Functiomed clinic.

Booking flow:
  1. get_services()          → list services from DB
  2. confirm_service(choice) → validate
  3. get_doctors(service)    → list doctors from DB
  4. confirm_doctor(choice)  → validate
  5. get_slots()             → list available slots from DB
  6. confirm_slot(choice)    → validate + store slot_id  ← FIX
  7. Ask full name
  8. confirm_name(name)
  9. confirm_booking()       → summary → YES/NO
  10a. YES → save_appointment()
  10b. NO  → cancel_booking()
"""

import logging
import os

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import Agent, AgentSession, JobContext, JobProcess, RunContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.plugins import deepgram, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from booking.rag_client import (
    ask_rag,
    get_available_services,
    get_doctors_for_service,
    get_slots_for_doctor,
    validate_service,
    validate_doctor,
    validate_slot_choice,
    save_booking_to_rag,
    check_rag_health,
)
from booking.session import BookingSession, BookingStep, get_session, clear_session

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _spoken_list(items: list[str]) -> str:
    """['A','B','C'] → 'A, B, and C'"""
    if not items:
        return "none"
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


# ─────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────

class FunctiomedAgent(Agent):

    def __init__(self, room_name: str):
        self.room_name = room_name
        self._booking  = get_session(room_name)
        self._room: rtc.Room | None = None

        super().__init__(instructions=self._instructions())

    def _instructions(self) -> str:
        return """
You are a warm, professional voice assistant for Functiomed, a medical clinic in Zurich, Switzerland.
You handle two types of requests:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TYPE 1 — QUESTIONS about the clinic
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When the user asks ANY question about services, prices, opening hours, location,
treatments, insurance, or doctors — call ask_knowledge_base().
NEVER answer from memory. Always call the tool.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TYPE 2 — BOOKING an appointment
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Follow this EXACT sequence — no skipping steps:

  1.  Call get_services()           → present list → wait for choice
  2.  Call confirm_service(choice)  → validate
  3.  Call get_doctors(service)     → present list → wait for choice
  4.  Call confirm_doctor(choice)   → validate
  5.  Call get_slots()              → present available times → wait for choice
  6.  Call confirm_slot(choice)     → validate
  7.  Ask "What is your full name?" → wait
  8.  Call confirm_name(name)
  9.  Call confirm_booking()        → read summary → ask YES or NO
  10a. YES → call save_appointment()
  10b. NO  → call cancel_booking()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LANGUAGE: Detect from first message. English → English. German → German.
VOICE: Short natural sentences. No markdown, bullets, or asterisks.
LISTS: Speak naturally. "We have Monday at 9 AM, Tuesday at 11 AM, and Wednesday at 2 PM."
"""

    # ── DataChannel ───────────────────────────────────────────

    async def _send_dc(self, available: list[str] = None):
        if self._room is None:
            return
        try:
            msg = self._booking.to_datachannel_msg(available=available or [])
            await self._room.local_participant.publish_data(msg, reliable=True)
            logger.debug("DataChannel sent: step=%s", self._booking.step.value)
        except Exception as e:
            logger.warning("DataChannel send failed: %s", e)

    # ──────────────────────────────────────────────────────────
    # TOOL 1 — Clinic Q&A via RAG
    # ──────────────────────────────────────────────────────────

    @function_tool
    async def ask_knowledge_base(self, context: RunContext, question: str) -> str:
        """
        Answer any question about Functiomed clinic using the knowledge base.
        Use for ALL questions: services, hours, location, prices, treatments, insurance.

        Args:
            question: The user's question exactly as they asked it.
        """
        logger.info("[TOOL] ask_knowledge_base: %s", question)
        return await ask_rag(question, voice=True)

    # ──────────────────────────────────────────────────────────
    # TOOL 2 — Get services from DB
    # ──────────────────────────────────────────────────────────

    @function_tool
    async def get_services(self, context: RunContext, no_input: str = "") -> str:
        """
        Get the list of services offered at Functiomed from the database.
        Call this at the start of every booking conversation.
        """
        logger.info("[TOOL] get_services")
        try:
            services = await get_available_services()
            logger.info("[TOOL] get_services: %r", services)

            if not isinstance(services, list):
                services = []

            self._booking.available_services = services
            self._booking.step = BookingStep.COLLECT_SERVICE
            self._booking.reset_retries()

            await self._send_dc(available=services)

            if not services:
                return (
                    "I couldn't load the services list right now. "
                    "Please tell me which service you're looking for."
                )

            spoken = _spoken_list(services)
            logger.info("[TOOL] spoken services: %r", spoken)
            return f"We offer the following services: {spoken}. Which would you like to book?"

        except Exception as e:
            logger.exception("[TOOL] get_services CRASHED: %s", e)
            return "I'm having trouble loading services right now. Please try again."

    # ──────────────────────────────────────────────────────────
    # TOOL 3 — Confirm service
    # ──────────────────────────────────────────────────────────

    @function_tool
    async def confirm_service(self, context: RunContext, user_said: str) -> str:
        """
        Validate the user's service choice.
        Call this after the user names a service.

        Args:
            user_said: What the user said when choosing a service.
        """
        logger.info("[TOOL] confirm_service: %s", user_said)

        matched = await validate_service(user_said, self._booking.available_services)

        if matched:
            self._booking.service_name = matched
            self._booking.step         = BookingStep.COLLECT_DOCTOR
            self._booking.reset_retries()
            await self._send_dc()
            logger.info("[BOOKING] Service confirmed: %s", matched)
            return f"SERVICE_CONFIRMED:{matched}"
        else:
            if self._booking.next_retry():
                self._booking.reset()
                await self._send_dc()
                return "I'm having trouble understanding. Let me start over. How can I help you?"
            return (
                f"I didn't catch that. Our services are: "
                f"{_spoken_list(self._booking.available_services)}. Which would you like?"
            )

    # ──────────────────────────────────────────────────────────
    # TOOL 4 — Get doctors from DB
    # ──────────────────────────────────────────────────────────

    @function_tool
    async def get_doctors(self, context: RunContext, service_name: str) -> str:
        """
        Get available practitioners for the chosen service from the database.
        Call this immediately after service is confirmed.

        Args:
            service_name: The confirmed service name.
        """
        logger.info("[TOOL] get_doctors: %s", service_name)

        try:
            doctors = await get_doctors_for_service(service_name)

            self._booking.available_doctors = doctors
            self._booking.step = BookingStep.COLLECT_DOCTOR
            self._booking.reset_retries()

            await self._send_dc(available=doctors)

            if not doctors:
                return (
                    f"For {service_name}, please tell me which practitioner you'd prefer, "
                    f"or say 'any' if you have no preference."
                )

            spoken = _spoken_list(doctors)
            return f"For {service_name}, the available practitioners are: {spoken}. Who would you prefer?"

        except Exception as e:
            logger.exception("[TOOL] get_doctors CRASHED: %s", e)
            return "I'm having trouble loading practitioners. Please try again."

    # ──────────────────────────────────────────────────────────
    # TOOL 5 — Confirm doctor
    # ──────────────────────────────────────────────────────────

    @function_tool
    async def confirm_doctor(self, context: RunContext, user_said: str) -> str:
        """
        Validate the user's doctor choice.
        Call this after the user names a practitioner.

        Args:
            user_said: What the user said when choosing a doctor.
        """
        logger.info("[TOOL] confirm_doctor: %s", user_said)

        if any(w in user_said.lower() for w in ["any", "egal", "anyone", "whoever"]):
            chosen = self._booking.available_doctors[0] if self._booking.available_doctors else user_said
            self._booking.doctor_name = chosen
            self._booking.step        = BookingStep.COLLECT_SLOT
            self._booking.reset_retries()
            await self._send_dc()
            return f"DOCTOR_CONFIRMED:{chosen}"

        matched = await validate_doctor(user_said, self._booking.available_doctors)

        if matched:
            self._booking.doctor_name = matched
            self._booking.step        = BookingStep.COLLECT_SLOT
            self._booking.reset_retries()
            await self._send_dc()
            logger.info("[BOOKING] Doctor confirmed: %s", matched)
            return f"DOCTOR_CONFIRMED:{matched}"
        else:
            if self._booking.next_retry():
                self._booking.doctor_name = user_said
                self._booking.step        = BookingStep.COLLECT_SLOT
                await self._send_dc()
                return f"DOCTOR_CONFIRMED:{user_said}"
            docs = _spoken_list(self._booking.available_doctors)
            return f"I didn't catch that. Available practitioners: {docs}. Who would you prefer?"

    # ──────────────────────────────────────────────────────────
    # TOOL 6 — Get available slots from DB
    # ──────────────────────────────────────────────────────────

    @function_tool
    async def get_slots(self, context: RunContext, no_input: str = "") -> str:
        """
        Get available appointment slots for the chosen doctor and service.
        Call this immediately after doctor is confirmed.
        """
        logger.info(
            "[TOOL] get_slots: service=%s doctor=%s",
            self._booking.service_name, self._booking.doctor_name,
        )

        if not self._booking.service_name or not self._booking.doctor_name:
            return "I need the service and doctor before checking slots. Let me go back."

        try:
            slots, spoken_options = await get_slots_for_doctor(
                service_name=self._booking.service_name,
                doctor_name=self._booking.doctor_name,
            )

            # FIX: store BOTH the full slot dicts AND the spoken labels on the session
            self._booking.available_slots       = slots
            self._booking.available_slot_labels = spoken_options

            self._booking.step = BookingStep.COLLECT_SLOT
            self._booking.reset_retries()

            await self._send_dc(available=spoken_options)

            if not slots:
                return (
                    f"Unfortunately there are no available slots for "
                    f"{self._booking.doctor_name} right now. "
                    f"Would you like to choose a different doctor or contact the clinic directly?"
                )

            spoken = _spoken_list(spoken_options)
            return f"Here are the available times: {spoken}. Which would you prefer?"

        except Exception as e:
            logger.exception("[TOOL] get_slots CRASHED: %s", e)
            return "I'm having trouble loading available slots. Please try again."

    # ──────────────────────────────────────────────────────────
    # TOOL 7 — Confirm slot choice
    # ──────────────────────────────────────────────────────────

    @function_tool
    async def confirm_slot(self, context: RunContext, user_said: str) -> str:
        """
        Validate the user's slot choice.
        Call this after the user picks an appointment time.

        Args:
            user_said: What the user said (e.g. "option 1", "Monday", "the first one").
        """
        logger.info("[TOOL] confirm_slot: %s", user_said)

        slots          = self._booking.available_slots
        # FIX: use stored labels from get_slots, not rebuilt locally
        spoken_options = self._booking.available_slot_labels

        # Safety fallback if labels weren't stored
        if not spoken_options:
            spoken_options = [
                f"Option {i+1}: {s['slot_date']} at {s['slot_time']}"
                for i, s in enumerate(slots)
            ]

        matched = validate_slot_choice(user_said, slots, spoken_options)

        if matched:
            # FIX: find index by slot ID not object identity
            idx = next(
                (i for i, s in enumerate(slots) if s["id"] == matched["id"]),
                0,
            )

            self._booking.slot_id    = matched["id"]
            self._booking.slot_date  = matched["slot_date"]
            self._booking.slot_time  = matched["slot_time"]
            self._booking.slot_label = (
                spoken_options[idx]
                if idx < len(spoken_options)
                else f"{matched['slot_date']} at {matched['slot_time']}"
            )
            self._booking.step = BookingStep.COLLECT_NAME
            self._booking.reset_retries()
            await self._send_dc()

            logger.info(
                "[BOOKING] Slot confirmed: id=%s date=%s time=%s label=%s",
                matched["id"], matched["slot_date"],
                matched["slot_time"], self._booking.slot_label,
            )
            return f"SLOT_CONFIRMED:{matched['slot_date']} at {matched['slot_time']}"

        else:
            if self._booking.next_retry():
                # Proceed without slot after max retries
                self._booking.slot_id    = None
                self._booking.slot_date  = None
                self._booking.slot_time  = None
                self._booking.slot_label = None
                self._booking.step = BookingStep.COLLECT_NAME
                await self._send_dc()
                return "SLOT_CONFIRMED:no_preference"

            options_spoken = _spoken_list(spoken_options)
            return (
                f"I didn't catch that. Available times are: {options_spoken}. "
                f"Which would you prefer?"
            )

    # ──────────────────────────────────────────────────────────
    # TOOL 8 — Save patient name
    # ──────────────────────────────────────────────────────────

    @function_tool
    async def confirm_name(self, context: RunContext, full_name: str) -> str:
        """
        Save the patient's full name.
        Call this after the user says their name.

        Args:
            full_name: The patient's full name as spoken.
        """
        logger.info("[TOOL] confirm_name: %s", full_name)

        if not full_name or len(full_name.strip()) < 2:
            return "I didn't catch your name. Could you repeat it please?"

        self._booking.patient_name = full_name.strip()
        self._booking.step         = BookingStep.CONFIRM
        self._booking.reset_retries()

        await self._send_dc()
        logger.info("[BOOKING] Name: %s", full_name)
        return f"NAME_CONFIRMED:{full_name}"

    # ──────────────────────────────────────────────────────────
    # TOOL 9 — Confirm booking summary
    # ──────────────────────────────────────────────────────────

    @function_tool
    async def confirm_booking(self, context: RunContext, no_input: str = "") -> str:
        """
        Read back the complete booking summary and ask YES or NO.
        Call this after service, doctor, slot, and name have all been collected.
        """
        logger.info("[TOOL] confirm_booking")

        if not self._booking.is_complete():
            missing = [
                k for k, v in [
                    ("service", self._booking.service_name),
                    ("doctor",  self._booking.doctor_name),
                    ("name",    self._booking.patient_name),
                ] if not v
            ]
            return f"I still need: {', '.join(missing)}."

        self._booking.step = BookingStep.CONFIRM
        await self._send_dc()
        return self._booking.summary()

    # ──────────────────────────────────────────────────────────
    # TOOL 10 — Save booking to DB
    # ──────────────────────────────────────────────────────────

    @function_tool
    async def save_appointment(self, context: RunContext, no_input: str = "") -> str:
        """
        Save the confirmed booking to the database.
        Call this ONLY after the user says YES to the booking summary.
        """
        logger.info("[TOOL] save_appointment")

        if not self._booking.is_complete():
            return "Booking details are incomplete. Let me start over."

        try:
            record = await save_booking_to_rag(
                service_name    = self._booking.service_name,
                doctor_name     = self._booking.doctor_name,
                patient_name    = self._booking.patient_name,
                slot_id         = self._booking.slot_id,
                language        = self._booking.language,
                session_summary = (
                    f"Service={self._booking.service_name}, "
                    f"Doctor={self._booking.doctor_name}, "
                    f"Slot={self._booking.slot_date or 'N/A'} {self._booking.slot_time or ''}, "
                    f"Patient={self._booking.patient_name}"
                ),
            )

            conf = record["confirmation_number"]
            self._booking.confirmation_number = conf
            self._booking.step                = BookingStep.DONE

            await self._send_dc()
            logger.info("[BOOKING] SAVED ✅ Confirmation: %s", conf)

            slot_part = ""
            if record.get("slot_date") and record.get("slot_time"):
                slot_part = f" on {record['slot_date']} at {record['slot_time']}"

            return (
                f"Your appointment is confirmed{slot_part}! "
                f"Your confirmation number is {conf}. "
                f"We look forward to seeing {self._booking.patient_name} "
                f"for {self._booking.service_name} with {self._booking.doctor_name}. "
                f"Is there anything else I can help you with?"
            )

        except Exception as e:
            logger.error("save_appointment failed: %s", e)
            return (
                "I'm sorry, there was a problem saving your appointment. "
                "Please call the clinic directly to confirm your booking."
            )

    # ──────────────────────────────────────────────────────────
    # TOOL 11 — Cancel booking
    # ──────────────────────────────────────────────────────────

    @function_tool
    async def cancel_booking(self, context: RunContext, no_input: str = "") -> str:
        """
        Cancel the current in-progress booking.
        Call when user says NO or explicitly asks to cancel.
        """
        logger.info("[TOOL] cancel_booking")
        self._booking.step = BookingStep.CANCELLED
        await self._send_dc()
        self._booking.reset()
        return "No problem, I've cancelled the booking. How else can I help you?"

    # ── Lifecycle ─────────────────────────────────────────────

    async def on_enter(self):
        logger.info("Agent entered room: %s", self.room_name)
        self._room = self.session.room if hasattr(self.session, "room") else None

        if not await check_rag_health():
            logger.warning("⚠️  Backend unreachable at %s", os.getenv("RAG_BACKEND_URL"))

        await self.session.generate_reply(
            instructions=(
                "Greet the user warmly. Say you're the Functiomed voice assistant. "
                "Tell them you can answer questions about the clinic or help book an appointment. "
                "Ask how you can help. Two sentences max."
            )
        )

    async def on_exit(self):
        logger.info("Agent exiting room: %s", self.room_name)
        clear_session(self.room_name)


# ─────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────

async def entrypoint(ctx: JobContext):
    logger.info("New room: %s", ctx.room.name)

    session = AgentSession(
        stt=deepgram.STT(model="nova-2"),
        llm=openai.LLM(
            model=os.getenv("AGENT_LLM_MODEL", "gpt-4o-mini"),
            temperature=0.4,
        ),
        tts=openai.TTS(voice="echo", speed=1.0),
        vad=ctx.proc.userdata.get("vad") or silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    agent = FunctiomedAgent(room_name=ctx.room.name)

    await session.start(
        room=ctx.room,
        agent=agent,
    )

    agent._room = ctx.room


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))