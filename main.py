"""
Agent Service — main.py
=========================
LiveKit voice agent for Functiomed clinic.

Key architectural decisions based on your answers:
  ✅ RAG backend   → http://localhost:8000  (HTTP calls for all knowledge + booking save)
  ✅ Bookings      → saved in RAG backend's SQLite via POST /bookings
  ✅ DataChannel   → sent to frontend after EVERY booking step change

DataChannel flow to React frontend:
  Agent confirms service → sends {"type":"booking_update","step":"collect_doctor",...}
  Agent confirms doctor  → sends {"type":"booking_update","step":"collect_name",...}
  Agent confirms name    → sends {"type":"booking_update","step":"confirm",...}
  Booking saved          → sends {"type":"booking_update","step":"done","data":{confirmation:...}}

Run:
    python main.py download-files   ← first time only
    python main.py console          ← test locally with mic
    python main.py start            ← production with LiveKit server
"""

import json
import logging
import os

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, AgentSession, JobContext, JobProcess, RunContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.plugins import deepgram, openai, silero, groq
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from booking.rag_client import (
    ask_rag,
    get_available_services,
    get_doctors_for_service,
    validate_service,
    validate_doctor,
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
# Agent
# ─────────────────────────────────────────────────────────────

class FunctiomedAgent(Agent):
    """
    Functiomed voice agent.

    Holds a reference to:
      - self._booking  : BookingSession (current booking state)
      - self._room     : LiveKit room  (for DataChannel publishing)

    The room reference is set in on_enter() because ctx.room
    is only available after the session starts.
    """

    def __init__(self, room_name: str):
        self.room_name = room_name
        self._booking  = get_session(room_name)
        self._room: rtc.Room | None = None   # set in on_enter

        super().__init__(instructions=self._instructions())

    # ── System prompt ─────────────────────────────────────────

    def _instructions(self) -> str:
        return """
You are a warm, professional voice assistant for Functiomed, a medical clinic in Zurich, Switzerland.
You handle two types of requests:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TYPE 1 — QUESTIONS about the clinic
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When the user asks ANY question about services, prices, opening hours, location,
treatments, insurance, or doctors — call ask_knowledge_base().
NEVER answer clinic questions from your own memory. Always call the tool.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TYPE 2 — BOOKING an appointment
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When the user wants to book, follow this EXACT sequence — no skipping steps:

  1. Call get_services()          → present list → wait for user to choose
  2. Call confirm_service(choice) → if valid, proceed; if not, ask again
  3. Call get_doctors(service)    → present list → wait for user to choose
  4. Call confirm_doctor(choice)  → if valid, proceed; if not, ask again
  5. Ask "What is your full name?" → wait for answer
  6. Call confirm_name(name)
  7. Call confirm_booking()       → reads back summary → ask YES or NO
  8a. User says YES → call save_appointment()
  8b. User says NO  → call cancel_booking()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LANGUAGE: Detect from first message. German → respond in German. English → English. Never mix.
VOICE: Short natural sentences. No markdown, no bullet points, no asterisks.
LISTS: Speak options naturally. "We offer physiotherapy, massage, and osteopathy."
"""

    # ── DataChannel helper ────────────────────────────────────

    async def _send_dc(self, available: list[str] = None):
        """
        Send current booking state to the React frontend via DataChannel.
        Called after every booking step change.
        """
        if self._room is None:
            return
        try:
            msg = self._booking.to_datachannel_msg(available=available or [])
            await self._room.local_participant.publish_data(msg, reliable=True)
            logger.debug("DataChannel sent: step=%s", self._booking.step.value)
        except Exception as e:
            logger.warning("DataChannel send failed: %s", e)

    # ──────────────────────────────────────────────────────────
    # TOOL 1 — Answer clinic questions via RAG
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
    # TOOL 2 — Get available services (booking step 1)
    # ──────────────────────────────────────────────────────────

    @function_tool
    async def get_services(self, context: RunContext) -> str:
        """
        Get the list of services offered at Functiomed from the knowledge base.
        Call this at the start of every booking conversation.
        """
        logger.info("[TOOL] get_services")

        services = await get_available_services()
        self._booking.available_services = services
        self._booking.step = BookingStep.COLLECT_SERVICE
        self._booking.reset_retries()

        # Notify frontend
        await self._send_dc(available=services)

        if not services:
            return (
                "I couldn't load the services list right now. "
                "Please tell me which service you're looking for."
            )

        spoken = _spoken_list(services)
        return f"We offer the following services: {spoken}. Which would you like to book?"

    # ──────────────────────────────────────────────────────────
    # TOOL 3 — Confirm service choice (booking step 2)
    # ──────────────────────────────────────────────────────────

    @function_tool
    async def confirm_service(self, context: RunContext, user_said: str) -> str:
        """
        Validate the user's service choice against available services.
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
            logger.info("[BOOKING] Service: %s", matched)
            return f"SERVICE_CONFIRMED:{matched}"
        else:
            if self._booking.next_retry():
                self._booking.reset()
                await self._send_dc()
                return "I'm having trouble understanding. Let me start over. How can I help you?"
            return (
                f"I didn't catch that. Available services are: "
                f"{_spoken_list(self._booking.available_services)}. Which would you like?"
            )

    # ──────────────────────────────────────────────────────────
    # TOOL 4 — Get doctors for service (booking step 3)
    # ──────────────────────────────────────────────────────────

    @function_tool
    async def get_doctors(self, context: RunContext, service_name: str) -> str:
        """
        Get available practitioners for the chosen service from the knowledge base.
        Call this immediately after the service is confirmed.

        Args:
            service_name: The confirmed service name.
        """
        logger.info("[TOOL] get_doctors: %s", service_name)

        doctors = await get_doctors_for_service(service_name)
        self._booking.available_doctors = doctors
        self._booking.step = BookingStep.COLLECT_DOCTOR
        self._booking.reset_retries()

        await self._send_dc(available=doctors)

        if not doctors:
            return (
                f"For {service_name}, please tell me which practitioner you'd like, "
                f"or say 'any' if you have no preference."
            )

        spoken = _spoken_list(doctors)
        return f"For {service_name}, the available practitioners are: {spoken}. Who would you prefer?"

    # ──────────────────────────────────────────────────────────
    # TOOL 5 — Confirm doctor choice (booking step 4)
    # ──────────────────────────────────────────────────────────

    @function_tool
    async def confirm_doctor(self, context: RunContext, user_said: str) -> str:
        """
        Validate the user's doctor/practitioner choice.
        Call this after the user names a practitioner.

        Args:
            user_said: What the user said when choosing a doctor.
        """
        logger.info("[TOOL] confirm_doctor: %s", user_said)

        # "any" preference → pick first available
        if any(w in user_said.lower() for w in ["any", "egal", "anyone", "whoever"]):
            chosen = self._booking.available_doctors[0] \
                if self._booking.available_doctors else user_said
            self._booking.doctor_name = chosen
            self._booking.step        = BookingStep.COLLECT_NAME
            self._booking.reset_retries()
            await self._send_dc()
            return f"DOCTOR_CONFIRMED:{chosen}"

        matched = await validate_doctor(user_said, self._booking.available_doctors)

        if matched:
            self._booking.doctor_name = matched
            self._booking.step        = BookingStep.COLLECT_NAME
            self._booking.reset_retries()
            await self._send_dc()
            logger.info("[BOOKING] Doctor: %s", matched)
            return f"DOCTOR_CONFIRMED:{matched}"
        else:
            if self._booking.next_retry():
                # Accept free-text after max retries
                self._booking.doctor_name = user_said
                self._booking.step        = BookingStep.COLLECT_NAME
                await self._send_dc()
                return f"DOCTOR_CONFIRMED:{user_said}"
            docs = _spoken_list(self._booking.available_doctors)
            return f"I didn't catch that. Available practitioners: {docs}. Who would you prefer?"

    # ──────────────────────────────────────────────────────────
    # TOOL 6 — Save patient name (booking step 5)
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
    # TOOL 7 — Read back summary (booking step 6)
    # ──────────────────────────────────────────────────────────

    @function_tool
    async def confirm_booking(self, context: RunContext) -> str:
        """
        Read back the complete booking summary and ask the user to confirm.
        Call this after service, doctor, and name have all been collected.
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
    # TOOL 8 — Save booking to RAG backend SQLite ✅
    # ──────────────────────────────────────────────────────────

    @function_tool
    async def save_appointment(self, context: RunContext) -> str:
        """
        Save the confirmed booking to the RAG backend database.
        Call this ONLY after the user has said YES to the booking summary.
        """
        logger.info("[TOOL] save_appointment")

        if not self._booking.is_complete():
            return "Booking details are incomplete. Let me start over."

        try:
            record = await save_booking_to_rag(
                service_name    = self._booking.service_name,
                doctor_name     = self._booking.doctor_name,
                patient_name    = self._booking.patient_name,
                language        = self._booking.language,
                session_summary = (
                    f"Service={self._booking.service_name}, "
                    f"Doctor={self._booking.doctor_name}, "
                    f"Patient={self._booking.patient_name}"
                ),
            )

            conf = record["confirmation_number"]
            self._booking.confirmation_number = conf
            self._booking.step                = BookingStep.DONE

            # Send final DataChannel update with confirmation number
            await self._send_dc()

            logger.info("[BOOKING] SAVED ✅ Confirmation: %s", conf)

            return (
                f"Your appointment is confirmed! "
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
    # TOOL 9 — Cancel booking ❌
    # ──────────────────────────────────────────────────────────

    @function_tool
    async def cancel_booking(self, context: RunContext) -> str:
        """
        Cancel the current in-progress booking.
        Call when user says NO to the confirmation, or explicitly asks to cancel.
        """
        logger.info("[TOOL] cancel_booking")
        self._booking.step = BookingStep.CANCELLED
        await self._send_dc()
        self._booking.reset()
        return "No problem, I've cancelled the booking. How else can I help you?"

    # ──────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────

    async def on_enter(self):
        """Called when agent becomes active in the room."""
        logger.info("Agent entered room: %s", self.room_name)

        # Store room reference for DataChannel publishing
        # session.room is accessible after session.start()
        self._room = self.session.room if hasattr(self.session, "room") else None

        # Warn if RAG is not reachable
        if not await check_rag_health():
            logger.warning("⚠️  RAG backend unreachable at %s",
                           os.getenv("RAG_BACKEND_URL"))

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

        # Agent LLM — drives conversation and tool calling
        # This is SEPARATE from the RAG LLM inside the RAG backend
        llm=groq.LLM(
            model=os.getenv("AGENT_LLM_MODEL", "llama-3.3-70b-versatile"),
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

    # Give the agent a reference to the room AFTER session starts
    agent._room = ctx.room


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _spoken_list(items: list[str]) -> str:
    """Turn a list into natural spoken English. ['A','B','C'] → 'A, B, and C'"""
    if not items:
        return "none"
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"