
"""
agent/main.py — Functiomed Voice Agent (Single LLM, Groq compatible)
"""

import asyncio
import logging
import os

from dotenv import load_dotenv
load_dotenv()

from livekit import agents
from livekit.agents import AgentSession, Agent, ChatContext, ChatMessage, RoomInputOptions
from livekit.plugins import openai, deepgram, silero
from livekit.agents import function_tool, RunContext

from booking.rag_client import retrieve_context, health_check
from booking.booking_client import (
    get_services,
    get_doctors_for_service,
    get_slots,
    save_appointment,
)

log = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are the friendly voice assistant for Functiomed, a medical clinic in Zurich.
You speak naturally and concisely — this is a voice conversation, so keep answers short and clear.

Your two responsibilities:
1. Answer questions about the clinic using the knowledge injected into your context.
2. Help patients book appointments using your booking tools.

Critical voice constraints:
- Never read out the retrieved context or quote long passages.
- Keep answers very short (aim for <= 2 sentences / ~40 words) unless the user asks for details.
- If you need to list options (services/doctors/slots), list only the top few and ask a follow-up question.

Critical booking constraints:
- For booking flows, use ONLY the booking tools (database-backed). Do not answer booking questions from retrieved context.
- When the user expresses booking intent, immediately call get_services.

Booking flow (follow this order strictly):
  1. get_services → confirm_service
  2. get_doctors  → confirm_doctor
  3. get_slots    → confirm_slot (IMPORTANT: pass the exact slot_id UUID shown in the list)
  4. confirm_name → save_appointment

Rules:
- Always start the conversation in English. Switch to German only if the user speaks German first.
- Never make up clinic information. If context is missing, say you will check with the team.
- Keep responses under 3 sentences for simple questions.
- For booking, confirm each step before moving to the next.
- When calling confirm_slot, always pass the exact slot_id UUID string from the get_slots list.
- Never lowercase or paraphrase service or doctor names — use them exactly as provided.
"""

# ── Booking state ─────────────────────────────────────────────

class BookingState:
    def __init__(self):
        self.service:      str | None = None
        # Store DB-matchable doctor name (doctors.full_name), without title prefixes.
        self.doctor:       str | None = None
        self.slot_id:      str | None = None
        self.slot_date:    str | None = None
        self.slot_time:    str | None = None
        self.patient_name: str | None = None
        self.step:         str        = "idle"
        self._services:    list       = []
        self._doctors:     list       = []
        self._slots:       list       = []


# ── Agent ─────────────────────────────────────────────────────

class FunctiomedAgent(Agent):
    def __init__(self, chat_ctx: ChatContext, mode: str = "rag") -> None:
        super().__init__(instructions=SYSTEM_PROMPT, chat_ctx=chat_ctx)
        self._booking = BookingState()
        self._mode = (mode or "rag").strip().lower()

    @property
    def mode(self) -> str:
        return self._mode

    def _is_booking_intent(self, text: str) -> bool:
        t = (text or "").lower()
        return any(
            kw in t
            for kw in (
                "book", "booking", "appointment", "schedule",
                "termin", "buchen", "buchung",
            )
        )

    # ── RAG injection ─────────────────────────────────────────

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        user_text = new_message.text_content or ""

        # Mode lock: booking sessions never inject RAG; RAG sessions never initiate booking.
        if self.mode == "booking":
            # Booking-only: never inject RAG, even for non-booking utterances.
            return
        if self.mode == "rag" and self._is_booking_intent(user_text):
            turn_ctx.add_message(
                role="assistant",
                content=(
                    "I can answer clinic questions in this session. "
                    "To book an appointment, please start a Booking session."
                ),
            )
            return

        # If the user is trying to book, do NOT inject RAG.
        # Long retrieved context increases tool-calling failures and can exceed TTS limits.
        booking_intent = self._is_booking_intent(user_text)
        if booking_intent:
            log.info("[RAG] Skipping — booking intent detected")

            # Force the LLM to start the DB-backed booking flow immediately.
            # (We still skip RAG injection here.)
            if self._booking.step in ("idle", "done"):
                self._booking.step = "idle"
                turn_ctx.add_message(
                    role="system",
                    content=(
                        "User wants to book an appointment. "
                        "Immediately call the get_services tool, then ask the user to choose a service."
                    ),
                )
            return

        if self._booking.step not in ("idle", "done"):
            log.info("[RAG] Skipping — booking in progress (step=%s)", self._booking.step)
            return

        context = await retrieve_context(user_text, top_k=5)
        if context:
            turn_ctx.add_message(
                # System message so it guides the LLM but is never treated as something to say aloud.
                role="system",
                content=f"[Clinic knowledge retrieved for this query]\n\n{context}",
            )
            log.info("[RAG] Injected %d chars of context", len(context))
        else:
            log.info("[RAG] No context retrieved or backend unavailable")

    # ── Booking tools ─────────────────────────────────────────
    # NOTE: Tools with no parameters must declare dummy: str = "" 
    # to satisfy Groq's strict JSON schema (no empty required array)

    @function_tool()
    async def get_services(self, context: RunContext, dummy: str = "") -> str:
        """Get all available clinic services."""
        if self.mode != "booking":
            return "Booking is disabled in this session. Please start a Booking session."
        services = await get_services()
        if not services:
            return "No services are currently available."
        self._booking._services = services
        names = [s["name"] for s in services]
        return f"Available services: {', '.join(names)}"

    @function_tool()
    async def confirm_service(self, context: RunContext, service_name: str) -> str:
        """Confirm the service the patient wants to book."""
        if self.mode != "booking":
            return "Booking is disabled in this session. Please start a Booking session."
        service_name = (service_name or "").strip()
        stored  = self._booking._services
        matched = next(
            (s for s in stored
             if s["name"].lower() == service_name.lower()
             or service_name.lower() in s["name"].lower()),
            None
        )
        exact_name            = (matched["name"] if matched else service_name).strip()
        self._booking.service = exact_name
        self._booking.step    = "service"
        return f"Service confirmed: {exact_name}. Let me find available doctors."

    @function_tool()
    async def get_doctors(self, context: RunContext, dummy: str = "") -> str:
        """Get doctors available for the selected service."""
        if self.mode != "booking":
            return "Booking is disabled in this session. Please start a Booking session."
        if not self._booking.service:
            return "Please select a service first."
        doctors = await get_doctors_for_service(self._booking.service)
        if not doctors:
            return f"No doctors available for {self._booking.service}."
        self._booking._doctors = doctors
        lines = []
        for d in doctors:
            title   = d.get("title", "").strip()
            name    = d.get("full_name", "").strip()
            display = f"{title} {name}".strip() if title else name
            lines.append(display)
        return f"Available doctors: {', '.join(lines)}"

    @function_tool()
    async def confirm_doctor(self, context: RunContext, doctor_name: str) -> str:
        """Confirm the doctor the patient wants to see."""
        if self.mode != "booking":
            return "Booking is disabled in this session. Please start a Booking session."
        stored  = self._booking._doctors
        matched = next(
            (d for d in stored
             if doctor_name.lower() in d.get("full_name", "").lower()
             or d.get("full_name", "").lower() in doctor_name.lower()),
            None
        )
        if matched:
            title      = matched.get("title", "").strip()
            full_name  = matched["full_name"].strip()
            # Display name can include title, but API/DB matching should use full_name only.
            display_name = f"{title} {full_name}".strip() if title else full_name
            db_name      = full_name
        else:
            display_name = doctor_name.strip()
            db_name      = doctor_name.strip()

        self._booking.doctor = db_name
        self._booking.step   = "doctor"
        return f"Doctor confirmed: {display_name}. Let me check available time slots."

    @function_tool()
    async def get_slots(self, context: RunContext, dummy: str = "") -> str:
        """Get available appointment slots for the selected service and doctor."""
        if self.mode != "booking":
            return "Booking is disabled in this session. Please start a Booking session."
        if not self._booking.service or not self._booking.doctor:
            return "Please select a service and doctor first."
        slots = await get_slots(self._booking.service, self._booking.doctor)
        if not slots:
            return "No available slots at this time. Please call the clinic directly."
        self._booking._slots = slots
        lines = []
        for i, s in enumerate(slots[:5], 1):
            lines.append(f"{i}. {s['slot_date']} at {s['slot_time']} — slot_id: {s['id']}")
        return (
            "Available slots:\n" + "\n".join(lines) +
            "\n\nWhen confirming, pass the exact slot_id UUID shown above."
        )

    @function_tool()
    async def confirm_slot(
        self,
        context: RunContext,
        slot_date: str,
        slot_time: str,
        slot_id: str,
    ) -> str:
        """Confirm the chosen slot. slot_id must be the UUID from get_slots, never a number."""
        if self.mode != "booking":
            return "Booking is disabled in this session. Please start a Booking session."
        stored_slots = self._booking._slots

        matched = next(
            (s for s in stored_slots if str(s["id"]) == str(slot_id)),
            None
        )

        if not matched:
            matched = next(
                (s for s in stored_slots
                 if s["slot_date"] == slot_date and s["slot_time"] == slot_time),
                None
            )
            if matched:
                log.warning(
                    "[SLOT] LLM passed slot_id=%s, resolved by date+time to UUID=%s",
                    slot_id, matched["id"]
                )

        if not matched:
            slots_available = "\n".join(
                f"  {s['slot_date']} at {s['slot_time']} (slot_id: {s['id']})"
                for s in stored_slots[:5]
            )
            return f"I couldn't find that slot. Please choose from:\n{slots_available}"

        self._booking.slot_id   = matched["id"]
        self._booking.slot_date = matched["slot_date"]
        self._booking.slot_time = matched["slot_time"]
        self._booking.step      = "slot"
        return (
            f"Slot confirmed: {matched['slot_date']} at {matched['slot_time']}. "
            "May I have your full name to complete the booking?"
        )

    @function_tool()
    async def confirm_name(self, context: RunContext, patient_name: str) -> str:
        """Confirm the patient's full name."""
        if self.mode != "booking":
            return "Booking is disabled in this session. Please start a Booking session."
        self._booking.patient_name = patient_name
        self._booking.step         = "name"
        b = self._booking
        return (
            f"Let me confirm: {b.service} with {b.doctor} "
            f"on {b.slot_date} at {b.slot_time} for {patient_name}. "
            "Shall I confirm this appointment?"
        )

    @function_tool()
    async def save_appointment(self, context: RunContext, dummy: str = "") -> str:
        """Save the appointment after the patient confirms all details."""
        if self.mode != "booking":
            return "Booking is disabled in this session. Please start a Booking session."
        b = self._booking

        missing = [
            field for field, val in {
                "service":      b.service,
                "doctor":       b.doctor,
                "slot_id":      b.slot_id,
                "patient_name": b.patient_name,
            }.items() if not val
        ]
        if missing:
            log.error("[SAVE] Missing fields: %s", missing)
            return (
                f"I'm missing some details: {', '.join(missing)}. "
                + ("Which slot would you like?" if "slot_id" in missing
                   else "Could you repeat your name?")
            )

        log.info(
            "[SAVE] service=%s doctor=%s slot_id=%s patient=%s",
            b.service, b.doctor, b.slot_id, b.patient_name
        )

        result = await save_appointment(
            service_name=b.service,
            doctor_name=b.doctor,
            patient_name=b.patient_name,
            slot_id=b.slot_id,
        )

        booking = result.get("booking") or result
        conf    = booking.get("confirmation_number")

        if conf:
            self._booking = BookingState()
            return (
                f"Your appointment is confirmed! "
                f"Confirmation number: {conf}. "
                "We look forward to seeing you at Functiomed."
            )

        log.error("[SAVE] Backend returned no confirmation: %s", result)
        return "There was a problem saving your booking. Please call the clinic directly."


# ── Entrypoint ────────────────────────────────────────────────

async def entrypoint(ctx: agents.JobContext):
    log.info("New room: %s", ctx.room.name)
    await ctx.connect()
    log.info("Agent entered room: %s", ctx.room.name)

    room_mode = "booking" if ctx.room.name.lower().endswith("-booking") else "rag"

    backend_ok = await health_check()
    if not backend_ok:
        log.warning("⚠️  Backend unreachable at %s", os.getenv("RAG_BACKEND_URL"))

    # If running the OpenAI LLM plugin, ensure we never pass a Groq-only model name.
    # LiveKit Cloud deployments often miss env vars; this keeps the agent from hard-failing with 404.
    requested_model = (os.getenv("AGENT_LLM_MODEL") or "").strip()
    openai_model = requested_model or "gpt-4o-mini"
    if openai_model.startswith("llama-") or openai_model.startswith("mixtral") or openai_model.startswith("gemma"):
        log.warning("AGENT_LLM_MODEL=%s is not an OpenAI model; falling back to gpt-4o-mini", openai_model)
        openai_model = "gpt-4o-mini"

    session = AgentSession(
        stt=deepgram.STT(model="nova-2-general"),
        llm=openai.LLM(
            model=openai_model,
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
        tts=openai.TTS(voice="alloy"),
        vad=silero.VAD.load(),
    )

    initial_ctx = ChatContext()
    initial_ctx.add_message(
        role="assistant",
        content="[System] Backend: " + ("online" if backend_ok else "offline"),
    )

    # Booking mode: preload services from DB so the user immediately sees real options.
    if room_mode == "booking":
        try:
            services = await get_services()
            if services:
                names = [s["name"] for s in services]
                initial_ctx.add_message(
                    role="assistant",
                    content=(
                        "Welcome to Functiomed booking. Which service would you like? "
                        f"Available services: {', '.join(names[:8])}" + ("." if len(names) <= 8 else ", and more.")
                    ),
                )
        except Exception as e:
            log.error("Preload services failed: %s", e)

    await session.start(
        room=ctx.room,
        agent=FunctiomedAgent(chat_ctx=initial_ctx, mode=room_mode),
        room_input_options=RoomInputOptions(noise_cancellation=True),
    )

    await session.generate_reply(
        instructions=(
            "Greet the patient warmly in English."
            if room_mode == "booking"
            else "Greet the patient warmly in English and ask how you can help them today."
        )
    )

    log.info("Agent exiting room: %s", ctx.room.name)


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint)
    )