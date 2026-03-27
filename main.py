"""
agent/main.py — Functiomed Voice Agent (Single LLM, Groq compatible)

KEY CHANGE: Added explicit confirmation gate.
  confirm_name  → presents full summary → sets step="awaiting_confirmation"
  confirm_booking → checks user said "yes" → then calls save_appointment
  save_appointment is NEVER called directly by the LLM anymore.

RAG FIX: Context is now injected as role="user" (not role="system").
  OpenAI silently ignores system messages added mid-conversation.
  Injecting as user-role with an explicit instruction prefix ensures
  the LLM actually reads and uses the retrieved clinic knowledge.
  
RAG FIX 2: Removed booking intent redirect in RAG mode.
  Previously, any question with booking keywords would get redirected
  to start a booking session instead of answering clinic questions.
  Now RAG mode properly answers all clinic-related questions.

MODE SWITCHING: Added dynamic mode switching between RAG and booking
  based on user intent. Agent can now handle both types of queries
  in the same session.
"""

import asyncio
import json
import logging
import os
import re
import time
from difflib import SequenceMatcher

from dotenv import load_dotenv
load_dotenv()

from livekit import agents
from livekit.agents import AgentSession, Agent, ChatContext, ChatMessage, RoomInputOptions, llm
from livekit.plugins import openai, deepgram, silero
from livekit.agents import function_tool, RunContext

from booking.rag_client import retrieve_context, health_check
from booking.booking_client import (
    get_services,
    get_doctors_for_service,
    get_slots,
    save_appointment,
)

# --- DEEPGRAM MONKEYPATCH FOR TRIPLE/DOUBLE PHONE NUMBERS ---
# Deepgram natively allows term replacement via query params, but LiveKit doesn't
# expose it. We monkeypatch the URL builder to include replacements for digits.
import livekit.plugins.deepgram._utils as dg_utils
from urllib.parse import urlencode

_orig_to_deepgram_url = dg_utils._to_deepgram_url
def _patched_to_deepgram_url(opts: dict, base_url: str, *, websocket: bool) -> str:
    url = _orig_to_deepgram_url(opts, base_url, websocket=websocket)
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    parts = []
    for i in range(10):
        # Use literal colon (not %3A) — Deepgram needs it as separator.
        # Encode spaces as '+' for query-string safety.
        for val in (
            f"triple {i}:{i}{i}{i}",
            f"double {i}:{i}{i}",
            f"triple {words[i]}:{i}{i}{i}",
            f"double {words[i]}:{i}{i}",
        ):
            parts.append("replace=" + val.replace(" ", "+"))
    extra_qs = "&".join(parts)
    return url + ("&" if "?" in url else "?") + extra_qs

dg_utils._to_deepgram_url = _patched_to_deepgram_url
# -----------------------------------------------------------

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Spoken-number normalizer (shared by RAG + booking paths)
# ─────────────────────────────────────────────────────────────

_SPOKEN_DIGIT_MAP: dict[str, str] = {
    "zero": "0", "oh": "0",
    "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}

def _normalize_spoken_numbers(text: str) -> str:
    """Expand spoken multiplier patterns into digit strings.

    Examples:
      'triple 5'    → '555'
      'double one'  → '11'
      'triple five' → '555'
      'double 2'    → '22'
    Works whether Deepgram's replace feature fired or not.
    """
    digit_alts = "|".join(_SPOKEN_DIGIT_MAP.keys())
    single_dig = rf"(?:{digit_alts}|\d)"

    def _expand(m: re.Match, times: int) -> str:
        raw = m.group(1).lower()
        return _SPOKEN_DIGIT_MAP.get(raw, raw) * times

    text = re.sub(
        rf"\btriple\s+({single_dig})\b",
        lambda m: _expand(m, 3),
        text, flags=re.IGNORECASE,
    )
    text = re.sub(
        rf"\bdouble\s+({single_dig})\b",
        lambda m: _expand(m, 2),
        text, flags=re.IGNORECASE,
    )
    return text

# ─────────────────────────────────────────────────────────────
# Affirmative / negative keyword sets (voice-safe)
# ─────────────────────────────────────────────────────────────

_AFFIRMATIVE_EN = {
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm",
    "confirmed", "correct", "right", "absolutely", "definitely",
    "please", "go ahead", "do it", "book it", "book", "proceed",
    "sounds good", "that's right", "that is right", "looks good",
    "yes please", "yes confirm", "confirm my booking", "confirm booking",
    "save", "save it", "agree", "affirmative",
}

_AFFIRMATIVE_DE = {
    "ja", "jep", "jo", "ok", "okay", "bitte", "bestätigen",
    "bestätigt", "korrekt", "richtig", "genau", "stimmt",
    "weiter", "buchen", "ja bitte", "ja genau", "ja korrekt",
    "ja bestätigen", "termin bestätigen", "termin buchen",
    "ja das ist richtig", "alles richtig", "passt",
}

_NEGATIVE_EN = {
    "no", "nope", "cancel", "stop", "abort", "don't", "do not",
    "not correct", "wrong", "incorrect", "wait", "change",
}

_NEGATIVE_DE = {
    "nein", "ne", "nö", "abbrechen", "stopp", "stop", "falsch",
    "nicht", "ändern", "warte", "moment",
}

_CANCEL_EN = {
    "cancel", "cancel booking", "stop booking", "abort", "do not confirm",
    "don't confirm", "do not proceed", "stop", "cancel it",
}

_CANCEL_DE = {
    "abbrechen", "buchung abbrechen", "stornieren", "nicht bestätigen",
    "nicht fortfahren", "stop", "stopp",
}


def _is_affirmative(text: str, lang: str) -> bool:
    t = (text or "").lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)
    words = set(t.split())
    pool = _AFFIRMATIVE_DE if lang == "de" else _AFFIRMATIVE_EN
    if words & pool:
        return True
    for phrase in pool:
        if " " in phrase and phrase in t:
            return True
    return False


def _is_negative(text: str, lang: str) -> bool:
    t = (text or "").lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)
    words = set(t.split())
    pool = _NEGATIVE_DE if lang == "de" else _NEGATIVE_EN
    if words & pool:
        return True
    for phrase in pool:
        if " " in phrase and phrase in t:
            return True
    return False


def _is_cancel_intent(text: str, lang: str) -> bool:
    t = (text or "").lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)
    words = set(t.split())
    pool = _CANCEL_DE if lang == "de" else _CANCEL_EN
    if words & pool:
        return True
    for phrase in pool:
        if " " in phrase and phrase in t:
            return True
    return False


# ── System prompt builder ──────────────────────────────────────

def build_system_prompt(lang: str, mode: str) -> str:
    mode = (mode or "rag").strip().lower()
    if lang == "de":
        language_instruction = (
            "WICHTIG: Du sprichst AUSSCHLIESSLICH Deutsch in JEDER Antwort, ohne Ausnahme. "
            "Wechsle niemals ins Englische, unabhängig davon, in welcher Sprache der Patient schreibt oder spricht. "
            "Alle Begrüßungen, Bestätigungen, Fragen und Informationen sind auf Deutsch."
        )
        rag_instruction = (
            "WICHTIG: Verwende AUSSCHLIESSLICH die Informationen, die dir im Kontext bereitgestellt werden, "
            "um Fragen über die Klinik zu beantworten. Erfinde keine Informationen."
        )
        language_note = "Sprache: Deutsch (fest konfiguriert)"
    else:
        language_instruction = (
            "IMPORTANT: You speak ONLY English in EVERY response, without exception. "
            "Never switch to German or any other language, regardless of what language the patient uses. "
            "All greetings, confirmations, questions, and information must be in English."
        )
        rag_instruction = (
            "IMPORTANT: When answering questions about the clinic, use ONLY the information provided "
            "in the context. Do not make up information."
        )
        language_note = "Language: English (fixed)"

    if mode == "rag":
        # RAG mode specific prompt
        return f"""You are the friendly health information assistant for Functiomed, a medical clinic in Zurich.
Your ONLY responsibility is to answer questions about the clinic using the clinic knowledge base provided in your context.

{language_instruction}
{rag_instruction}

Critical constraints:
- You are in INFORMATION mode. Use the provided clinic knowledge base context to answer ALL questions, including those about services, treatments, clinic timings, and which doctors work in which departments.
- Never read out the retrieved context verbatim; summarize it naturally for a voice conversation.
- Keep answers very short (aim for <= 2 sentences / ~40 words).
- If the patient wants to book an appointment, inform them they should switch to the 'Booking' section of the application. Do NOT attempt to use booking tools yourself.
- If the answer is not in the provided context, politely say you don't know and suggest they contact the clinic directly.
- {language_note}
"""

    else:
        # Booking mode specific prompt
        return f"""You are the friendly booking assistant for Functiomed.
Your ONLY responsibility is to help patients book appointments using your booking tools.

{language_instruction}
{language_note}

Critical booking constraints:
- Use ONLY the booking tools (database-backed) for all requests. 
- Do NOT answer general clinic questions — if asked, politely remind the patient that you are the booking assistant and can only help with appointments.
- First message must be only a greeting + "how can I assist you in booking?"
- Do NOT list services in the first greeting.
- get_services is your ONLY source of service information. Call it whenever the user asks about services or wants to book.

Booking flow — follow this EXACT order, NEVER skip a step:
  STEP 1: call get_services     → present options → call confirm_service
  STEP 2: call get_doctors      → present options → call confirm_doctor
  STEP 3: call get_slots        → present options → call confirm_slot  (pass exact slot_id UUID)
  STEP 4: call confirm_name     → read booking summary back to patient → ask "shall I confirm?"
  STEP 5: call confirm_booking  → ONLY after patient says YES (yes, ok, confirm, go ahead, etc.)

Confirmation gate (STEP 5):
- After confirm_name, read the full booking summary and explicitly ask the patient to confirm.
- Wait for the user to verbally confirm with a 'yes' or equivalent before calling confirm_booking.
- If the user wants to change something, ask what they want to change and go back to that step.

CRITICAL tool rules:
- NEVER call get_slots before confirm_doctor has succeeded.
- NEVER call get_doctors before confirm_service has succeeded.
- NEVER call confirm_booking before confirm_name has succeeded.
- NEVER call save_appointment directly (always use confirm_booking).
- AMBIGUITY: If the user provides a service or doctor name that is ambiguous (e.g., 'Haneef'), you MUST ask the user for clarification. Do NOT automatically select one.
"""

# ── Booking state ─────────────────────────────────────────────

class BookingState:
    def __init__(self):
        self.service:      str | None = None
        self.doctor:       str | None = None
        self.slot_id:      str | None = None
        self.slot_date:    str | None = None
        self.slot_time:    str | None = None
        self.patient_name: str | None = None
        # step progression:
        # idle → collect_service → service → collect_doctor → doctor
        # → collect_slot → slot → collect_name → name
        # → awaiting_confirmation → done
        self.step:         str        = "idle"
        self._services:    list       = []
        self._doctors:     list       = []
        self._slots:       list       = []


# ── Agent ─────────────────────────────────────────────────────

class FunctiomedAgent(Agent):
    def __init__(self, chat_ctx: ChatContext, mode: str = "rag", lang: str = "en", room=None) -> None:
        super().__init__(instructions=build_system_prompt(lang, mode), chat_ctx=chat_ctx)
        self._booking = BookingState()
        self._mode = (mode or "rag").strip().lower()
        self._lang = (lang or "en").strip().lower()
        self._room = room

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def lang(self) -> str:
        return self._lang

    def _is_booking_intent(self, text: str) -> bool:
        t = (text or "").lower()
        return any(
            kw in t
            for kw in (
                "book", "booking", "appointment", "schedule",
                "termin", "buchen", "buchung",
            )
        )

    def _norm(self, s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _get_ranked_matches(self, query: str, choices: list[str]) -> list[tuple[str, float]]:
        q = self._norm(query)
        if not q or not choices:
            return []
        
        results = []
        for c in choices:
            c_norm = self._norm(c)
            if not c_norm:
                continue
            
            # Base fuzzy score
            score = SequenceMatcher(None, q, c_norm).ratio()
            
            # Substring boost: if the query is a whole word or significant part
            # This helps cases like 'Haneef' matching 'Dr Haneef Ullah'
            if q in c_norm:
                words = c_norm.split()
                if q in words:
                    # Perfect word match gets a high floor
                    score = max(score, 0.85)
                else:
                    # Partial substring match gets a good floor
                    score = max(score, 0.75)
            
            results.append((c, score))
            
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _best_match(self, query: str, choices: list[str]) -> tuple[str | None, float]:
        """Legacy wrapper for single best match."""
        matches = self._get_ranked_matches(query, choices)
        if not matches:
            return None, 0.0
        return matches[0]

    async def _publish_state(self, ctx: RunContext, available: list | None = None) -> None:
        b = self._booking
        step_map = {
            "idle":                   "idle",
            "collect_service":        "collect_service",
            "service":                "collect_doctor",
            "collect_doctor":         "collect_doctor",
            "doctor":                 "collect_slot",
            "collect_slot":           "collect_slot",
            "slot":                   "collect_name",
            "collect_name":           "collect_name",
            "name":                   "confirm",
            "awaiting_confirmation":  "confirm",
            "processing":             "processing",
            "done":                   "done",
        }
        payload = {
            "type": "booking_update",
            "step": step_map.get(b.step, "idle"),
            "data": {
                "service": b.service,
                "doctor":  b.doctor,
                "slot": (
                    f"{b.slot_date} {b.slot_time}".strip()
                    if b.slot_date and b.slot_time else None
                ),
                "name": b.patient_name,
            },
            "available": available or [],
        }
        try:
            if not self._room:
                raise RuntimeError("Room unavailable for booking_update publish")
            await self._room.local_participant.publish_data(
                json.dumps(payload).encode(),
                reliable=True,
            )
        except Exception as e:
            log.error("[STATE] Failed to publish booking_update: %s", e)

    async def _publish_cancelled_state(self) -> None:
        payload = {
            "type": "booking_update",
            "step": "cancelled",
            "data": {"service": None, "doctor": None, "slot": None, "name": None},
            "available": [],
        }
        try:
            if not self._room:
                raise RuntimeError("Room unavailable for cancelled-state publish")
            await self._room.local_participant.publish_data(
                json.dumps(payload).encode(),
                reliable=True,
            )
        except Exception as e:
            log.error("[STATE] Failed to publish cancelled state: %s", e)

    async def _publish_mode_change(self, new_mode: str) -> None:
        """Notify frontend when agent switches modes"""
        payload = {
            "type": "mode_change",
            "mode": new_mode,
        }
        try:
            if not self._room:
                raise RuntimeError("Room unavailable for mode_change publish")
            await self._room.local_participant.publish_data(
                json.dumps(payload).encode(),
                reliable=True,
            )
            log.info(f"[MODE] Published mode change to: {new_mode}")
        except Exception as e:
            log.error("[MODE] Failed to publish mode change: %s", e)

    # ── RAG injection ─────────────────────────────────────────

    async def llm_node(
        self,
        chat_ctx: ChatContext,
        tools: list[llm.Tool],
        model_settings,
    ):
        """Intercept LLM generation to inject RAG context or handle booking gates for both voice and text."""
        last_user_msg = next((m for m in reversed(chat_ctx.items) if isinstance(m, ChatMessage) and m.role == "user"), None)
        
        if last_user_msg and not getattr(last_user_msg, "_intercepted", False):
            last_user_msg._intercepted = True  # Prevent re-intercepting on tool callbacks
            user_text = last_user_msg.text_content or getattr(last_user_msg, "content", "")
            if isinstance(user_text, list): user_text = str(user_text)

            # ── Normalise spoken number patterns (triple 5→555, double 1→11) ──
            _raw_text = user_text
            user_text = _normalize_spoken_numbers(user_text)
            if user_text != _raw_text:
                log.info("[NORMALIZE] %r → %r", _raw_text[:80], user_text[:80])
                if hasattr(last_user_msg, "content"):
                    if isinstance(last_user_msg.content, list):
                        last_user_msg.content = [user_text]
                    else:
                        last_user_msg.content = user_text

            log.info(f"[DEBUG] Agent mode: {self.mode}, Booking step: {self._booking.step}")
            log.info(f"[DEBUG] User said: {user_text[:100]}")

            # ── Confirmation-gate intercept ───────────────────────
            if self._booking.step == "awaiting_confirmation":
                if _is_affirmative(user_text, self._lang):
                    instr = (
                        "Der Patient hat die Buchung bestätigt. Rufe jetzt sofort das confirm_booking-Tool auf."
                        if self._lang == "de" else
                        "The patient has confirmed the booking. Call the confirm_booking tool now."
                    )
                elif _is_negative(user_text, self._lang):
                    self._booking.step = "awaiting_cancel_or_change"
                    instr = (
                        "Der Patient hat die Zusammenfassung nicht bestätigt. "
                        "Frage jetzt klar nach: Möchten Sie die Buchung komplett abbrechen oder "
                        "möchten Sie ein Feld ändern (Service, Arzt, Termin oder Name)? "
                        "Wenn der Patient abbrechen möchte, bestätige kurz den Abbruch."
                        if self._lang == "de" else
                        "The patient did not confirm the summary. Ask clearly: do you want to "
                        "cancel the booking entirely, or change a field (service, doctor, slot, or name)? "
                        "If the patient wants to cancel, acknowledge cancellation briefly."
                    )
                else:
                    instr = (
                        "Die Antwort des Patienten war unklar. Frage ihn erneut höflich, "
                        "ob er den Termin bestätigen möchte."
                        if self._lang == "de" else
                        "The patient's response was unclear. Politely ask again whether they "
                        "want to confirm the appointment."
                    )
                chat_ctx.items.append(ChatMessage(role="system", content=[instr]))

            # ── Ambiguity intercept (Doctor) ──────────────────────
            elif self.mode == "booking" and self._booking.step == "collect_doctor" and self._booking._doctors:
                choices = []
                for d in self._booking._doctors:
                    title = (d.get("title") or "").strip()
                    full  = (d.get("full_name") or "").strip()
                    display = f"{title} {full}".strip() if title else full
                    choices.append(display)
                
                matches = self._get_ranked_matches(user_text, choices)
                # If multiple matches are found with good scores (>= 0.72)
                ambiguous = [name for name, s in matches if s >= 0.72]
                if len(ambiguous) > 1:
                    log.info("[INTERCEPT] Doctor ambiguity detected for %r: %s", user_text, ambiguous)
                    instr = (
                        f"Der Patient hat '{user_text}' gesagt. Dies ist mehrdeutig zwischen {', '.join(ambiguous)}. "
                        "Frage den Patienten bitte höflich, welchen dieser Ärzte er meint. Wähle NICHT selbst einen aus."
                        if self._lang == "de" else
                        f"The patient said '{user_text}'. This is ambiguous between {', '.join(ambiguous)}. "
                        "Please politely ask the patient which of these doctors they mean. Do NOT select one yourself."
                    )
                    chat_ctx.items.append(ChatMessage(role="system", content=[instr]))

            # ── Ambiguity intercept (Service) ─────────────────────
            elif self.mode == "booking" and self._booking.step == "collect_service" and self._booking._services:
                choices = [s.get("name", "").strip() for s in self._booking._services if s.get("name")]
                matches = self._get_ranked_matches(user_text, choices)
                ambiguous = [name for name, s in matches if s >= 0.72]
                if len(ambiguous) > 1:
                    log.info("[INTERCEPT] Service ambiguity detected for %r: %s", user_text, ambiguous)
                    instr = (
                        f"Der Patient hat '{user_text}' gesagt. Dies ist mehrdeutig zwischen {', '.join(ambiguous)}. "
                        "Frage den Patienten bitte, welchen Dienst er meint."
                        if self._lang == "de" else
                        f"The patient said '{user_text}'. This is ambiguous between {', '.join(ambiguous)}. "
                        "Please ask the patient which service they mean."
                    )
                    chat_ctx.items.append(ChatMessage(role="system", content=[instr]))

            elif self._booking.step == "awaiting_cancel_or_change":
                if _is_cancel_intent(user_text, self._lang):
                    await self._publish_cancelled_state()
                    self._booking = BookingState()
                    instr = (
                        "Der Patient möchte die Buchung abbrechen. Bestätige kurz den Abbruch "
                        "und biete an, später eine neue Buchung zu starten."
                        if self._lang == "de" else
                        "The patient wants to cancel the booking. Briefly confirm cancellation "
                        "and offer to start a new booking later."
                    )
                else:
                    instr = (
                        "Der Patient möchte die Buchung nicht abbrechen. Frage, welches Feld geändert werden soll "
                        "(Service, Arzt, Termin oder Name), und gehe dann zum passenden Schritt zurück."
                        if self._lang == "de" else
                        "The patient does not want to cancel. Ask which field should be changed "
                        "(service, doctor, slot, or name), then return to the appropriate step."
                    )
                chat_ctx.items.append(ChatMessage(role="system", content=[instr]))

            # ── RAG mode: handle questions ──────────────────────────
            elif self.mode == "rag" and self._booking.step in ("idle", "done"):
                t0_rag = time.perf_counter()
                context = await retrieve_context(user_text, top_k=5)
                dt_rag_ms = (time.perf_counter() - t0_rag) * 1000.0
                log.info("[TIMER][rag] retrieve_context %.1fms", dt_rag_ms)
                
                if context:
                    combined = f"{context}\n\nPatient question: {user_text}"
                    if hasattr(last_user_msg, "content"):
                        if isinstance(last_user_msg.content, list):
                            last_user_msg.content = [combined]
                        else:
                            last_user_msg.content = combined
                    log.info("[RAG] Injected %d chars of context for query: %r", len(context), user_text[:60])
                else:
                    log.warning("[RAG] No context retrieved — LLM will answer from its own knowledge for: %r", user_text[:60])

        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            yield chunk

    # ── Booking tools ─────────────────────────────────────────

    @function_tool()
    async def get_services(self, context: RunContext, dummy: str = "") -> str:
        """Get all available clinic services. ONLY call this if the user is explicitly starting a booking flow."""
        if self.mode != "booking":
            if self._lang == "de":
                return "Buchungen sind in dieser Sitzung deaktiviert. Bitte starten Sie eine Buchungs-Sitzung."
            return "Booking is disabled in this session. Please start a Booking session."
        log.info("[GET_SERVICES] fetching services from backend")
        t0 = time.perf_counter()
        try:
            services = await get_services()
        except Exception as e:
            log.error("[GET_SERVICES] backend exception: %s", e)
            if self._lang == "de":
                return "Es gab einen technischen Fehler. Bitte versuchen Sie es erneut."
            return "There was a technical error fetching services. Please try again."
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if not services:
            log.warning("[GET_SERVICES] backend returned no services")
            if self._lang == "de":
                return "Derzeit sind keine Leistungen verfügbar."
            return "No services are currently available."
        self._booking._services = services
        self._booking.step = "collect_service"
        names = [s["name"] for s in services]
        log.info("[GET_SERVICES] returning %d services: %s", len(names), names)
        log.info("[TIMER][booking] get_services %.1fms (count=%d)", dt_ms, len(names))
        await self._publish_state(context, available=names)
        return f"Available services: {', '.join(names)}"

    @function_tool()
    async def confirm_service(self, context: RunContext, service_name: str) -> str:
        """Confirm the service the patient wants. Must be called after get_services."""
        if self.mode != "booking":
            if self._lang == "de":
                return "Buchungen sind in dieser Sitzung deaktiviert. Bitte starten Sie eine Buchungs-Sitzung."
            return "Booking is disabled in this session. Please start a Booking session."
        service_name = (service_name or "").strip()
        stored = self._booking._services
        log.info("[CONFIRM_SERVICE] received=%r stored_count=%d", service_name, len(stored))
        choices = [s.get("name", "").strip() for s in stored if s.get("name")]
        direct = next((c for c in choices if self._norm(c) == self._norm(service_name)), None)
        if direct:
            exact_name = direct
        else:
            matches = self._get_ranked_matches(service_name, choices)
            log.info("[CONFIRM_SERVICE] found %d matches", len(matches))
            
            if not matches:
                best, score = None, 0.0
            else:
                best, score = matches[0]

            # Check for ambiguity if we have multiple good matches
            ambiguous = [name for name, s in matches if s >= 0.72 and (score - s) < 0.1]
            
            if len(ambiguous) > 1:
                log.info("[CONFIRM_SERVICE] Ambiguity detected: %s", ambiguous)
                self._booking.step = "collect_service"
                await self._publish_state(context, available=choices)
                if self._lang == "de":
                    return f"Ich habe mehrere Dienste gefunden, die zu '{service_name}' passen könnten: {', '.join(ambiguous)}. Welchen meinten Sie?"
                return f"I found multiple services that could match '{service_name}': {', '.join(ambiguous)}. Which one did you mean?"

            if best and score >= 0.72:
                exact_name = best
            else:
                self._booking.step = "collect_service"
                await self._publish_state(context, available=choices)
                log.warning("[CONFIRM_SERVICE] no match for %r — re-asking", service_name)
                if self._lang == "de":
                    return (
                        f"Ich habe verstanden: {service_name}. "
                        "Könnten Sie den Servicenamen wiederholen oder aus den Optionen in der Seitenleiste wählen?"
                    )
                return (
                    f"I heard: {service_name}. "
                    "Could you repeat the service name, or choose from the options in the sidebar?"
                )

        self._booking.service = exact_name
        self._booking.step    = "service"
        log.info("[CONFIRM_SERVICE] stored service=%r step=service", exact_name)
        await self._publish_state(context)

        # ── Background prefetch: start fetching doctors NOW so get_doctors hits cache ──
        async def _prefetch_doctors() -> None:
            try:
                log.info("[PREFETCH] Fetching doctors for service=%r in background", exact_name)
                t0 = time.perf_counter()
                await get_doctors_for_service(exact_name)
                log.info("[PREFETCH] Doctors ready in %.0fms", (time.perf_counter()-t0)*1000)
            except Exception as exc:
                log.warning("[PREFETCH] doctors prefetch failed: %s", exc)
        asyncio.ensure_future(_prefetch_doctors())

        if self._lang == "de":
            return f"Service bestätigt: {exact_name}. Ich suche jetzt nach verfügbaren Ärzten."
        return f"Service confirmed: {exact_name}. Let me find available doctors."

    @function_tool()
    async def get_doctors(self, context: RunContext, dummy: str = "") -> str:
        """Get doctors available for the confirmed service. Call after confirm_service."""
        if self.mode != "booking":
            if self._lang == "de":
                return "Buchungen sind in dieser Sitzung deaktiviert. Bitte starten Sie eine Buchungs-Sitzung."
            return "Booking is disabled in this session. Please start a Booking session."
        log.info("[GET_DOCTORS] service=%r", self._booking.service)
        if not self._booking.service:
            log.error("[GET_DOCTORS] GUARD HIT — service is None")
            if self._lang == "de":
                return "Bitte wählen Sie zuerst einen Service aus."
            return "Please select a service first."
        t0 = time.perf_counter()
        try:
            doctors = await get_doctors_for_service(self._booking.service)
        except Exception as e:
            log.error("[GET_DOCTORS] backend exception: %s", e)
            if self._lang == "de":
                return "Es gab einen technischen Fehler beim Abrufen der Ärzte. Bitte versuchen Sie es erneut."
            return "There was a technical error fetching doctors. Please try again."
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if not doctors:
            log.warning("[GET_DOCTORS] no doctors for service=%r", self._booking.service)
            if self._lang == "de":
                return f"Für {self._booking.service} sind keine Ärzte verfügbar."
            return f"No doctors available for {self._booking.service}."
        self._booking._doctors = doctors
        lines = []
        for d in doctors:
            title   = d.get("title", "").strip()
            name    = d.get("full_name", "").strip()
            display = f"{title} {name}".strip() if title else name
            lines.append(display)
        log.info("[GET_DOCTORS] returning %d doctors: %s", len(lines), lines)
        log.info("[TIMER][booking] get_doctors %.1fms (count=%d)", dt_ms, len(lines))
        self._booking.step = "collect_doctor"  # ← ensure step is set for llm_node intercept
        await self._publish_state(context, available=lines)
        return f"Available doctors: {', '.join(lines)}"

    @function_tool()
    async def confirm_doctor(self, context: RunContext, doctor_name: str) -> str:
        """Confirm the doctor the patient wants to see. Must be called before get_slots."""
        if self.mode != "booking":
            if self._lang == "de":
                return "Buchungen sind in dieser Sitzung deaktiviert. Bitte starten Sie eine Buchungs-Sitzung."
            return "Booking is disabled in this session. Please start a Booking session."

        stored = self._booking._doctors
        log.info("[CONFIRM_DOCTOR] received=%r stored_count=%d service=%r",
                 doctor_name, len(stored), self._booking.service)

        choices = []
        for d in stored:
            title = (d.get("title") or "").strip()
            full  = (d.get("full_name") or "").strip()
            if not full:
                continue
            display = f"{title} {full}".strip() if title else full
            choices.append(display)

        log.info("[CONFIRM_DOCTOR] parsed_choices=%s", choices)

        # 1. Direct match check
        direct = next((c for c in choices if self._norm(c) == self._norm(doctor_name)), None)
        if direct:
            display_name = direct
        else:
            # 2. Ranked match check
            matches = self._get_ranked_matches(doctor_name, choices)
            log.info("[CONFIRM_DOCTOR] found %d matches", len(matches))
            
            if not matches:
                best, score = None, 0.0
            else:
                best, score = matches[0]

            # 3. Check for ambiguity
            ambiguous = [name for name, s in matches if s >= 0.72 and (score - s) < 0.1]
            
            if len(ambiguous) > 1:
                log.info("[CONFIRM_DOCTOR] Ambiguity detected: %s", ambiguous)
                self._booking.step = "collect_doctor"
                await self._publish_state(context, available=choices)
                if self._lang == "de":
                    return f"Ich habe mehrere Ärzte gefunden, die zu '{doctor_name}' passen könnten: {', '.join(ambiguous)}. Welchen meinten Sie?"
                return f"I found multiple doctors that could match '{doctor_name}': {', '.join(ambiguous)}. Which one did you mean?"

            if best and score >= 0.72:
                display_name = best
            else:
                self._booking.step = "collect_doctor"
                await self._publish_state(context, available=choices)
                log.warning("[CONFIRM_DOCTOR] no match for %r — re-asking", doctor_name)
                if self._lang == "de":
                    return (
                        f"Ich habe verstanden: {doctor_name}. "
                        "Könnten Sie den Namen des Arztes wiederholen oder aus den Optionen in der Seitenleiste wählen?"
                    )
                return (
                    f"I heard: {doctor_name}. "
                    "Could you repeat the doctor name, or choose from the options in the sidebar?"
                )

        self._booking.doctor = display_name
        self._booking.step   = "doctor"
        log.info("[CONFIRM_DOCTOR] stored doctor=%r step=doctor", self._booking.doctor)
        await self._publish_state(context)

        # ── Background prefetch: start fetching slots NOW so get_slots hits cache ──
        _svc = self._booking.service
        _doc = display_name
        async def _prefetch_slots() -> None:
            try:
                log.info("[PREFETCH] Fetching slots for service=%r doctor=%r in background", _svc, _doc)
                t0 = time.perf_counter()
                await get_slots(_svc, _doc)
                log.info("[PREFETCH] Slots ready in %.0fms", (time.perf_counter()-t0)*1000)
            except Exception as exc:
                log.warning("[PREFETCH] slots prefetch failed: %s", exc)
        asyncio.ensure_future(_prefetch_slots())

        if self._lang == "de":
            return f"Arzt bestätigt: {display_name}. Ich prüfe jetzt die verfügbaren Termine."
        return f"Doctor confirmed: {display_name}. Let me check available time slots."

    @function_tool()
    async def get_slots(self, context: RunContext, dummy: str = "") -> str:
        """Get available appointment slots. Only call this AFTER confirm_doctor has been called."""
        if self.mode != "booking":
            if self._lang == "de":
                return "Buchungen sind in dieser Sitzung deaktiviert. Bitte starten Sie eine Buchungs-Sitzung."
            return "Booking is disabled in this session. Please start a Booking session."

        log.info("[GET_SLOTS] service=%r doctor=%r step=%r",
                 self._booking.service, self._booking.doctor, self._booking.step)

        if not self._booking.service:
            log.error("[GET_SLOTS] GUARD HIT — service is None, step=%r", self._booking.step)
            if self._lang == "de":
                return "Bitte wählen Sie zuerst einen Service aus."
            return "Service not selected yet. Please call confirm_service first."

        if not self._booking.doctor:
            log.error("[GET_SLOTS] GUARD HIT — doctor is None, step=%r", self._booking.step)
            if self._lang == "de":
                return "Bitte wählen Sie zuerst einen Arzt aus."
            return "Doctor not selected yet. Please call confirm_doctor first."

        t0 = time.perf_counter()
        try:
            slots = await get_slots(self._booking.service, self._booking.doctor)
        except Exception as e:
            log.error("[GET_SLOTS] backend exception: %s", e)
            if self._lang == "de":
                return "Es gab einen technischen Fehler beim Abrufen der Termine. Bitte versuchen Sie es erneut."
            return "There was a technical error fetching slots. Please try again."
        dt_ms = (time.perf_counter() - t0) * 1000.0

        log.info("[GET_SLOTS] backend returned %d slots", len(slots) if slots else 0)

        if not slots:
            log.warning("[GET_SLOTS] no slots for service=%r doctor=%r",
                        self._booking.service, self._booking.doctor)
            if self._lang == "de":
                return "Derzeit sind keine Termine verfügbar. Bitte rufen Sie die Klinik direkt an."
            return "No available slots at this time. Please call the clinic directly."

        self._booking._slots = slots
        self._booking.step   = "collect_slot"  # ← ensure step is set
        lines = []
        for i, s in enumerate(slots[:5], 1):
            lines.append(f"{i}. {s['slot_date']} at {s['slot_time']} — slot_id: {s['id']}")
        slot_labels = [f"{s['slot_date']} at {s['slot_time']}" for s in slots[:5]]
        await self._publish_state(context, available=slot_labels)
        log.info("[GET_SLOTS] returning %d slots to LLM", len(lines))
        log.info("[TIMER][booking] get_slots %.1fms (count=%d)", dt_ms, len(slots))
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
            if self._lang == "de":
                return "Buchungen sind in dieser Sitzung deaktiviert. Bitte starten Sie eine Buchungs-Sitzung."
            return "Booking is disabled in this session. Please start a Booking session."
        stored_slots = self._booking._slots

        matched = next(
            (s for s in stored_slots if str(s["id"]) == str(slot_id)),
            None,
        )

        if not matched:
            matched = next(
                (s for s in stored_slots
                 if s["slot_date"] == slot_date and s["slot_time"] == slot_time),
                None,
            )
            if matched:
                log.warning(
                    "[SLOT] LLM passed slot_id=%s, resolved by date+time to UUID=%s",
                    slot_id, matched["id"],
                )

        if not matched:
            slots_available = "\n".join(
                f"  {s['slot_date']} at {s['slot_time']} (slot_id: {s['id']})"
                for s in stored_slots[:5]
            )
            if self._lang == "de":
                return f"Ich konnte diesen Termin nicht finden. Bitte wählen Sie aus:\n{slots_available}"
            return f"I couldn't find that slot. Please choose from:\n{slots_available}"

        self._booking.slot_id   = matched["id"]
        self._booking.slot_date = matched["slot_date"]
        self._booking.slot_time = matched["slot_time"]
        self._booking.step      = "slot"
        await self._publish_state(context)
        if self._lang == "de":
            return (
                f"Termin bestätigt: {matched['slot_date']} um {matched['slot_time']}. "
                "Darf ich Ihren vollständigen Namen für die Buchung erfahren?"
            )
        return (
            f"Slot confirmed: {matched['slot_date']} at {matched['slot_time']}. "
            "May I have your full name to complete the booking?"
        )

    @function_tool()
    async def confirm_name(self, context: RunContext, patient_name: str) -> str:
        """
        Confirm the patient's full name.
        After this, read the booking summary and ask the patient to say YES or NO.
        Do NOT call confirm_booking yet — wait for the patient's verbal confirmation.
        """
        if self.mode != "booking":
            if self._lang == "de":
                return "Buchungen sind in dieser Sitzung deaktiviert. Bitte starten Sie eine Buchungs-Sitzung."
            return "Booking is disabled in this session. Please start a Booking session."

        self._booking.patient_name = patient_name
        self._booking.step         = "awaiting_confirmation"   # ← gate is now active
        await self._publish_state(context)

        b = self._booking
        if self._lang == "de":
            return (
                f"Danke. Ich habe Ihren Namen als: {patient_name} verstanden. "
                "Falls die Schreibweise nicht korrekt ist, buchstabieren Sie bitte Ihren vollständigen Namen langsam (Buchstabe für Buchstabe). "
                f"Vor der Bestätigung hier die vollständige Zusammenfassung: Service {b.service}, Arzt {b.doctor}, "
                f"Termin {b.slot_date} um {b.slot_time}, Patientenname {patient_name}. "
                "Soll ich diese Buchung jetzt bestätigen? Sagen Sie Ja zum Bestätigen oder Abbrechen / Ändern."
            )
        return (
            f"Thanks. I heard your name as: {patient_name}. "
            "If the spelling is not correct, please spell your full name slowly, letter by letter. "
            f"Before I confirm, here is your full booking summary: service {b.service}, doctor {b.doctor}, "
            f"slot {b.slot_date} at {b.slot_time}, patient name {patient_name}. "
            "Do you want me to confirm this booking now? Say yes to confirm, or say cancel / change."
        )

    @function_tool()
    async def confirm_booking(self, context: RunContext, dummy: str = "") -> str:
        """
        Called ONLY after the patient has verbally said YES to the booking summary.
        This saves the appointment. Never call this before confirm_name has been completed
        and the patient has explicitly confirmed.
        """
        if self.mode != "booking":
            if self._lang == "de":
                return "Buchungen sind in dieser Sitzung deaktiviert. Bitte starten Sie eine Buchungs-Sitzung."
            return "Booking is disabled in this session. Please start a Booking session."

        b = self._booking

        # Safety guard — must be in the awaiting_confirmation step
        if b.step != "awaiting_confirmation":
            log.warning(
                "[CONFIRM_BOOKING] Called at wrong step=%r — rejecting. "
                "confirm_name must be called first.", b.step
            )
            if self._lang == "de":
                return (
                    "Ich kann die Buchung noch nicht abschließen. "
                    "Bitte stellen Sie sicher, dass alle Schritte abgeschlossen sind."
                )
            return (
                "I can't complete the booking yet. "
                "Please make sure all steps have been completed first."
            )

        missing = [
            field for field, val in {
                "service":      b.service,
                "doctor":       b.doctor,
                "slot_id":      b.slot_id,
                "patient_name": b.patient_name,
            }.items() if not val
        ]
        if missing:
            log.error("[CONFIRM_BOOKING] Missing fields: %s", missing)
            if self._lang == "de":
                return (
                    f"Es fehlen noch einige Angaben: {', '.join(missing)}. "
                    + ("Welchen Termin möchten Sie?" if "slot_id" in missing
                       else "Könnten Sie Ihren Namen wiederholen?")
                )
            return (
                f"I'm missing some details: {', '.join(missing)}. "
                + ("Which slot would you like?" if "slot_id" in missing
                   else "Could you repeat your name?")
            )

        log.info(
            "[CONFIRM_BOOKING] Saving — service=%s doctor=%s slot_id=%s patient=%s",
            b.service, b.doctor, b.slot_id, b.patient_name,
        )

        t0_total = time.perf_counter()
        t0_save  = time.perf_counter()

        # Let the UI show immediate progress while we wait for the backend save.
        b.step = "processing"
        await self._publish_state(context)

        result = await save_appointment(
            service_name=b.service,
            doctor_name=b.doctor,
            patient_name=b.patient_name,
            slot_id=b.slot_id,
        )
        dt_save_ms = (time.perf_counter() - t0_save) * 1000.0
        log.info("[TIMER][booking] confirm_booking save_appointment %.1fms", dt_save_ms)

        booking = result.get("booking") or result
        conf    = booking.get("confirmation_number")

        if conf:
            try:
                if not self._room:
                    raise RuntimeError("Room unavailable for booking done publish")
                await self._room.local_participant.publish_data(
                    json.dumps({
                        "type": "booking_update",
                        "step": "done",
                        "data": {
                            "service":      b.service,
                            "doctor":       b.doctor,
                            "slot":         (
                                f"{b.slot_date} {b.slot_time}".strip()
                                if b.slot_date and b.slot_time else None
                            ),
                            "name":         b.patient_name,
                            "confirmation": conf,
                        },
                        "available": [],
                    }).encode(),
                    reliable=True,
                )
            except Exception as e:
                log.error("[STATE] Failed to publish done state: %s", e)

            self._booking = BookingState()

            log.info(
                "[TIMER][booking] confirm_booking total %.1fms (confirmation=%s)",
                (time.perf_counter() - t0_total) * 1000.0, conf,
            )
            if self._lang == "de":
                return (
                    f"Ihr Termin ist bestätigt! "
                    f"Bestätigungsnummer: {conf}. "
                    "Wir freuen uns darauf, Sie bei Functiomed zu sehen."
                )
            return (
                f"Your appointment is confirmed! "
                f"Confirmation number: {conf}. "
                "We look forward to seeing you at Functiomed."
            )

        log.error("[CONFIRM_BOOKING] Backend returned no confirmation: %s", result)
        log.info(
            "[TIMER][booking] confirm_booking total %.1fms (no confirmation)",
            (time.perf_counter() - t0_total) * 1000.0,
        )
        if self._lang == "de":
            return "Bei der Speicherung Ihrer Buchung ist ein Problem aufgetreten. Bitte rufen Sie die Klinik direkt an."
        return "There was a problem saving your booking. Please call the clinic directly."

    # ── save_appointment is intentionally NOT exposed as a function_tool.
    #    The LLM must use confirm_booking instead, which contains the gate check.


# ── Entrypoint ────────────────────────────────────────────────

async def entrypoint(ctx: agents.JobContext):
    log.info("New room: %s", ctx.room.name)
    await ctx.connect()
    log.info("Agent entered room: %s", ctx.room.name)

    room_name  = (ctx.room.name or "").lower()
    room_parts = room_name.split("-")
    room_mode  = "booking" if "booking" in room_parts else "rag"
    
    # Debug logging
    log.info(f"[DEBUG] Room name: {room_name}")
    log.info(f"[DEBUG] Room parts: {room_parts}")
    log.info(f"[DEBUG] Parsed mode: {room_mode}")
    log.info(f"[DEBUG] Contains 'booking'? {'booking' in room_parts}")
    log.info(f"[DEBUG] Contains 'rag'? {'rag' in room_parts}")

    room_lang = "en"
    if "de" in room_parts:
        room_lang = "de"
    elif "en" in room_parts:
        room_lang = "en"

    log.info("[ROOM] mode=%s lang=%s room=%s", room_mode, room_lang, room_name)

    backend_ok = await health_check()
    if not backend_ok:
        log.warning("⚠️  Backend unreachable at %s", os.getenv("RAG_BACKEND_URL"))

    requested_model = (os.getenv("AGENT_LLM_MODEL") or "").strip()
    openai_model    = requested_model or "gpt-4o"
    if openai_model.startswith(("llama-", "mixtral", "gemma")):
        log.warning("AGENT_LLM_MODEL=%s is not an OpenAI model; falling back to gpt-4o", openai_model)
        openai_model = "gpt-4o"
    log.info("[LLM] Using model: %s", openai_model)

    initial_ctx = ChatContext()
    initial_ctx.add_message(
        role="assistant",
        content="[System] Backend: " + ("online" if backend_ok else "offline"),
    )

    stt_kwargs = {
        "model":           "nova-2-general",
        "interim_results": True,
        "punctuate":       True,
        "smart_format":    True,
        "numerals":        True,
        "language":        room_lang,
    }

    log.info("[STT] mode=%s language=%s config=%s", room_mode, room_lang, stt_kwargs)

    session = AgentSession(
        stt=deepgram.STT(**stt_kwargs),
        llm=openai.LLM(
            model=openai_model,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.3,           # deterministic, better for booking flows
            max_completion_tokens=200, # short answers = less TTS to generate
        ),
        # Deepgram TTS: streaming=True → first audio chunk plays immediately,
        # no waiting for the full response like openai.TTS (streaming=False)
        tts=deepgram.TTS(model="aura-2-andromeda-en"),
        vad=silero.VAD.load(),
    )

    await session.start(
        room=ctx.room,
        agent=FunctiomedAgent(chat_ctx=initial_ctx, mode=room_mode, lang=room_lang, room=ctx.room),
        room_input_options=RoomInputOptions(noise_cancellation=True),
    )

    if room_mode == "booking":
        try:
            await ctx.room.local_participant.publish_data(
                json.dumps({
                    "type": "booking_update",
                    "step": "idle",
                    "data": {"service": None, "doctor": None, "slot": None, "name": None},
                    "available": [],
                }).encode(),
                reliable=True,
            )
        except Exception as e:
            log.error("[STATE] Failed to publish initial idle state: %s", e)

        # Warm the booking client cache while we greet the user.
        # This removes the first-call latency for `/clinic/services`.
        try:
            asyncio.create_task(get_services()).add_done_callback(
                lambda fut: log.info(
                    "[WARMUP][booking] get_services warmup done (ok=%s)",
                    not fut.cancelled() and fut.exception() is None,
                )
            )
        except Exception as e:
            log.warning("[WARMUP][booking] failed to start get_services warmup: %s", e)

    if room_mode == "booking":
        if room_lang == "de":
            greeting_instructions = (
                "Begrüße den Patienten herzlich auf Deutsch, stelle dich als Buchungsassistent von Functiomed vor "
                "und frage: Wie kann ich Ihnen heute bei der Terminbuchung helfen? "
                "Liste keine Leistungen auf, es sei denn, der Patient fragt danach."
            )
        else:
            greeting_instructions = (
                "Greet the patient warmly in English, introduce yourself as the Functiomed booking assistant, "
                "and ask: how can I assist you in booking today? "
                "Do not list any services unless the user asks to book or asks what services are available."
            )
    else:
        if room_lang == "de":
            greeting_instructions = (
                "Begrüße den Patienten herzlich auf Deutsch und frage, wie du ihm heute helfen kannst."
            )
        else:
            greeting_instructions = (
                "Greet the patient warmly in English and ask how you can help them today."
            )

    await session.generate_reply(instructions=greeting_instructions)

    log.info("Agent exiting room: %s", ctx.room.name)


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint)
    )