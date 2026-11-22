import os
import time
import random
import logging
from typing import List, Dict, Any, Optional
import httpx
from openai import OpenAI
from app.config import settings

logger = logging.getLogger(__name__)

# System prompt adhering strictly to prompt requirements:
SYSTEM_PROMPT = """You are a warm, professional, real-time voice customer support agent for AI Adventures (aiadventures.in), a hands-on, code-first AI and machine learning training institute in Pune, Maharashtra.

<rules>
1. Answer ONLY from the <knowledge_base> and executed tool results provided.
2. If the user asks about anything not in the knowledge base (e.g. unverified curriculum, unverified batch timings, unconfirmed placement claims, EMI, discounts, or policies marked TODO), say clearly: "I don't have that verified in my notes right now, let me connect you with an admissions counselor" and trigger a human handoff. NEVER guess or invent facts.
3. Keep answers concise: 1 to 3 short, natural spoken sentences.
4. Spoken delivery rules:
   - NO markdown formatting (no asterisks, bullet points, headers, or brackets).
   - NO raw URLs read aloud (say "our website ai adventures dot in").
   - Read prices and phone numbers in natural spoken form (e.g. "fifteen thousand rupees per month").
5. Financial and high-impact actions:
   - You NEVER autonomously issue refunds, change fees, or settle payment disputes.
   - For refunds or fee disputes, create a support ticket and inform the caller that a human manager will assist them.
6. Booking demo sessions: Always confirm the preferred time slot out loud before confirming the booking.
</rules>
"""

class LLMClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
        self.base_url = base_url or settings.GROQ_BASE_URL
        self.fast_model = settings.GROQ_FAST_MODEL
        self.escalation_model = settings.GROQ_ESCALATION_MODEL
        self.client: Optional[OpenAI] = None
        self.live_models: List[str] = []
        self.is_offline = not bool(self.api_key and self.api_key != "your_groq_api_key_here")

        if not self.is_offline:
            try:
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                self._verify_and_select_models()
            except Exception as e:
                logger.warning(f"Could not connect to Groq at startup: {e}. Falling back to offline policy.")
                self.is_offline = True
        else:
            logger.info("Operating in deterministic offline policy mode (no GROQ_API_KEY provided).")

    def _verify_and_select_models(self):
        """Call GET /v1/models and verify configured models exist; fall back safely if deprecated."""
        try:
            resp = httpx.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5.0
            )
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                self.live_models = [m.get("id") for m in data]
                logger.info(f"Retrieved {len(self.live_models)} live models from Groq.")
                
                # Check fast model
                if self.fast_model not in self.live_models:
                    logger.warning(f"Configured fast model '{self.fast_model}' not found in live Groq models.")
                    # Pick suitable live fallback
                    candidates = [m for m in self.live_models if "8b" in m or "instant" in m or "20b" in m or "gpt" in m]
                    if candidates:
                        self.fast_model = candidates[0]
                        logger.info(f"Fallback fast model selected: {self.fast_model}")

                # Check escalation model
                if self.escalation_model not in self.live_models:
                    logger.warning(f"Configured escalation model '{self.escalation_model}' not in live Groq models.")
                    candidates = [m for m in self.live_models if "70b" in m or "120b" in m or "versatile" in m]
                    if candidates:
                        self.escalation_model = candidates[0]
                        logger.info(f"Fallback escalation model selected: {self.escalation_model}")
            else:
                logger.warning(f"Groq /models returned {resp.status_code}. Using configured defaults.")
        except Exception as e:
            logger.warning(f"Failed to query Groq /models: {e}. Using configured defaults.")

    def complete(self, messages: List[Dict[str, str]], escalate: bool = False, max_retries: int = 3) -> str:
        """Execute chat completion with exponential backoff and jitter for 429s,
        or deterministic offline policy if offline.
        """
        if self.is_offline or not self.client:
            return self._offline_policy_response(messages)

        model = self.escalation_model if escalate else self.fast_model
        delay = 1.0

        for attempt in range(max_retries):
            try:
                start_t = time.perf_counter()
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=250
                )
                logger.debug(f"LLM call ({model}) took {(time.perf_counter() - start_t)*1000:.1f}ms")
                return response.choices[0].message.content or ""
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate limit" in err_str:
                    jitter = random.uniform(0.1, 0.5)
                    sleep_time = delay + jitter
                    logger.warning(f"Groq rate limit hit (attempt {attempt+1}/{max_retries}). Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                    delay *= 2
                else:
                    logger.error(f"Groq inference failed: {e}")
                    raise e

        # If retries exhausted, degrade gracefully
        return "I am experiencing high network traffic right now. Let me connect you directly to our human admissions desk."

    def _offline_policy_response(self, messages: List[Dict[str, str]]) -> str:
        """Deterministic offline policy for zero-quota, millisecond test execution."""
        user_query = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_query = m.get("content", "").lower()
                break

        # Check for refund / payment dispute requests
        if "refund" in user_query or "cancel" in user_query or "money back" in user_query:
            return "I cannot process refunds or fee adjustments directly. I have raised a priority ticket for our accounts team, and an advisor will contact you shortly."

        if "machine learning" in user_query or "ml" in user_query:
            return "The Machine Learning program is 6 months long with a fee of 15,000 rupees per month. Our counselors can confirm the latest batch schedule."

        if "python" in user_query:
            return "Our beginner Python course is 15,000 rupees per month. Please note that this is an unverified estimate, and our team will confirm the exact details."

        if "data science" in user_query:
            return "The Data Science program has a duration of 6 months, and the fee is approximately 25,000 rupees per month, subject to official confirmation."

        if "courses" in user_query or "what do you offer" in user_query or "topics" in user_query:
            return "AI Adventures offers hands-on programs in Python, Data Science, Machine Learning, Deep Learning, and Computer Vision based in Pune."

        if "location" in user_query or "address" in user_query or "where" in user_query:
            return "We are located at 2nd Floor, 1284 Atre House, Congress Bhavan Road, JM Road, near Balgandharv Rangmandir, Pune 411005."

        if "lookup" in user_query or "enquiry" in user_query or "status" in user_query:
            return "Let me check that enquiry for you right now."

        # Unverified topic -> triggers human handoff default
        if any(w in user_query for w in ["curriculum", "syllabus", "batch timing", "placement", "guarantee", "emi", "discount", "scholarship"]):
            return "I don't have that verified in my notes right now, so let me connect you with an admissions counselor who can provide the exact details."

        # Default fallback
        return "AI Adventures is a hands-on AI and machine learning training institute in Pune. How can I assist you with our courses today?"

# Global default LLM client
llm_client = LLMClient()
