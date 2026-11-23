"""
Warm Handoff Builder — Phase 4 Safety.

When the agent decides to hand off to a human, it assembles a structured payload
containing everything the human agent needs to pick up the call without re-asking
the caller to repeat themselves.

The payload is:
  - JSON (for CRM / ticketing system ingestion)
  - PII-redacted for the transcript field (safe to store)
  - Linked to the audit log path for a full trail

The human sees:
  - Caller's name (if looked up), enquiry ID, course interest
  - Reason for handoff (KB miss / tool error / high-impact action / rate limit)
  - Concise conversation summary (from ShortTermMemory)
  - The last 4 raw turns verbatim (for context)
  - The agent's last spoken reply (so the human can continue seamlessly)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from app.safety.pii import redact_for_storage
from app.config import settings


@dataclass
class HandoffPayload:
    session_id: str
    handoff_reason: str
    handoff_category: str           # "kb_miss" | "tool_error" | "high_impact" | "rate_limit" | "injection" | "low_confidence"
    caller_name: Optional[str]
    enquiry_id: Optional[str]
    course_interest: Optional[str]
    conversation_summary: str
    recent_turns: List[Dict[str, str]]   # [{role, content}, ...] — PII-redacted
    last_agent_reply: str
    audit_log_path: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_spoken(self) -> str:
        """Human-readable handoff announcement (spoken to caller)."""
        reason_phrases = {
            "kb_miss": "I need to connect you with someone who can give you verified details.",
            "tool_error": "I'm having trouble accessing your records right now.",
            "high_impact": "This requires approval from our team.",
            "rate_limit": "I'm experiencing a temporary delay.",
            "injection": "I cannot process that request.",
            "low_confidence": "I want to make sure you get accurate information.",
        }
        phrase = reason_phrases.get(self.handoff_category, "Let me connect you with our team.")
        name_part = f" {self.caller_name}" if self.caller_name else ""
        return (
            f"One moment{name_part}. {phrase} "
            "I'm connecting you to an AI Adventures admissions counselor now. "
            "They'll have a summary of our conversation."
        )


class WarmHandoffBuilder:
    """
    Builds the HandoffPayload from current session state.
    Called by the speak_gate / agent node when handoff=True.
    """

    def build(
        self,
        session_id: str,
        handoff_reason: str,
        handoff_category: str,
        conversation_history: List[Dict],
        summary: str,
        last_agent_reply: str,
        caller_name: Optional[str] = None,
        enquiry_id: Optional[str] = None,
        course_interest: Optional[str] = None,
    ) -> HandoffPayload:
        # Redact PII from recent turns before storing in payload
        redacted_turns = []
        for turn in conversation_history[-6:]:  # last 6 turns max
            redacted_turns.append({
                "role": turn.get("role", "unknown"),
                "content": redact_for_storage(str(turn.get("content", turn.get("agent_reply", "")))),
            })

        audit_path = os.path.join(settings.BASE_DIR, "audit_logs", f"{session_id}.jsonl")

        return HandoffPayload(
            session_id=session_id,
            handoff_reason=handoff_reason,
            handoff_category=handoff_category,
            caller_name=caller_name,
            enquiry_id=enquiry_id,
            course_interest=course_interest,
            conversation_summary=redact_for_storage(summary),
            recent_turns=redacted_turns,
            last_agent_reply=redact_for_storage(last_agent_reply),
            audit_log_path=audit_path,
        )


# Global singleton
warm_handoff_builder = WarmHandoffBuilder()
