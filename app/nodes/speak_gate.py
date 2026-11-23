"""
Speak Gate — Phase 4 upgraded.

Responsibilities:
1. Barge-in handling (unchanged from Phase 2/3)
2. Audit log integration — log every reply, handoff, and tool call result
3. Warm handoff payload construction on handoff=True
4. Memory update (add turn to ShortTermMemory)
"""
import time
import logging
from typing import Dict, Any, Optional

from app.graph.state import VoiceAgentState
from app.safety.audit import audit_registry
from app.safety.handoff import warm_handoff_builder
from app.safety.memory import memory_registry
from app.safety.pii import redact_for_storage

logger = logging.getLogger(__name__)


def n_speak_gate(state: VoiceAgentState) -> VoiceAgentState:
    """
    The speak gate:
    1. If barge_in → cut, re-plan.
    2. Otherwise: log to audit, build handoff payload if needed, update memory.
    """
    start_t = time.perf_counter()
    barge = state.get("barge_in")
    latencies = dict(state.get("latency_ms", {}))
    session_id = state.get("session_id", "unknown")
    transcript = state.get("transcript", "")
    reply = state.get("reply", "")
    handoff = state.get("handoff", False)
    handoff_reason = state.get("handoff_reason", "")
    handoff_category = state.get("handoff_category", "kb_miss")
    tool_results = state.get("tool_results", [])
    pii_safe_transcript = state.get("pii_redacted_transcript") or redact_for_storage(transcript)

    audit = audit_registry.get(session_id)
    memory = memory_registry.get(session_id)

    # ── Barge-in ───────────────────────────────────────────────────────────
    if barge:
        audit.log_barge_in(barge_text=barge, transcript_ref=pii_safe_transcript)
        elapsed = (time.perf_counter() - start_t) * 1000
        latencies["speak_gate"] = elapsed
        return {
            "interrupted": True,
            "transcript": barge,
            "barge_in": None,
            "context": None,
            "decision": None,
            "reply": None,
            "tool_calls": [],
            "tool_results": [],
            "steps": 0,
            "latency_ms": latencies,
        }

    # ── Audit: tool call results ───────────────────────────────────────────
    for tr in tool_results:
        audit.log_tool_result(
            tool_name=tr.get("tool", "unknown"),
            result=tr.get("result"),
            error=tr.get("error"),
            transcript_ref=pii_safe_transcript,
        )

    # ── Audit: reply ──────────────────────────────────────────────────────
    if reply:
        audit.log_reply(
            reply=reply,
            handoff=handoff,
            transcript_ref=pii_safe_transcript,
        )

    # ── Audit: handoff ────────────────────────────────────────────────────
    handoff_payload_dict: Optional[Dict] = None
    if handoff:
        audit.log_handoff(
            reason=handoff_reason,
            summary=memory.build_context(),
            transcript_ref=pii_safe_transcript,
        )

        # Build warm handoff payload
        # Extract caller info from most recent tool result (if available)
        caller_name = None
        enquiry_id = None
        course_interest = None
        for tr in reversed(tool_results):
            res = tr.get("result", {})
            if isinstance(res, dict):
                caller_name = caller_name or res.get("first_name")
                course_interest = course_interest or res.get("course_interest")
                if tr.get("tool") == "enquiry_lookup" and tr.get("args", {}).get("phone_or_enquiry_id"):
                    enquiry_id = enquiry_id or tr.get("args", {}).get("phone_or_enquiry_id")

        payload = warm_handoff_builder.build(
            session_id=session_id,
            handoff_reason=handoff_reason,
            handoff_category=handoff_category or "kb_miss",
            conversation_history=[
                {"role": "user", "content": t.caller, "agent_reply": t.agent}
                for t in memory.as_state().recent_turns
            ],
            summary=memory.build_context(),
            last_agent_reply=reply or "",
            caller_name=caller_name,
            enquiry_id=enquiry_id,
            course_interest=course_interest,
        )
        handoff_payload_dict = payload.to_dict()
        logger.info(f"[{session_id}] Warm handoff payload built: reason={handoff_reason}")

    # ── Memory: add this turn ──────────────────────────────────────────────
    if transcript and reply:
        tool_name = tool_results[-1].get("tool") if tool_results else None
        memory.add_turn(
            caller=pii_safe_transcript,
            agent=reply or "",
            handoff=handoff,
            tool_used=tool_name,
        )

    elapsed = (time.perf_counter() - start_t) * 1000
    latencies["speak_gate"] = elapsed

    updates: Dict = {"latency_ms": latencies}
    if handoff_payload_dict:
        updates["handoff_payload"] = handoff_payload_dict

    return updates


def route_speak_gate(state: VoiceAgentState) -> str:
    """Route after speak gate: loop back to retrieve if interrupted, else finalize response."""
    if state.get("interrupted") and state.get("reply") is None:
        return "retrieve"
    return "respond"
