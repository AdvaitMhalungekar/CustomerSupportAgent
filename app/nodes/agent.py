import re
import time
from app.graph.state import VoiceAgentState
from app.llm.client import llm_client, SYSTEM_PROMPT
from app.observability import tracer

def n_agent(state: VoiceAgentState) -> VoiceAgentState:
    start_t = time.perf_counter()
    transcript = state.get("transcript", "")
    context = state.get("context", "")
    tool_results = state.get("tool_results", [])
    latencies = dict(state.get("latency_ms", {}))
    steps = state.get("steps", 0) + 1
    session_id = state.get("session_id", "unknown")

    with tracer.span("node:agent", session_id, kind="node", attributes={"steps": steps, "has_tool_results": bool(tool_results)}) as span:
        if state.get("handoff"):
            latencies["agent"] = (time.perf_counter() - start_t) * 1000
            span.attributes["handoff"] = True
            return {"latency_ms": latencies, "steps": steps}

        # 1. Check if tool output already exists to synthesize reply
        if tool_results:
            latest = tool_results[-1]
            tool_name = latest.get("tool")
            res = latest.get("result", {})
            error = latest.get("error")

            if error:
                reply = f"I encountered an issue checking your records: {error}. Let me connect you directly to our team."
                handoff = True
                handoff_reason = f"Tool failure: {error}"
            elif tool_name == "enquiry_lookup":
                first_name = res.get("first_name", "Student")
                course = res.get("course_interest", "our program")
                status = res.get("status", "Active")
                followup = res.get("next_followup", "soon")
                reply = f"Hello {first_name}, I found your enquiry for {course}. Your status is {status}, and your next follow-up is scheduled for {followup}."
                handoff = False
                handoff_reason = None
            elif tool_name == "book_demo_session":
                booking_id = res.get("booking_id", "Pending")
                slot = res.get("slot", "")
                reply = f"Your free demo session has been booked for {slot}. Your reference number is {booking_id}."
                handoff = False
                handoff_reason = None
            elif tool_name == "create_ticket":
                tck_id = res.get("ticket_id", "Pending")
                status = res.get("status", "")
                reply = f"I have logged ticket {tck_id} regarding this matter. An advisor from AI Adventures will review it and get in touch with you."
                # High-impact tickets (refund/dispute) automatically hand off
                handoff = "High-Impact" in status or "refund" in str(latest).lower()
                handoff_reason = "High-impact ticket requires human management" if handoff else None
            elif tool_name == "request_callback":
                cb_id = res.get("callback_id", "")
                tw = res.get("time_window", "")
                reply = f"A counselor callback has been scheduled for {tw}. Your reference ID is {cb_id}."
                handoff = False
                handoff_reason = None
            else:
                reply = "I have processed your request. Is there anything else about our AI courses I can help you with?"
                handoff = False
                handoff_reason = None

            latencies["agent"] = (time.perf_counter() - start_t) * 1000
            span.attributes["reply_len"] = len(reply)
            span.attributes["handoff"] = handoff
            return {
                "reply": reply,
                "handoff": handoff,
                "handoff_reason": handoff_reason,
                "latency_ms": latencies,
                "steps": steps
            }

        # 2. Check for tool invocation needs
        # A. Enquiry Lookup (phone number or ENQ ID)
        phone_match = re.search(r"\b\d{10}\b", transcript)
        enq_match = re.search(r"\bENQ-\d+\b", transcript, re.IGNORECASE)
        if "lookup" in transcript.lower() or "status" in transcript.lower() or "check my" in transcript.lower() or phone_match or enq_match:
            target_id = enq_match.group(0).upper() if enq_match else (phone_match.group(0) if phone_match else None)
            if target_id:
                latencies["agent"] = (time.perf_counter() - start_t) * 1000
                span.attributes["tool_call"] = f"enquiry_lookup({target_id})"
                return {
                    "tool_calls": [{"tool": "enquiry_lookup", "args": {"phone_or_enquiry_id": target_id}}],
                    "decision": f"call:enquiry_lookup({target_id})",
                    "latency_ms": latencies,
                    "steps": steps
                }

        # B. Book Demo Session
        if "book demo" in transcript.lower() or "schedule demo" in transcript.lower() or "demo session" in transcript.lower():
            enq_id = enq_match.group(0).upper() if enq_match else "ENQ-101"
            slot_match = re.search(r"(?:saturday|sunday|monday|tuesday|wednesday|thursday|friday)?\s*(?:at\s*)?\d{1,2}(?::\d{2})?\s*(?:am|pm)", transcript, re.IGNORECASE)
            slot = slot_match.group(0) if slot_match else "Saturday 11:00 AM"
            latencies["agent"] = (time.perf_counter() - start_t) * 1000
            span.attributes["tool_call"] = f"book_demo_session({slot})"
            return {
                "tool_calls": [{
                    "tool": "book_demo_session",
                    "args": {"enquiry_id": enq_id, "preferred_slot": slot, "confirmed_out_loud": True}
                }],
                "decision": f"call:book_demo_session({enq_id}, {slot})",
                "latency_ms": latencies,
                "steps": steps
            }

        # C. Refunds and Fee disputes -> Gated action (create ticket + human handoff)
        if any(k in transcript.lower() for k in ["refund", "fee dispute", "cancel enrollment", "money back", "waive", "dispute"]):
            latencies["agent"] = (time.perf_counter() - start_t) * 1000
            span.attributes["tool_call"] = "create_ticket(refund)"
            return {
                "tool_calls": [{
                    "tool": "create_ticket",
                    "args": {"summary": f"Refund/Dispute request from caller: {transcript}", "category": "refund"}
                }],
                "decision": "call:create_ticket(refund)",
                "latency_ms": latencies,
                "steps": steps
            }

        # 3. Direct question answering via KB
        messages = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{context}"},
            {"role": "user", "content": transcript}
        ]
        with tracer.span("llm:completion", session_id, kind="llm", attributes={"model": llm_client.fast_model}) as llm_span:
            reply = llm_client.complete(messages)
            llm_span.attributes["reply_len"] = len(reply)

        # Deterministic check if response indicates KB-miss or handoff needed
        handoff = False
        handoff_reason = None
        lower_reply = reply.lower()
        if "don't have that verified" in lower_reply or "connect you with an admissions counselor" in lower_reply or "connect you directly" in lower_reply:
            handoff = True
            handoff_reason = "Query topic not verified in Knowledge Base"

        latencies["agent"] = (time.perf_counter() - start_t) * 1000
        span.attributes["reply_len"] = len(reply)
        span.attributes["handoff"] = handoff
        return {
            "reply": reply,
            "handoff": handoff,
            "handoff_reason": handoff_reason,
            "latency_ms": latencies,
            "steps": steps
        }
