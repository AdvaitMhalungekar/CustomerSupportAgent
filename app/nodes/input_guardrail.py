import time
from app.graph.state import VoiceAgentState
from app.guardrails.input_guard import check_input_safety

def n_input_guardrail(state: VoiceAgentState) -> VoiceAgentState:
    start_t = time.perf_counter()
    transcript = state.get("transcript", "")
    latencies = dict(state.get("latency_ms", {}))

    is_safe, violation, sanitized = check_input_safety(transcript)
    elapsed = (time.perf_counter() - start_t) * 1000
    latencies["input_guardrail"] = elapsed

    if not is_safe:
        return {
            "transcript": sanitized,
            "handoff": True,
            "handoff_reason": f"Input safety triggered: {violation}",
            "reply": "I am unable to process that instruction. Let me transfer you to our admissions desk.",
            "latency_ms": latencies
        }

    return {
        "transcript": sanitized,
        "latency_ms": latencies
    }
