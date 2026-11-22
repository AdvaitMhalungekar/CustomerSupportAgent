import time
import json
from app.graph.state import VoiceAgentState
from app.guardrails.groundedness import verify_groundedness

def n_output_guardrail(state: VoiceAgentState) -> VoiceAgentState:
    start_t = time.perf_counter()
    reply = state.get("reply", "")
    context = state.get("context", "")
    tool_results = state.get("tool_results", [])
    latencies = dict(state.get("latency_ms", {}))

    tool_context = json.dumps(tool_results) if tool_results else ""
    is_grounded, ungrounded_tokens = verify_groundedness(reply, context, tool_context)
    elapsed = (time.perf_counter() - start_t) * 1000
    latencies["output_guardrail"] = elapsed

    if not is_grounded:
        return {
            "reply": "I want to ensure all course numbers and details are completely verified, so let me connect you with an admissions counselor right away.",
            "handoff": True,
            "handoff_reason": f"Groundedness guardrail triggered: ungrounded items {ungrounded_tokens}",
            "latency_ms": latencies
        }

    return {
        "latency_ms": latencies
    }
