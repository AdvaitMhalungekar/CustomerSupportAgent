import time
from typing import Dict, Any
from app.graph.state import VoiceAgentState

def n_speak_gate(state: VoiceAgentState) -> VoiceAgentState:
    """The speak gate checks whether the caller barged in while we formed the reply.
    If so, cut the turn short, clear the half-formed reply, and re-plan against what was said.
    """
    start_t = time.perf_counter()
    barge = state.get("barge_in")
    latencies = dict(state.get("latency_ms", {}))
    elapsed = (time.perf_counter() - start_t) * 1000
    latencies["speak_gate"] = elapsed

    if barge:
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
            "latency_ms": latencies
        }

    return {"latency_ms": latencies}

def route_speak_gate(state: VoiceAgentState) -> str:
    """Route after speak gate: loop back to retrieve if interrupted, else finalize response."""
    if state.get("interrupted") and state.get("reply") is None:
        return "retrieve"
    return "respond"
