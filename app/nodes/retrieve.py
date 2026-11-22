import time
from app.graph.state import VoiceAgentState
from app.kb.provider import StaticKBProvider

_kb_provider = StaticKBProvider()

def n_retrieve(state: VoiceAgentState) -> VoiceAgentState:
    start_t = time.perf_counter()
    transcript = state.get("transcript", "")
    latencies = dict(state.get("latency_ms", {}))

    context = _kb_provider.get_context(transcript)
    elapsed = (time.perf_counter() - start_t) * 1000
    latencies["retrieve"] = elapsed

    return {
        "context": context,
        "latency_ms": latencies
    }
