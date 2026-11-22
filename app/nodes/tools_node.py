import time
from typing import Dict, Any
from app.graph.state import VoiceAgentState
from app.tools.registry import tools_registry, ToolError, SafetyPolicyViolation

def n_tools(state: VoiceAgentState) -> VoiceAgentState:
    start_t = time.perf_counter()
    tool_calls = state.get("tool_calls", [])
    results = list(state.get("tool_results", []))
    latencies = dict(state.get("latency_ms", {}))

    for tc in tool_calls:
        tool_name = tc.get("tool")
        args = tc.get("args", {})
        err = None
        res = None

        # Execute with single retry policy
        for attempt in range(2):
            try:
                if tool_name == "enquiry_lookup":
                    res = tools_registry.enquiry_lookup(**args)
                elif tool_name == "book_demo_session":
                    res = tools_registry.book_demo_session(**args)
                elif tool_name == "create_ticket":
                    res = tools_registry.create_ticket(**args)
                elif tool_name == "request_callback":
                    res = tools_registry.request_callback(**args)
                else:
                    raise ToolError(f"Unknown tool: {tool_name}")
                err = None
                break
            except Exception as e:
                err = str(e)
                time.sleep(0.05)

        results.append({
            "tool": tool_name,
            "args": args,
            "result": res,
            "error": err
        })

    latencies["tools"] = (time.perf_counter() - start_t) * 1000
    return {
        "tool_results": results,
        "tool_calls": [],  # clear pending
        "latency_ms": latencies
    }
