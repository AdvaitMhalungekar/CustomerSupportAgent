from langgraph.graph import StateGraph, START, END
from app.graph.state import VoiceAgentState
from app.nodes.input_guardrail import n_input_guardrail
from app.nodes.retrieve import n_retrieve
from app.nodes.agent import n_agent
from app.nodes.tools_node import n_tools
from app.nodes.output_guardrail import n_output_guardrail
from app.nodes.speak_gate import n_speak_gate, route_speak_gate

def route_agent(state: VoiceAgentState) -> str:
    """Route from agent: if pending tool_calls, run tools; otherwise run output guardrail."""
    if state.get("tool_calls") and len(state.get("tool_calls", [])) > 0:
        return "tools"
    return "output_guardrail"

def create_voice_agent_graph():
    builder = StateGraph(VoiceAgentState)

    # Register nodes
    builder.add_node("input_guardrail", n_input_guardrail)
    builder.add_node("retrieve", n_retrieve)
    builder.add_node("agent", n_agent)
    builder.add_node("tools", n_tools)
    builder.add_node("output_guardrail", n_output_guardrail)
    builder.add_node("speak_gate", n_speak_gate)

    # Edges
    builder.add_edge(START, "input_guardrail")
    builder.add_edge("input_guardrail", "retrieve")
    builder.add_edge("retrieve", "agent")

    builder.add_conditional_edges(
        "agent",
        route_agent,
        {
            "tools": "tools",
            "output_guardrail": "output_guardrail"
        }
    )

    builder.add_edge("tools", "agent")
    builder.add_edge("output_guardrail", "speak_gate")

    builder.add_conditional_edges(
        "speak_gate",
        route_speak_gate,
        {
            "retrieve": "retrieve",
            "respond": END
        }
    )

    return builder.compile()

# Global compiled graph
voice_graph = create_voice_agent_graph()
