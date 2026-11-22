from typing import TypedDict, Optional, List, Dict, Any

class VoiceAgentState(TypedDict, total=False):
    transcript: str
    session_id: str
    context: Optional[str]
    messages: List[Dict[str, str]]
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    decision: Optional[str]
    reply: Optional[str]
    handoff: bool
    handoff_reason: Optional[str]
    handoff_payload: Optional[Dict[str, Any]]
    barge_in: Optional[str]
    interrupted: bool
    steps: int
    latency_ms: Dict[str, float]
    budget_ms: float
