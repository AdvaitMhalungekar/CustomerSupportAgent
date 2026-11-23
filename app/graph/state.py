from typing import TypedDict, Optional, List, Dict, Any

class VoiceAgentState(TypedDict, total=False):
    # Core pipeline
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
    handoff_category: Optional[str]        # Phase 4: "kb_miss"|"tool_error"|"high_impact"|...
    handoff_payload: Optional[Dict[str, Any]]
    barge_in: Optional[str]
    interrupted: bool
    steps: int
    latency_ms: Dict[str, float]
    budget_ms: float

    # Phase 4 Safety fields
    consent_given: bool                    # True after caller ACKs consent notice
    consent_status: Optional[str]          # "granted"|"refused"|"re_prompt"|"pending"
    pii_redacted_transcript: Optional[str] # transcript after storage-level PII redaction
    safety_blocked: bool                   # True if input safety blocked the turn
    safety_violation: Optional[str]        # description of violation
    conversation_summary: Optional[str]    # from ShortTermMemory
