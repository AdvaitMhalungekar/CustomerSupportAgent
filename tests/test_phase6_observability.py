"""
Phase 6 Tests — Observability, Spans & OpenTelemetry Integration.

Tests:
- Span creation and recording via contextmanager
- Trace span attributes (session_id, node name, duration_ms, kind)
- Error status handling on exceptions in spans
- LocalSpanCollector buffering & filtering by trace_id
- Node and Tool level automatic span generation
- Phoenix collector endpoint configuration
"""
import pytest
from app.observability import tracer, Span, LocalSpanCollector
from app.graph.builder import voice_graph
from app.graph.state import VoiceAgentState
from app.config import settings


def test_span_context_manager_records_duration():
    collector = LocalSpanCollector()
    tracer_inst = tracer
    tracer_inst.collector = collector

    with tracer_inst.span("test_node", "sess-123", kind="node", attributes={"test_key": "val"}) as s:
        import time
        time.sleep(0.01)

    spans = collector.get_spans("sess-123")
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "test_node"
    assert span.kind == "node"
    assert span.duration_ms >= 8.0
    assert span.status == "OK"
    assert span.attributes["test_key"] == "val"


def test_span_records_error_status_on_exception():
    collector = LocalSpanCollector()
    tracer.collector = collector

    with pytest.raises(ValueError):
        with tracer.span("failing_node", "sess-err", kind="node") as s:
            raise ValueError("Simulated node failure")

    spans = collector.get_spans("sess-err")
    assert len(spans) == 1
    span = spans[0]
    assert span.status == "ERROR"
    assert "Simulated node failure" in (span.error_message or "")


def test_graph_turn_emits_node_and_llm_spans():
    collector = LocalSpanCollector()
    tracer.collector = collector
    session_id = "trace-turn-test"

    state: VoiceAgentState = {
        "transcript": "What is the fee for the Machine Learning course?",
        "session_id": session_id,
        "latency_ms": {},
        "tool_calls": [],
        "tool_results": [],
        "steps": 0,
        "interrupted": False,
        "handoff": False,
    }
    voice_graph.invoke(state)

    spans = collector.get_spans(session_id)
    span_names = [s.name for s in spans]
    assert "node:agent" in span_names
    assert "llm:completion" in span_names


def test_tool_execution_emits_tool_span():
    collector = LocalSpanCollector()
    tracer.collector = collector
    session_id = "trace-tool-test"

    state: VoiceAgentState = {
        "transcript": "Look up my enquiry 9876543210",
        "session_id": session_id,
        "latency_ms": {},
        "tool_calls": [],
        "tool_results": [],
        "steps": 0,
        "interrupted": False,
        "handoff": False,
    }
    voice_graph.invoke(state)

    spans = collector.get_spans(session_id)
    span_names = [s.name for s in spans]
    assert any("tool:enquiry_lookup" in name for name in span_names)


def test_phoenix_config_defaults():
    assert "localhost:6006" in settings.PHOENIX_COLLECTOR_ENDPOINT
    assert settings.PHOENIX_PROJECT_NAME == "ai-adventures-voice-support"
