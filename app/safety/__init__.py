"""
Phase 4 — Safety layer.

Modules:
  consent.py      — ConsentManager: captures and verifies caller consent
  audit.py        — ImmutableAuditLog: append-only JSONL log per session
  pii.py          — PII redactor for stored transcripts (extended from input_guard)
  memory.py       — ShortTermMemory: sliding window + summarisation of old turns
  handoff.py      — WarmHandoffBuilder: assembles the structured handoff payload
"""
