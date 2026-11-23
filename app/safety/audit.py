"""
Immutable Audit Log — Phase 4 Safety.

Every agent action (tool call, handoff, reply, consent event) is appended to
a per-session JSONL file. The log is append-only: existing lines are never
modified or deleted. Each entry carries:
  - ISO timestamp
  - session_id
  - event_type
  - payload (action details)
  - transcript_ref (the caller utterance that triggered it)
  - sha256 hash of (prev_line + this_line) for chain integrity

The log directory is <BASE_DIR>/audit_logs/. Each session gets its own file:
  audit_logs/<session_id>.jsonl

In production, replace the filesystem backend with an append-only object store
(e.g. GCS Bucket with Object Versioning, S3 WORM). The interface stays the same.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.config import settings

logger = logging.getLogger(__name__)

AUDIT_DIR = os.path.join(settings.BASE_DIR, "audit_logs")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class AuditEntry:
    session_id: str
    event_type: str            # "consent" | "tool_call" | "tool_result" | "reply" | "handoff" | "barge_in" | "safety_block"
    payload: Dict[str, Any]
    transcript_ref: str        # caller utterance that triggered this event (may be redacted)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    chain_hash: str = ""       # sha256(prev_line + this_line_without_hash)


class ImmutableAuditLog:
    """
    Append-only per-session audit log backed by a local JSONL file.
    Every write is fsync'd; lines are never updated or removed.
    """

    def __init__(self, session_id: str, log_dir: Optional[str] = None):
        self.session_id = session_id
        self.log_dir = log_dir or AUDIT_DIR
        os.makedirs(self.log_dir, exist_ok=True)
        self._path = os.path.join(self.log_dir, f"{session_id}.jsonl")
        self._last_line: str = ""  # for chaining

    # ── Public API ──────────────────────────────────────────────────────────

    def log_consent(self, outcome: str, transcript_ref: str):
        self._append(AuditEntry(
            session_id=self.session_id,
            event_type="consent",
            payload={"outcome": outcome},
            transcript_ref=transcript_ref,
        ))

    def log_tool_call(self, tool_name: str, args: Dict, transcript_ref: str):
        self._append(AuditEntry(
            session_id=self.session_id,
            event_type="tool_call",
            payload={"tool": tool_name, "args": args},
            transcript_ref=transcript_ref,
        ))

    def log_tool_result(self, tool_name: str, result: Any, error: Optional[str], transcript_ref: str):
        self._append(AuditEntry(
            session_id=self.session_id,
            event_type="tool_result",
            payload={"tool": tool_name, "result": result, "error": error},
            transcript_ref=transcript_ref,
        ))

    def log_reply(self, reply: str, handoff: bool, transcript_ref: str):
        self._append(AuditEntry(
            session_id=self.session_id,
            event_type="reply",
            payload={"reply_len": len(reply), "handoff": handoff},
            transcript_ref=transcript_ref,
        ))

    def log_handoff(self, reason: str, summary: str, transcript_ref: str):
        self._append(AuditEntry(
            session_id=self.session_id,
            event_type="handoff",
            payload={"reason": reason, "summary": summary},
            transcript_ref=transcript_ref,
        ))

    def log_safety_block(self, violation: str, transcript_ref: str):
        self._append(AuditEntry(
            session_id=self.session_id,
            event_type="safety_block",
            payload={"violation": violation},
            transcript_ref=transcript_ref,
        ))

    def log_barge_in(self, barge_text: str, transcript_ref: str):
        self._append(AuditEntry(
            session_id=self.session_id,
            event_type="barge_in",
            payload={"barge_text": barge_text[:200]},
            transcript_ref=transcript_ref,
        ))

    def read_all(self) -> list:
        """Read all entries (for inspection / test assertions)."""
        if not os.path.exists(self._path):
            return []
        entries = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries

    def verify_chain(self) -> bool:
        """
        Verify the hash chain is intact.
        Returns True if every entry's chain_hash matches sha256(prev_line + base_line).
        """
        entries = self.read_all()
        prev_line = ""
        for entry in entries:
            stored_hash = entry.get("chain_hash", "")
            # Reconstruct the entry without chain_hash to compute expected hash
            base = {k: v for k, v in entry.items() if k != "chain_hash"}
            base_line = json.dumps(base, separators=(",", ":"), ensure_ascii=False)
            expected = _sha256(prev_line + base_line)
            if stored_hash != expected:
                logger.error(f"Audit chain broken at entry: {entry.get('timestamp')}")
                return False
            prev_line = json.dumps(entry, separators=(",", ":"), ensure_ascii=False)
        return True

    # ── Internal ────────────────────────────────────────────────────────────

    def _append(self, entry: AuditEntry):
        """Compute chain hash and append to JSONL file (fsync'd)."""
        base = {
            "session_id": entry.session_id,
            "event_type": entry.event_type,
            "payload": entry.payload,
            "transcript_ref": entry.transcript_ref,
            "timestamp": entry.timestamp,
        }
        base_line = json.dumps(base, separators=(",", ":"), ensure_ascii=False)
        chain_hash = _sha256(self._last_line + base_line)
        entry.chain_hash = chain_hash

        full = dict(base, chain_hash=chain_hash)
        line = json.dumps(full, separators=(",", ":"), ensure_ascii=False)

        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            self._last_line = line
        except OSError as e:
            logger.error(f"Audit log write failed: {e}")


class AuditLogRegistry:
    """Registry of per-session audit logs (lazy creation)."""

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = log_dir
        self._logs: Dict[str, ImmutableAuditLog] = {}

    def get(self, session_id: str) -> ImmutableAuditLog:
        # Check global AUDIT_DIR dynamically if self.log_dir is None
        active_dir = self.log_dir or AUDIT_DIR
        if session_id not in self._logs or self._logs[session_id].log_dir != active_dir:
            self._logs[session_id] = ImmutableAuditLog(session_id, log_dir=active_dir)
        return self._logs[session_id]

    def clear(self):
        self._logs.clear()


# Global registry
audit_registry = AuditLogRegistry()
