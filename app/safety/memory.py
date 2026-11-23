"""
Short-Term Memory with Turn Summarisation — Phase 4 Safety.

Problem: resending the full conversation history every turn is wasteful and
hits Groq's tokens-per-minute limit quickly.

Solution:
  - Keep the last N turns verbatim (default N=4).
  - Summarise older turns into a single compact paragraph using the offline
    policy (or Groq if available, sparingly — summarisation is cheap).
  - The summary + recent turns together form the conversation context injected
    into each LLM call.

This keeps the prompt under ~800 tokens per turn even for long calls.

Design constraints:
  - ShortTermMemory is immutable-append (turns are never edited, only summarised).
  - Summary is refreshed lazily — only when the window slides.
  - In offline mode, summarisation is a deterministic string join (zero quota).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

VERBATIM_WINDOW = 4          # keep last N turns verbatim
MAX_SUMMARY_TOKENS = 120     # target summary length in words (rough proxy for tokens)


@dataclass
class Turn:
    index: int
    caller: str               # caller's (redacted) transcript
    agent: str                # agent's reply
    handoff: bool = False
    tool_used: Optional[str] = None


@dataclass
class MemoryState:
    summary: str = ""         # compressed summary of turns older than window
    recent_turns: List[Turn] = field(default_factory=list)
    total_turns: int = 0


class ShortTermMemory:
    """
    Sliding-window conversation memory with lazy summarisation.

    Usage:
        mem = ShortTermMemory()
        mem.add_turn(caller_text, agent_reply)
        context_str = mem.build_context()   # inject into LLM messages
    """

    def __init__(self, window: int = VERBATIM_WINDOW):
        self._window = window
        self._all_turns: List[Turn] = []
        self._summary: str = ""
        self._needs_summarise: bool = False

    def add_turn(
        self,
        caller: str,
        agent: str,
        handoff: bool = False,
        tool_used: Optional[str] = None,
    ):
        turn = Turn(
            index=len(self._all_turns),
            caller=caller,
            agent=agent,
            handoff=handoff,
            tool_used=tool_used,
        )
        self._all_turns.append(turn)
        # Mark for lazy summarisation when window slides
        if len(self._all_turns) > self._window:
            self._needs_summarise = True

    def build_context(self, llm_client=None) -> str:
        """
        Return the conversation context string for injection into the LLM prompt.
        Format:
          [Summary of earlier turns]
          Recent conversation:
          Caller: ...
          Agent: ...
          ...
        """
        if self._needs_summarise:
            self._refresh_summary(llm_client)
            self._needs_summarise = False

        recent = self._all_turns[-self._window:]
        parts: List[str] = []

        if self._summary:
            parts.append(f"<conversation_summary>\n{self._summary}\n</conversation_summary>")

        if recent:
            parts.append("<recent_turns>")
            for t in recent:
                parts.append(f"Caller: {t.caller}")
                parts.append(f"Agent: {t.agent}")
                if t.tool_used:
                    parts.append(f"[Tool used: {t.tool_used}]")
            parts.append("</recent_turns>")

        return "\n".join(parts)

    def build_messages(self) -> List[Dict[str, str]]:
        """
        Return a list of message dicts (role/content) for the LLM,
        covering the recent verbatim window only.
        """
        recent = self._all_turns[-self._window:]
        messages = []
        for t in recent:
            messages.append({"role": "user", "content": t.caller})
            messages.append({"role": "assistant", "content": t.agent})
        return messages

    def as_state(self) -> MemoryState:
        return MemoryState(
            summary=self._summary,
            recent_turns=self._all_turns[-self._window:],
            total_turns=len(self._all_turns),
        )

    def clear(self):
        self._all_turns.clear()
        self._summary = ""
        self._needs_summarise = False

    # ── Internal ───────────────────────────────────────────────────────────

    def _refresh_summary(self, llm_client=None):
        """Summarise old turns (before the current window)."""
        old_turns = self._all_turns[:-self._window]
        if not old_turns:
            return

        # Build a compact representation of old turns
        old_text = "\n".join(
            f"Turn {t.index + 1}: Caller asked '{t.caller[:80]}'. "
            f"Agent replied '{t.agent[:80]}'."
            + (f" Handoff triggered." if t.handoff else "")
            for t in old_turns
        )

        if llm_client and not llm_client.is_offline:
            # Live mode: use LLM to summarise (sparingly — only on window slide)
            try:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a call summarizer. Summarize the following customer support "
                            "conversation turns into 2-3 short sentences. Focus on: what the caller "
                            "asked, what was resolved, and any pending actions. Keep under 100 words."
                        ),
                    },
                    {"role": "user", "content": old_text},
                ]
                summary = llm_client.complete(messages)
                self._summary = summary.strip()[:600]  # hard cap
                logger.debug(f"Turn summary (LLM): {self._summary[:80]}...")
                return
            except Exception as e:
                logger.warning(f"LLM summarisation failed: {e}. Using extractive fallback.")

        # Offline / fallback: extractive summary (deterministic, zero quota)
        lines = []
        for t in old_turns[-3:]:  # keep only the last 3 old turns in extractive
            lines.append(
                f"Earlier: caller asked about '{t.caller[:60]}'; "
                f"agent replied '{t.agent[:60]}'."
            )
        self._summary = " ".join(lines)
        logger.debug(f"Turn summary (extractive): {self._summary[:80]}...")


class SessionMemoryRegistry:
    """Registry of per-session ShortTermMemory instances."""

    def __init__(self):
        self._memories: Dict[str, ShortTermMemory] = {}

    def get(self, session_id: str) -> ShortTermMemory:
        if session_id not in self._memories:
            self._memories[session_id] = ShortTermMemory()
        return self._memories[session_id]

    def clear(self, session_id: str):
        if session_id in self._memories:
            del self._memories[session_id]


# Global registry
memory_registry = SessionMemoryRegistry()
