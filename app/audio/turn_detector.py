"""Turn detector abstraction.

Two implementations behind a config flag (USE_SEMANTIC_TURN_DETECTOR):

1. SilenceTimerDetector (default):
   Fires after SILENCE_TIMEOUT_MS of contiguous silence following speech.
   Simple, predictable, low latency. Tradeoff: may cut off slow thinkers.

2. SemanticTurnDetector:
   Runs a lightweight local model (or heuristic) to decide if the user
   has finished their thought even before silence timeout.
   Reduces dead-air for fast speakers; can mis-fire on mid-sentence pauses.
   Set USE_SEMANTIC_TURN_DETECTOR=true in .env to enable.

Both implement the same TurnDetector interface so the WebSocket server
swaps them without code changes.
"""
from __future__ import annotations

import re
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


class TurnDecision:
    """Result of a turn-end check."""
    __slots__ = ("end_of_turn", "confidence", "method", "dead_air_ms")

    def __init__(
        self,
        end_of_turn: bool,
        confidence: float = 1.0,
        method: str = "silence_timer",
        dead_air_ms: float = 0.0,
    ):
        self.end_of_turn = end_of_turn
        self.confidence = confidence
        self.method = method
        self.dead_air_ms = dead_air_ms

    def __repr__(self):
        return (
            f"TurnDecision(end={self.end_of_turn}, "
            f"conf={self.confidence:.2f}, method={self.method}, "
            f"dead_air={self.dead_air_ms:.0f}ms)"
        )


class TurnDetector(ABC):
    """Abstract turn detector — both implementations share this interface."""

    @abstractmethod
    def push_vad_result(self, is_speech: bool, frame_ms: float = 30.0) -> TurnDecision:
        """
        Push one VAD frame result (is_speech, frame duration).
        Returns a TurnDecision — check .end_of_turn to know when to send to ASR.
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset state after a turn has been emitted."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class SilenceTimerDetector(TurnDetector):
    """
    Classic silence-timer endpointing.
    Ends the turn after `silence_timeout_ms` ms of continuous silence
    following any speech activity.

    Latency profile:
      + Very low additional latency beyond the silence timeout itself.
      + Zero CPU overhead.
      - Adds the full timeout as dead-air after every utterance.
    """

    def __init__(self, silence_timeout_ms: int = None):
        self._timeout_ms = silence_timeout_ms or settings.SILENCE_TIMEOUT_MS
        self._in_speech = False
        self._silence_ms = 0.0
        self._speech_ms = 0.0

    @property
    def name(self) -> str:
        return "silence_timer"

    def push_vad_result(self, is_speech: bool, frame_ms: float = 30.0) -> TurnDecision:
        if is_speech:
            self._in_speech = True
            self._silence_ms = 0.0
            self._speech_ms += frame_ms
            return TurnDecision(False, 1.0, self.name)
        else:
            if self._in_speech:
                self._silence_ms += frame_ms
                if self._silence_ms >= self._timeout_ms:
                    dead_air = self._silence_ms
                    return TurnDecision(
                        True, 1.0, self.name, dead_air_ms=dead_air
                    )
            return TurnDecision(False, 0.0, self.name)

    def reset(self):
        self._in_speech = False
        self._silence_ms = 0.0
        self._speech_ms = 0.0


# ─── Semantic Turn Detector ─────────────────────────────────────────────────

# Heuristic patterns that strongly signal utterance completion
_COMPLETION_PATTERNS = [
    r"\?$",                             # ends with question mark
    r"\.$",                             # ends with period
    r"!\s*$",                           # exclamation
    r"\bthank(?:s| you)\b",            # "thanks" / "thank you"
    r"\bplease\b.{0,20}$",             # ends with "please ..."
    r"\bthat'?s? (?:all|it)\b",        # "that's all"
    r"\bbye\b|\bgoodbye\b",            # call closing
    r"\bwait\b|\bactually\b",          # explicit redirect / barge-in signal
    r"\d{10}\b",                        # phone number (complete)
    r"(?:enq-?)\d+",                   # enquiry ID
]
_INCOMPLETE_PATTERNS = [
    r"\band\b$",     # trailing conjunction
    r"\bbut\b$",
    r"\bor\b$",
    r"\bso\b$",
    r"\blike\b$",
    r"\buhh?\b$",   # filler hesitation
    r"\bumm?\b$",
]

_COMPILED_COMPLETE = [re.compile(p, re.IGNORECASE) for p in _COMPLETION_PATTERNS]
_COMPILED_INCOMPLETE = [re.compile(p, re.IGNORECASE) for p in _INCOMPLETE_PATTERNS]


class SemanticTurnDetector(TurnDetector):
    """
    Lightweight heuristic semantic turn detector.

    Fires as soon as:
      1. A short silence window (early_silence_ms) has elapsed AND
      2. The partial transcript contains semantic completion signals.

    Falls back to silence_timeout_ms if no semantic signal is detected.

    This approximates the LiveKit-style open-weight turn detector without
    requiring a GPU or separate model server. A real model (e.g. the
    LiveKit turn-detector ONNX) can be dropped in by overriding
    _semantic_complete().

    Latency profile:
      + Can end turns 100-400ms earlier than pure silence timer.
      - Occasional false ends on mid-sentence pauses with complete-sounding fragments.
      - Needs partial transcript, so adds coupling to ASR path.
    """

    def __init__(
        self,
        silence_timeout_ms: int = None,
        early_silence_ms: int = 200,
    ):
        self._timeout_ms = silence_timeout_ms or settings.SILENCE_TIMEOUT_MS
        self._early_ms = early_silence_ms
        self._in_speech = False
        self._silence_ms = 0.0
        self._speech_ms = 0.0
        self._partial_text: str = ""

    @property
    def name(self) -> str:
        return "semantic"

    def update_partial(self, text: str):
        """Call this with partial ASR text as it arrives."""
        self._partial_text = text.strip()

    def push_vad_result(self, is_speech: bool, frame_ms: float = 30.0) -> TurnDecision:
        if is_speech:
            self._in_speech = True
            self._silence_ms = 0.0
            self._speech_ms += frame_ms
            return TurnDecision(False, 1.0, self.name)

        if not self._in_speech:
            return TurnDecision(False, 0.0, self.name)

        self._silence_ms += frame_ms

        # Early-fire if semantically complete after short silence
        if self._silence_ms >= self._early_ms:
            if self._semantic_complete(self._partial_text):
                logger.debug(
                    f"SemanticTurnDetector: early fire at "
                    f"{self._silence_ms:.0f}ms (partial='{self._partial_text}')"
                )
                return TurnDecision(
                    True,
                    confidence=0.85,
                    method="semantic_early",
                    dead_air_ms=self._silence_ms,
                )

        # Hard fallback: full silence timeout
        if self._silence_ms >= self._timeout_ms:
            return TurnDecision(
                True,
                confidence=1.0,
                method="semantic_timeout",
                dead_air_ms=self._silence_ms,
            )

        return TurnDecision(False, 0.0, self.name)

    def reset(self):
        self._in_speech = False
        self._silence_ms = 0.0
        self._speech_ms = 0.0
        self._partial_text = ""

    def _semantic_complete(self, text: str) -> bool:
        if not text or len(text.split()) < 2:
            return False
        for pat in _COMPILED_INCOMPLETE:
            if pat.search(text):
                return False
        for pat in _COMPILED_COMPLETE:
            if pat.search(text):
                return True
        # Minimum length heuristic: > 5 words without incomplete signal
        if len(text.split()) >= 5:
            return True
        return False


def build_turn_detector() -> TurnDetector:
    """Factory: returns the configured turn detector based on settings."""
    if settings.USE_SEMANTIC_TURN_DETECTOR:
        logger.info("Using SemanticTurnDetector (heuristic mode)")
        return SemanticTurnDetector()
    else:
        logger.info(f"Using SilenceTimerDetector (timeout={settings.SILENCE_TIMEOUT_MS}ms)")
        return SilenceTimerDetector()
