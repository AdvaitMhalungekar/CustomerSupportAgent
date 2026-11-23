"""
Phase 3: Turn Detection Comparison & Latency Tuning

Provides:
1. TurnDetectorBenchmark — offline comparison of silence_timer vs semantic detector
   on a synthetic transcript corpus. Measures:
   - dead-air duration (ms added before turn fires)
   - false early-fire rate (cut-off mid-sentence)
   - end-of-turn recall (detected when utterance genuinely ended)

2. TTFAProfiler — wraps the graph + TTS to record component breakdowns.
   In offline/stub mode: returns synthetic latencies for CI.
   In live mode: measures real Groq RTT separately from TTS first-chunk.

3. PromptCachingAdvisor — inspects system prompt stability across turns and
   reports whether provider-side prompt caching is likely to help.
"""
from __future__ import annotations

import time
import logging
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from app.audio.turn_detector import (
    SilenceTimerDetector,
    SemanticTurnDetector,
    TurnDecision,
)
from app.config import settings

logger = logging.getLogger(__name__)


# ─── Synthetic Corpus ────────────────────────────────────────────────────────

# Each entry: (utterance_text, expected_complete: bool, mid_sentence: bool)
# mid_sentence=True means the utterance is intentionally incomplete (slow speaker)
# so a *good* detector should NOT fire early.
TURN_TEST_CORPUS: List[Tuple[str, bool, bool]] = [
    # Complete utterances — detector SHOULD fire
    ("What is the fee for the Data Science course?", True, False),
    ("I want to book a demo session please.", True, False),
    ("Can you tell me about the Machine Learning program?", True, False),
    ("Thank you for the information.", True, False),
    ("What is the address of AI Adventures?", True, False),
    ("My phone number is 9876543210.", True, False),
    ("That's all, goodbye.", True, False),
    ("Tell me about the Python beginner course.", True, False),
    ("How long is the Data Science program?", True, False),
    ("What courses do you offer?", True, False),

    # Incomplete mid-sentence fragments — detector should NOT fire early
    # (they would be followed by more speech)
    ("I want to know about the", False, True),    # trailing conjunction context
    ("The fee is", False, True),                   # incomplete
    ("Okay so", False, True),                      # filler
    ("What about", False, True),                   # trailing
    ("I was wondering if you could", False, True), # long incomplete

    # Backchannels — should be recognized as non-turns (backchannel, not full turn)
    ("hmm", False, False),
    ("okay", False, False),
    ("yeah", False, False),
    ("uh", False, False),
    ("mhm", False, False),
]


# ─── Detector Benchmark ─────────────────────────────────────────────────────

@dataclass
class DetectorResult:
    """Per-utterance detection result."""
    utterance: str
    expected_complete: bool
    is_mid_sentence: bool
    fired: bool
    dead_air_ms: float
    method: str
    correct: bool  # fired==expected_complete (for complete) OR ~fired (for mid-sentence)


@dataclass
class BenchmarkReport:
    """Summary of a detector benchmark run."""
    detector_name: str
    n_complete: int
    n_incomplete: int
    n_backchannel: int
    recall_complete: float        # fraction of complete utterances correctly detected
    false_early_rate: float       # fraction of incomplete utterances incorrectly fired
    backchannel_fire_rate: float  # fraction of backchannels that (incorrectly) fired as turns
    mean_dead_air_ms: float
    p50_dead_air_ms: float
    p95_dead_air_ms: float
    details: List[DetectorResult] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {
            "detector": self.detector_name,
            "recall_complete_%": round(self.recall_complete * 100, 1),
            "false_early_rate_%": round(self.false_early_rate * 100, 1),
            "backchannel_fire_rate_%": round(self.backchannel_fire_rate * 100, 1),
            "mean_dead_air_ms": round(self.mean_dead_air_ms, 1),
            "p50_dead_air_ms": round(self.p50_dead_air_ms, 1),
            "p95_dead_air_ms": round(self.p95_dead_air_ms, 1),
            "n_complete": self.n_complete,
            "n_incomplete": self.n_incomplete,
            "n_backchannel": self.n_backchannel,
        }


def _simulate_vad_frames(text: str, speech_wpm: int = 130, frame_ms: float = 30.0) -> List[bool]:
    """
    Simulate a VAD frame sequence for a text utterance.
    Returns list of is_speech bools representing the utterance followed by silence.

    Speech timing: words at ~130 wpm; silence_timeout_ms after.
    """
    n_words = max(1, len(text.split()))
    speech_duration_ms = (n_words / speech_wpm) * 60_000
    n_speech_frames = max(1, int(speech_duration_ms / frame_ms))

    # Add silence: 800ms of silence to eventually trigger even silence_timer at 600ms
    n_silence_frames = int(900 / frame_ms)  # 30 frames of silence

    frames = [True] * n_speech_frames + [False] * n_silence_frames
    return frames


def run_detector_benchmark(
    corpus: Optional[List[Tuple[str, bool, bool]]] = None,
    silence_timeout_ms: int = 600,
    semantic_early_ms: int = 200,
) -> Tuple[BenchmarkReport, BenchmarkReport]:
    """
    Run both detectors on the corpus and return (silence_timer_report, semantic_report).
    """
    corpus = corpus or TURN_TEST_CORPUS
    detectors = [
        ("silence_timer", SilenceTimerDetector(silence_timeout_ms=silence_timeout_ms)),
        ("semantic", SemanticTurnDetector(silence_timeout_ms=silence_timeout_ms, early_silence_ms=semantic_early_ms)),
    ]

    reports = []
    for det_name, detector in detectors:
        results: List[DetectorResult] = []

        for utterance, expected_complete, is_mid_sentence in corpus:
            detector.reset()

            # For semantic detector, feed partial transcript
            if hasattr(detector, "update_partial"):
                detector.update_partial(utterance)  # type: ignore

            frames = _simulate_vad_frames(utterance)
            fired = False
            dead_air_ms = 0.0
            method = "none"

            for frame_is_speech in frames:
                decision = detector.push_vad_result(frame_is_speech, frame_ms=30.0)
                if decision.end_of_turn:
                    fired = True
                    dead_air_ms = decision.dead_air_ms
                    method = decision.method
                    break

            # Correctness logic:
            # - For complete utterances: fired should be True (recall)
            # - For incomplete mid-sentence: fired should be False (don't cut off)
            # - Backchannels (not complete, not mid-sentence): treated as a special "don't-fire" case
            if is_mid_sentence:
                correct = not fired  # incomplete should NOT fire
            elif not expected_complete and not is_mid_sentence:
                # Backchannels: a fired=True here is a false-positive turn
                correct = not fired
            else:
                correct = fired  # complete utterance should fire

            results.append(DetectorResult(
                utterance=utterance,
                expected_complete=expected_complete,
                is_mid_sentence=is_mid_sentence,
                fired=fired,
                dead_air_ms=dead_air_ms,
                method=method,
                correct=correct,
            ))

        # Compute stats
        complete_cases = [r for r in results if r.expected_complete and not r.is_mid_sentence]
        incomplete_cases = [r for r in results if r.is_mid_sentence]
        backchannel_cases = [r for r in results if not r.expected_complete and not r.is_mid_sentence]
        dead_airs = [r.dead_air_ms for r in results if r.fired and r.dead_air_ms > 0]

        recall = sum(1 for r in complete_cases if r.fired) / max(1, len(complete_cases))
        false_early = sum(1 for r in incomplete_cases if r.fired) / max(1, len(incomplete_cases))
        bc_fire = sum(1 for r in backchannel_cases if r.fired) / max(1, len(backchannel_cases))

        dead_airs_sorted = sorted(dead_airs) if dead_airs else [0.0]
        p50 = statistics.median(dead_airs_sorted)
        p95_idx = int(len(dead_airs_sorted) * 0.95)
        p95 = dead_airs_sorted[min(p95_idx, len(dead_airs_sorted) - 1)]
        mean_da = statistics.mean(dead_airs_sorted)

        reports.append(BenchmarkReport(
            detector_name=det_name,
            n_complete=len(complete_cases),
            n_incomplete=len(incomplete_cases),
            n_backchannel=len(backchannel_cases),
            recall_complete=recall,
            false_early_rate=false_early,
            backchannel_fire_rate=bc_fire,
            mean_dead_air_ms=mean_da,
            p50_dead_air_ms=p50,
            p95_dead_air_ms=p95,
            details=results,
        ))

    return reports[0], reports[1]


def print_comparison_table(silence_report: BenchmarkReport, semantic_report: BenchmarkReport):
    """Print a formatted comparison table to stdout."""
    print("\n" + "=" * 72)
    print("  PHASE 3 — TURN DETECTOR COMPARISON")
    print("=" * 72)
    header = f"  {'Metric':<35} {'silence_timer':>15} {'semantic':>15}"
    print(header)
    print("-" * 72)

    rows = [
        ("Recall (complete utterances) %", f"{silence_report.recall_complete*100:.1f}%", f"{semantic_report.recall_complete*100:.1f}%"),
        ("False early-fire rate %", f"{silence_report.false_early_rate*100:.1f}%", f"{semantic_report.false_early_rate*100:.1f}%"),
        ("Backchannel false-fire rate %", f"{silence_report.backchannel_fire_rate*100:.1f}%", f"{semantic_report.backchannel_fire_rate*100:.1f}%"),
        ("Dead-air p50 (ms)", f"{silence_report.p50_dead_air_ms:.0f}", f"{semantic_report.p50_dead_air_ms:.0f}"),
        ("Dead-air p95 (ms)", f"{silence_report.p95_dead_air_ms:.0f}", f"{semantic_report.p95_dead_air_ms:.0f}"),
        ("Dead-air mean (ms)", f"{silence_report.mean_dead_air_ms:.0f}", f"{semantic_report.mean_dead_air_ms:.0f}"),
    ]

    for label, sv, semv in rows:
        print(f"  {label:<35} {sv:>15} {semv:>15}")

    print("-" * 72)
    print(f"  Corpus size: {silence_report.n_complete} complete, "
          f"{silence_report.n_incomplete} incomplete, "
          f"{silence_report.n_backchannel} backchannels")
    print("=" * 72)
    print()


# ─── Latency Profiler ────────────────────────────────────────────────────────

@dataclass
class ComponentLatency:
    """Per-component latency for a single turn."""
    asr_ms: float = 0.0
    agent_graph_ms: float = 0.0
    tts_first_chunk_ms: float = 0.0
    groq_rtt_ms: float = 0.0        # LLM-only Groq network roundtrip
    endpointing_dead_air_ms: float = 0.0
    ttfa_ms: float = 0.0            # end-of-speech → first audio byte
    turn_detector: str = "unknown"
    offline: bool = True


class TTFAProfiler:
    """
    Wraps the graph invocation to measure TTFA and component latencies.

    In offline mode: performs a dry-run of the graph with stub LLM and
    returns synthetic latencies (all sub-10ms). Reports honestly.

    In live mode: invokes real Groq and measures wall-clock latencies.
    Separates Groq RTT (LLM-only network hop) from total agent time.
    """

    def __init__(self):
        from app.graph.builder import voice_graph
        from app.llm.client import llm_client
        self._graph = voice_graph
        self._llm_client = llm_client
        self._records: List[ComponentLatency] = []

    def profile_turn(
        self,
        transcript: str,
        session_id: str = "profiler",
        turn_detector_name: str = "silence_timer",
        dead_air_ms: float = 0.0,
    ) -> ComponentLatency:
        """Profile one full turn through the graph + stub TTS synthesis."""
        from app.graph.state import VoiceAgentState
        from app.audio.tts import tts_provider

        t_start = time.perf_counter()

        state: VoiceAgentState = {
            "transcript": transcript,
            "session_id": session_id,
            "latency_ms": {},
            "tool_calls": [],
            "tool_results": [],
            "steps": 0,
            "interrupted": False,
            "handoff": False,
        }

        t_agent_start = time.perf_counter()
        final_state = self._graph.invoke(state)
        agent_ms = (time.perf_counter() - t_agent_start) * 1000

        reply = final_state.get("reply", "")
        node_latencies = final_state.get("latency_ms", {})

        # TTS first-chunk synthesis (offline: stub returns silence instantly)
        t_tts_start = time.perf_counter()
        sentences = []
        for chunk in tts_provider.stream_sentences(reply or "one moment"):
            sentences.append(chunk)
            break  # only measure first chunk
        tts_first_chunk_ms = (time.perf_counter() - t_tts_start) * 1000

        ttfa_ms = (time.perf_counter() - t_start) * 1000 + dead_air_ms

        lat = ComponentLatency(
            asr_ms=0.0,  # ASR not invoked in text-mode profiling
            agent_graph_ms=agent_ms,
            tts_first_chunk_ms=tts_first_chunk_ms,
            groq_rtt_ms=node_latencies.get("agent", 0.0),
            endpointing_dead_air_ms=dead_air_ms,
            ttfa_ms=ttfa_ms,
            turn_detector=turn_detector_name,
            offline=self._llm_client.is_offline,
        )
        self._records.append(lat)
        return lat

    def report(self) -> Dict:
        """Return a summary dict of all profiled turns."""
        if not self._records:
            return {"n": 0, "note": "No turns profiled."}

        def pct(vals, p):
            if not vals:
                return 0.0
            s = sorted(vals)
            idx = max(0, min(int(len(s) * p / 100), len(s) - 1))
            return round(s[idx], 1)

        ttfas = [r.ttfa_ms for r in self._records]
        agents = [r.agent_graph_ms for r in self._records]
        ttss = [r.tts_first_chunk_ms for r in self._records]
        dead_airs = [r.endpointing_dead_air_ms for r in self._records if r.endpointing_dead_air_ms > 0]

        is_offline = any(r.offline for r in self._records)
        return {
            "n": len(self._records),
            "ttfa_p50_ms": pct(ttfas, 50),
            "ttfa_p95_ms": pct(ttfas, 95),
            "ttfa_mean_ms": round(statistics.mean(ttfas), 1),
            "agent_graph_p50_ms": pct(agents, 50),
            "tts_first_chunk_p50_ms": pct(ttss, 50),
            "dead_air_p50_ms": pct(dead_airs, 50) if dead_airs else "N/A",
            "ttfa_target_ms": settings.TTFA_TARGET_MS,
            "meets_target_%": round(100 * sum(1 for t in ttfas if t <= settings.TTFA_TARGET_MS) / len(ttfas), 1),
            "mode": "OFFLINE (stub LLM — sub-ms graph, 0ms Groq RTT)" if is_offline else "LIVE (real Groq API)",
        }

    def print_report(self):
        rpt = self.report()
        print("\n" + "=" * 65)
        print("  PHASE 3 — TTFA LATENCY PROFILE")
        print("=" * 65)
        for k, v in rpt.items():
            if k != "mode":
                print(f"  {k:<35}: {v}")
        print(f"\n  NOTE: {rpt.get('mode', '')}")
        print("=" * 65 + "\n")


# ─── Prompt Caching Advisor ──────────────────────────────────────────────────

class PromptCachingAdvisor:
    """
    Checks whether the system prompt prefix is stable across turns.
    Provider-side prompt caching (Groq/OpenAI) requires the system prompt
    to be byte-identical at the start of every request.

    Strategy:
    - Load KB at startup, inject once into the system prompt prefix.
    - Never modify the prefix mid-session (append per-turn context as a
      separate 'user' message, not into the system message).
    - Log the prefix length so engineers can monitor cache efficiency.
    """

    def __init__(self):
        from app.llm.client import SYSTEM_PROMPT
        from app.kb.provider import KnowledgeProvider
        from app.config import settings
        import os

        self._base_prefix = SYSTEM_PROMPT
        try:
            kb = KnowledgeProvider()
            kb_text = kb.get_context("")
            self._full_prefix = f"{SYSTEM_PROMPT}\n\n<knowledge_base>\n{kb_text}\n</knowledge_base>"
        except Exception:
            self._full_prefix = SYSTEM_PROMPT

        self._prefix_hash = hash(self._full_prefix)
        self._prefix_len_tokens = len(self._full_prefix.split())  # rough estimate

    def check_stability(self, messages: List[Dict]) -> Tuple[bool, str]:
        """
        Given a messages list, check if the system message prefix is stable.
        Returns (is_stable, reason).
        """
        if not messages:
            return False, "Empty messages list"
        sys_msg = next((m for m in messages if m.get("role") == "system"), None)
        if sys_msg is None:
            return False, "No system message found"
        content = sys_msg.get("content", "")
        is_stable = hash(content) == self._prefix_hash
        return is_stable, ("stable — eligible for provider-side caching" if is_stable
                           else "UNSTABLE — prefix changed, cache will miss")

    def report(self) -> Dict:
        return {
            "prefix_len_approx_tokens": self._prefix_len_tokens,
            "prefix_hash": str(self._prefix_hash),
            "recommendation": (
                "Keep system prompt prefix byte-identical across turns. "
                "Per-turn context (tool results, conversation history) should be "
                "appended as user/assistant messages, NOT edited into the system message. "
                f"Estimated prefix: ~{self._prefix_len_tokens} tokens. "
                "Provider caching kicks in after the first call with this prefix."
            ),
        }
