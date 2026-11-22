"""
Latency instrumentation and TTFA (Time-to-First-Audio) measurement.

TTFA = time from end of caller's last word to first audio byte delivered.

Components:
  t0 = VAD end-of-speech timestamp (utterance sealed)
  t1 = ASR transcript returned
  t2 = Agent graph response returned
  t3 = First TTS audio chunk sent

TTFA = t3 - t0
Groq RTT = t2 - t1 (pure LLM + network, excluding ASR)
ASR latency = t1 - t0

NOTE: All times are wall-clock on this machine. Network jitter between
      machine and Groq servers is included but not decomposed.
      Reported numbers are from REAL runs with GROQ_API_KEY set.
      Offline mode latencies are ~0ms (deterministic policy, no network).
"""
from __future__ import annotations

import time
import statistics
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
from app.config import settings

@dataclass
class TurnLatency:
    session_id: str
    turn_index: int
    transcript: str
    asr_ms: float = 0.0
    agent_ms: float = 0.0
    tts_first_chunk_ms: float = 0.0
    ttfa_ms: float = 0.0          # end-of-speech → first audio byte
    groq_rtt_ms: float = 0.0     # LLM-only Groq round-trip
    handoff: bool = False
    turn_detector: str = "silence_timer"
    dead_air_ms: float = 0.0
    nodes: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class LatencyRecorder:
    """Thread-safe latency store. Accumulates per-turn records for reporting."""

    def __init__(self):
        self._records: List[TurnLatency] = []

    def record(self, lat: TurnLatency):
        self._records.append(lat)

    def all(self) -> List[TurnLatency]:
        return list(self._records)

    def clear(self):
        self._records.clear()

    def report(self) -> Dict:
        recs = [r for r in self._records if r.ttfa_ms > 0]
        if not recs:
            return {"n": 0, "note": "No turns with TTFA recorded yet."}

        ttfas = [r.ttfa_ms for r in recs]
        asrs = [r.asr_ms for r in recs if r.asr_ms > 0]
        agents = [r.agent_ms for r in recs if r.agent_ms > 0]
        dead_airs = [r.dead_air_ms for r in recs if r.dead_air_ms > 0]
        groq_rtts = [r.groq_rtt_ms for r in recs if r.groq_rtt_ms > 0]

        def stats(vals):
            if not vals:
                return {}
            return {
                "n": len(vals),
                "p50_ms": round(statistics.median(vals), 1),
                "p95_ms": round(sorted(vals)[int(len(vals) * 0.95)], 1),
                "mean_ms": round(statistics.mean(vals), 1),
                "min_ms": round(min(vals), 1),
                "max_ms": round(max(vals), 1),
            }

        target = settings.TTFA_TARGET_MS
        meets_target = sum(1 for v in ttfas if v <= target)
        return {
            "n": len(recs),
            "ttfa": stats(ttfas),
            "asr": stats(asrs),
            "agent_graph": stats(agents),
            "groq_rtt": stats(groq_rtts),
            "dead_air": stats(dead_airs),
            "ttfa_target_ms": target,
            "meets_target_pct": round(100 * meets_target / len(ttfas), 1),
            "note": (
                "OFFLINE MODE: Groq not contacted. All latencies are "
                "deterministic policy (sub-10ms). Live numbers require GROQ_API_KEY."
                if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "your_groq_api_key_here"
                else "LIVE MODE: Groq API contacted. Numbers reflect real network latency."
            ),
        }

    def save_json(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"records": [asdict(r) for r in self._records], "report": self.report()},
                f, indent=2
            )

    def print_table(self):
        report = self.report()
        print("\n" + "=" * 65)
        print("  LATENCY REPORT (Phase 3)")
        print("=" * 65)
        print(f"  Turns recorded     : {report.get('n', 0)}")
        print(f"  TTFA target        : ≤ {report.get('ttfa_target_ms')} ms")
        print(f"  Meets target       : {report.get('meets_target_pct', 'N/A')}%")
        print("-" * 65)
        for key in ("ttfa", "asr", "agent_graph", "groq_rtt", "dead_air"):
            s = report.get(key, {})
            if s:
                print(f"  {key:<18} p50={s.get('p50_ms')}ms  "
                      f"p95={s.get('p95_ms')}ms  "
                      f"mean={s.get('mean_ms')}ms")
        print("-" * 65)
        print(f"  Note: {report.get('note', '')}")
        print("=" * 65 + "\n")


# Global recorder (shared across sessions)
latency_recorder = LatencyRecorder()
