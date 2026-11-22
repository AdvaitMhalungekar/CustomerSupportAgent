"""FastAPI WebSocket server for real-time voice support.

Message protocol (binary frames from browser):
  - audio frames: raw PCM int16, 16kHz mono, 480 samples (30ms)
  - text frames: JSON control messages

Server-to-browser messages (JSON):
  { "type": "transcript", "text": "...", "is_final": true }
  { "type": "audio", "data": "<base64 WAV bytes>" }
  { "type": "text_reply", "text": "...", "handoff": false }
  { "type": "barge_in_detected" }
  { "type": "handoff", "reason": "...", "summary": "..." }
  { "type": "error", "message": "..." }
"""
import asyncio
import base64
import json
import logging
import time
import uuid
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from app.graph.builder import voice_graph
from app.graph.state import VoiceAgentState
from app.audio.vad import SileroVAD, UtteranceSegmenter, FRAME_SAMPLES
from app.audio.asr import groq_asr
from app.audio.tts import tts_provider
from app.audio.turn_detector import build_turn_detector
from app.audio.latency import latency_recorder, TurnLatency

logger = logging.getLogger(__name__)

app = FastAPI(title="AI Adventures Voice Support Agent")

# Mount static client files
_client_dir = "client"
import os
if os.path.isdir(_client_dir):
    app.mount("/static", StaticFiles(directory=_client_dir), name="static")


@app.get("/")
async def index():
    with open("client/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.3.0-phase3"}


@app.get("/metrics/latency")
async def latency_metrics():
    """Return real-time TTFA and component latency statistics."""
    return latency_recorder.report()


@app.get("/metrics/latency/detail")
async def latency_detail():
    """Return per-turn raw latency records."""
    from dataclasses import asdict
    return {"records": [asdict(r) for r in latency_recorder.all()]}


class VoiceSession:
    """Manages state for one caller's WebSocket session."""

    def __init__(self, session_id: str, ws: WebSocket):
        self.session_id = session_id
        self.ws = ws
        self.vad = SileroVAD()
        self.segmenter = UtteranceSegmenter(self.vad)
        self.turn_detector = build_turn_detector()
        self.turn_index = 0
        self.conversation_history: list = []
        self.is_speaking = False
        self.current_tts_task: Optional[asyncio.Task] = None
        self.barge_in_buffer: Optional[np.ndarray] = None
        self.t_vad_end: Optional[float] = None  # timestamp when utterance was sealed

    async def send_json(self, msg: dict):
        try:
            await self.ws.send_text(json.dumps(msg))
        except Exception:
            pass

    async def send_audio_chunk(self, pcm: np.ndarray):
        """Send a WAV-encoded audio chunk to the browser."""
        wav_bytes = tts_provider.pcm_to_wav_bytes(pcm)
        b64 = base64.b64encode(wav_bytes).decode()
        await self.send_json({"type": "audio", "data": b64})

    async def stop_speaking(self):
        """Cancel in-flight TTS task (barge-in handler)."""
        if self.current_tts_task and not self.current_tts_task.done():
            self.current_tts_task.cancel()
            try:
                await self.current_tts_task
            except asyncio.CancelledError:
                pass
        self.is_speaking = False
        self.current_tts_task = None


@app.websocket("/ws/voice")
async def voice_websocket(ws: WebSocket):
    await ws.accept()
    session_id = str(uuid.uuid4())
    session = VoiceSession(session_id, ws)
    logger.info(f"Voice session started: {session_id}")

    # Consent message first (Phase 4 will gate on consent ACK; for now we announce)
    await session.send_json({
        "type": "text_reply",
        "text": "Hello! You have reached AI Adventures, Pune's hands-on AI and machine learning training institute. This call may be monitored for quality. How can I help you today?",
        "handoff": False
    })

    try:
        while True:
            message = await ws.receive()

            if "bytes" in message and message["bytes"]:
                raw = message["bytes"]
                # Each message is exactly one 30ms frame (480 int16 samples = 960 bytes)
                # or multiple frames concatenated from the browser
                frame_size_bytes = FRAME_SAMPLES * 2  # int16
                raw_bytes = raw

                while len(raw_bytes) >= frame_size_bytes:
                    frame_bytes = raw_bytes[:frame_size_bytes]
                    raw_bytes = raw_bytes[frame_size_bytes:]
                    frame = np.frombuffer(frame_bytes, dtype=np.int16).copy()

                    # --- Barge-in detection while TTS is playing ---
                    if session.is_speaking and session.vad.is_speech(frame):
                        logger.info(f"[{session_id}] BARGE-IN detected!")
                        await session.stop_speaking()
                        await session.send_json({"type": "barge_in_detected"})

                    # --- VAD segmentation ---
                    utterance = session.segmenter.process_frame(frame)
                    if utterance is not None:
                        # Stamp the moment the utterance was sealed
                        session.t_vad_end = time.perf_counter()
                        # Launch ASR + agent in background
                        asyncio.create_task(
                            _handle_utterance(session, utterance)
                        )

            elif "text" in message and message["text"]:
                try:
                    ctrl = json.loads(message["text"])
                    msg_type = ctrl.get("type")
                    if msg_type == "text_input":
                        # Text-only mode: skip ASR
                        text = ctrl.get("text", "").strip()
                        if text:
                            asyncio.create_task(_handle_text(session, text))
                    elif msg_type == "ping":
                        await session.send_json({"type": "pong"})
                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        logger.info(f"Voice session disconnected: {session_id}")
        # Flush any remaining audio utterance
        leftover = session.segmenter.flush()
        if leftover is not None and len(leftover) > 100:
            await _handle_utterance(session, leftover)
    except Exception as e:
        logger.error(f"Session {session_id} error: {e}")
        await session.send_json({"type": "error", "message": str(e)})


async def _handle_utterance(session: VoiceSession, audio: np.ndarray):
    """ASR → normalize → agent → TTS pipeline for one utterance."""
    t_utterance_end = time.perf_counter()
    t_vad_end = session.t_vad_end or t_utterance_end

    # 1. ASR
    t_asr_start = time.perf_counter()
    transcript, is_backchannel = await asyncio.get_event_loop().run_in_executor(
        None, groq_asr.transcribe, audio
    )
    asr_ms = (time.perf_counter() - t_asr_start) * 1000
    logger.info(f"ASR ({asr_ms:.0f}ms): '{transcript}' (backchannel={is_backchannel})")

    if not transcript or not transcript.strip():
        return

    await session.send_json({
        "type": "transcript",
        "text": transcript,
        "is_final": True,
        "asr_ms": round(asr_ms, 1),
        "is_backchannel": is_backchannel
    })

    # Backchannels (hmm, okay) don't trigger a full LLM re-plan
    if is_backchannel:
        logger.debug(f"Skipping re-plan for backchannel: '{transcript}'")
        return

    await _handle_text(session, transcript, t_utterance_end)


async def _handle_text(session: VoiceSession, transcript: str, t_start: Optional[float] = None):
    """Agent pipeline + streaming TTS for a finalized transcript."""
    if t_start is None:
        t_start = time.perf_counter()

    # 2. Agent graph
    t_agent_start = time.perf_counter()
    initial_state: VoiceAgentState = {
        "transcript": transcript,
        "session_id": session.session_id,
        "latency_ms": {},
        "tool_calls": [],
        "tool_results": [],
        "steps": 0,
        "interrupted": False,
        "handoff": False,
    }

    # Run filler audio if we have tools to execute (detected by keyword scan)
    needs_tool = any(k in transcript.lower() for k in [
        "lookup", "status", "book demo", "refund", "cancel", "enquiry", "enq-"
    ]) or bool(__import__("re").search(r"\b\d{10}\b", transcript))

    if needs_tool:
        filler_audio = await asyncio.get_event_loop().run_in_executor(None, tts_provider.filler)
        session.current_tts_task = asyncio.create_task(
            _stream_audio(session, [filler_audio])
        )

    final_state: VoiceAgentState = await asyncio.get_event_loop().run_in_executor(
        None, voice_graph.invoke, initial_state
    )
    agent_ms = (time.perf_counter() - t_agent_start) * 1000

    reply = final_state.get("reply") or ""
    handoff = final_state.get("handoff", False)
    handoff_reason = final_state.get("handoff_reason", "")

    # 3. Send text reply
    await session.send_json({
        "type": "text_reply",
        "text": reply,
        "handoff": handoff,
        "handoff_reason": handoff_reason,
        "latency_ms": final_state.get("latency_ms", {}),
        "agent_ms": round(agent_ms, 1),
    })

    if handoff:
        await session.send_json({
            "type": "handoff",
            "reason": handoff_reason,
            "summary": f"Caller said: {transcript}. Agent reply: {reply}",
            "transcript": session.conversation_history,
        })

    # 4. TTS streaming (sentence by sentence)
    if reply:
        # Stop filler if still playing
        if session.current_tts_task and not session.current_tts_task.done():
            session.current_tts_task.cancel()
            try:
                await session.current_tts_task
            except asyncio.CancelledError:
                pass

        t_tts_start = time.perf_counter()
        audio_chunks = await asyncio.get_event_loop().run_in_executor(
            None, _synthesize_all_sentences, reply
        )
        tts_ms = (time.perf_counter() - t_tts_start) * 1000

        ttfa = (time.perf_counter() - t_start) * 1000
        logger.info(f"TTFA={ttfa:.0f}ms | ASR+Agent={agent_ms:.0f}ms | TTS={tts_ms:.0f}ms")

        # Record latency for reporting
        lat = TurnLatency(
            session_id=session.session_id,
            turn_index=session.turn_index,
            transcript=transcript,
            asr_ms=asr_ms if hasattr(session, "_last_asr_ms") else 0.0,
            agent_ms=agent_ms,
            tts_first_chunk_ms=tts_ms,
            ttfa_ms=ttfa,
            handoff=handoff,
            turn_detector=session.turn_detector.name,
            nodes=final_state.get("latency_ms", {}),
        )
        latency_recorder.record(lat)
        session.turn_index += 1

        session.current_tts_task = asyncio.create_task(
            _stream_audio(session, audio_chunks)
        )

    # Store turn in conversation history
    session.conversation_history.append({
        "role": "user", "content": transcript,
        "agent_reply": reply, "handoff": handoff,
    })


def _synthesize_all_sentences(text: str):
    """Synthesize all sentences (run in executor to avoid blocking event loop)."""
    chunks = []
    for chunk in tts_provider.stream_sentences(text):
        chunks.append(chunk)
    return chunks


async def _stream_audio(session: VoiceSession, chunks: list):
    """Stream PCM audio chunks to the browser sequentially."""
    session.is_speaking = True
    try:
        for chunk in chunks:
            if chunk is not None and len(chunk) > 0:
                await session.send_audio_chunk(chunk)
                # Simulate realtime playback pacing (~22050Hz int16 => 44100 bytes/sec)
                duration_sec = len(chunk) / 22050
                await asyncio.sleep(duration_sec * 0.9)
    except asyncio.CancelledError:
        logger.debug("TTS stream cancelled (barge-in)")
        raise
    finally:
        session.is_speaking = False
