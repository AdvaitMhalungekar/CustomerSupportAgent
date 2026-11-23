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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import httpx

from app.config import settings
from app.graph.builder import voice_graph
from app.graph.state import VoiceAgentState
from app.audio.vad import SileroVAD, UtteranceSegmenter, FRAME_SAMPLES
from app.audio.asr import groq_asr
from app.audio.tts import tts_provider
from app.audio.turn_detector import build_turn_detector
from app.audio.latency import latency_recorder, TurnLatency
from app.audio.twilio_codec import (
    ulaw_to_pcm16_16k,
    pcm16_to_ulaw_8k,
    ulaw_to_base64_payload,
    base64_payload_to_ulaw,
)

# Configure root logger to output to both console and voice_agent.log file
log_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
file_handler = logging.FileHandler("voice_agent.log", encoding="utf-8")
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)
if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root_logger.handlers):
    root_logger.addHandler(console_handler)

logger = logging.getLogger("voice_app")

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
    return {
        "status": "ok",
        "version": "0.3.0-phase3",
        "twilio_configured": bool(settings.TWILIO_ACCOUNT_SID and settings.TWILIO_AUTH_TOKEN),
        "public_url": settings.PUBLIC_URL,
    }


@app.get("/logs")
async def get_recent_logs(lines: int = 100):
    """Return the recent server log entries for live telephony debugging."""
    log_file = "voice_agent.log"
    if not os.path.exists(log_file):
        return {"logs": ["Log file not created yet."]}
    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
            return {"total_lines": len(all_lines), "logs": all_lines[-lines:]}
    except Exception as e:
        return {"error": str(e)}
    return {
        "status": "ok",
        "version": "0.3.0-phase3",
        "twilio_configured": bool(settings.TWILIO_ACCOUNT_SID and settings.TWILIO_AUTH_TOKEN),
        "public_url": settings.PUBLIC_URL,
    }


class OutboundCallRequest(BaseModel):
    phone_number: str
    name: Optional[str] = None


@app.post("/api/call")
async def initiate_outbound_call(payload: OutboundCallRequest):
    """Trigger an outbound call using Twilio REST API."""
    to_number = payload.phone_number.strip().replace(" ", "").replace("-", "")
    if not to_number.startswith("+"):
        # Default to India (+91) if 10 digits provided, else require +
        if len(to_number) == 10 and to_number.isdigit():
            to_number = f"+91{to_number}"
        else:
            to_number = f"+{to_number}"

    if not settings.TWILIO_ACCOUNT_SID or not settings.TWILIO_AUTH_TOKEN:
        return JSONResponse(
            status_code=400,
            content={"error": "TWILIO_ACCOUNT_SID or TWILIO_AUTH_TOKEN not configured in .env"}
        )

    if not settings.PUBLIC_URL or "your-ngrok" in settings.PUBLIC_URL:
        return JSONResponse(
            status_code=400,
            content={"error": "PUBLIC_URL is not set to your active ngrok URL in .env"}
        )

    base_url = settings.PUBLIC_URL.rstrip("/")
    twiml_url = f"{base_url}/twilio/twiml"

    twilio_api_url = f"https://api.twilio.com/2010-04-01/Accounts/{settings.TWILIO_ACCOUNT_SID}/Calls.json"
    data = {
        "To": to_number,
        "From": settings.TWILIO_PHONE_NUMBER.strip().replace(" ", ""),
        "Url": twiml_url,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                twilio_api_url,
                data=data,
                auth=(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN),
            )
            resp_data = resp.json()

            if resp.status_code >= 400:
                logger.error(f"Twilio error: {resp_data}")
                return JSONResponse(
                    status_code=resp.status_code,
                    content={"error": resp_data.get("message", "Twilio call initiation failed"), "details": resp_data}
                )

            return {
                "status": "initiated",
                "call_sid": resp_data.get("sid"),
                "to": to_number,
                "message": f"Calling {to_number} from {settings.TWILIO_PHONE_NUMBER}..."
            }
    except Exception as e:
        logger.error(f"Error making outbound Twilio call: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.api_route("/twilio/twiml", methods=["GET", "POST"])
async def twilio_twiml(request: Request):
    """Return TwiML XML connecting the call to our WebSocket Media Stream."""
    # Determine the public wss host
    public_url = settings.PUBLIC_URL.strip().rstrip("/")
    if public_url.startswith("https://"):
        ws_url = public_url.replace("https://", "wss://") + "/ws/twilio"
    elif public_url.startswith("http://"):
        ws_url = public_url.replace("http://", "ws://") + "/ws/twilio"
    else:
        host = request.headers.get("host", "localhost:8000")
        ws_url = f"wss://{host}/ws/twilio"

    twiml_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}" />
    </Connect>
</Response>"""
    return Response(content=twiml_xml, media_type="application/xml")


@app.get("/metrics/latency")
async def latency_metrics():
    """Return real-time TTFA and component latency statistics."""
    return latency_recorder.report()


@app.get("/metrics/latency/detail")
async def latency_detail():
    """Return per-turn raw latency records."""
    from dataclasses import asdict
    return {"records": [asdict(r) for r in latency_recorder.all()]}


@app.get("/metrics/traces")
async def get_traces():
    """Return all recorded OpenTelemetry/Phoenix trace spans."""
    from app.observability import tracer
    spans = [s.to_dict() for s in tracer.collector.get_spans()]
    return {"total_spans": len(spans), "spans": spans}


@app.get("/metrics/traces/{session_id}")
async def get_session_traces(session_id: str):
    """Return trace spans for a specific session."""
    from app.observability import tracer
    spans = [s.to_dict() for s in tracer.collector.get_spans(trace_id=session_id)]
    return {"session_id": session_id, "spans": spans}


class VoiceSession:
    """Manages state for one caller's WebSocket session."""

    def __init__(self, session_id: str, ws: WebSocket, is_twilio: bool = False):
        self.session_id = session_id
        self.ws = ws
        self.is_twilio = is_twilio
        self.stream_sid: Optional[str] = None
        self.vad = SileroVAD()
        # Telephony uses slightly lower silence timeout (500ms) for snappy response
        self.segmenter = UtteranceSegmenter(self.vad, silence_timeout_ms=500 if is_twilio else 600)
        self.turn_detector = build_turn_detector()
        self.turn_index = 0
        self.conversation_history: list = []
        self.is_speaking = False
        self.current_tts_task: Optional[asyncio.Task] = None
        self.barge_in_buffer: Optional[np.ndarray] = None
        self.t_vad_end: Optional[float] = None  # timestamp when utterance was sealed
        self._leftover_bytes: bytes = b""
        self._packet_count: int = 0
        self._speech_frame_count: int = 0

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

    async def send_twilio_audio_chunk(self, pcm: np.ndarray, stream_sid: str):
        """Send mu-law 8kHz audio chunk to Twilio Media Stream in small frames (~20ms / 160 bytes)."""
        # Piper or SAPI outputs 22050Hz
        sr = getattr(tts_provider, "sample_rate", 22050)
        ulaw_bytes = pcm16_to_ulaw_8k(pcm, orig_sr=sr)
        if not ulaw_bytes:
            return

        # Twilio Media Streams expect audio in 20ms packets (160 bytes of 8kHz mu-law)
        FRAME_BYTES = 160
        for i in range(0, len(ulaw_bytes), FRAME_BYTES):
            chunk = ulaw_bytes[i:i + FRAME_BYTES]
            payload_b64 = ulaw_to_base64_payload(chunk)
            await self.send_json({
                "event": "media",
                "streamSid": stream_sid,
                "media": {
                    "payload": payload_b64
                }
            })
            # Pace ~20ms per packet
            await asyncio.sleep(0.018)

    async def send_twilio_clear(self, stream_sid: str):
        """Send clear event to interrupt Twilio buffered audio (barge-in)."""
        await self.send_json({
            "event": "clear",
            "streamSid": stream_sid
        })

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
        if self.is_twilio and self.stream_sid:
            await self.send_twilio_clear(self.stream_sid)


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
                # Each message contains 30ms frames (480 int16 samples = 960 bytes)
                frame_size_bytes = FRAME_SAMPLES * 2  # int16
                raw_bytes = session._leftover_bytes + raw

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

                session._leftover_bytes = raw_bytes

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


@app.websocket("/ws/twilio")
async def twilio_websocket(ws: WebSocket):
    """Twilio Media Streams WebSocket endpoint."""
    await ws.accept()
    session_id = str(uuid.uuid4())
    session = VoiceSession(session_id, ws, is_twilio=True)
    logger.info(f"Twilio call connected: session {session_id}")

    try:
        while True:
            msg_str = await ws.receive_text()
            if not msg_str:
                continue

            msg = json.loads(msg_str)
            event = msg.get("event")

            if event == "start":
                start_data = msg.get("start", {})
                session.stream_sid = msg.get("streamSid") or start_data.get("streamSid")
                call_sid = start_data.get("callSid")
                logger.info(f"Twilio stream started: streamSid={session.stream_sid}, callSid={call_sid}")

                # Greet user over phone
                greeting = "Hello! You have reached AI Adventures, Pune's hands-on AI and machine learning training institute. How can I help you today?"
                audio_chunks = await asyncio.get_event_loop().run_in_executor(
                    None, _synthesize_all_sentences, greeting
                )
                session.current_tts_task = asyncio.create_task(
                    _stream_audio(session, audio_chunks)
                )

            elif event == "media":
                media = msg.get("media", {})
                payload_b64 = media.get("payload")
                if not payload_b64:
                    continue

                session._packet_count += 1
                ulaw_bytes = base64_payload_to_ulaw(payload_b64)
                # Decode 8kHz mu-law to 16kHz linear PCM int16
                pcm_16k = ulaw_to_pcm16_16k(ulaw_bytes)
                if len(pcm_16k) == 0:
                    continue

                # Frame processing: 480 samples @ 16kHz = 30ms
                raw_bytes = session._leftover_bytes + pcm_16k.tobytes()
                frame_size_bytes = FRAME_SAMPLES * 2

                while len(raw_bytes) >= frame_size_bytes:
                    frame_bytes = raw_bytes[:frame_size_bytes]
                    raw_bytes = raw_bytes[frame_size_bytes:]
                    frame = np.frombuffer(frame_bytes, dtype=np.int16).copy()

                    is_sp = session.vad.is_speech(frame)
                    if is_sp:
                        session._speech_frame_count += 1

                    # Log every 100 packets to monitor audio stream alive
                    if session._packet_count % 100 == 0:
                        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
                        logger.info(f"[{session_id}] Audio alive: packets={session._packet_count}, current frame RMS={rms:.1f}, speech_frames={session._speech_frame_count}")

                    # Barge-in: caller starts speaking while TTS is playing
                    if session.is_speaking and is_sp:
                        logger.info(f"[{session_id}] Telephony BARGE-IN detected! Interrupting audio.")
                        await session.stop_speaking()

                    # VAD segmentation
                    utterance = session.segmenter.process_frame(frame)
                    if utterance is not None:
                        logger.info(f"[{session_id}] Telephony utterance detected! Length: {len(utterance)} samples ({len(utterance)/16000:.2f}s). Dispatching to ASR...")
                        session.t_vad_end = time.perf_counter()
                        asyncio.create_task(_handle_utterance(session, utterance))

                session._leftover_bytes = raw_bytes

            elif event == "stop":
                logger.info(f"Twilio stream stopped: {session.stream_sid}")
                # Flush pending utterance on stop if caller hung up or finished
                leftover = session.segmenter.flush()
                if leftover is not None and len(leftover) > 1600:
                    asyncio.create_task(_handle_utterance(session, leftover))
                break

    except WebSocketDisconnect:
        logger.info(f"Twilio call disconnected: {session_id}")
    except Exception as e:
        logger.error(f"Twilio session {session_id} error: {e}")
    finally:
        await session.stop_speaking()


async def _handle_utterance(session: VoiceSession, audio: np.ndarray):
    """ASR → normalize → agent → TTS pipeline for one utterance."""
    t_utterance_end = time.perf_counter()
    t_vad_end = session.t_vad_end or t_utterance_end

    # 1. ASR
    t_asr_start = time.perf_counter()
    try:
        transcript, is_backchannel = await asyncio.get_event_loop().run_in_executor(
            None, groq_asr.transcribe, audio
        )
    except Exception as e:
        logger.error(f"[{session.session_id}] ASR failed with exception: {e}", exc_info=True)
        return

    asr_ms = (time.perf_counter() - t_asr_start) * 1000
    logger.info(f"[{session.session_id}] ASR ({asr_ms:.0f}ms): '{transcript}' (backchannel={is_backchannel})")

    if not transcript or not transcript.strip():
        logger.info(f"[{session.session_id}] ASR returned empty transcript. Ignoring.")
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
        logger.info(f"[{session.session_id}] Synthesizing agent reply with TTS: '{reply[:60]}...'")
        audio_chunks = await asyncio.get_event_loop().run_in_executor(
            None, _synthesize_all_sentences, reply
        )
        tts_ms = (time.perf_counter() - t_tts_start) * 1000
        logger.info(f"[{session.session_id}] TTS generated {len(audio_chunks)} sentence chunks ({tts_ms:.0f}ms)")

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
    """Stream PCM audio chunks to browser or Twilio sequentially."""
    session.is_speaking = True
    try:
        for chunk in chunks:
            if chunk is not None and len(chunk) > 0:
                if session.is_twilio:
                    if session.stream_sid:
                        await session.send_twilio_audio_chunk(chunk, session.stream_sid)
                else:
                    await session.send_audio_chunk(chunk)
                    # Simulate realtime playback pacing (~22050Hz int16 => 44100 bytes/sec)
                    duration_sec = len(chunk) / 22050
                    await asyncio.sleep(duration_sec * 0.9)
    except asyncio.CancelledError:
        logger.debug("TTS stream cancelled (barge-in)")
        raise
    finally:
        session.is_speaking = False
