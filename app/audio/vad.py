"""Silero VAD wrapper using ONNX runtime (open source, runs locally on CPU).
Processes 30ms frames at 16kHz (480 samples per frame).
Returns speech probability per frame; segments audio into utterances.
"""
import numpy as np
import os
import logging
from typing import List, Tuple, Optional
import onnxruntime as ort
from app.config import settings

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_SAMPLES = 480  # 30ms @ 16kHz
SPEECH_THRESHOLD = 0.5
SILENCE_THRESHOLD = 0.35


class SileroVAD:
    """Silero VAD v5 ONNX wrapper.
    Model inputs: (input[1,N], state[2,1,128], sr[]) -> (output[1,1], stateN[2,1,128])
    """
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.path.join(settings.MODELS_DIR, "silero_vad.onnx")
        self._sess: Optional[ort.InferenceSession] = None
        # v5 uses a single consolidated state tensor [2, batch, 128]
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._input_names: List[str] = []
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            logger.warning(f"Silero VAD model not found at {self.model_path}. Using stub.")
            return
        try:
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            self._sess = ort.InferenceSession(
                self.model_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"]
            )
            self._input_names = [i.name for i in self._sess.get_inputs()]
            self._output_names = [o.name for o in self._sess.get_outputs()]
            logger.info(f"Silero VAD loaded. inputs={self._input_names} outputs={self._output_names}")
        except Exception as e:
            logger.warning(f"Failed to load Silero VAD: {e}. Using stub mode.")
            self._sess = None

    def reset_state(self):
        """Reset recurrent state between utterances."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def get_speech_prob(self, audio_frame: np.ndarray) -> float:
        """Run VAD on a single 480-sample (30ms) frame. Returns speech prob 0.0-1.0."""
        if self._sess is None:
            # Energy-based stub
            rms = float(np.sqrt(np.mean(audio_frame.astype(np.float32) ** 2)))
            return 1.0 if rms > 500 else 0.0

        frame = audio_frame.astype(np.float32) / 32768.0
        frame = frame[np.newaxis, :]  # (1, 480)
        sr = np.array(SAMPLE_RATE, dtype=np.int64)

        # v5 API: inputs are (input, state, sr) -> (output, stateN)
        feed = {"input": frame, "state": self._state, "sr": sr}
        outs = self._sess.run(["output", "stateN"], feed)
        prob = float(outs[0].squeeze())
        self._state = outs[1]
        return prob

    def is_speech(self, audio_frame: np.ndarray) -> bool:
        return self.get_speech_prob(audio_frame) >= SPEECH_THRESHOLD


class UtteranceSegmenter:
    """
    Accumulates VAD frame decisions and emits complete utterance buffers
    based on a configurable silence timeout.
    """
    def __init__(
        self,
        vad: SileroVAD,
        silence_timeout_ms: int = 600,
        pre_buffer_ms: int = 200,
    ):
        self.vad = vad
        self.silence_timeout_frames = max(1, silence_timeout_ms // 30)
        self.pre_buffer_frames = max(1, pre_buffer_ms // 30)

        self._buffer: List[np.ndarray] = []
        self._pre_buffer: List[np.ndarray] = []
        self._in_speech = False
        self._silence_count = 0

    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Feed a 30ms audio frame; returns a complete utterance as np.ndarray
        when speech ends, or None if still accumulating.
        """
        is_speech = self.vad.is_speech(frame)

        if is_speech:
            if not self._in_speech:
                # Prepend pre-buffer to catch leading edge
                self._buffer = list(self._pre_buffer)
                self._in_speech = True
            self._buffer.append(frame)
            self._silence_count = 0
            self._pre_buffer = []
        else:
            if self._in_speech:
                self._buffer.append(frame)
                self._silence_count += 1
                if self._silence_count >= self.silence_timeout_frames:
                    # Utterance ended — emit
                    utterance = np.concatenate(self._buffer)
                    self._buffer = []
                    self._in_speech = False
                    self._silence_count = 0
                    self._pre_buffer = []
                    return utterance
            else:
                # Keep rolling pre-buffer for voice onset capture
                self._pre_buffer.append(frame)
                if len(self._pre_buffer) > self.pre_buffer_frames:
                    self._pre_buffer.pop(0)

        return None

    def flush(self) -> Optional[np.ndarray]:
        """Force emit any pending speech (e.g., on WebSocket close)."""
        if self._buffer:
            utterance = np.concatenate(self._buffer)
            self._buffer = []
            self._in_speech = False
            self._silence_count = 0
            return utterance
        return None

    def is_speech_active(self) -> bool:
        return self._in_speech
