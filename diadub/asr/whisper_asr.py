import logging
from pathlib import Path
from typing import List, Dict, Any

from .base_asr import BaseASR

log = logging.getLogger("diadub.asr.whisper")

try:
    from faster_whisper import WhisperModel
except ImportError:
    log.warning(
        "faster-whisper not installed. Install via `pip install faster-whisper`"
    )


class WhisperASR(BaseASR):
    """ASR wrapper around faster-whisper."""

    def __init__(self, model_id="openai/whisper-base", device="cuda"):
        self.model_id = model_id
        self.device = device
        self.model = WhisperModel(model_id, device=device)
        log.info(f"Loaded Whisper model: {model_id}")

    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        log.info(f"Transcribing {audio_path}")
        segments, info = self.model.transcribe(audio_path)
        results = []
        for seg in segments:
            results.append(
                {
                    "text": seg.text.strip(),
                    "start": seg.start,
                    "end": seg.end,
                    "confidence": getattr(seg, "avg_logprob", 0.0),
                }
            )
        log.info(f"Transcribed {len(results)} segments")
        return results


def load_model(spec: dict, device="cuda"):
    """Factory called by ModelRegistry."""
    model_id = spec.get("model_id", "openai/whisper-base")
    return WhisperASR(model_id=model_id, device=device)
