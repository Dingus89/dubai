"""
pyannote_diarizer.py

Lightweight wrapper around pyannote.audio speaker diarization pipeline.

Interface:
    class PyannoteDiarizer:
        def __init__(self, model_id="pyannote/speaker-diarization", device="cuda")
        def diarize(self, audio_path: str) -> List[Dict]:
            returns list of dicts: {"speaker": str, "start": float, "end": float, "confidence": float_or_None}

A factory function `load_model(spec, device="cuda")` is provided so ModelRegistry can use this module.
"""

from pathlib import Path
import logging
from typing import List, Dict, Optional

log = logging.getLogger("diadub.diarization.pyannote")

try:
    # pyannote Pipelines (modern API)
    from pyannote.audio import Pipeline

    _HAS_PYANNOTE = True
except Exception as e:
    log.error("Failed to import pyannote.audio.Pipeline: %s", e, exc_info=True)
    Pipeline = None
    _HAS_PYANNOTE = False


class PyannoteDiarizer:
    def __init__(
        self, model_id: str = "pyannote/speaker-diarization", device: str = "cuda"
    ):
        """
        model_id: HF pipeline id, e.g. "pyannote/speaker-diarization"
        device: "cuda" or "cpu"
        """
        if not _HAS_PYANNOTE:
            raise RuntimeError(
                "pyannote.audio is not installed. Install with `pip install pyannote.audio` "
                "and ensure torch is available."
            )
        self.model_id = model_id
        self.device = device
        log.info(
            f"Loading pyannote diarization pipeline: {model_id} (device={device})")
        try:
            # By default Pipeline.from_pretrained will use GPU if available; device selection
            # is controlled by device argument in from_pretrained for some versions.
            # We pass the model id directly; pyannote will handle the rest.
            # If the HF model requires authentication, the user must set HF_TOKEN in env.
            from os import environ
            auth_token = environ.get("HF_TOKEN")
            self.pipeline = Pipeline.from_pretrained(model_id, use_auth_token=auth_token)
        except Exception as exc:
            log.warning(
                "Failed to load pyannote pipeline '%s': %s. "
                "This may be because you need to accept the user conditions on Hugging Face "
                "and/or provide an auth token via the HF_TOKEN environment variable.",
                model_id, exc
            )
            self.pipeline = None
    def diarize(self, audio_path: str) -> List[Dict[str, Optional[float]]]:
        """
        Perform diarization on the given audio file.

        Returns a list of segments:
            [
              {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.23, "confidence": None},
              ...
            ]

        Note: pyannote's diarization output does not always include segment-level confidence.
        """
        if not self.pipeline:
            log.warning("Pyannote pipeline not available, skipping diarization.")
            return []

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        log.info("Running diarization on %s", audio_path)
        try:
            diarization = self.pipeline(
                {"uri": str(audio_path.stem), "audio": str(audio_path)}
            )
        except Exception as exc:
            log.exception("Diarization pipeline failed: %s", exc)
            raise

        segments: List[Dict[str, Optional[float]]] = []

        # diarization is a pyannote.core.Annotation-like object
        # iterate tracks with labels
        try:
            # itertracks(yield_label=True) yields (segment, track, label) depending on version.
            # In many examples, use diarization.itertracks(yield_label=True)
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start = float(turn.start)
                end = float(turn.end)
                segments.append(
                    {
                        "speaker": str(speaker),
                        "start": start,
                        "end": end,
                        # pyannote diarization segments typically don't expose confidence per segment
                        "confidence": None,
                    }
                )
        except Exception:
            # fallback: iterate over diarization.items()
            try:
                for segment, track in diarization.get_timeline().support():
                    # This fallback is less structured; prefer itertracks
                    pass
            except Exception:
                # Best-effort: iterate annotation items
                for segment, label in diarization.itersegments():
                    segments.append(
                        {
                            "speaker": str(label),
                            "start": float(segment.start),
                            "end": float(segment.end),
                            "confidence": None,
                        }
                    )

        log.info("Diarization produced %d segments", len(segments))
        return segments


def load_model(spec: dict, device: str = "cuda"):
    """
    Factory for ModelRegistry.
    spec example:
    {
      "backend": "pyannote",
      "model_id": "pyannote/speaker-diarization",
      "device": "cuda"
    }
    """
    if not _HAS_PYANNOTE:
        raise RuntimeError(
            "pyannote.audio is not available. Install it to use diarization."
        )
    model_id = spec.get("model_id", "pyannote/speaker-diarization")
    device = spec.get("device", device)
    return PyannoteDiarizer(model_id=model_id, device=device)
