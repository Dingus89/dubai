"""
diadub/alignment/word_aligner.py

Provides a simple forced-alignment interface using faster-whisper if available,
with a fallback lightweight aligner for testing. Designed to be replaceable by
Montreal Forced Aligner (MFA) later without changing the pipeline API.

Main function:
    align_words(audio_path: str, text: str, max_word_gap: float = 0.15) -> List[Dict]

Returned per-word dict:
    {
      "word": "Hello",
      "start": 12.300,
      "end": 12.620,
      "duration": 0.320,
      "loudness_db": -18.3,
      "mean_f0": 123.4
    }
"""
from .align_utils import rms_db, mean_f0_in_interval, clamp_times, split_text_words
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import numpy as np
import subprocess

log = logging.getLogger("diadub.alignment.word_aligner")

# optional dependencies
try:
    from faster_whisper import WhisperModel
    _HAS_FASTER_WHISPER = True
except Exception:
    WhisperModel = None
    _HAS_FASTER_WHISPER = False

try:
    import soundfile as sf
except Exception:
    sf = None

try:
    import librosa
    _HAS_LIBROSA = True
except Exception:
    librosa = None
    _HAS_LIBROSA = False


# simple cached model holder to avoid reloading
_faster_whisper_model = None


def _load_faster_whisper(model_name: str = "large-v3", device: str = "cuda"):
    global _faster_whisper_model
    if _faster_whisper_model is None:
        if not _HAS_FASTER_WHISPER:
            raise RuntimeError("faster-whisper is not installed")
        _faster_whisper_model = WhisperModel(
            model_name, device=device, compute_type="float16")
    return _faster_whisper_model


def _naive_uniform_align(duration: float, text: str) -> List[Dict[str, Any]]:
    """
    Very simple fallback: split text into words and distribute across duration uniformly.
    """
    words = split_text_words(text)
    n = len(words)
    if n == 0:
        return []
    per = duration / n
    out = []
    t = 0.0
    for w in words:
        start = t
        end = t + per
        out.append({"word": w, "start": start,
                   "end": end, "duration": end - start})
        t = end
    return out


def align_words(audio_path: str, text: str, max_word_gap: float = 0.15, model_name: str = "large-v3", device: str = "cuda") -> List[Dict[str, Any]]:
    """
    Align words in `text` to `audio_path`. Returns list of word dicts with start/end.
    Strategy:
      1. If faster-whisper is available, use it to get word-level timestamps.
      2. Otherwise, fall back to a naive uniform splitter.
      3. Enrich intervals with loudness_db and mean_f0 if librosa/soundfile is available.
    """
    p = Path(audio_path)
    if not p.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    duration = None
    try:
        if sf is not None:
            info = sf.info(str(p))
            duration = info.frames / info.samplerate
        elif _HAS_LIBROSA:
            y, sr = librosa.load(str(p), sr=None, mono=True)
            duration = len(y) / sr
    except Exception:
        duration = None

    words_aligned: List[Dict[str, Any]] = []

    # Attempt faster-whisper alignment
    if _HAS_FASTER_WHISPER:
        try:
            model = _load_faster_whisper(model_name=model_name, device=device)
            segments, info = model.transcribe(
                str(p), word_timestamps=True, vad_filter=True)
            # segments is iterable of objects with .words list (word, start, end)
            for seg in segments:
                if hasattr(seg, "words") and seg.words:
                    for w in seg.words:
                        words_aligned.append({
                            "word": w.word.strip(),
                            "start": float(w.start),
                            "end": float(w.end),
                            "duration": float(w.end - w.start)
                        })
            # If no word timestamps returned, fall through to naive
            if not words_aligned:
                raise RuntimeError("No word timestamps from faster-whisper")
        except Exception as e:
            log.debug("Faster-whisper alignment failed or unavailable: %s", e)
            words_aligned = []
    # Fallback uniform split across segment duration
    if not words_aligned:
        if duration is None:
            # try to get duration via ffprobe
            try:
                import shlex
                cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{str(p)}"'
                res = subprocess.check_output(cmd, shell=True).decode().strip()
                duration = float(res)
            except Exception:
                duration = None
        if duration is None:
            # set duration to 1s to avoid divide by zero; align uniformly
            duration = 1.0
        words = _naive_uniform_align(duration, text)
        # shift uniform align to segment start 0..duration, user will offset later
        words_aligned = words

    # enrich with loudness and pitch
    try:
        if _HAS_LIBROSA and sf is not None:
            y, sr = librosa.load(str(p), sr=None, mono=True)
            for w in words_aligned:
                s_frame = int(max(0, round(w["start"] * sr)))
                e_frame = int(min(len(y), round(w["end"] * sr)))
                if e_frame <= s_frame:
                    w["loudness_db"] = None
                    w["mean_f0"] = None
                else:
                    chunk = y[s_frame:e_frame]
                    w["loudness_db"] = rms_db(chunk)
                    try:
                        w["mean_f0"] = mean_f0_in_interval(chunk, sr)
                    except Exception:
                        w["mean_f0"] = None
    except Exception as e:
        log.debug("Failed to compute per-word loudness/pitch: %s", e)

    # Clamp to valid times and ensure durations
    words_aligned = clamp_times(
        words_aligned, total_duration=duration, min_gap=max_word_gap)

    return words_aligned
