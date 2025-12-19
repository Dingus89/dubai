"""
diadub/alignment/align_utils.py

Helper utilities for word alignment, loudness, pitch, and time clamping.
"""
from typing import List, Dict, Any
import numpy as np
import math

# lightweight RMS->dB


def rms_db(y):
    try:
        import numpy as np
        rms = (np.mean(y**2))**0.5
        if rms <= 1e-9:
            return -120.0
        return 20.0 * math.log10(rms + 1e-12)
    except Exception:
        return None


def mean_f0_in_interval(y, sr):
    """
    Estimate mean fundamental frequency using librosa.yin if available.
    y: numpy array segment
    sr: sample rate
    """
    try:
        import librosa
        f0 = librosa.yin(y, fmin=50, fmax=600, sr=sr)
        f0 = f0[~np.isnan(f0)]
        if len(f0) == 0:
            return None
        return float(np.mean(f0))
    except Exception:
        return None


def split_text_words(text: str):
    # simple splitter that retains punctuation as part of word if attached
    parts = [w.strip() for w in text.replace(
        "\n", " ").split(" ") if w.strip()]
    return parts


def clamp_times(words: List[Dict[str, Any]], total_duration: float = None, min_gap: float = 0.05) -> List[Dict[str, Any]]:
    """
    Ensure that word start/end times are monotonic and within [0, total_duration].
    If gaps are very small or negative, adjust by distributing small amounts.
    """
    if not words:
        return words
    # enforce order
    for i in range(len(words)):
        if "start" not in words[i] or "end" not in words[i]:
            continue
        if words[i]["end"] < words[i]["start"]:
            words[i]["end"] = words[i]["start"] + \
                max(0.01, words[i].get("duration", 0.05))
    # clamp within total_duration
    if total_duration is not None:
        for w in words:
            if w["start"] < 0:
                w["start"] = 0.0
            if w["end"] > total_duration:
                w["end"] = total_duration
            w["duration"] = max(0.0, w["end"] - w["start"])
    # ensure non-overlap by nudging
    for i in range(1, len(words)):
        prev = words[i - 1]
        cur = words[i]
        if cur["start"] < prev["end"]:
            # push cur forward slightly
            shift = prev["end"] - cur["start"] + 0.001
            cur["start"] += shift
            cur["end"] += shift
            cur["duration"] = max(0.0, cur["end"] - cur["start"])
    # merge tiny gaps
    for i in range(len(words) - 1):
        if words[i + 1]["start"] - words[i]["end"] < min_gap:
            # expand previous slightly or compress next
            mid = (words[i]["end"] + words[i + 1]["start"]) / 2.0
            words[i]["end"] = mid
            words[i + 1]["start"] = mid
            words[i]["duration"] = words[i]["end"] - words[i]["start"]
            words[i + 1]["duration"] = words[i +
                                             1]["end"] - words[i + 1]["start"]
    return words
