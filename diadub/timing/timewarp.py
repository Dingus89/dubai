"""
diadub/timing/timewarp.py

Utilities to compute per-word stretch factors to map a "natural" TTS rendering
onto a target timing (from original audio). This implements the Option C flow:
 - Generate TTS naturally (no hard constraints)
 - Compute per-word durations for TTS output and target script
 - Compute stretch factors per word (clamped)
 - Apply small per-word time-stretching then stitch with small crossfades
"""
from pathlib import Path
from typing import List, Dict, Any
import os
import tempfile
import shutil
import math
import logging
from .rubberband_wrapper import time_stretch_segment

log = logging.getLogger("diadub.timing.timewarp")


def compute_stretch_factors(tts_words: List[Dict[str, Any]], target_words: List[Dict[str, Any]], max_stretch: float = 1.25, min_stretch: float = 0.8):
    """
    Given two lists of word dicts having 'duration' or computed durations, produce stretch factors.
    tts_words and target_words must align by word order or include mapping externally.
    Returns list of factors matching tts_words length.
    """
    factors = []
    n = min(len(tts_words), len(target_words))
    for i in range(n):
        t_dur = max(0.001, float(tts_words[i].get("duration", 0.001)))
        tgt_dur = max(0.001, float(target_words[i].get("duration", 0.001)))
        raw = tgt_dur / t_dur
        clamped = max(min_stretch, min(max_stretch, raw))
        factors.append(float(clamped))
    # If lengths differ, append 1.0 for extras
    if len(tts_words) > n:
        factors.extend([1.0] * (len(tts_words) - n))
    return factors


def apply_per_word_timewarp(tts_audio_path: str, tts_words: List[Dict[str, Any]], target_words: List[Dict[str, Any]], out_path: str, crossfade_ms: int = 10, tmpdir: str = None):
    """
    Splits tts_audio_path into per-word segments based on tts_words timings (start_abs/end_abs),
    applies per-word stretching to match target_words durations, and concatenates them into out_path.
    crossfade_ms: milliseconds of crossfade between adjacent segments to reduce clicks.
    """
    inp = Path(tts_audio_path)
    if not inp.exists():
        raise FileNotFoundError(f"TTS audio not found: {tts_audio_path}")
    tmp = Path(tmpdir) if tmpdir else Path(
        tempfile.mkdtemp(prefix="timewarp_"))
    tmp.mkdir(parents=True, exist_ok=True)
    pieces = []
    try:
        import soundfile as sf
        import numpy as np
        # Load full TTS audio
        y, sr = sf.read(str(inp))
        # If tts_words have absolute times, they should be relative to the TTS audio start; otherwise use cumulative
        # We'll treat tts_words times as relative (start and end exist)
        factors = compute_stretch_factors(tts_words, target_words)
        for i, w in enumerate(tts_words):
            s = float(w.get("start", 0.0))
            e = float(w.get("end", s + w.get("duration", 0.0)))
            s_idx = int(round(s * sr))
            e_idx = int(round(e * sr))
            segment = y[s_idx:e_idx]
            seg_path = tmp / f"seg_{i:04d}.wav"
            sf.write(str(seg_path), segment, sr)
            # apply stretch
            factor = factors[i] if i < len(factors) else 1.0
            out_seg = tmp / f"seg_{i:04d}_stretched.wav"
            try:
                time_stretch_segment(str(seg_path), str(
                    out_seg), stretch=factor, method="auto")
            except Exception as exc:
                log.warning("stretch failed for segment %d: %s", i, exc)
                shutil.copy2(str(seg_path), str(out_seg))
            pieces.append(str(out_seg))
        # concatenate pieces with crossfade
        # Simple concatenation using numpy with crossfade
        out_audio = None
        for idx, pth in enumerate(pieces):
            seg_y, _ = sf.read(pth)
            if out_audio is None:
                out_audio = seg_y
            else:
                # crossfade last crossfade_ms milliseconds
                cf = int(round((crossfade_ms / 1000.0) * sr))
                if cf <= 0:
                    out_audio = np.concatenate([out_audio, seg_y])
                else:
                    a_tail = out_audio[-cf:]
                    b_head = seg_y[:cf]
                    # linear crossfade
                    ramp = np.linspace(0, 1, cf)[:, None]
                    mix = (1 - ramp) * a_tail + ramp * b_head
                    out_audio = np.concatenate(
                        [out_audio[:-cf], mix, seg_y[cf:]])
        # write final
        if out_audio is None:
            raise RuntimeError("No output audio produced in timewarp")
        sf.write(str(out_path), out_audio, sr)
    finally:
        # cleanup tmp
        try:
            shutil.rmtree(tmp)
        except Exception:
            pass
    return str(out_path)
