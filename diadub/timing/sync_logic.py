"""
diadub/timing/sync_logic.py

High-level sync logic to take:
 - generated TTS audio and its word timings
 - target script word timings (from original audio)
and produce a final time-warped TTS audio that aligns to the target.

Functions:
 - align_tts_to_target(tts_audio, tts_words, target_words, out_path, options)
"""
from typing import List, Dict, Any
from pathlib import Path
import logging
from .timewarp import apply_per_word_timewarp, compute_stretch_factors

log = logging.getLogger("diadub.timing.sync_logic")


def align_tts_to_target(tts_audio_path: str, tts_words: List[Dict[str, Any]], target_words: List[Dict[str, Any]], out_path: str, options: Dict[str, Any] = None) -> str:
    """
    Align TTS audio to target word timings.
    options may include:
      - max_stretch, min_stretch (floats)
      - crossfade_ms (int)
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    opts = options or {}
    max_stretch = float(opts.get("max_stretch", 1.25))
    min_stretch = float(opts.get("min_stretch", 0.8))
    crossfade_ms = int(opts.get("crossfade_ms", 10))
    # ensure tts_words and target_words are same length ideally; if not, best-effort mapping by index
    if not tts_words or not target_words:
        raise ValueError("tts_words and target_words required")

    # Compute stretch factors (we reuse function from timewarp)
    factors = compute_stretch_factors(
        tts_words, target_words, max_stretch=max_stretch, min_stretch=min_stretch)
    # apply per-word stretching and stitching
    out = apply_per_word_timewarp(
        tts_audio_path, tts_words, target_words, out_path, crossfade_ms=crossfade_ms)
    log.info("Aligned TTS written to: %s", out)
    return out
