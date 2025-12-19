"""
diadub/alignment/align_wrapper.py
Universal wrapper to choose between faster-whisper, MFA, or naive aligners
based on configuration and availability. Exposes align_words_universal()
which returns the standard word dict shape.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import os

log = logging.getLogger("diadub.alignment.align_wrapper")

# Try to import faster-whisper aligner
try:
    from .word_aligner import align_words as _whisper_align
    _HAS_WHISPER_ALIGN = True
except Exception:
    _whisper_align = None
    _HAS_WHISPER_ALIGN = False

# Try to import MFA aligner
try:
    from .mfa_aligner import align_with_mfa as _mfa_align
    _HAS_MFA_ALIGN = True
except Exception:
    _mfa_align = None
    _HAS_MFA_ALIGN = False


def align_words_universal(audio_path: str, text: str, prefer: str = "mfa", **kwargs) -> List[Dict[str, Any]]:
    """
    Universal aligner selection.
    prefer: 'mfa'|'whisper'|'auto' - selection preference
    kwargs forwarded to underlying aligners (e.g., mfa_model_path)
    """
    prefer = prefer.lower() if isinstance(prefer, str) else "auto"
    tried = []
    # try preferred first
    if prefer in ("mfa", "auto") and _HAS_MFA_ALIGN:
        try:
            return _mfa_align(audio_path, text, **kwargs)
        except Exception as e:
            log.debug("MFA align failed: %s", e)
            tried.append("mfa")
    if prefer in ("whisper", "auto") and _HAS_WHISPER_ALIGN:
        try:
            return _whisper_align(audio_path, text, **kwargs)
        except Exception as e:
            log.debug("Whisper align failed: %s", e)
            tried.append("whisper")
    # fallback attempts
    if prefer == "mfa" and _HAS_WHISPER_ALIGN:
        try:
            return _whisper_align(audio_path, text, **kwargs)
        except Exception as e:
            log.debug("Fallback whisper align also failed: %s", e)
            tried.append("whisper")
    if prefer == "whisper" and _HAS_MFA_ALIGN:
        try:
            return _mfa_align(audio_path, text, **kwargs)
        except Exception as e:
            log.debug("Fallback MFA align also failed: %s", e)
            tried.append("mfa")
    # last resort: naive uniform aligner from word_aligner
    try:
        from .word_aligner import _naive_uniform_align as _naive
        duration = kwargs.get("duration", 1.0)
        return _naive(duration, text)
    except Exception as e:
        log.error(
            "All aligners failed; returning empty list. Tried: %s; error: %s", tried, e)
        return []
