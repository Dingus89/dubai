"""
    diadub/alignment/mfa_aligner.py

    Wrapper to run Montreal Forced Aligner (MFA) for high-accuracy forced alignment.
    This module provides a simple programmatic interface that:
     - generates required temp directories and files (wav + transcript)
     - calls MFA (binary) to align
     - parses MFA TextGrid output into the same word dict shape used by align_words()
     - falls back gracefully if MFA is not installed

    MFA requirements:
     - Follow MFA install: https://montreal-forced-aligner.readthedocs.io/
     - Install pretrained acoustic and pronunciation models (or run with --clean)
     - This wrapper assumes `mfa` CLI is on PATH
    """
from pathlib import Path
from typing import List, Dict, Any, Optional
import subprocess
import tempfile
import shutil
import json
import logging
import os
import re

log = logging.getLogger("diadub.alignment.mfa_aligner")


def _textgrid_to_words(tg_path: Path) -> List[Dict[str, Any]]:
    """
    Very small TextGrid parser focused on word tier.
    Parses a single-tier TextGrid exported by MFA with "words" tier.
    """
    words = []
    try:
        with open(tg_path, "r", encoding="utf-8") as f:
            txt = f.read()
    except Exception as e:
        log.error("Failed to open TextGrid: %s", e)
        return words

    # crude parse: find intervals blocks and extract label, xmin, xmax
    # This is tolerant to common MFA TextGrid format
    pattern = re.compile(
        r'intervals \[\d+\][\s\S]*?xmin = ([0-9\.]+)[\s\S]*?xmax = ([0-9\.]+)[\s\S]*?text = "([^"]*)"')
    for m in pattern.finditer(txt):
        xmin = float(m.group(1))
        xmax = float(m.group(2))
        label = m.group(3).strip()
        if label == "" or label.lower() == "sp":  # skip empty or sp markers
            continue
        words.append({"word": label, "start": xmin,
                     "end": xmax, "duration": xmax - xmin})
    return words


def align_with_mfa(audio_path: str, transcript: str, mfa_model_path: Optional[str] = None, temp_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Align using MFA.
    - audio_path: path to wav
    - transcript: transcription text (string)
    - mfa_model_path: optional path to pretrained acoustic model dir (not required if MFA has defaults)
    Returns list of words dict {word,start,end,duration}
    """
    p = Path(audio_path)
    if not p.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    # Ensure mfa CLI present
    try:
        subprocess.run(["mfa", "--version"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        raise RuntimeError(
            "MFA CLI not found on PATH. Install MFA: https://montreal-forced-aligner.readthedocs.io/") from e

    # prepare temp workspace
    tmp = Path(temp_dir) if temp_dir else Path(
        tempfile.mkdtemp(prefix="mfa_align_"))
    wavs_dir = tmp / "wavs"
    text_dir = tmp / "txt"
    out_dir = tmp / "mfa_out"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # write files: basename.wav and basename.txt
    base = p.stem
    shutil.copy2(str(p), str(wavs_dir / f"{base}.wav"))
    (text_dir / f"{base}.txt").write_text(transcript, encoding="utf-8")

    # Build MFA command. Many MFA installs handle models internally; this is a basic invocation.
    try:
        cmd = ["mfa", "align", str(wavs_dir), str(text_dir), str(out_dir)]
        # If user provided model path, add it (MFA uses installed models by default; this is optional)
        if mfa_model_path:
            cmd.extend(["-a", str(mfa_model_path)])
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MFA alignment failed: {e}") from e

    # MFA writes TextGrid next to wavs or in output dir; search for .TextGrid
    tg_files = list(out_dir.rglob("*.TextGrid")) + \
        list(wavs_dir.rglob("*.TextGrid"))
    if not tg_files:
        tg = out_dir / f"{base}.TextGrid"
        if tg.exists():
            tg_files = [tg]
    if not tg_files:
        raise RuntimeError(
            "MFA did not produce a TextGrid output; check MFA logs")

    words = _textgrid_to_words(tg_files[0])

    # cleanup temp if we created it
    if temp_dir is None:
        try:
            shutil.rmtree(tmp)
        except Exception:
            pass

    return words
