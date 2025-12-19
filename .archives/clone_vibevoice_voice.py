"""
Starter script for extracting embeddings for VibeVoice-style models.
This script does not implement VibeVoice training.
Prepares audio & extracts mel features to feed VibeVoice embedding extractor.
Requirements: librosa, soundfile, numpy
"""

import argparse
from pathlib import Path
import librosa
import numpy as np
import json


def extract_mels(src_dir, out_json):
    p = Path(src_dir)
    files = sorted(p.glob("*.wav"))
    entries = []
    for f in files:
        y, sr = librosa.load(str(f), sr=48000, mono=True)
        # Compute mel or log-mel using librosa
        mels = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
        log_mel = librosa.power_to_db(mels).mean(axis=1).tolist()
        entries.append({"file": str(f), "mel_mean": log_mel})
    Path(out_json).write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print("Wrote embeddings-like json to", out_json)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python scripts/clone_vibevoice_voice.py <samples_dir> <out.json>")
        sys.exit(1)
    extract_mels(sys.argv[1], sys.argv[2])
