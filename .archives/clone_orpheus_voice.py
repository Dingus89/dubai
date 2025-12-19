"""
Starter script: prepare an Orpheus/Coqui-style voice package.

- Expects a folder with WAV samples.
- Normalizes and resamples to 48k mono (pipeline uses 48k).
- Writes metadata JSON and copies files into voices/{voice_id}/samples/.
- (Optional) attempts to extract mel features for downstream adaptation.
"""

import argparse
from pathlib import Path
import shutil
import json
import subprocess


def normalize_and_resample(src: Path, dst: Path, sr=48000):
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i",
           str(src), "-ac", "1", "-ar", str(sr), str(dst)]
    subprocess.run(cmd, check=True)


def main(src_dir, voice_id):
    s = Path(src_dir)
    if not s.exists():
        raise FileNotFoundError(src_dir)
    out = Path("voices") / voice_id
    samples = out / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    # copy and resample WAVs
    for i, f in enumerate(sorted(s.glob("*.wav"))):
        dst = samples / f"{i:03d}.wav"
        normalize_and_resample(f, dst, sr=48000)
    # write metadata
    meta = {"id": voice_id, "notes": "", "created": None}
    out_meta = out / "metadata.json"
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Prepared voice:", voice_id, "samples:",
          len(list(samples.glob('*.wav'))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="Directory containing wav samples")
    parser.add_argument("voice_id", help="voice id (e.g. male_30_smooth)")
    args = parser.parse_args()
    main(args.src, args.voice_id)
