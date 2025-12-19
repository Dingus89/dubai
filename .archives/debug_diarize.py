"""
Quick script to test the pyannote diarizer.
Place a short wav in data/samples/test.wav (or modify path below).
Run: python scripts/debug_diarize.py
"""

import logging
from pathlib import Path
from diadub.models.registry import ModelRegistry
from diadub.ffmpeg_utils import extract_audio

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("debug.diarize")


def main():
    sample_video = Path("data/samples/test.mp4")
    # if a video exists, extract audio, otherwise allow direct wav path
    if sample_video.exists():
        audio_path = extract_audio(str(sample_video))
    else:
        # fallback to direct wav in samples
        audio_path = "data/samples/test.wav"

    registry = ModelRegistry()
    try:
        diarizer = registry.get("diarization")
    except KeyError:
        log.error(
            "No diarization model configured in models.json under 'diarization'.")
        return

    segments = diarizer.diarize(audio_path)
    log.info("Diarization result (first 20 segments):")
    for seg in segments[:20]:
        log.info("[%0.2f - %0.2f] %s", seg["start"],
                 seg["end"], seg["speaker"])


if __name__ == "__main__":
    main()
