"""
Generate synthetic audio/video samples for DiaDub tests.
Produces:
  data/samples/test_tone.wav
  data/samples/test_bg.wav
  data/samples/test_dialog.wav
  data/samples/test.mp4
"""

import os
from pathlib import Path
from pydub.generators import Sine
from pydub import AudioSegment

SAMPLES = Path("data/samples")
SAMPLES.mkdir(parents=True, exist_ok=True)


def make_tone(freq=440, dur_ms=2000, name="test_tone.wav"):
    tone = Sine(freq).to_audio_segment(duration=dur_ms).apply_gain(-6)
    path = SAMPLES / name
    tone.export(path, format="wav")
    return path


def make_bg(name="test_bg.wav"):
    bg = AudioSegment.silent(duration=5000)
    # add two short tones as "events"
    bg = bg.overlay(
        Sine(220).to_audio_segment(duration=300).apply_gain(-15), position=500
    )
    bg = bg.overlay(
        Sine(330).to_audio_segment(duration=300).apply_gain(-15), position=3000
    )
    path = SAMPLES / name
    bg.export(path, format="wav")
    return path


def make_dialog(name="test_dialog.wav"):
    # synthetic "speech" waveform
    seg = Sine(880).to_audio_segment(duration=700).apply_gain(-8)
    silent = AudioSegment.silent(duration=300)
    dialog = seg + silent + seg
    path = SAMPLES / name
    dialog.export(path, format="wav")
    return path


def make_video(name="test.mp4"):
    """Combine silent 5s video + audio using ffmpeg."""
    bg = make_bg()
    out = SAMPLES / name
    os.system(
        f"ffmpeg -y -f lavfi -i color=c=black:s=320x240:d=5 -i {bg} "
        f"-shortest -c:v libx264 -c:a aac -pix_fmt yuv420p {out}"
    )
    return out


if __name__ == "__main__":
    make_tone()
    make_bg()
    make_dialog()
    make_video()
    print("Synthetic test assets created in data/samples/")
