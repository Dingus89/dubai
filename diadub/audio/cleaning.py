from pathlib import Path
import pyloudnorm as pyln
import soundfile as sf
import numpy as np
import logging
import subprocess
import os

log = logging.getLogger("diadub.audio.cleaning")


def normalize_loudness(in_wav: str, out_wav: str, target_lufs: float = -23.0):
    data, sr = sf.read(in_wav)
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(data)
    loud_normed = pyln.normalize.loudness(data, loudness, target_lufs)
    sf.write(out_wav, loud_normed, sr)
    return out_wav


def remove_leading_trailing_silence(in_wav: str, out_wav: str, top_db: int = 30):
    # use ffmpeg's silenceremove for robust result
    cmd = [
        "ffmpeg", "-y", "-i", in_wav,
        "-af", f"silenceremove=start_periods=1:start_duration=0.1:start_threshold=-{top_db}dB:stop_periods=1:stop_duration=0.1:stop_threshold=-{top_db}dB",
        out_wav
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out_wav


def smooth_transients(in_wav: str, out_wav: str, sr_out: int = 48000):
    # simple low-pass filter transient smoothing (uses sox if available)
    sox = shutil.which("sox")
    if sox:
        cmd = ["sox", in_wav, out_wav, "lowpass", "6000"]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return out_wav
    else:
        # fallback: just copy
        shutil.copy2(in_wav, out_wav)
        return out_wav


def clean_tts_file(in_wav: str, out_wav: str, target_lufs: float = -23.0):
    tmp1 = str(Path(out_wav).with_suffix(".norm.wav"))
    try:
        normalize_loudness(in_wav, tmp1, target_lufs=target_lufs)
        remove_leading_trailing_silence(tmp1, out_wav)
    except Exception as e:
        log.warning("TTS cleaning failed (%s), copying raw", e)
        shutil.copy2(in_wav, out_wav)
    finally:
        try:
            if Path(tmp1).exists():
                Path(tmp1).unlink()
        except Exception:
            pass
    return out_wav
