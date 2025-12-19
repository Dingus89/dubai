import numpy as np
import logging
from pathlib import Path

log = logging.getLogger("diadub.audio_analysis")

try:
    import librosa

    _HAS_LIBROSA = True
except Exception:
    _HAS_LIBROSA = False
try:
    import soundfile as sf

    _HAS_SOUNDFILE = True
except Exception:
    _HAS_SOUNDFILE = False
try:
    import pyloudnorm as pyln

    _HAS_PYLOUD = True
except Exception:
    _HAS_PYLOUD = False


def analyze_segment(wav_path: str) -> dict:
    out = {
        "duration": 0.0,
        "mean_f0": None,
        "pitch_median": None,
        "loudness_db": None,
        "lufs": None,
        "words_per_sec": None,
    }
    p = Path(wav_path)
    if not p.exists():
        return out
    try:
        if _HAS_SOUNDFILE:
            data, sr = sf.read(str(p))
            data = np.mean(data, 1) if data.ndim > 1 else data
        elif _HAS_LIBROSA:
            data, sr = librosa.load(str(p), sr=22050, mono=True)
        else:
            return out
        dur = len(data) / sr
        out["duration"] = dur
        rms = np.sqrt(np.mean(data**2))
        out["loudness_db"] = 20 * np.log10(rms + 1e-9)
        if _HAS_PYLOUD:
            try:
                out["lufs"] = float(pyln.Meter(sr).integrated_loudness(data))
            except:
                pass
        if _HAS_LIBROSA:
            try:
                f0 = librosa.yin(data, 50, 600, sr=sr)
                f0 = f0[~np.isnan(f0)]
                if len(f0):
                    out["mean_f0"] = float(np.mean(f0))
                    out["pitch_median"] = float(np.median(f0))
            except:
                pass
    except Exception as e:
        log.debug("analyze_segment failed: %s", e)
    return out
