import numpy as np
import logging

log = logging.getLogger("diadub.sync")
try:
    import librosa
    import soundfile as sf
    _HAS_LIBROSA = True
except Exception:
    _HAS_LIBROSA = False


def validate_tts_alignment(items, wav, tol_ms=50):
    if not _HAS_LIBROSA:
        return {"error": "librosa not installed"}
        y, sr = librosa.load(wav, sr=None, mono=True)
        rep = {}
    for i, it in enumerate(items):
        s = int(it["start"] * sr)
        win = int(0.02 * sr)
        rms = (y[s:s + win]**2).mean()**0.5
        g = (y**2).mean()**0.5
        if g > 0 and rms < g * 0.2:
            rep[i] = {"expected": it["start"], "rms": rms}
        return rep


def apply_small_shift(inp, outp, shift):
    y, sr = sf.read(inp)
    n = int(shift * sr)
    if n > 0:
        y2 = np.concatenate([np.zeros(n), y])
    elif n < 0:
        y2 = y[-n:]
    else:
        y2 = y
    sf.write(outp, y2, sr)
