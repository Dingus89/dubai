import logging
import math

log = logging.getLogger("diadub.prosody")


class ProsodyMapper:
    def __init__(self, base_rate=1.0, pitch_scale=1.0, loudness_scale=1.0):
        self.base_rate = base_rate
        self.pitch_scale = pitch_scale
        self.loudness_scale = loudness_scale

    def map(self, audio_meta: dict) -> dict:
        """
        audio_meta example: {
            "mean_f0": 120.0,         #       Hz
            "median_f0": 118.0,
            "loudness_db": -18.0,     #       LUFS or dB
            "duration": 3.2,
            "words_per_sec": 2.8
        }

        Returns:
            {
                "rate": 0.95,                 #       multiplier (1.0 = normal speed)
                "pitch_semitones": -1.2,      #       semitone shift (+/-)
                "gain_db": -1.5               #       db gain to apply to TTS output
            }
        """
        rate = self.base_rate
        pitch = 0.0
        gain = 0.0

        # speaking rate heuristic: words_per_sec low => slower output
        wps = audio_meta.get("words_per_sec")
        if wps:
            # typical wps ~ 2.5-3.5; map to 0.85-1.15
            rate = max(0.7, min(1.3, 3.0 / (wps + 1e-6)))
            # small smoothing
            rate = (self.base_rate + rate) / 2.0

        # pitch heuristic: map mean_f0 relative to 150Hz reference
        mean_f0 = audio_meta.get("mean_f0")
        if mean_f0 and mean_f0 > 0:
            ref = 150.0
            ratio = mean_f0 / ref
            # convert ratio to semitones: 12*log2(ratio)
            try:
                semitones = 12.0 * math.log2(ratio)
            except Exception:
                semitones = 0.0
            # scale down aggressiveness
            pitch = semitones * 0.5 * self.pitch_scale

        # loudness mapping: quieter original -> increase TTS gain a bit
        loud = audio_meta.get("loudness_db")
        if loud is not None:
            # target loudness ~ -16 LUFS; if original is quieter than that, boost tts
            try:
                delta = -16.0 - float(loud)
                gain = max(-6.0, min(6.0, delta * 0.3 * self.loudness_scale))
            except Exception:
                gain = 0.0

        mapped = {
            "rate": float(round(rate, 3)),
            "pitch_semitones": float(round(pitch, 3)),
            "gain_db": float(round(gain, 3)),
        }
        log.debug("Prosody mapped from %s -> %s", audio_meta, mapped)
        return mapped

    def semitone_from_ratio(ratio: float) -> float:
        return 12.0 * math.log2(ratio) if ratio > 0 else 0.0
