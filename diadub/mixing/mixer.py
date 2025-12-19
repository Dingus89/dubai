"""
mixing/mixer.py
Implements ducking and gain-matched mixing of dialogue with original audio.

Dependencies: pydub
"""

import logging
from pathlib import Path
from pydub import AudioSegment, effects

log = logging.getLogger("diadub.mixing")


class Mixer:
    def __init__(self, duck_db: float = 8.0, fade_ms: int = 50):
        """
        duck_db: how much to reduce background when speech is present
        fade_ms: fade in/out time for ducking envelope
        """
        self.duck_db = duck_db
        self.fade_ms = fade_ms

    def mix(self, bg_path: str, dialog_path: str, out_path: str) -> str:
        """Overlay dialog over background with ducking."""
        bg = AudioSegment.from_file(
            bg_path).set_frame_rate(48000).set_channels(2)
        dialog = (
            AudioSegment.from_file(dialog_path).set_frame_rate(
                48000).set_channels(2)
        )

        # Normalize both
        bg = effects.normalize(bg)
        dialog = effects.normalize(dialog)

        log.info(
            "Applying ducking of %.1f dB with fade %d ms", self.duck_db, self.fade_ms
        )

        # Create envelope: reduce background where dialog amplitude > -inf
        envelope = dialog.split_to_mono()[0].apply_gain(
            0)  # use left channel as mask
        step = 100  # 100 ms steps
        out = AudioSegment.silent(duration=len(bg))
        for pos in range(0, len(bg), step):
            seg_bg = bg[pos: pos + step]
            seg_dialog = dialog[pos: pos + step]
            if seg_dialog.rms > 200:  # speech present
                seg_bg = seg_bg - self.duck_db
            out = out + seg_bg

        # smooth edges
        out = out.fade_in(self.fade_ms).fade_out(self.fade_ms)

        mixed = out.overlay(dialog)
        mixed = effects.normalize(mixed)
        mixed.export(out_path, format="wav")
        log.info("Exported mixed track: %s", out_path)
        return out_path
