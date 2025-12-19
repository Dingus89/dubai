import pytest
from pathlib import Path
import numpy as np
import os
from pydub.generators import Sine
from pydub import AudioSegment
from diadub.prosody.prosody_mapper import ProsodyMapper

# ensure test data dir
SAMPLES = Path("data/samples")
SAMPLES.mkdir(parents=True, exist_ok=True)


def make_synthetic_reference(path: Path, freq=220, dur_ms=3000):
    t = Sine(freq).to_audio_segment(duration=dur_ms).apply_gain(-6)
    t.export(str(path), format="wav")


def make_synthetic_tts(path: Path, freq=440, dur_ms=2000):
    t = Sine(freq).to_audio_segment(duration=dur_ms).apply_gain(-3)
    t.export(str(path), format="wav")


def test_prosody_mapper_basic():
    mapper = ProsodyMapper()
    meta = {"mean_f0": 120.0, "loudness_db": -20.0, "words_per_sec": 2.5}
    out = mapper.map(meta)
    assert "rate" in out and "pitch_semitones" in out and "gain_db" in out


@pytest.mark.parametrize("per_subsegment", [False, True])
def test_align_and_adjust_roundtrip(per_subsegment, tmp_path):
    ref = tmp_path / "ref.wav"
    tts = tmp_path / "tts.wav"
    out = tmp_path / "out.wav"
    make_synthetic_reference(ref)
    make_synthetic_tts(tts)
    # import aligner
    from diadub.alignment.aligner import align_and_adjust, _wav_duration

    # call align_and_adjust
    adjusted = align_and_adjust(
        str(tts), str(ref), str(out),
        per_subsegment=per_subsegment, max_stretch=1.5
    )
    assert Path(adjusted).exists()
    # duration check: adjusted should be close to reference duration
    ref_d = _wav_duration(str(ref))
    out_d = _wav_duration(str(adjusted))
    # allow 15% tolerance
    assert abs(out_d - ref_d) / max(1e-6, ref_d) < 0.60
