from pathlib import Path
import pytest
from diadub.pipeline import Pipeline
from .gen_sample_audio import make_video


@pytest.mark.order("last")
def test_pipeline_run(tmp_path):
    video = make_video()
    outdir = tmp_path / "out"
    pipe = Pipeline()
    result = pipe.run(str(video), out_dir=str(outdir), dry_run=True)
    assert "error" in result and result["error"] == "no_segments"
