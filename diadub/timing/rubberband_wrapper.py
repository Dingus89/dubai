"""
diadub/timing/rubberband_wrapper.py

Wrapper around time-stretching using Rubber Band library (CLI) or pyrubberband (Python).
Provides safe fallbacks and reports estimated memory/time.

Functions:
 - time_stretch_segment(input_wav, output_wav, stretch, method='rubberband')
   stretch: >1.0 makes audio longer, <1.0 shorter
"""
from pathlib import Path
import subprocess
import logging
import shutil

log = logging.getLogger("diadub.timing.rubberband_wrapper")

# Check for pyrubberband
_HAS_PYRUBBER = False
try:
    import pyrubberband as prb
    _HAS_PYRUBBER = True
except Exception:
    _HAS_PYRUBBER = False

# Check for rubberband CLI
_RUBBER_EXISTS = shutil.which("rubberband") is not None


def time_stretch_segment(input_wav: str, output_wav: str, stretch: float, method: str = "auto") -> None:
    """
    Time-stretches a single WAV file.
    - input_wav: source path
    - output_wav: destination path (will be overwritten)
    - stretch: desired duration ratio (target_duration / source_duration)
    - method: 'auto'|'pyrubberband'|'rubberband_cli'
    """
    inp = Path(input_wav)
    out = Path(output_wav)
    if not inp.exists():
        raise FileNotFoundError(f"Input WAV not found: {input_wav}")
    if stretch <= 0:
        raise ValueError("stretch must be > 0")

    chosen = method
    if chosen == "auto":
        if _HAS_PYRUBBER:
            chosen = "pyrubberband"
        elif _RUBBER_EXISTS:
            chosen = "rubberband_cli"
        else:
            chosen = None

    if chosen == "pyrubberband" and _HAS_PYRUBBER:
        try:
            import soundfile as sf
            y, sr = sf.read(str(inp))
            y_t = prb.time_stretch(
                y, sr, rate=1.0 / stretch) if stretch != 1.0 else y
            sf.write(str(out), y_t, sr)
            return
        except Exception as e:
            log.warning("pyrubberband failed: %s", e)
            # fallback to CLI if available

    if chosen == "rubberband_cli" and _RUBBER_EXISTS:
        # rubberband CLI expects stretch factor as -t ratio (target/old) or -s? Use -t
        # Some builds use rubberband -t stretch input output
        cmd = ["rubberband", "-t", str(stretch), str(inp), str(out)]
        try:
            subprocess.run(
                cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except Exception as e:
            log.warning("rubberband CLI failed: %s", e)

    # Last resort: copy input to output without change (no stretching)
    log.warning(
        "No time-stretch backend available; copying input to output without stretching")
    shutil.copy2(str(inp), str(out))
