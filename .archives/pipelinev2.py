"""
pipeline.py

High-level orchestration for the diadub pipeline:
  1) Calls script_generator.generate_script(...) to create an enriched script JSON
  2) For each line: ask the TTS engine to synthesize (modular manager)
  3) Time-warp TTS to target timings using diadub.timing.sync_logic.align_tts_to_target
  4) Place each aligned segment into a delayed layer using ffmpeg adelay
  5) Mix the delayed TTS layer with original audio to produce final dubbed WAV
  6) Clean up temp files (by default)

Notes:
 - The pipeline intentionally delegates actual TTS to the tts_engine_manager,
   which picks the correct engine from config/models.json (defaults to Orpheus).
 - The pipeline writes intermediate files to out_dir/tmp_diadub/
 - Final output is a single WAV: <base_name>_dubbed.wav
"""

from pathlib import Path
import json
import os
import logging
import subprocess
import shutil
import tempfile
from typing import Dict, Any, List

log = logging.getLogger("diadub.pipeline")
logging.basicConfig(level=logging.INFO)

# Local modules we expect to exist
try:
    from script_generator import generate_script
except Exception:
    # If running from package, import package path
    try:
        from diadub.script.script_generator import generate_script
    except Exception:
        raise

# TTS manager (modular)
try:
    from models.tts_engine_manager import TTSManager
except Exception:
    TTSManager = None

# Timewarp/alignment helper
try:
    from diadub.timing.sync_logic import align_tts_to_target
except Exception:
    align_tts_to_target = None

# helpers


def _safe_run(cmd: List[str], raise_on_err: bool = True):
    log.debug("RUN: %s", " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        log.error("Command failed: %s\nSTDOUT: %s\nSTDERR: %s", " ".join(
            cmd), res.stdout.decode(errors="ignore"), res.stderr.decode(errors="ignore"))
        if raise_on_err:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return res


def _ffmpeg_delay(input_wav: str, output_wav: str, delay_ms: int):
    """
    Creates a version of input_wav prefixed by silence/delay_ms using ffmpeg adelay.
    We use 'adelay' filter with single channel mapping. Format: adelay=ms|ms (for each channel).
    """
    # ensure delay is non-negative integer
    delay_ms = max(0, int(round(delay_ms)))
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_wav),
        "-af", f"adelay={delay_ms}|{delay_ms},volume=1.0",
        "-ar", "48000", "-ac", "1",
        str(output_wav)
    ]
    _safe_run(cmd)


def _ffmpeg_mix_base_and_layers(base_audio: str, delayed_layers: List[str], out_path: str):
    """
    Mix base_audio with all delayed_layers using ffmpeg amix.
    Accepts many inputs and uses amix to sum them. Keeps duration=longest.
    """
    inputs = []
    inputs.append("-i")
    inputs.append(str(base_audio))
    for dl in delayed_layers:
        inputs.append("-i")
        inputs.append(str(dl))

    # Build filter_complex: amix=inputs=N channels=1:duration=longest
    total_inputs = 1 + len(delayed_layers)
    filter_complex = f"amix=inputs={total_inputs}:duration=longest:dropout_transition=0"
    cmd = ["ffmpeg", "-y"] + inputs + ["-filter_complex",
                                       filter_complex, "-ar", "48000", "-ac", "1", str(out_path)]
    _safe_run(cmd)

# load config


def load_models_config(config_path: str = "config/models.json") -> Dict[str, Any]:
    cfg = {"tts": {"engine": "orpheus", "model_path": ""}}
    try:
        p = Path(config_path)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        log.warning("Failed to load models config %s", config_path)
    return cfg

# pipeline


def run_pipeline(
    input_audio: str,
    out_dir: str,
    base_name: str,
    srt_path: str = None,
    language: str = "en",
    keep_temp: bool = False,
    **kwargs
) -> str:
    """
    High-level pipeline runner.

    Returns path to final dubbed wav.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_root = out_dir / "tmp_diadub"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    # 1) Generate script JSON
    log.info("Generating script JSON")
    script_path = generate_script(
        out_dir=str(out_dir),
        base_name=base_name,
        audio_path=input_audio,
        srt_path=srt_path,
        language=language,
        use_local_polish=kwargs.get("use_local_polish", True),
        groq_use=kwargs.get("groq_use", False),
    )
    log.info("Script path: %s", script_path)
    script_obj = json.loads(Path(script_path).read_text(encoding="utf-8"))

    # 2) Initialize TTS manager
    cfg = load_models_config()
    tts_manager = None
    if TTSManager is not None:
        tts_manager = TTSManager(cfg.get("tts", {}))
    else:
        log.warning(
            "TTSManager not available; pipeline cannot synthesize. Exiting.")
        raise RuntimeError("TTSManager module missing")

    # 3) For each line: synthesize, time-warp, create delayed file
    delayed_layers = []
    aligned_paths = []
    try:
        for idx, line in enumerate(script_obj.get("lines", [])):
            log.info("Processing line %d / %d", idx+1,
                     len(script_obj.get("lines", [])))
            tts_instruction = line.get("tts_instruction")
            if not tts_instruction:
                log.info("No TTS instruction for line %d; skipping", idx)
                continue

            # Synthesize using selected engine (this returns path to raw natural TTS and word timings if available)
            try:
                tts_result = tts_manager.synthesize(
                    tts_instruction, tmp_root=str(tmp_root))
                # tts_result: {"tts_path": "/path/to/tts.wav", "tts_words": [ {start,end,duration,word}, ... ] }
                tts_path = tts_result.get("tts_path")
                tts_words = tts_result.get("tts_words", [])
            except Exception as e:
                log.error("TTS synth failed for line %d: %s", idx, e)
                continue

            if not tts_path or not Path(tts_path).exists():
                log.error("TTS engine did not produce audio for line %d", idx)
                continue

            # target words = line["tts_instruction"]["target_words"] (absolute times in original audio)
            target_words = line.get(
                "tts_instruction", {}).get("target_words", [])
            # If align_tts_to_target available, use it to produce aligned wav for this segment
            aligned_segment_path = tmp_root / \
                f"{base_name}_line_{idx:04d}_aligned.wav"
            try:
                if align_tts_to_target is not None and target_words:
                    # try to map tts_words -> target_words by index; engines that output tts_words should provide relative timings
                    # align_tts_to_target will time-warp the natural TTS render to fit the target timings
                    aligned = align_tts_to_target(
                        tts_audio_path=str(tts_path),
                        tts_words=tts_words if tts_words else target_words,
                        target_words=target_words,
                        out_path=str(aligned_segment_path),
                        options={"max_stretch": 1.25,
                                 "min_stretch": 0.8, "crossfade_ms": 12}
                    )
                    aligned_path = str(aligned)
                else:
                    # If no aligner or no target words, assume tts_path is acceptable and copy
                    aligned_path = str(aligned_segment_path)
                    shutil.copy2(str(tts_path), aligned_path)
                # record path
                line["aligned_tts_path"] = aligned_path
                aligned_paths.append(aligned_path)
            except Exception as e:
                log.error("Time-warp align failed for line %d: %s", idx, e)
                # fallback to copying the raw tts
                fallback = tmp_root / f"{base_name}_line_{idx:04d}_raw.wav"
                shutil.copy2(str(tts_path), str(fallback))
                line["aligned_tts_path"] = str(fallback)
                aligned_paths.append(str(fallback))

            # Create delayed version (absolute placement) using adelay filter
            try:
                # compute delay in ms from line.start
                seg_start = float(line.get("start", 0.0))
                delay_ms = int(round(seg_start * 1000.0))
                delayed_out = tmp_root / \
                    f"{base_name}_line_{idx:04d}_delayed.wav"
                _ffmpeg_delay(line["aligned_tts_path"],
                              str(delayed_out), delay_ms)
                delayed_layers.append(str(delayed_out))
            except Exception as e:
                log.error(
                    "Failed to create delayed layer for line %d: %s", idx, e)

        # 4) Mix original audio with delayed layers
        final_out = out_dir / f"{base_name}_dubbed.wav"
        log.info("Mixing final audio to %s (this may take a while)", final_out)
        _ffmpeg_mix_base_and_layers(
            input_audio, delayed_layers, str(final_out))
        log.info("Final dubbed audio written to %s", final_out)

        # 5) Update script JSON with aligned_tts_path for each line and write final script
        script_obj["meta"]["dubbed_path"] = str(final_out)
        final_script_path = out_dir / f"{base_name}_script_with_audio.json"
        Path(final_script_path).write_text(json.dumps(
            script_obj, indent=2, ensure_ascii=False), encoding="utf-8")
    finally:
        if not keep_temp:
            try:
                shutil.rmtree(str(tmp_root))
            except Exception:
                pass

    return str(final_out)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run full diadub pipeline (script -> TTS -> align -> mix)")
    parser.add_argument("--input", required=True,
                        help="Input audio path (wav)")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--base", required=True, help="Base name for outputs")
    parser.add_argument("--srt", default=None, help="Optional SRT file")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep temporary files")
    args = parser.parse_args()
    run_pipeline(args.input, args.out, args.base,
                 srt_path=args.srt, keep_temp=args.keep_temp)
