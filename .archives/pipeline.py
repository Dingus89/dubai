from tqdm import tqdm
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import json
import time
import tempfile
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any
from diadub.models.tts_engine_manager import TTSEngineManager
from diadub import utils

log = logging.getLogger("diadub.pipeline")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------
# FFmpeg helpers (kept/merged from earlier)
# ---------------------------------------------------------------------


def _safe_run(cmd: List[str], raise_on_err: bool = True):
    log.debug("RUN: %s", " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        log.error(
            "Command failed: %s\nSTDOUT: %s\nSTDERR: %s",
            " ".join(cmd),
            res.stdout.decode(errors="ignore"),
            res.stderr.decode(errors="ignore"),
        )
        if raise_on_err:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return res


def _ffmpeg_fade_in_out(src: str, dst: str, fade_ms: int = 8, sr: int = 48000):
    """
    Apply tiny fade in/out to avoid clicks on segment boundaries.
    """
    fade_s = max(0.001, fade_ms / 1000.0)
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-af", f"afade=t=in:st=0:d={fade_s},afade=t=out:st=0:d={fade_s}",
        "-ar", str(sr), "-ac", "1",
        str(dst)
    ]
    _safe_run(cmd)


def _ffmpeg_delay(input_wav: str, output_wav: str, delay_ms: int):
    """
    Creates a version of input_wav prefixed by silence/delay_ms using ffmpeg adelay.
    Use afade to reduce clicks.
    """
    delay_ms = max(0, int(round(delay_ms)))
    cmd = [
        "ffmpeg", "-y", "-i", str(input_wav),
        "-af", f"adelay={delay_ms}|{delay_ms},afade=t=in:st=0:d=0.01,afade=t=out:st=0:d=0.01",
        "-ar", "48000", "-ac", "1",
        str(output_wav)
    ]
    _safe_run(cmd)


def _ffmpeg_mix(inputs: List[str], out_path: str):
    """
    Mix multiple inputs into a single WAV using amix.
    """
    if not inputs:
        raise ValueError("No inputs to mix")
    cmd = ["ffmpeg", "-y"]
    for i in inputs:
        cmd += ["-i", str(i)]
    total_inputs = len(inputs)
    filter_complex = f"amix=inputs={total_inputs}:duration=longest:dropout_transition=0"
    cmd += ["-filter_complex", filter_complex,
            "-ar", "48000", "-ac", "1", str(out_path)]
    _safe_run(cmd)


def _ffmpeg_mix_base_and_layers(base_audio: str, delayed_layers: List[str], out_path: str):
    """
    Mix base_audio with all delayed_layers using ffmpeg amix. Keeps duration=longest.
    """
    inputs = ["-i", str(base_audio)]
    for dl in delayed_layers:
        inputs.extend(["-i", str(dl)])
    total_inputs = 1 + len(delayed_layers)
    filter_complex = f"amix=inputs={total_inputs}:duration=longest:dropout_transition=0"
    cmd = ["ffmpeg", "-y"] + inputs + ["-filter_complex",
                                       filter_complex, "-ar", "48000", "-ac", "1", str(out_path)]
    _safe_run(cmd)


def _ffmpeg_sidechain_duck(base_audio: str, tts_mix: str, out_path: str, makeup_gain_db: float = 0.0):
    """
    Attempt to duck base_audio using tts_mix as the sidechain input.
    Uses ffmpeg's sidechaincompress filter: [0:a][1:a]sidechaincompress=...
    If sidechaincompress isn't supported, raise RuntimeError to let caller fall back.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(base_audio),
        "-i", str(tts_mix),
        "-filter_complex",
        "[0:a][1:a]sidechaincompress=threshold=0.1:ratio=8:attack=5:release=100",
        "-ar", "48000", "-ac", "1",
        str(out_path)
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        stderr = res.stderr.decode(errors="ignore")
        raise RuntimeError(f"sidechaincompress failed: {stderr}")
    return out_path


def _ffmpeg_reduce_base_gain(base_audio: str, out_path: str, gain_db: float = -6.0):
    """
    Fallback: reduce base audio overall gain by gain_db dB.
    """
    cmd = [
        "ffmpeg", "-y", "-i", str(base_audio),
        "-af", f"volume={gain_db}dB",
        "-ar", "48000", "-ac", "1",
        str(out_path)
    ]
    _safe_run(cmd)
    return out_path

# ---------------------------------------------------------------------
# Config loader (kept)
# ---------------------------------------------------------------------


def load_models_config(config_path: str = "config/models.json") -> Dict[str, Any]:
    cfg = {"tts": {"engine": "vibevoice", "model_path": "microsoft/VibeVoice-1.5B",
                   "device": "cuda", "voice_routing": {}}}
    try:
        p = Path(config_path)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        log.warning("Failed to load models config %s", config_path)
    return cfg


# ---------------------------------------------------------------------
# Optional imports & fallbacks (preserve registry/backends behavior)
# ---------------------------------------------------------------------
# extract_audio / replace_audio helpers (may come from project utilities)
try:
    from diadub.ffmpeg_utils import extract_audio, replace_audio
except Exception:
    def extract_audio(video_path: str, out_wav: str):
        cmd = ["ffmpeg", "-y", "-i",
               str(video_path), "-ar", "48000", "-ac", "1", "-vn", str(out_wav)]
        _safe_run(cmd)

    def replace_audio(video_path: str, audio_path: str, output_path: str):
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-i", str(
            audio_path), "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", str(output_path)]
        _safe_run(cmd)

# Model registry (optional)
try:
    from diadub.models.registry import ModelRegistry
except Exception:
    ModelRegistry = None

# Logging setup optional
try:
    from diadub.utils.logging_config import setup_logging
except Exception:
    setup_logging = None

# AudioSegment fallback for assembly
try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None

# checkpoint manager optional
try:
    from diadub.storage.checkpoint import CheckpointManager
except Exception:
    CheckpointManager = None

# alignment helpers
try:
    from diadub.timing.sync_logic import align_tts_to_target
except Exception:
    align_tts_to_target = None

try:
    from diadub.alignment.aligner import align_and_adjust, post_tts_pitch_shift
except Exception:
    align_and_adjust = None
    post_tts_pitch_shift = None

# TTS manager
try:
    from diadub.models.tts_engine_manager import TTSManager
except Exception:
    TTSManager = None

# Prosody mapper optional
try:
    from diadub.prosody.prosody_mapper import ProsodyMapper
except Exception:
    ProsodyMapper = None

try:
    from models.separation.demucs_wrapper import demucs_separate
except Exception:
    try:
        from diadub.models.separation.demucs_wrapper import demucs_separate
    except Exception:
        demucs_separate = None

try:
    from diadub.audio.cleaning import clean_tts_file
except Exception:
    clean_tts_file = None

try:
    from diadub.lipsync.viseme_mapper import generate_viseme_timing
except Exception:
    generate_viseme_timing = None

try:
    from diadub.lipsync.forced_align import run_mfa_align
except Exception:
    run_mfa_align = None

# queue manager for GPU/CPU scheduling (we will instantiate later)
try:
    from diadub.tools.queue_manager import QueueManager
except Exception:
    QueueManager = None


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------
DEPENDENCIES = {
    "extract_audio": [],
    "asr": ["extract_audio"],
    "diarization": ["extract_audio"],
    "translation": ["asr"],
    "generate_script": ["translation"],
    "tts": ["generate_script"],
    "assemble_dialogue": ["tts"],
    "mix": ["assemble_dialogue", "extract_audio"],
    "mux": ["mix"],
}

# -------------------------
# Separation cache & helpers
# -------------------------


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    p = Path(path)
    if not p.exists():
        return ""
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seconds_to_srt_time(s: float) -> str:
    ms = int(round(s * 1000))
    hours = ms // 3_600_000
    ms %= 3_600_000
    minutes = ms // 60_000
    ms %= 60_000
    seconds = ms // 1000
    ms %= 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"

# ---------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------


class Pipeline:
    STAGES = [
        "extract_audio",
        "asr",
        "diarization",
        "translation",
        "generate_script",
        "tts",
        "assemble_dialogue",
        "mix",
        "mux",
    ]

    def __init__(
        self,
        model_config: str = "config/models.json",
        device: str = "cuda",
        temp_dir: str = "data/cache",
        checkpoint_path: Optional[str] = None,
        resume: bool = False,
    ):
        # Per-seg separation cache and flags
        self.use_per_segment_separation = True
        # key: seg_path -> { "vocals": "/path", "checksum": "...", "meta": {...}}
        self.separation_cache = {}
        # queue manager (use to throttle GPU tasks)
        self.queue_mgr = QueueManager(cpu_workers=min(8, (os.cpu_count(
        ) or 4)//2), gpu_count=1, gpu_mem_threshold_mb=1200) if QueueManager else None
        self.device = device
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = Path(
            checkpoint_path) if checkpoint_path else self.temp_dir / "pipeline.checkpoint.json"
        self.resume = resume

        # checkpoint manager
        self.ckpt = CheckpointManager(
            str(self.checkpoint_path)) if CheckpointManager else None
        if resume and self.checkpoint_path.exists() and self.ckpt is not None:
            try:
                self.ckpt.load()
                log.info("Loaded checkpoint from %s", self.checkpoint_path)
            except Exception as e:
                log.warning(
                    "Failed to load checkpoint (will start fresh): %s", e)

        # lazy backends
        self.asr = None
        self.diarizer = None
        self.translator = None
        self.tts_manager = None
        self.tts_backend = None
        self.emotion_model = None

    # -----------------------------------------------------------------
    # Checkpoint helpers
    # -----------------------------------------------------------------

    def _stage_done(self, name: str) -> bool:
        if self.resume and self.ckpt is not None:
            return self.ckpt.is_done(name)
        return False

    def get_dependent_stages(self, failed_stage: str) -> list:
        dependents = set()

        def recurse(target):
            for stage, deps in DEPENDENCIES.items():
                if target in deps and stage not in dependents:
                    dependents.add(stage)
                    recurse(stage)
        recurse(failed_stage)
        return sorted(dependents, key=lambda s: self.STAGES.index(s))

    # -----------------------------------------------------------------
    # Core run method (unified & fixed)
    # -----------------------------------------------------------------
    def _run_extract_audio(self, video_path: str, out_dir: Path, base_stem: str) -> Path:
        if not self._stage_done("extract_audio"):
            audio_out = out_dir / f"{base_stem}.wav"
            if not audio_out.exists():
                log.info("Extracting audio from %s -> %s",
                         video_path, audio_out)
                extract_audio(str(video_path), str(audio_out))
            else:
                log.info("Using cached audio %s", audio_out)
            audio_path = audio_out
            if self.ckpt:
                self.ckpt.set_artifact("audio_path", str(audio_path))
                self.ckpt.set_stage_done(
                    "extract_audio", {"audio_path": str(audio_path)})
        else:
            audio_path = Path(self.ckpt.get_artifact(
                "audio_path")) if self.ckpt else audio_path
            log.info("Resuming: using audio from checkpoint: %s", audio_path)
        return audio_path

    def _run_asr(self, audio_path: Path, srt_path: Optional[str]) -> List[Dict]:
        if not self._stage_done("asr"):
            segments: List[Dict] = []
            if self.asr:
                try:
                    if hasattr(self.asr, "infer") and callable(getattr(self.asr, "infer")):
                        asr_out = self.asr.infer(str(audio_path))
                        if isinstance(asr_out, dict) and "segments" in asr_out:
                            segments = [
                                {"text": s.get("text", "").strip(), "start": float(s.get("start", 0.0)), "end": float(
                                    s.get("end", 0.0)), "confidence": s.get("confidence", None)}
                                for s in asr_out.get("segments", [])
                            ]
                        elif isinstance(asr_out, dict) and "text" in asr_out:
                            segments = [{"text": asr_out.get("text", ""), "start": 0.0, "end": float(
                                self._get_duration(str(audio_path))), "confidence": None}]
                        else:
                            segments = []
                    else:
                        raw_segments = self.asr.transcribe(str(audio_path))
                        segments = [
                            {"text": seg.get("text", "").strip(), "start": float(seg.get("start", 0.0)), "end": float(
                                seg.get("end", 0.0)), "confidence": seg.get("confidence", None)}
                            for seg in raw_segments
                        ]
                except Exception as exc:
                    log.exception("ASR failed: %s", exc)
                    segments = []
            else:
                log.warning("No ASR backend. Trying to parse SRT.")
                if srt_path and Path(srt_path).exists():
                    segments = self._parse_srt(srt_path)
            if self.ckpt:
                self.ckpt.set_artifact("segments", segments)
                self.ckpt.set_stage_done("asr", {"segments": segments})
        else:
            segments = self.ckpt.get_artifact(
                "segments", []) if self.ckpt else []
            log.info("Resuming: loaded %d segments from checkpoint.",
                     len(segments))
        return segments

    def _run_diarization(self, audio_path: Path, segments: List[Dict]) -> List[Dict]:
        if not self._stage_done("diarization"):
            diarize_segs: List[Dict] = []
            if self.diarizer:
                try:
                    diarize_segs = self.diarizer.diarize(str(audio_path))
                    self._attach_speakers(segments, diarize_segs)
                except Exception as exc:
                    log.exception("Diarization failed: %s", exc)
            if self.ckpt:
                self.ckpt.set_artifact("diarize_segments", diarize_segs)
                self.ckpt.set_stage_done(
                    "diarization", {"diarize_segments": diarize_segs})
        else:
            diarize_segs = self.ckpt.get_artifact(
                "diarize_segments", []) if self.ckpt else []
            self._attach_speakers(segments, diarize_segs)
        return segments

    def _run_translation(self, segments: List[Dict]) -> List[Dict]:
        if not self._stage_done("translation"):
            try:
                from tqdm import tqdm
                bar = tqdm(total=len(segments), desc="Translating", unit="seg")
            except Exception:
                bar = None

            for i, seg in enumerate(segments):
                txt = seg.get("text", "")
                if self.translator and txt.strip():
                    try:
                        seg["translated"] = (
                            self.translator.translate_text(txt)
                            if hasattr(self.translator, "translate_text")
                            else self.translator.translate_lines([seg])[0].get("translated", txt)
                        )
                    except Exception:
                        try:
                            if hasattr(self.translator, "translate_lines"):
                                res = self.translator.translate_lines([seg])
                                seg["translated"] = res[0].get(
                                    "translated", txt)
                            else:
                                seg["translated"] = txt
                        except Exception:
                            seg["translated"] = txt
                else:
                    seg["translated"] = txt

                if bar:
                    bar.update(1)
                try:
                    if self.ckpt:
                        self.ckpt.set_progress(
                            "translation", i + 1, len(segments))
                except Exception:
                    pass

            if bar:
                bar.close()
            if self.ckpt:
                self.ckpt.set_artifact("translated_segments", segments)
                self.ckpt.set_stage_done(
                    "translation", {"translated_segments": segments})
        else:
            segments = self.ckpt.get_artifact(
                "translated_segments", segments) if self.ckpt else segments
        return segments

    def _run_generate_script(self, out_dir: Path, base_stem: str, audio_path: Path, translated_srt_path: Path, segments: List[Dict], groq_use: bool, groq_api_key: Optional[str], language: str) -> Path:
        script_path = out_dir / f"{base_stem}_script.json"
        if not self._stage_done("generate_script"):
            try:
                try:
                    from diadub.script.script_generator import generate_script
                    emotion_model = self.emotion_model
                    groq_flag = groq_use
                    groq_key = groq_api_key
                    script_file = generate_script(
                        out_dir=str(out_dir),
                        base_name=base_stem,
                        audio_path=str(audio_path),
                        srt_path=(
                            str(translated_srt_path)
                            if translated_srt_path.exists() else None
                        ),
                        asr_segments=segments,
                        emotion_model=emotion_model,
                        groq_use=groq_flag,
                        groq_api_key=groq_key,
                        language=language,
                        use_local_polish=True,
                    )
                    script_path = Path(script_file)
                    log.info("Script generated at %s", script_path)
                except Exception as e:
                    log.warning(
                        "Script generator not available or failed: %s. Falling back to segment-based script.", e)
                    fallback = {"metadata": {"base_name": base_stem, "created_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ"), "audio_path": str(audio_path)}, "lines": []}
                    for i, seg in enumerate(segments):
                        fallback["lines"].append({
                            "index": i,
                            "start": seg.get("start", 0.0),
                            "end": seg.get("end", seg.get("start", 0.0) + 0.5),
                            "text": seg.get("translated", seg.get("text", "")),
                            "analysis": {"duration": seg.get("end", seg.get("start", 0.0)) - seg.get("start", 0.0)},
                        })
                    script_path.write_text(json.dumps(
                        fallback, indent=2, ensure_ascii=False))
                if self.ckpt:
                    self.ckpt.set_artifact("script_path", str(script_path))
            except Exception as e:
                log.exception("generate_script stage failed: %s", e)
            if self.ckpt:
                self.ckpt.set_stage_done(
                    "generate_script", {"script_path": str(script_path)})
        else:
            script_path = Path(self.ckpt.get_artifact(
                "script_path", str(script_path))) if self.ckpt else script_path
            log.info("Resuming: script at %s", script_path)
        return script_path

    def _run_tts(self, out_dir: Path, base_stem: str, script_path: Path, segments: List[Dict]) -> List[Dict]:
        synthesized_segments: List[Dict] = []
        if not self._stage_done("tts"):
            try:
                from tqdm import tqdm
                bar = tqdm(total=len(segments),
                           desc="Synthesizing voices", unit="seg")
            except Exception:
                bar = None

            prev = self.ckpt.get_progress(
                "tts") or {"current": 0} if self.ckpt else {"current": 0}
            start_index = int(prev.get("current", 0))

            # prepare prosody mapper
            try:
                prosody_mapper = ProsodyMapper() if ProsodyMapper else None
            except Exception:
                prosody_mapper = None

            # parse script into tts items if possible
            try:
                from diadub.script.script_parser import parse_for_tts
                tts_items = parse_for_tts(str(script_path))
            except Exception:
                # fallback: create items from segments
                tts_items = []
                for i, seg in enumerate(segments):
                    tts_items.append({
                        "index": i,
                        "text": seg.get("translated", seg.get("text", "")),
                        "start": seg.get("start", 0.0),
                        "end": seg.get("end", seg.get("start", 0.0) + 0.5),
                        "duration": seg.get("end", seg.get("start", 0.0)) - seg.get("start", 0.0),
                        "prosody": {"rate": 1.0, "pitch_semitones": 0.0, "gain_db": 0.0},
                        "voice": None,
                        "speaker": seg.get("speaker"),
                        "emotion": None,
                    })

            total_segments = len(tts_items)

            # Temporary dir for alignment/delays (pipelinev2 style)
            tmp_root = out_dir / "tmp_diadub"
            if tmp_root.exists():
                shutil.rmtree(tmp_root)
            tmp_root.mkdir(parents=True, exist_ok=True)

            for i in range(start_index, total_segments):
                item = tts_items[i]
                text_to_speak = item.get("text", "").strip()
                if not text_to_speak:
                    if bar:
                        bar.update(1)
                    try:
                        if self.ckpt:
                            self.ckpt.set_progress(
                                "tts", i + 1, total_segments)
                    except Exception:
                        pass
                    continue

                # deterministic filenames
                fname_raw = f"{base_stem}_seg_{i:04d}_tts_raw.wav"
                fname_pitch = f"{base_stem}_seg_{i:04d}_tts_pitch.wav"
                fname_adjusted = f"{base_stem}_seg_{i:04d}_adjusted.wav"
                wav_path_raw = self.temp_dir / fname_raw
                wav_path_pitch = self.temp_dir / fname_pitch
                wav_path_adjusted = self.temp_dir / fname_adjusted

                # choose clean segment (prefer cached separated vocals)
                seg_orig = self.temp_dir / f"{base_stem}_seg_{i:04d}_orig.wav"
                clean_seg_path = seg_orig
                cache_entry = self.separation_cache.get(str(seg_orig))
                if cache_entry and cache_entry.get("vocals") and Path(cache_entry["vocals"]).exists():
                    # verify checksum matches current file
                    cur_chk = compute_sha256(str(seg_orig))
                    if cache_entry.get("checksum") == cur_chk:
                        clean_seg_path = Path(cache_entry["vocals"])
                        log.debug(
                            f"[seg {i}] using cached separated vocals: {clean_seg_path}")
                    else:
                        log.debug(
                            f"[seg {i}] checksum mismatch; will re-separate later.")
                else:
                    # fallback: try to run separation synchronously for this single segment
                    try:
                        sep_out = out_dir / "seg_separated" / f"seg_{i:04d}"
                        sep_out.mkdir(parents=True, exist_ok=True)
                        sep_info = demucs_separate(
                            str(seg_orig), str(sep_out), stems="vocals")
                        if sep_info and sep_info.get("vocals"):
                            clean_seg_path = Path(sep_info["vocals"])
                            self.separation_cache[str(seg_orig)] = sep_info
                    except Exception as e:
                        log.warning(
                            f"[seg {i}] per-seg separation failed: {e}; using original segment")

                # analyze for prosody
                audio_meta = None
                try:
                    from diadub.audio_analysis.audio_features import analyze_segment
                    audio_meta = analyze_segment(segment_audio_tmp)
                except Exception:
                    audio_meta = {"duration": item.get(
                        "duration", item.get("end", 0.0) - item.get("start", 0.0))}

                # prosody mapping
                prosody_params = {}
                if prosody_mapper is not None:
                    try:
                        prosody_params = prosody_mapper.map(audio_meta)
                        script_prosody = item.get("prosody", {})
                        for k in ("rate", "pitch_semitones", "gain_db"):
                            if k in script_prosody and script_prosody[k] is not None:
                                prosody_params[k] = float(script_prosody[k])
                    except Exception as e:
                        log.debug("Prosody mapping failed: %s", e)
                        prosody_params = {}

                        # call TTS via TTSEngineManager
                out = None
                tts_words = []
                try:
                    # --------- TTS via QueueManager + cleaning + alignment + visemes ----------
                    # prepare tts call values
                    text_to_synthesize = item.get("text_polished") or item.get(
                        "translated") or item.get("text")
                    tts_hint = item.get("tts_hint") or {
                        "voice": item.get("voice") or item.get("speaker")}
                    # wrapper for actual synth call (worker executes on GPU)

                    def _tts_worker(text, tts_hint, tmp_root_local):
                        # Use the TTSEngineManager instance (self.tts_engine)
                        # TTSEngineManager should expose a method synthesize_text(text, hint, out_dir) returning path or bytes
                        return self.tts_engine.synthesize(text=text, hint=tts_hint, out_dir=str(tmp_root_local))

                    try:
                        if self.queue_mgr:
                            fut = self.queue_mgr.submit_gpu(
                                _tts_worker, text_to_synthesize, tts_hint, tmp_root, gpu_index=0)
                            tts_res = fut.result()
                        else:
                            tts_res = _tts_worker(
                                text_to_synthesize, tts_hint, tmp_root)
                    except Exception as e:
                        log.exception("TTS job failed for seg %d: %s", i, e)
                        # mark progress and continue
                        if bar:
                            bar.update(1)
                        if self.ckpt:
                            self.ckpt.set_progress("tts", i+1, total_segments)
                        continue

                    # normalize result into wav_path_raw
                    try:
                        # tts_res can be path, bytes, numpy array -- handle typical cases
                        if isinstance(tts_res, (str, Path)):
                            ttp = Path(tts_res)
                            if ttp.exists():
                                shutil.copy2(str(ttp), str(wav_path_raw))
                            else:
                                raise RuntimeError(
                                    "TTS returned path but file missing: " + str(ttp))
                        elif isinstance(tts_res, bytes):
                            wav_path_raw.write_bytes(tts_res)
                        else:
                            try:
                                import numpy as _np
                                import soundfile as sf
                                if hasattr(tts_res, "numpy"):
                                    arr = tts_res.numpy()
                                    sf.write(str(wav_path_raw), arr, 48000)
                                elif isinstance(tts_res, _np.ndarray):
                                    sf.write(str(wav_path_raw), tts_res, 48000)
                                else:
                                    raise RuntimeError(
                                        "Unsupported TTS return type")
                            except Exception as e:
                                raise

                    except Exception as e:
                        log.exception(
                            "Failed to normalize TTS output for seg %d: %s", i, e)
                        if bar:
                            bar.update(1)
                        if self.ckpt:
                            self.ckpt.set_progress("tts", i+1, total_segments)
                        continue

                    # cleaning: normalize loudness & trim
                    base_for_alignment = str(wav_path_raw)
                    if clean_tts_file:
                        try:
                            _cleaned = self.temp_dir / \
                                f"{base_stem}_seg_{i:04d}_tts_clean.wav"
                            clean_tts_file(str(wav_path_raw), str(
                                _cleaned), target_lufs=-23.0)
                            base_for_alignment = str(_cleaned)
                        except Exception as e:
                            log.warning(
                                "clean_tts_file failed for seg %d: %s -- using raw.", i, e)

                    # forced alignment (prefer MFA)
                    word_entries = []
                    try:
                        # prefer MFA
                        if run_mfa_align:
                            try:
                                # write transcript one-line
                                tmp_lab = tmp_root / \
                                    f"{base_stem}_seg_{i:04d}.lab"
                                tmp_lab.write_text(
                                    text_to_synthesize.strip(), encoding="utf-8")
                                mfa_out = tmp_root / "mfa" / f"seg_{i:04d}"
                                mfa_out.mkdir(parents=True, exist_ok=True)
                                word_entries = run_mfa_align(
                                    base_for_alignment, str(tmp_lab), str(mfa_out)) or []
                            except Exception as me:
                                log.debug("MFA fail seg %d: %s", i, me)
                                word_entries = []

                        # final fallback: naive per-word distribution
                        if not word_entries:
                            # get tokens
                            words = text_to_synthesize.split()
                            dur = max(0.001, self._get_duration(
                                base_for_alignment))
                            per = dur / max(1, len(words))
                            t = 0.0
                            for w in words:
                                word_entries.append({"word": w, "start": round(t, 6), "end": round(
                                    t+per, 6), "duration": round(per, 6), "phones": []})
                                t += per

                    except Exception as e:
                        log.debug("Alignment general fail seg %d: %s", i, e)
                        word_entries = []

                    # viseme mapping
                    try:
                        if generate_viseme_timing and word_entries:
                            vdat = generate_viseme_timing(word_entries)
                            line_ph = vdat.get("phonemes")
                            line_vs = vdat.get("visemes")
                            # attach to script line (existing s_obj)
                            # you already later write aligned_tts_path -> do same adding visemes to script_json
                            ln["phonemes"] = line_ph
                            ln["visemes"] = line_vs
                    except Exception as e:
                        log.debug("Viseme mapping failed seg %d: %s", i, e)

                    # finalize synthesized_segments as before
                    synthesized_segments.append({
                        "wav": base_for_alignment,
                        "start": item.get("start"),
                        "end": item.get("end"),
                        "speaker": item.get("speaker"),
                        "index": item.get("index"),
                        "words": word_entries,
                    })
                    # progress
                    if self.ckpt:
                        self.ckpt.set_artifact(
                            "synthesized_segments", synthesized_segments)
                        self.ckpt.set_progress("tts", i+1, total_segments)
                    if bar:
                        bar.update(1)
            if self.ckpt:
                self.ckpt.set_artifact(
                    "synthesized_segments", synthesized_segments)
                self.ckpt.set_stage_done(
                    "tts", {"synthesized_segments": synthesized_segments})
        else:
            synthesized_segments = self.ckpt.get_artifact(
                "synthesized_segments", []) if self.ckpt else []
        return synthesized_segments

    def _run_assemble_dialogue(self, out_dir: Path, base_stem: str, audio_path: Path, synthesized_segments: List[Dict]) -> Path:
        dialogue_path = out_dir / f"{base_stem}_dialogue.wav"
        if not self._stage_done("assemble_dialogue"):
            if synthesized_segments and AudioSegment is not None:
                try:
                    self._assemble_and_export_dialogue(
                        str(audio_path), synthesized_segments, str(dialogue_path))
                    if self.ckpt:
                        self.ckpt.set_artifact(
                            "dialogue_path", str(dialogue_path))
                except Exception as e:
                    log.exception("Assemble dialogue failed: %s", e)
            else:
                log.info("No synthesized segments to assemble.")
            if self.ckpt:
                self.ckpt.set_stage_done("assemble_dialogue", {
                                         "dialogue_path": str(dialogue_path)})
        else:
            dialogue_path = Path(self.ckpt.get_artifact(
                "dialogue_path", str(dialogue_path))) if self.ckpt else dialogue_path
        return dialogue_path

    def _run_mix(self, out_dir: Path, base_stem: str, audio_path: Path, dialogue_path: Path, synthesized_segments: List[Dict]) -> Path:
        mixed_path = out_dir / f"{base_stem}_dialogue_mixed.wav"
        if not self._stage_done("mix"):
            try:
                from diadub.mixing.mixer import Mixer
                mixer = Mixer()
                if dialogue_path.exists():
                    mixer.mix(str(audio_path), str(
                        dialogue_path), str(mixed_path))
                    if self.ckpt:
                        self.ckpt.set_artifact("mixed_path", str(mixed_path))
                else:
                    log.info("Dialogue track missing; skipping mixer.")
            except Exception as exc:
                log.warning(
                    "Mixer not available or failed: %s. Falling back to ffmpeg mix.", exc)
                try:
                    tmp_root = out_dir / "tmp_diadub"
                    tmp_root.mkdir(parents=True, exist_ok=True)

                    delayed_layers = []
                    for seg in synthesized_segments:
                        try:
                            seg_start = float(seg.get("start", 0.0))
                            delay_ms = int(round(seg_start * 1000.0))
                            delayed_out = tmp_root / \
                                f"{base_stem}_seg_{int(seg.get('index', 0)):04d}_delayed.wav"
                            temp_fade = tmp_root / \
                                f"{base_stem}_seg_{int(seg.get('index', 0)):04d}_faded.wav"
                            _ffmpeg_fade_in_out(
                                seg.get("wav"), str(temp_fade), fade_ms=8)
                            _ffmpeg_delay(str(temp_fade), str(
                                delayed_out), delay_ms)
                            delayed_layers.append(str(delayed_out))
                        except Exception as e:
                            log.debug(
                                "Failed to create delayed layer for seg %s: %s", seg.get('index'), e)

                    if not delayed_layers:
                        raise RuntimeError(
                            "No delayed layers created for fallback mix")

                    tts_mix = tmp_root / f"{base_stem}_tts_mix.wav"
                    log.info("Mixing %d delayed layers into single tts mix", len(
                        delayed_layers))
                    _ffmpeg_mix(delayed_layers, str(tts_mix))

                    ducked_base = tmp_root / f"{base_stem}_base_ducked.wav"
                    try:
                        _ffmpeg_sidechain_duck(
                            str(audio_path), str(tts_mix), str(ducked_base))
                        log.info("Applied sidechain ducking to base audio")
                    except Exception as e:
                        log.warning("Sidechain duck failed or unsupported: %s. Falling back to static reduction (gain=%sdB)", e, float(
                            -6.0))
                        _ffmpeg_reduce_base_gain(str(audio_path), str(
                            ducked_base), gain_db=float(-6.0))

                    _ffmpeg_mix([str(ducked_base), str(tts_mix)],
                                str(mixed_path))
                    if self.ckpt:
                        self.ckpt.set_artifact("mixed_path", str(mixed_path))
                except Exception as e:
                    log.exception("Fallback ffmpeg mixing failed: %s", e)
            if self.ckpt:
                self.ckpt.set_stage_done(
                    "mix", {"mixed_path": str(mixed_path)})
        else:
            mixed_path = Path(self.ckpt.get_artifact(
                "mixed_path", str(mixed_path))) if self.ckpt else mixed_path
        return mixed_path

    def _run_mux(self, out_dir: Path, base_stem: str, video_path: str, mixed_path: Path, synthesized_segments: List[Dict]) -> Path:
        dub_video_path = out_dir / f"{base_stem}_dub.mp4"
        if not self._stage_done("mux"):
            try:
                audio_to_mux = mixed_path if Path(
                    mixed_path).exists() else None
                if audio_to_mux and video_path:
                    # Apply lipsync
                    self._apply_lipsync(
                        str(video_path), synthesized_segments, str(dub_video_path))

                    replace_audio(str(dub_video_path), str(
                        audio_to_mux), output_path=str(dub_video_path))
                    if self.ckpt:
                        self.ckpt.set_artifact(
                            "dub_video", str(dub_video_path))
                else:
                    log.info("No assembled audio available to mux; skipping mux.")
            except Exception as exc:
                log.exception("Failed to mux: %s", exc)
            if self.ckpt:
                self.ckpt.set_stage_done(
                    "mux", {"dub_video": str(dub_video_path)})
        else:
            dub_video_path = Path(self.ckpt.get_artifact(
                "dub_video", str(dub_video_path))) if self.ckpt else dub_video_path
        return dub_video_path

    def run(
        self,
        out_dir: str = "output",
        srt_path: Optional[str] = None,
        video_path: Optional[str] = None,
    ) -> Dict:
        """
        Run the full pipeline. Returns a dict with artifact paths or error.
        - If video_path is provided, will extract audio and mux final video.
        - If input_audio provided (wav), will use that instead.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # input selection: prefer input_audio; if not, use video_path -> extract audio
        if video_path:
            video_path_obj = Path(video_path)
            if not video_path_obj.exists():
                raise FileNotFoundError(f"Video not found: {video_path}")
            base_stem = video_path_obj.stem
        else:
            raise ValueError(
                "video_path must be provided")

        # Initialize TTS Engine Manager
        self.tts_engine = TTSEngineManager()

        # progress indicator preparation
        try:
            from tqdm import tqdm
            progress = tqdm(total=len(self.STAGES), desc="Pipeline")
            for s in self.STAGES:
                if self._stage_done(s):
                    progress.update(1)
        except Exception:
            progress = None

        result = {"video": str(video_path)
                  if video_path else None, "artifacts": {}}

        # Artifact integrity check before resume
        if self.resume and self.ckpt is not None:
            ok = self.ckpt.verify_artifacts(
                deep=False, log_path=str(self.temp_dir / "integrity.log"))
            if not ok:
                log.warning(
                    "Integrity check failed: some artifacts missing or small. Re-running dependent stages.")
                failed_keys = []
                for k, v in self.ckpt.to_dict().get("artifacts", {}).items():
                    if isinstance(v, str):
                        p = Path(v)
                        if not p.exists() or p.stat().st_size < 1024:
                            failed_keys.append(k)
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, str):
                                p = Path(item)
                                if not p.exists() or p.stat().st_size < 1024:
                                    failed_keys.append(k)
                failed_keys = list(set(failed_keys))
                to_invalidate = set()
                for fk in failed_keys:
                    key_to_stage = {
                        "audio_path": "extract_audio",
                        "segments": "asr",
                        "diarize_segments": "diarization",
                        "translated_segments": "translation",
                        "synthesized_segments": "tts",
                        "dialogue_path": "assemble_dialogue",
                        "mixed_path": "mix",
                        "dub_video": "mux",
                        "translated_srt": "translation",
                        "script_path": "generate_script",
                    }
                    stage = key_to_stage.get(fk)
                    if stage:
                        to_invalidate.add(stage)
                        to_invalidate.update(self.get_dependent_stages(stage))
                    else:
                        to_invalidate.update(self.STAGES)
                for stage in to_invalidate:
                    log.info("Invalidating stage due to dependency: %s", stage)
                    self.ckpt.invalidate_stage(stage)

        def _parallel_separate_worker(args):
            idx, seg_path, out_root, checksum = args
            try:
                sep_out = out_root / f"seg_{idx:04d}"
                sep_out.mkdir(parents=True, exist_ok=True)
                sep_info = demucs_separate(
                    str(seg_path), str(sep_out), stems="vocals")
                if sep_info:
                    sep_info["checksum"] = checksum
                return (idx, sep_info, None)
            except Exception as e:
                return (idx, None, str(e))

        # STAGES
        audio_path = self._run_extract_audio(video_path, out_dir, base_stem)
        if progress:
            progress.update(1)

        segments = self._run_asr(audio_path, srt_path)
        if progress:
            progress.update(1)

        segments = self._run_diarization(audio_path, segments)
        if progress:
            progress.update(1)

        segments = self._run_translation(segments)
        if progress:
            progress.update(1)

        translated_srt_path = out_dir / f"{base_stem}_translated.srt"
        self._write_srt(segments, translated_srt_path)
        if self.ckpt:
            self.ckpt.set_artifact("translated_srt", str(translated_srt_path))

        script_path = self._run_generate_script(
            out_dir, base_stem, audio_path, translated_srt_path, segments, dry_run, None, "en")
        if progress:
            progress.update(1)

        synthesized_segments = self._run_tts(
            out_dir, base_stem, script_path, segments)
        if progress:
            progress.update(1)

        dialogue_path = self._run_assemble_dialogue(
            out_dir, base_stem, audio_path, synthesized_segments)
        if progress:
            progress.update(1)

        mixed_path = self._run_mix(
            out_dir, base_stem, audio_path, dialogue_path, synthesized_segments)
        if progress:
            progress.update(1)

        dub_video_path = self._run_mux(
            out_dir, base_stem, video_path, mixed_path, synthesized_segments)
        if progress:
            progress.update(1)

        if progress:
            progress.close()

        result["artifacts"] = self.ckpt.to_dict().get(
            "artifacts", {})
        return result

    def _apply_lipsync(self, video_path: str, synthesized_segments: List[Dict], output_path: str):
        try:
            from diadub.lipsync.viseme_mapper import apply_visemes_to_video
            apply_visemes_to_video(
                video_path, synthesized_segments, output_path)
        except Exception as e:
            log.warning(
                "Failed to apply lipsync: %s. Copying original video.", e)
            shutil.copy2(video_path, output_path)

    # ---------------------
    # Helper methods below
    # ---------------------
    def _extract_segment(self, full_audio: str, start_s: float, end_s: float, out_wav: str):
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(full_audio),
            "-ss",
            f"{start_s:.3f}",
            "-to",
            f"{end_s:.3f}",
            "-ar",
            "48000",
            "-ac",
            "1",
            "-vn",
            str(out_wav),
        ]
        subprocess.run(cmd, check=True)

    def _parse_srt(self, srt_path: str) -> List[Dict]:
        srt_path = Path(srt_path)
        out = []
        if not srt_path.exists():
            return out
        text = srt_path.read_text(encoding="utf-8")
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        for p in parts:
            lines = p.splitlines()
            # skip indexes
            if len(lines) >= 2:
                ts = lines[1]
                if "-->" not in ts:
                    continue
                start_txt, end_txt = [t.strip() for t in ts.split("-->")]

                def parse_time_text(s):
                    s = s.replace(",", ".").strip()
                    parts = s.split(":")
                    if len(parts) == 3:
                        h, m, sec = parts
                        return int(h) * 3600 + int(m) * 60 + float(sec)
                    elif len(parts) == 2:
                        m, sec = parts
                        return int(m) * 60 + float(sec)
                    else:
                        try:
                            return float(s)
                        except:
                            return 0.0

                try:
                    start = parse_time_text(start_txt)
                    end = parse_time_text(end_txt)
                except Exception:
                    start = 0.0
                    end = 0.0
                body = " ".join(lines[2:]) if len(lines) > 2 else ""
                out.append({"text": body, "start": start,
                           "end": end, "confidence": None})
        return out

    def _attach_speakers(self, asr_segs: List[Dict], diarize_segs: List[Dict]):
        for seg in asr_segs:
            best = None
            best_overlap = 0.0
            s_start, s_end = seg["start"], seg["end"]
            for d in diarize_segs:
                d_start, d_end = d["start"], d["end"]
                overlap_start = max(s_start, d_start)
                overlap_end = min(s_end, d_end)
                overlap = max(0.0, overlap_end - overlap_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best = d
            if best and best_overlap > 0.0:
                seg["speaker"] = best.get("speaker")
            else:
                seg["speaker"] = None

    def _write_srt(self, segments: List[Dict], out_path: Path):
        srt_lines = []
        for i, seg in enumerate(segments, start=1):
            start = seconds_to_srt_time(seg["start"])
            end = seconds_to_srt_time(seg["end"])
            text = seg.get("translated") or seg.get("text") or ""
            if seg.get("speaker"):
                text = f"[{seg['speaker']}] {text}"
            srt_lines.append(f"{i}\n{start} --> {end}\n{text}\n\n")
        out_path.write_text("".join(srt_lines), encoding="utf-8")

    def _assemble_and_export_dialogue(self, original_audio_path: str, synth_segments: List[Dict], out_wav_path: str):
        if AudioSegment is None:
            raise RuntimeError("pydub not available")

        original = AudioSegment.from_file(original_audio_path)
        total_len_ms = len(original)
        base = AudioSegment.silent(duration=total_len_ms)

        for seg in synth_segments:
            try:
                wav = AudioSegment.from_file(seg["wav"])
            except Exception as e:
                log.warning(
                    "Failed to load synthesized segment %s: %s", seg.get("wav"), e)
                continue
            start_ms = int(round(seg["start"] * 1000))
            overlay = wav
            seg_duration_ms = int(round((seg["end"] - seg["start"]) * 1000))
            if len(overlay) > seg_duration_ms + 500:
                log.debug("TTS audio longer than original: %s (tts %d ms vs seg %d ms)",
                          seg["wav"], len(overlay), seg_duration_ms)
            base = base.overlay(overlay, position=start_ms)
        base = base.set_frame_rate(48000).set_channels(2)
        base.export(out_wav_path, format="wav")

    def _get_duration(self, wav_path: str) -> float:
        try:
            import soundfile as sf
            info = sf.info(wav_path)
            return info.frames / info.samplerate
        except Exception:
            return 0.0
