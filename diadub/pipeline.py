from tqdm import tqdm
import torch
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
from typing import List, Dict, Optional, Any, Callable
from diadub.models.tts_engine_manager import TTSEngineManager
from diadub import utils
from diadub.progress.progress_manager import ProgressManager
from diadub.lipsync.wav2lip_queue_worker import Wav2LipQueueManager
# --- GUI & external runner integration imports (ADD HERE) ---
import time
import json
from typing import Callable, Any, Dict, Optional

# Wav2Lip wrapper used by optional lip-sync stage
try:
    from diadub.models.wav2lip.wav2lip_wrapper import run_wav2lip
except Exception:
    run_wav2lip = None

pm = ProgressManager.get()
log = logging.getLogger("diadub.pipeline")
logging.basicConfig(level=logging.INFO)

# --- Progress emit helpers used by run_full_pipeline (ADD HERE if not present) ---


def _emit_progress(cb: Optional[Callable[[str, float], None]], stage: str, pct: float):
    """Safely call a progress callback (stage name, pct 0.0-1.0 or 0-100)."""
    if cb is None:
        return
    try:
        # normalise pct to 0.0..1.0 if user passed 0-100
        if pct is None:
            cb(stage, None)
            return
        p = float(pct)
        if p > 1.0:
            p = max(0.0, min(100.0, p)) / 100.0
        cb(stage, p)
    except Exception:
        # don't allow GUI failures to crash the pipeline
        try:
            cb(stage, None)
        except Exception:
            pass


def _emit_eta(cb: Optional[Callable[[str, float], None]], stage: str, eta_seconds: Optional[float]):
    """Safely call an ETA callback in seconds."""
    if cb is None:
        return
    try:
        cb(stage, float(eta_seconds) if eta_seconds is not None else None)
    except Exception:
        try:
            cb(stage, None)
        except Exception:
            pass

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
            raise RuntimeError(f"Command failed: {" ".join(cmd)}")
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
    from tools.queue_manager import QueueManager
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
        self.models_config = load_models_config(model_config)
        self.device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
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
    # Backend initialization
    # -----------------------------------------------------------------
    def _ensure_backends(self):
        # ASR
        if "asr" in self.models_config and "default" in self.models_config["asr"]:
            asr_config = self.models_config["asr"]["default"]
            backend = asr_config.get("backend")
            if backend == "whisperx":
                try:
                    from diadub.asr.whisper_asr import WhisperASR
                    self.asr = WhisperASR(model_id=asr_config.get("model_id"), device=self.device)
                    log.info("ASR backend available.")
                except Exception as exc:
                    log.warning("ASR backend not available: %s", exc)
        
        # Diarization
        if "diarization" in self.models_config and "default" in self.models_config["diarization"]:
            diar_config = self.models_config["diarization"]["default"]
            backend = diar_config.get("backend")
            if backend == "pyannote":
                try:
                    from diadub.diarization.pyannote_diarizer import PyannoteDiarizer
                    self.diarizer = PyannoteDiarizer(model_id=diar_config.get("model_id"), device=self.device)
                    log.info("Diarization backend available.")
                except Exception as exc:
                    log.warning("Diarization backend not available: %s", exc)

        # Translation
        if "translation" in self.models_config and "default" in self.models_config["translation"]:
            trans_config = self.models_config["translation"]["default"]
            backend = trans_config.get("backend")
            if backend == "hf_translator":
                try:
                    from diadub.translation.hf_translator import HFTranslator
                    self.translator = HFTranslator(model_id=trans_config.get("model_id"), device=self.device)
                    log.info("Translation backend available.")
                except Exception as exc:
                    log.warning("Translation backend not available: %s", exc)

        # Emotion
        if "asr_wav2vec_emotion" in self.models_config:
            try:
                from diadub.models.stt.wav2vec_emotion import Wav2VecEmotionSTT
                self.emotion_model = Wav2VecEmotionSTT(model_id=self.models_config["asr_wav2vec_emotion"].get("model_id"), device=self.device)
                log.info("Emotion model available.")
            except Exception as e:
                self.emotion_model = None
                log.warning("Could not load emotion model: %s", e)

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
    def run(
        self,
        input_audio: Optional[str] = None,
        out_dir: str = "output",
        base_name: Optional[str] = None,
        srt_path: Optional[str] = None,
        language: str = "en",
        keep_temp: bool = False,
        video_path: Optional[str] = None,
        dry_run: bool = False,
        groq_use: bool = False,
        groq_api_key: Optional[str] = None,
        duck_amount_db: float = -6.0,
    ) -> Dict:
        """
        Run the full pipeline. Returns a dict with artifact paths or error.
        - If video_path is provided, will extract audio and mux final video.
        - If input_audio provided (wav), will use that instead.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # input selection: prefer input_audio; if not, use video_path -> extract audio
        if input_audio:
            audio_path = Path(input_audio)
            if not audio_path.exists():
                raise FileNotFoundError(
                    f"Input audio not found: {input_audio}")
            base_stem = base_name if base_name else audio_path.stem
            video_path_obj = Path(video_path) if video_path else None
        elif video_path:
            video_path_obj = Path(video_path)
            if not video_path_obj.exists():
                raise FileNotFoundError(f"Video not found: {video_path}")
            base_stem = base_name if base_name else video_path_obj.stem
            audio_path = out_dir / f"{base_stem}.wav"
        else:
            raise ValueError(
                "Either input_audio or video_path must be provided")

        self._ensure_backends()
        # Initialize TTS Engine Manager
        self.tts_engine = TTSEngineManager(orpheus_path="diadub/models/orpheus-3b-0.1-ft-UD-Q8_K_XL.gguf")

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

        pm.emit("pipeline", "Starting pipeline", 0.01)

        # -------------------------
        # STAGE 1: extract_audio
        # -------------------------
        if not self._stage_done("extract_audio"):
            if video_path:
                audio_out = out_dir / f"{base_stem}.wav"
                if not audio_out.exists():
                    log.info("Extracting audio from %s -> %s",
                             video_path, audio_out)
                    extract_audio(str(video_path), str(audio_out))
                else:
                    log.info("Using cached audio %s", audio_out)
                audio_path = audio_out
            else:
                # audio_path already set to input_audio
                log.info("Using provided audio %s", audio_path)
            if self.ckpt:
                self.ckpt.set_artifact("audio_path", str(audio_path))
                self.ckpt.set_stage_done(
                    "extract_audio", {"audio_path": str(audio_path)})
            if progress:
                progress.update(1)
        else:
            audio_path = Path(self.ckpt.get_artifact(
                "audio_path")) if self.ckpt else audio_path
            log.info("Resuming: using audio from checkpoint: %s", audio_path)
            if progress:
                progress.update(0)

        pm.emit("extract_audio", "Audio extracted",
                0.10, {"audio_path": str(audio_path)})

        # -------------------------
        # STAGE 2: ASR
        # -------------------------
        segments: List[Dict] = []
        asr_result = None
        if not self._stage_done("asr"):
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
                            asr_result = asr_out
                        elif isinstance(asr_out, dict) and "text" in asr_out:
                            segments = [{"text": asr_out.get("text", ""), "start": 0.0, "end": float(
                                self._get_duration(str(audio_path))), "confidence": None}]
                            asr_result = asr_out
                        else:
                            segments = []
                            asr_result = asr_out
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
            if progress:
                progress.update(1)
        else:
            segments = self.ckpt.get_artifact(
                "segments", []) if self.ckpt else []
            log.info("Resuming: loaded %d segments from checkpoint.",
                     len(segments))
            if progress:
                progress.update(0)

        pm.emit("asr", "ASR complete", 0.2, {"num_segments": len(segments)})
        if not segments:
            log.warning("No segments available after ASR; aborting.")
            if progress:
                progress.close()
            return {"error": "no_segments"}

        # -------------------------
        # STAGE 3: Diarization
        # -------------------------
        diarize_segs: List[Dict] = []
        if not self._stage_done("diarization"):
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
            if progress:
                progress.update(1)
        else:
            diarize_segs = self.ckpt.get_artifact(
                "diarize_segments", []) if self.ckpt else []
            self._attach_speakers(segments, diarize_segs)
            if progress:
                progress.update(0)
        pm.emit("diarization", "Diarization complete", 0.3, {
                "num_diarization_segments": len(diarize_segs)})

        # -------------------------
        # STAGE 4: Translation
        # -------------------------
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
            if progress:
                progress.update(1)
        else:
            segments = self.ckpt.get_artifact(
                "translated_segments", segments) if self.ckpt else segments
            if progress:
                progress.update(0)

        pm.emit("translation", "Translation complete", 0.4)
        # Write translated SRT artifact
        translated_srt_path = out_dir / f"{base_stem}_translated.srt"
        self._write_srt(segments, translated_srt_path)
        if self.ckpt:
            self.ckpt.set_artifact("translated_srt", str(translated_srt_path))

        # -------------------------
        # STAGE 5: generate_script (use script_generator if available)
        # -------------------------
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
                        srt_path=(str(translated_srt_path)
                                  if translated_srt_path.exists() else None),
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
            if progress:
                progress.update(1)
        else:
            script_path = Path(self.ckpt.get_artifact(
                "script_path", str(script_path))) if self.ckpt else script_path
            log.info("Resuming: script at %s", script_path)
            if progress:
                progress.update(0)
        pm.emit("generate_script", "Script generated",
                0.5, {"script_path": str(script_path)})

        # -------------------------
        # STAGE 6: TTS (synthesize -> align -> collect synthesized_segments)
        # -------------------------
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

            # -----------------------------
            # Parallel pre-separation (per-segment)
            # -----------------------------
            if self.use_per_segment_separation and demucs_separate:
                sep_dir = out_dir / "seg_separated"
                sep_dir.mkdir(parents=True, exist_ok=True)
                to_process = []
                for idx, seg in enumerate(segments):
                    seg_orig = self.temp_dir /
                        f"{base_stem}_seg_{idx:04d}_orig.wav"
                    # ensure seg exists or create one now
                    if not seg_orig.exists():
                        try:
                            self._extract_segment(str(audio_path), seg.get("start", 0.0), seg.get(
                                "end", seg.get("start", 0.0) + 0.5), str(seg_orig))
                        except Exception:
                            continue
                    checksum = compute_sha256(str(seg_orig))
                    cache_entry = self.separation_cache.get(str(seg_orig))
                    if cache_entry and cache_entry.get("checksum") == checksum:
                        continue
                    to_process.append((idx, seg_orig, sep_dir, checksum))

                if to_process:
                    workers = min(8, max(1, os.cpu_count()//2))
                    log.info("Parallel separation: scheduling %d segments (workers=%d)", len(
                        to_process), workers)
                    futures = []
                    with ThreadPoolExecutor(max_workers=workers) as ex:
                        for task in to_process:
                            futures.append(
                                ex.submit(_parallel_separate_worker, task))
                        for fut in tqdm(as_completed(futures), total=len(futures), desc="Demucs separation"):
                            idx, sep_info, err = fut.result()
                            seg_orig = self.temp_dir /
                                f"{base_stem}_seg_{idx:04d}_orig.wav"
                            if sep_info:
                                self.separation_cache[str(seg_orig)] = sep_info
                                log.info("[seg %d] separated and cached", idx)
                            else:
                                log.warning(
                                    "[seg %d] separation failed: %s", idx, err)

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
                            self.ckpt.set_progress("tts", i + 1, total_segments)
                    except Exception:
                        pass
                    continue

                synthesis_successful = False
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        # Start of logic for one segment
                        fname_raw = f"{base_stem}_seg_{i:04d}_tts_raw.wav"
                        wav_path_raw = self.temp_dir / fname_raw
                        fname_adjusted = f"{base_stem}_seg_{i:04d}_adjusted.wav"
                        wav_path_adjusted = self.temp_dir / fname_adjusted

                        seg_orig = self.temp_dir / f"{base_stem}_seg_{i:04d}_orig.wav"
                        clean_seg_path = seg_orig
                        cache_entry = self.separation_cache.get(str(seg_orig))
                        if cache_entry and cache_entry.get("vocals") and Path(cache_entry["vocals"]).exists():
                            cur_chk = compute_sha256(str(seg_orig))
                            if cache_entry.get("checksum") == cur_chk:
                                clean_seg_path = Path(cache_entry["vocals"])
                        
                        text_to_synthesize = item.get("text_polished") or item.get("translated") or item.get("text")
                        tts_hint = item.get("tts_hint") or {"voice": item.get("voice") or item.get("speaker")}

                        def _tts_worker(text, tts_hint, tmp_root_local):
                            return self.tts_engine.synthesize_text(text=text, hint=tts_hint, out_dir=str(tmp_root_local))

                        if self.queue_mgr:
                            fut = self.queue_mgr.submit_gpu(_tts_worker, text_to_synthesize, tts_hint, tmp_root, gpu_index=0)
                            tts_res = fut.result()
                        else:
                            tts_res = _tts_worker(text_to_synthesize, tts_hint, tmp_root)
                        
                        shutil.copy2(str(tts_res), str(wav_path_raw))
                        base_for_alignment = str(wav_path_raw)

                        if clean_tts_file:
                            try:
                                _cleaned = self.temp_dir / f"{base_stem}_seg_{i:04d}_tts_clean.wav"
                                clean_tts_file(str(wav_path_raw), str(_cleaned), target_lufs=-23.0)
                                base_for_alignment = str(_cleaned)
                            except Exception as e:
                                log.warning(f"clean_tts_file failed for seg {i}: {e} -- using raw.")

                        word_entries = []
                        if run_mfa_align:
                            try:
                                tmp_lab = tmp_root / f"{base_stem}_seg_{i:04d}.lab"
                                tmp_lab.write_text(text_to_synthesize.strip(), encoding="utf-8")
                                mfa_out = tmp_root / "mfa" / f"seg_{i:04d}"
                                mfa_out.mkdir(parents=True, exist_ok=True)
                                word_entries = run_mfa_align(base_for_alignment, str(tmp_lab), str(mfa_out)) or []
                            except Exception as me:
                                log.debug(f"MFA fail seg {i}: {me}")
                        
                        if not word_entries and align_tts_to_target:
                            tts_words = item.get("tts_words") or []
                            if tts_words:
                                try:
                                    word_entries = align_tts_to_target(
                                        tts_audio_path=base_for_alignment,
                                        tts_words=tts_words,
                                        target_words=[w["word"] for w in tts_words],
                                        out_path=str(wav_path_adjusted),
                                        options={"max_stretch": 1.5, "min_stretch": 0.7, "crossfade_ms": 12},
                                    ) or []
                                    base_for_alignment = str(wav_path_adjusted)
                                except Exception as ae:
                                    log.debug(f"align_tts_to_target fallback failed seg {i}: {ae}")

                        if generate_viseme_timing and word_entries:
                            try:
                                vdat = generate_viseme_timing(word_entries)
                                item["phonemes"] = vdat.get("phonemes")
                                item["visemes"] = vdat.get("visemes")
                            except Exception as e:
                                log.debug(f"Viseme mapping failed seg {i}: {e}")

                        synthesized_segments.append({
                            "wav": base_for_alignment, "start": item.get("start"), "end": item.get("end"),
                            "speaker": item.get("speaker"), "index": item.get("index"), "words": word_entries,
                        })

                        synthesis_successful = True
                        log.info(f"Segment {i} synthesis succeeded on attempt {attempt + 1}.")
                        break
                    except Exception as e:
                        log.warning(f"TTS/align attempt {attempt + 1}/{max_retries} for segment {i} failed: {e}")
                        if attempt < max_retries - 1:
                            time.sleep(2)
                        else:
                            log.exception(f"All {max_retries} attempts failed for segment {i}. Skipping.")
                
                if bar:
                    bar.update(1)

                if self.ckpt:
                    if synthesis_successful:
                        self.ckpt.set_artifact("synthesized_segments", synthesized_segments)
                    self.ckpt.set_progress("tts", i + 1, total_segments)
                
                if not synthesis_successful:
                    continue

        # -------------------------
        # STAGE 7: assemble_dialogue
        # -------------------------
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
            if progress:
                progress.update(1)
        else:
            if progress:
                progress.update(0)
            dialogue_path = Path(self.ckpt.get_artifact(
                "dialogue_path", str(dialogue_path))) if self.ckpt else dialogue_path
        pm.emit("assemble_dialogue", "Dialogue assembled",
                0.8, {"dialogue_path": str(dialogue_path)})

        # -------------------------
        # STAGE 8: mix (ducking)
        # -------------------------
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
                            delayed_out = tmp_root /
                                f"{base_stem}_seg_{int(seg.get('index', 0)):04d}_delayed.wav"
                            temp_fade = tmp_root /
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
                            duck_amount_db))
                        _ffmpeg_reduce_base_gain(str(audio_path), str(
                            ducked_base), gain_db=float(duck_amount_db))

                    _ffmpeg_mix([str(ducked_base), str(tts_mix)],
                                str(mixed_path))
                    if self.ckpt:
                        self.ckpt.set_artifact("mixed_path", str(mixed_path))
                except Exception as e:
                    log.exception("Fallback ffmpeg mixing failed: %s", e)
            if self.ckpt:
                self.ckpt.set_stage_done(
                    "mix", {"mixed_path": str(mixed_path)})
            if progress:
                progress.update(1)
        else:
            if progress:
                progress.update(0)
            mixed_path = Path(self.ckpt.get_artifact(
                "mixed_path", str(mixed_path))) if self.ckpt else mixed_path
        pm.emit("mix", "Audio mixed", 0.9, {"mixed_path": str(mixed_path)})

        # -------------------------
        # STAGE 9: mux
        # -------------------------
        dub_video_path = out_dir / f"{base_stem}_dub.mp4"
        if not self._stage_done("mux"):
            try:
                audio_to_mux = mixed_path if Path(
                    mixed_path).exists() else dialogue_path
                if Path(audio_to_mux).exists() and video_path:
                    replace_audio(str(video_path), str(
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
            if progress:
                progress.update(1)
        else:
            if progress:
                progress.update(0)
            dub_video_path = Path(self.ckpt.get_artifact(
                "dub_video", str(dub_video_path))) if self.ckpt else dub_video_path
        pm.emit("mux", "Video muxed", 1.0, {
                "final_video": str(dub_video_path)})

        if progress:
            progress.close()

        result["artifacts"] = self.ckpt.to_dict().get(
            "artifacts", {}) if self.ckpt else {}
        pm.emit("pipeline", "Pipeline complete", 1.0)
        return result

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

    def run_full_pipeline(
        video_path: str,
        output_path: str,
        srt_path: Optional[str] = None,
        use_wav2lip: bool = True,
        save_speaker_clips: bool = False,
        gpu_safe: bool = True,
        tts_model: Optional[str] = None,
        wav2lip_checkpoint: Optional[str] = None,
        progress_cb: Optional[Callable[[str, float], None]] = None,
        eta_cb: Optional[Callable[[str, float], None]] = None,
    ) -> Dict[str, Any]:
        """
        Lightweight orchestrator wrapper around the Pipeline class for GUI usage.

        - Uses your existing Pipeline class (does not replace it).
        - Emits coarse per-stage progress via progress_cb(stage, pct) where pct is 0.0..1.0
        - Calls run_wav2lip(...) if use_wav2lip is True and run_wav2lip is available.
        - Returns a summary dict with 'stages' key describing completion and final output path.
        """

        # Create pipeline using existing class (reuse model config path and device if available)
        # Use sensible defaults -- do not modify Pipeline internals.
        try:
            # If your file defines Pipeline in module scope, instantiate it
            pl = Pipeline(model_config="config/models.json",
                          device=("cuda" if gpu_safe else "cpu"))
        except Exception as e:
            # fallback attempt -- instantiate without args
            try:
                pl = Pipeline()
            except Exception as e2:
                raise RuntimeError(f"Failed to construct Pipeline: {e} / {e2}")

        # Stage list we advertise to GUI (coarse)
        stage_names = [
            "extract_audio", "asr", "diarization", "translation",
            "generate_script", "tts", "assemble_dialogue", "mix", "mux"
        ]

        # Start overall
        _emit_progress(progress_cb, "pipeline", 0.0)
        _emit_eta(eta_cb, "pipeline", None)

        base_name = Path(video_path).stem
        out_dir = str(Path(output_path).parent)

        # call Pipeline.run: it accepts video_path or input_audio, out_dir etc.
        # map parameters: use video_path -> pipeline.run(video_path=..., out_dir=...)
        try:
            _emit_progress(progress_cb, "extract_audio", 0.0)
            res = pl.run(
                input_audio=None,
                out_dir=out_dir,
                base_name=base_name,
                srt_path=srt_path,
                language="en",
                keep_temp=False,
                video_path=video_path,
                dry_run=False,
                groq_use=False,
                groq_api_key=None,
                duck_amount_db=-6.0,
            )
        except Exception as e:
            _emit_progress(progress_cb, "pipeline", 1.0)
            raise

        # coarse mark stages as completed for GUI (we cannot know inner percent without deeper hooks)
        for st in stage_names:
            _emit_progress(progress_cb, st, 1.0)

        # If pipeline returned a mixed/muxed result, prefer that as final output
        final_audio = None
        final_video = None
        try:
            artifacts = {}
            if isinstance(pl.ckpt, object) and hasattr(pl.ckpt, "to_dict"):
                artifacts = pl.ckpt.to_dict().get("artifacts", {})
            elif isinstance(res, dict):
                artifacts = res.get("artifacts", {}) or {}
            else:
                artifacts = {}
        except Exception:
            artifacts = {}

        final_video = artifacts.get("dub_video") or artifacts.get(
            "mixed_path") or artifacts.get("dialogue_path") or output_path

        # Optionally run wav2lip if requested and available
        if use_wav2lip and run_wav2lip is not None:
            _emit_progress(progress_cb, "wav2lip", 0.0)
            try:
                wav2lip_out = str(
                    Path(out_dir) / f"{base_name}_wav2lip_out.mp4")
                run_wav2lip(
                    video_path=str(final_video or video_path),
                    audio_path=str(artifacts.get("mixed_path") or artifacts.get(
                        "dialogue_path") or Path(out_dir) / f"{base_name}_dubbed_audio.wav"),
                    output_path=wav2lip_out,
                    checkpoint_path=wav2lip_checkpoint or str(
                        Path("diadub/models/wav2lip/checkpoints/wav2lip_gan.pth")),
                    device=("cuda" if gpu_safe else "cpu"),
                    progress_cb=lambda stage, pct: _emit_progress(
                        progress_cb, f"wav2lip:{stage}", pct),
                    eta_cb=lambda stage, eta: _emit_eta(
                        eta_cb, f"wav2lip:{stage}", eta),
                )
                final_video = wav2lip_out
                _emit_progress(progress_cb, "wav2lip", 1.0)
            except Exception as e:
                # log but continue -- pipeline already produced an output
                _emit_progress(progress_cb, "wav2lip", 0.0)

        # Final mux / move to requested output path if needed
        try:
            # If pl.run already produced a dub_video artifact, keep it; otherwise, leave provided output_path
            if final_video and str(final_video) != str(output_path):
                # attempt a safe copy/move to the requested location
                try:
                    from shutil import copy2
                    copy2(str(final_video), str(output_path))
                    final_video = output_path
                except Exception:
                    pass
        except Exception:
            pass

        _emit_progress(progress_cb, "pipeline", 1.0)
        _emit_eta(eta_cb, "pipeline", 0.0)

        return {"stages": [{"name": s, "progress": 1.0} for s in stage_names], "output": str(final_video)}