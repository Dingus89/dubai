"""
This file keeps the class-based Pipeline from pipeline.py
(stages, checkpointing, ModelRegistry, mixer, mux) while adopting
the TTSManager and align_tts_to_target alignment from pipelinev2.py.
It also includes ffmpeg helper functions from pipelinev2 for optional
delay/amix mixing and maintains the comprehensive TTS prosody/pitch/align
logic from pipeline.py, falling back to previous aligners when needed.
Added ducking + crossfade helpers from the user's ducking file.
"""

import logging
import os
import json
import time
import tempfile
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any

log = logging.getLogger("diadub.pipeline")

# ------------------
# FFmpeg helpers
# ------------------


def _safe_run(cmd: List[str], raise_on_err: bool = True):
    log.debug("RUN: %s", " ".join(cmd))
    res = subprocess.run(
        cmd, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    if res.returncode != 0:
        log.error("
            Command failed: %s,
            STDOUT: %s,
            STDERR: %s", " ".join(cmd),
            res.stdout.decode(errors="ignore"),
            res.stderr.decode(errors="ignore"))
        if raise_on_err:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return res


def _ffmpeg_fade_in_out(
    src: str, dst: str, fade_ms: int = 8, sr: int = 48000):
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
    cmd += ["-filter_complex", filter_complex, "-ar", "48000", "-ac", "1", str(out_path)]
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
    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", filter_complex, "-ar", "48000", "-ac", "1", str(out_path)]
    _safe_run(cmd)


def _ffmpeg_sidechain_duck(
    base_audio: str, tts_mix: str, out_path: str, makeup_gain_db: float = 0.0):
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
# Config
# ---------------------------------------------------------------------
def load_models_config(config_path: str = "config/models.json") -> Dict[str, Any]:
    cfg = {"tts": {"engine": "orpheus", "model_path": "", "device": "cuda", "voice_routing": {}}}
    try:
        p = Path(config_path)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        log.warning("Failed to load models config %s", config_path)
    return cfg

# ------------------
# Existing pipeline.py imports and helpers
# ------------------
try:
    from .ffmpeg_utils import extract_audio, replace_audio
except Exception:
    # fallback stubs if package layout differs
    def extract_audio(video_path: str, out_wav: str):
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-ar", "48000", "-ac", "1", "-vn", str(out_wav)]
        _safe_run(cmd)

    def replace_audio(video_path: str, audio_path: str, output_path: str):
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-i", str(audio_path), "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", str(output_path)]
        _safe_run(cmd)

try:
    from .models.registry import ModelRegistry
except Exception:
    ModelRegistry = None

try:
    from .utils.logging_config import setup_logging
except Exception:
    setup_logging = None

try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None

# checkpoint manager
try:
    from .storage.checkpoint import CheckpointManager
except Exception:
    CheckpointManager = None

# align_tts_to_target from pipelinev2, fallback to align_and_adjust
try:
    from diadub.timing.sync_logic import align_tts_to_target
except Exception:
    align_tts_to_target = None

try:
    from diadub.alignment.aligner import align_and_adjust
except Exception:
    align_and_adjust = None

# from pipelinev2: TTSManager - prefer it for TTS
try:
    from models.tts_engine_manager import TTSManager
except Exception:
    TTSManager = None

# other optional components used by pipeline.py
try:
    from diadub.prosody.prosody_mapper import ProsodyMapper
except Exception:
    ProsodyMapper = None

log = logging.getLogger("diadub.pipeline")

# Dependency graph for selective invalidation (updated with generate_script)
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


def seconds_to_srt_time(s: float) -> str:
    ms = int(round(s * 1000))
    hours = ms // 3_600_000
    ms %= 3_600_000
    minutes = ms // 60_000
    ms %= 60_000
    seconds = ms // 1000
    ms %= 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


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
        model_config: str = "models.json",
        device: str = "cuda",
        temp_dir: str = "data/cache",
        checkpoint_path: Optional[str] = None,
        resume: bool = False,
    ):
        self.registry = ModelRegistry(
            config_path=model_config, device=device) if ModelRegistry else None
        self.device = device
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path else self.temp_dir / "pipeline.checkpoint.json"
        )
        self.resume = resume

        # load checkpoint manager
        self.ckpt = CheckpointManager(str(
            self.checkpoint_path)) if CheckpointManager else None
        if resume and self.checkpoint_path.exists() and self.ckpt is not None:
            try:
                self.ckpt.load()
                log.info("Loaded checkpoint from %s", self.checkpoint_path)
            except Exception as e:
                log.warning("Failed to load checkpoint (will start fresh): %s", e)

        # lazy backends
        self.asr = None
        self.diarizer = None
        self.translator = None
        # prefer TTSManager for TTS, fallback to registry.get('tts')
        self.tts_manager = None
        self.tts_backend = None
        # optional emotion model
        self.emotion_model = None

    def _ensure_backends(self):
        # ASR
        if self.registry:
            try:
                self.asr = self.registry.get("asr")
                log.info("ASR backend available.")
            except Exception as exc:
                log.warning("ASR backend not available: %s", exc)

            # Diarization
            try:
                self.diarizer = self.registry.get("diarization")
                log.info("Diarization backend available.")
            except Exception as exc:
                log.warning("Diarization backend not available: %s", exc)

            # Translator
            try:
                self.translator = self.registry.get("translation")
                log.info("Translation backend available.")
            except Exception as exc:
                log.warning("Translation backend not available: %s", exc)

            # TTS fallback backend if TTSManager not used
            try:
                self.tts_backend = self.registry.get("tts")
                log.info("TTS backend available in registry.")
            except Exception as exc:
                log.warning("TTS backend not available in registry: %s", exc)

            # Emotion model (optional)
            try:
                self.emotion_model = self.registry.get("asr_wav2vec_emotion")
                log.info("Emotion model available.")
            except Exception:
                self.emotion_model = None

        # Initialize TTSManager from pipelinev2 config if available
        cfg = load_models_config()
        try:
            if TTSManager is not None:
                self.tts_manager = TTSManager(cfg.get("tts", {}))
                log.info("TTSManager initialized from config.")
            else:
                log.info(
                    "TTSManager not available; will use registry TTS if present.")
        except Exception as e:
            log.warning("Failed to init TTSManager: %s", e)
            self.tts_manager = None

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

    def _extract_segment(
        self, full_audio: str, start_s: float, end_s: float, out_wav: str):
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

    def run_pipeline(
        input_audio: str,
        out_dir: str = "output",
        base_name: str,
        srt_path: str = None,
        language: str = "en",
        keep_temp: bool = False,
        self,
        video_path: str,
        srt_path: Optional[str] = None,
        dry_run: bool = False,
        groq_use: bool = False,
        groq_api_key: Optional[str] = None,
        duck_amount_db: float = -6.0,
    ) -> Dict:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        video_path = Path(video_path)
        base_stem = video_path.stem

        self._ensure_backends()

        out_dir = Path(out_dir)
        audio_path=input_audio,
        srt_path=srt_path,
        language=language,
        use_local_polish=kwargs.get("use_local_polish", True),
        groq_use=kwargs.get("groq_use", False),
        )


        script_obj = json.loads(Path(script_path).read_text(encoding="utf-8"))


        cfg = load_models_config()
        if not TTSManager:
        raise RuntimeError("TTSManager missing")
        tts_manager = TTSManager(cfg.get("tts", {}))


        delayed_layers = []
        aligned_paths = []


        for idx, line in enumerate(script_obj.get("lines", [])):
        tts_instruction = line.get("tts_instruction")
        if not tts_instruction:
        continue


        tts_result = tts_manager.synthesize(tts_instruction, tmp_root=str(tmp_root))
        tts_path = tts_result.get("tts_path")
        tts_words = tts_result.get("tts_words", [])


        if not tts_path or not Path(tts_path).exists():
        continue


        target_words = tts_instruction.get("target_words", [])
        aligned_segment_path = tmp_root / f"{base_name}_line_{idx:04d}_aligned.wav"


        if align_tts_to_target and target_words:
        try:
        aligned_output = align_tts_to_target(
        tts_audio_path=str(tts_path),
        tts_words=tts_words if tts_words else target_words,
        target_words=target_words,
        out_path=str(aligned_segment_path),
        options={"max_stretch": 1.25, "min_stretch": 0.8, "crossfade_ms": 12},
        )
        aligned_path = str(aligned_output)
        except Exception:
        shutil.copy2(tts_path, aligned_segment_path)
        aligned_path = str(aligned_segment_path)
        else:
        shutil.copy2(tts_path, aligned_segment_path)
        aligned_path = str(aligned_segment_path)


        line["aligned_tts_path"] = aligned_path
        aligned_paths.append(aligned_path)


        faded = tmp_root / f"{base_name}_line_{idx:04d}_faded.wav"
        _ffmpeg_fade_in_out(aligned_path, str(faded))


        seg_start = float(line.get("start", 0.0))
        delay_ms = int(round(seg_start * 1000.0))
        delayed_out = tmp_root / f"{base_name}_line_{idx:04d}_delayed.wav"
        _ffmpeg_delay(str(faded), str(delayed_out), delay_ms)
        delayed_layers.append(str(delayed_out))


        if not delayed_layers:
        raise RuntimeError("No TTS layers generated")


        tts_mix = tmp_root / f"{base_name}_tts_mix.wav"
        _ffmpeg_mix(delayed_layers, str(tts_mix))


        ducked_base = tmp_root /

        # progress helper
        try:
            from tqdm import tqdm

            progress = tqdm(total=len(self.STAGES), desc="Pipeline")
            for s in self.STAGES:
                if self._stage_done(s):
                    progress.update(1)
        except Exception:
            progress = None

        result = {"video": str(video_path), "artifacts": {}}

        # --- Artifact integrity check before resume ---
        if self.resume and self.ckpt is not None:
            ok = self.ckpt.verify_artifacts(deep=False, log_path=str(
                self.temp_dir / "integrity.log"))
            if not ok:
                log.warning("""
                    Integrity check failed: some artifacts missing or small.
                    Re-running dependent stages.""")
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

        # STAGE 1: extract_audio
        if not self._stage_done("extract_audio"):
            audio_path = out_dir / f"{base_stem}.wav"
            if not audio_path.exists():
                log.info("Extracting audio from %s -> %s", video_path, audio_path)
                extract_audio(str(video_path), str(audio_path))
            else:
                log.info("Using cached audio %s", audio_path)
            if self.ckpt:
                self.ckpt.set_artifact("audio_path", str(audio_path))
                self.ckpt.set_stage_done("extract_audio", {"audio_path": str(audio_path)})
            if progress:
                progress.update(1)
        else:
            audio_path = Path(self.ckpt.get_artifact(
                "audio_path")) if self.ckpt else out_dir / f"{base_stem}.wav"
            log.info("Resuming: using audio from checkpoint: %s", audio_path)
            if progress:
                progress.update(0)

        # STAGE 2: ASR
        segments = []
        asr_result = None
        if not self._stage_done("asr"):
            if self.asr:
                try:
                    if hasattr(self.asr, "infer") and callable(
                        getattr(self.asr, "infer")):
                        asr_out = self.asr.infer(str(audio_path))
                        if isinstance(asr_out, dict) and "segments" in asr_out:
                            segments = [
                                {"text": s.get("text", "").strip(),
                                    "start": float(s.get("start", 0.0)),
                                    "end": float(s.get("end", 0.0)),
                                    "confidence": s.get("confidence", None)}
                                for s in asr_out.get("segments", [])
                            ]
                            asr_result = asr_out
                        elif isinstance(asr_out, dict) and "text" in asr_out:
                            segments = [{
                                "text": asr_out.get("text", ""),
                                "start": 0.0, "end": float(
                                    self._get_duration(str(audio_path))),
                                "confidence": None}]
                            asr_result = asr_out
                        else:
                            segments = []
                            asr_result = asr_out
                    else:
                        raw_segments = self.asr.transcribe(str(audio_path))
                        segments = [
                            {"text": seg.get("text", "").strip(),
                                "start": float(seg.get("start", 0.0)),
                                "end": float(seg.get("end", 0.0)),
                                "confidence": seg.get("confidence", None)}
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
            segments = self.ckpt.get_artifact("segments", []) if self.ckpt else []
            log.info(
                "Resuming: loaded %d segments from checkpoint.", len(segments))
            if progress:
                progress.update(0)

        if not segments:
            log.warning("No segments available after ASR; aborting.")
            if progress:
                progress.close()
            return {"error": "no_segments"}

        # STAGE 3: Diarization
        diarize_segs = []
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

        # STAGE 4: Translation
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
                            else self.translator.translate_lines([seg])[0].get(
                                "translated", txt)
                        )
                    except Exception:
                        try:
                            if hasattr(self.translator, "translate_lines"):
                                res = self.translator.translate_lines([seg])
                                seg["translated"] = res[0].get("translated", txt)
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
                        self.ckpt.set_progress("translation", i + 1, len(segments))
                except Exception:
                    pass

            if bar:
                bar.close()
            if self.ckpt:
                self.ckpt.set_artifact("translated_segments", segments)
                self.ckpt.set_stage_done("translation", {
                    "translated_segments": segments})
            if progress:
                progress.update(1)
        else:
            segments = self.ckpt.get_artifact(
                "translated_segments", segments) if self.ckpt else segments
            if progress:
                progress.update(0)

        # write translated srt artifact (idempotent)
        translated_srt_path = out_dir / f"{base_stem}_translated.srt"
        self._write_srt(segments, translated_srt_path)
        if self.ckpt:
            self.ckpt.set_artifact("translated_srt", str(translated_srt_path))

        # STAGE 5: generate_script (NEW)
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
                        srt_path=(str(
                            translated_srt_path) if translated_srt_path.exists() else None),
                        asr_segments=segments,
                        emotion_model=emotion_model,
                        groq_use=groq_flag,
                        groq_api_key=groq_key,
                    )
                    script_path = Path(script_file)
                    log.info("Script generated at %s", script_path)
                except Exception as e:
                    log.warning("""
                        Script generator not available or failed: %s.
                        Falling back to segment-based script.""", e)
                    fallback = {"metadata": {
                        "base_name": base_stem,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "audio_path": str(audio_path)}, "lines": []}
                    for i, seg in enumerate(segments):
                        fallback["lines"].append({
                            "index": i,
                            "start": seg.get("start", 0.0),
                            "end": seg.get("end", seg.get("start", 0.0) + 0.5),
                            "text": seg.get("translated", seg.get("text", "")),
                            "analysis": {"duration": seg.get("end", seg.get(
                                "start", 0.0)) - seg.get("start", 0.0)},
                        })
                    script_path.write_text(json.dumps(
                        fallback, indent=2, ensure_ascii=False))
                if self.ckpt:
                    self.ckpt.set_artifact("script_path", str(script_path))
            except Exception as e:
                log.exception("generate_script stage failed: %s", e)
            if self.ckpt:
                self.ckpt.set_stage_done(
                    "generate_script",
                    {"script_path": str(script_path)})
            if progress:
                progress.update(1)
        else:
            script_path = Path(self.ckpt.get_artifact(
                "script_path", str(script_path))) if self.ckpt else script_path
            log.info("Resuming: script at %s", script_path)
            if progress:
                progress.update(0)

        # STAGE 6: TTS — merged logic
        synthesized_segments = []
        if not self._stage_done("tts"):
            try:
                from tqdm import tqdm
                bar = tqdm(total=len(
                    segments), desc="Synthesizing voices", unit="seg")
            except Exception:
                bar = None

            prev = self.ckpt.get_progress("tts") or {
                "current": 0} if self.ckpt else {"current": 0}
            start_index = int(prev.get("current", 0))

            # prepare prosody mapper
            prosody_mapper = None
            try:
                from diadub.prosody.prosody_mapper import ProsodyMapper
                prosody_mapper = ProsodyMapper()
            except Exception:
                prosody_mapper = None

            # parse script into tts items
            try:
                from diadub.script.script_parser import parse_for_tts
                tts_items = parse_for_tts(str(script_path))
            except Exception:
                tts_items = []
                for i, seg in enumerate(segments):
                    tts_items.append({
                        "index": i,
                        "text": seg.get("translated", seg.get("text", "")),
                        "start": seg.get("start", 0.0),
                        "end": seg.get("end", seg.get("start", 0.0) + 0.5),
                        "duration": seg.get(
                            "end", seg.get("start", 0.0)) - seg.get("start", 0.0),
                        "prosody": {
                            "rate": 1.0, "pitch_semitones": 0.0, "gain_db": 0.0},
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
                            self.ckpt.set_progress("tts", i + 1, total_segments)
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

                # extract original segment for analysis and alignment
                segment_audio_tmp = str(self.temp_dir / f"{base_stem}_seg_{i:04d}_orig.wav")
                try:
                    self._extract_segment(str(audio_path), item.get(
                        "start", 0.0), item.get("end",
                        item.get("start", 0.0) + 0.5), segment_audio_tmp)
                except Exception as e:
                    log.debug(
                        "Failed to extract precise segment audio; using full audio: %s", e)
                    segment_audio_tmp = str(audio_path)

                # analyze for prosody
                audio_meta = None
                try:
                    from diadub.audio_analysis.audio_features import analyze_segment
                    audio_meta = analyze_segment(segment_audio_tmp)
                except Exception:
                    audio_meta = {"duration": item.get("duration", item.get(
                        "end", 0.0) - item.get("start", 0.0))}

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

                # call TTS via TTSManager (preferred) or registry backend
                out = None
                try:
                    if self.tts_manager is not None:
                        # TTSManager.synthesize expected to mirror pipelinev2
                        tts_result = self.tts_manager.synthesize({
                            "text": text_to_speak, "prosody": prosody_params,
                            "voice": item.get("voice")}, tmp_root=str(tmp_root))
                        # normalize to return types similar to pipeline.py expectations
                        if isinstance(tts_result, dict) and "tts_path" in tts_result:
                            out = tts_result.get("tts_path")
                            tts_words = tts_result.get("tts_words", [])
                        else:
                            out = tts_result
                            tts_words = []
                    elif self.tts_backend is not None:
                        try:
                            out = self.tts_backend.synth_text(
                                text_to_speak, voice_id=item.get("voice"),
                                prosody_params=prosody_params)
                        except TypeError:
                            out = self.tts_backend.synth_text(text_to_speak)
                        tts_words = []
                    else:
                        raise RuntimeError("No TTS backend available")
                except Exception as e:
                    log.exception("TTS synth error for item %d: %s", i, e)
                    if bar:
                        bar.update(1)
                    try:
                        if self.ckpt:
                            self.ckpt.set_progress("tts", i + 1, total_segments)
                    except Exception:
                        pass
                    continue

                # normalize output to wav file at wav_path_raw
                try:
                    if isinstance(out, (str, Path)):
                        retp = Path(out)
                        if retp.exists():
                            retp.replace(wav_path_raw)
                        else:
                            log.warning("TTS returned a path but file missing: %s", retp)
                    elif isinstance(out, bytes):
                        wav_path_raw.write_bytes(out)
                    else:
                        handled = False
                        try:
                            import numpy as np
                            if hasattr(out, "numpy"):
                                arr = out.numpy()
                                import soundfile as sf
                                sf.write(str(wav_path_raw), arr, samplerate=48000)
                                handled = True
                            elif isinstance(out, np.ndarray):
                                import soundfile as sf
                                sf.write(str(wav_path_raw), out, samplerate=48000)
                                handled = True
                        except Exception:
                            handled = False
                        if not handled:
                            log.warning("Unsupported TTS return type: %s", type(out))
                except Exception as e:
                    log.exception("Failed to write TTS output for item %d: %s", i, e)
                    if bar:
                        bar.update(1)
                    try:
                        if self.ckpt:
                            self.ckpt.set_progress("tts", i + 1, total_segments)
                    except Exception:
                        pass
                    continue

                # optional post pitch shift
                try:
                    pitch_steps = float(prosody_params.get(
                        "pitch_semitones", 0.0)) if prosody_params else 0.0
                except Exception:
                    pitch_steps = 0.0

                base_for_alignment = str(wav_path_raw)
                if abs(pitch_steps) > 0.01:
                    try:
                        from diadub.alignment.aligner import post_tts_pitch_shift
                        post_tts_pitch_shift(str(wav_path_raw), str(
                            wav_path_pitch), pitch_steps)
                        base_for_alignment = str(wav_path_pitch)
                    except Exception as e:
                        log.debug("Post pitch shift failed: %s", e)
                        base_for_alignment = str(wav_path_raw)

                # Alignment: prefer align_tts_to_target (for target info)
                synthesized_wav = str(wav_path_raw)
                try:
                    # compile target words from script line if available
                    target_words = item.get("tts_instruction", {}).get(
                        "target_words") if isinstance(item, dict) else None
                    # align_tts_to_target exists and tts_manager word timings
                    if align_tts_to_target is not None and target_words:
                        try:
                            # tts_words may have been returned earlier
                            aligned_out = align_tts_to_target(
                                tts_audio_path=base_for_alignment,
                                tts_words=tts_words if 'tts_words' in locals() and tts_words else target_words,
                                target_words=target_words,
                                out_path=str(wav_path_adjusted),
                                options={"max_stretch": 1.5, "min_stretch": 0.7, "crossfade_ms": 12},
                            )
                            synthesized_wav = str(aligned_out)
                        except Exception as e:
                            log.debug("align_tts_to_target failed: %s", e)
                            synthesized_wav = base_for_alignment
                    else:
                        # fallback to align_and_adjust if available
                        if align_and_adjust is not None:
                            try:
                                align_and_adjust(
                                    base_for_alignment, segment_audio_tmp,
                                    str(wav_path_adjusted), max_stretch=1.5,
                                    per_subsegment=True)
                                synthesized_wav = str(wav_path_adjusted)
                            except Exception as e:
                                log.exception(
                                    """Alignment adjust failed for item %d: %s.
                                    Falling back to raw TTS.""", i, e)
                                synthesized_wav = str(wav_path_raw)
                        else:
                            synthesized_wav = str(wav_path_raw)
                except Exception as e:
                    log.exception(
                        "Unexpected error during alignment for item %d: %s", i, e)
                    synthesized_wav = str(wav_path_raw)

                synthesized_segments.append({
                    "wav": synthesized_wav,
                    "start": item.get("start"),
                    "end": item.get("end"),
                    "speaker": item.get("speaker"),
                    "index": item.get("index"),
                })

                try:
                    if self.ckpt:
                        self.ckpt.set_artifact("synthesized_segments", synthesized_segments)
                        self.ckpt.set_progress("tts", i + 1, total_segments)
                except Exception:
                    pass

                if bar:
                    bar.update(1)

            if bar:
                bar.close()

            # write out script with aligned_tts_path per-line when possible
            try:
                # If script file is JSON and has lines, attach aligned paths
                s_obj = json.loads(script_path.read_text(
                    encoding="utf-8")) if script_path.exists() else None
                if s_obj and isinstance(s_obj.get("lines"), list):
                    # naive mapping by index
                    for seg in synthesized_segments:
                        idx = int(seg.get("index", -1))
                        if 0 <= idx < len(s_obj.get("lines")):
                            s_obj["lines"][idx]["aligned_tts_path"] = seg.get("wav")
                    final_script_path = out_dir / f"{base_stem}_script_with_audio.json"
                    final_script_path.write_text(
                        json.dumps(s_obj, indent=2,
                        ensure_ascii=False), encoding="utf-8")
                    if self.ckpt:
                        self.ckpt.set_artifact(
                            "script_with_audio", str(final_script_path))
            except Exception:
                pass

            if self.ckpt:
                self.ckpt.set_artifact(
                    "synthesized_segments", synthesized_segments)
                self.ckpt.set_stage_done("tts", {
                    "synthesized_segments": synthesized_segments})
            if progress:
                progress.update(1)
        else:
            synthesized_segments = self.ckpt.get_artifact(
                "synthesized_segments", []) if self.ckpt else []
            if progress:
                progress.update(0)

        # STAGE 7: assemble_dialogue
        dialogue_path = out_dir / f"{base_stem}_dialogue.wav"
        if not self._stage_done("assemble_dialogue"):
            if synthesized_segments and AudioSegment is not None:
                try:
                    self._assemble_and_export_dialogue(str(
                        audio_path), synthesized_segments, str(dialogue_path))
                    if self.ckpt:
                        self.ckpt.set_artifact("dialogue_path", str(dialogue_path))
                except Exception as e:
                    log.exception("Assemble dialogue failed: %s", e)
            else:
                log.info("No synthesized segments to assemble.")
            if self.ckpt:
                self.ckpt.set_stage_done(
                    "assemble_dialogue", {"dialogue_path": str(dialogue_path)})
            if progress:
                progress.update(1)
        else:
            if progress:
                progress.update(0)
            dialogue_path = Path(self.ckpt.get_artifact(
                "dialogue_path",
                str(dialogue_path))) if self.ckpt else dialogue_path

        # STAGE 8: mix (ducking) — fallback to ffmpeg amix + adelay + ducking
        mixed_path = out_dir / f"{base_stem}_dialogue_mixed.wav"
        if not self._stage_done("mix"):
            try:
                from .mixing.mixer import Mixer
                mixer = Mixer()
                if dialogue_path.exists():
                    mixer.mix(str(audio_path), str(dialogue_path), str(mixed_path))
                    if self.ckpt:
                        self.ckpt.set_artifact("mixed_path", str(mixed_path))
                else:
                    log.info("Dialogue track missing; skipping mixer.")
            except Exception as exc:
                log.warning("""
                    Mixer not available or failed: %s. Falling back to ffmpeg
                    mix if synthesized segments exist.""", exc)
                try:
                    # Ensure tmp_root exists
                    tmp_root = out_dir / "tmp_diadub"
                    tmp_root.mkdir(parents=True, exist_ok=True)

                    # Create delayed layers for each synthesized segment
                    delayed_layers = []
                    for seg in synthesized_segments:
                        try:
                            seg_start = float(seg.get("start", 0.0))
                            delay_ms = int(round(seg_start * 1000.0))
                            delayed_out = tmp_root / f"{base_stem}_seg_{int(
                                seg.get('index',0)):04d}_delayed.wav"
                            # apply small fade + delay to avoid clicks
                            temp_fade = tmp_root / f"{base_stem}_seg_{
                                int(seg.get('index',0)):04d}_faded.wav"
                            _ffmpeg_fade_in_out(seg.get("wav"), str(temp_fade), fade_ms=8)
                            _ffmpeg_delay(str(temp_fade), str(delayed_out), delay_ms)
                            delayed_layers.append(str(delayed_out))
                        except Exception as e:
                            log.debug(
                                "Failed to create delayed layer for seg %s: %s",
                                seg.get('index'), e)

                    if not delayed_layers:
                        raise RuntimeError(
                            "No delayed layers created for fallback mix")

                    # Mix delayed layers into a single TTS mix
                    tts_mix = tmp_root / f"{base_stem}_tts_mix.wav"
                    log.info(
                        "Mixing %d delayed layers into single tts mix", len(delayed_layers))
                    _ffmpeg_mix(delayed_layers, str(tts_mix))

                    # Attempt sidechain ducking; if unavailable, fallback to static reduction
                    ducked_base = tmp_root / f"{base_stem}_base_ducked.wav"
                    try:
                        _ffmpeg_sidechain_duck(str(
                            audio_path), str(tts_mix), str(ducked_base))
                        log.info("Applied sidechain ducking to base audio")
                    except Exception as e:
                        log.warning(
                            """Sidechain duck failed or unsupported: %s.
                            Falling back to static reduction (gain=%sdB)""", e, float(duck_amount_db))
                        _ffmpeg_reduce_base_gain(str(audio_path), str(
                            ducked_base), gain_db=float(duck_amount_db))

                    # Mix ducked base and tts_mix into final mixed track
                    _ffmpeg_mix([str(ducked_base), str(tts_mix)], str(mixed_path))
                    if self.ckpt:
                        self.ckpt.set_artifact("mixed_path", str(mixed_path))
                except Exception as e:
                    log.exception("Fallback ffmpeg mixing failed: %s", e)
            if self.ckpt:
                self.ckpt.set_stage_done("mix", {"mixed_path": str(mixed_path)})
            if progress:
                progress.update(1)
        else:
            if progress:
                progress.update(0)
            mixed_path = Path(self.ckpt.get_artifact("mixed_path", str(
                mixed_path))) if self.ckpt else mixed_path

        # STAGE 9: mux
        dub_video_path = out_dir / f"{base_stem}_dub.mp4"
        if not self._stage_done("mux"):
            try:
                audio_to_mux = mixed_path if Path(mixed_path).exists() else dialogue_path
                if Path(audio_to_mux).exists():
                    replace_audio(str(video_path), str(
                        audio_to_mux), output_path=str(dub_video_path))
                    if self.ckpt:
                        self.ckpt.set_artifact("dub_video", str(dub_video_path))
                else:
                    log.info("No assembled audio available to mux; skipping mux.")
            except Exception as exc:
                log.exception("Failed to mux: %s", exc)
            if self.ckpt:
                self.ckpt.set_stage_done("mux", {"dub_video": str(dub_video_path)})
            if progress:
                progress.update(1)
        else:
            if progress:
                progress.update(0)
            dub_video_path = Path(self.ckpt.get_artifact("dub_video", str(
                dub_video_path))) if self.ckpt else dub_video_path

        if progress:
            progress.close()

        result["artifacts"] = self.ckpt.to_dict().get("artifacts", {}) if self.ckpt else {}
        return result

    # ---------------------
    # Helper methods below
    # ---------------------
    def _parse_srt(self, srt_path: str) -> List[Dict]:
        srt_path = Path(srt_path)
        out = []
        if not srt_path.exists():
            return out
        text = srt_path.read_text(encoding="utf-8")
        parts = [p.strip() for p in text.split("

") if p.strip()]
        for p in parts:
            lines = p.splitlines()
            if len(lines) >= 2:
                ts = lines[1]
                if "-->" not in ts:
                    continue
                start_txt, end_txt = [t.strip() for t in ts.split("-->")]

                def parse(s):
                    h, m, rest = s.split(":")
                    sec, ms = rest.split(",")
                    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000.0

                try:
                    start = parse(start_txt)
                    end = parse(end_txt)
                except Exception:
                    start = 0.0
                    end = 0.0
                body = " ".join(lines[2:])
                out.append({"text": body, "start": start, "end": end, "confidence": None})
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
            srt_lines.append(f"{i}
{start} --> {end}
{text}

")
        out_path.write_text("".join(srt_lines), encoding="utf-8")

    def _assemble_and_export_dialogue(
            self, original_audio_path: str,
            synth_segments: List[Dict],
            out_wav_path: str):
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
                log.debug(
                    "TTS audio longer than original: %s (tts %d ms vs seg %d ms)",
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

