import hashlib
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from diadub.models.tts_engine_manager import TTSEngineManager
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
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------
# Local helper imports (robust fallbacks)
# ---------------------------------------------------------------------
# media helpers (if installed as separate module)
try:
    from diadub.utils.media import extract_audio, replace_audio, extract_audio_segment, get_audio_duration
except Exception:
    # Minimal safe fallbacks (should be overridden by diadub.utils.media)
    def extract_audio(video_path: str, out_wav: str):
        cmd = ["ffmpeg", "-y", "-i",
               str(video_path), "-ar", "48000", "-ac", "1", "-vn", str(out_wav)]
        subprocess.run(cmd, check=True)

    def replace_audio(video_path: str, audio_path: str, output_path: str):
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-i", str(
            audio_path), "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", str(output_path)]
        subprocess.run(cmd, check=True)

    def extract_audio_segment(src: str, dst: str, start: float, end: float):
        dur = max(0.001, float(end) - float(start))
        cmd = ["ffmpeg", "-y", "-i",
               str(src), "-ss", f"{float(start):.3f}", "-t", f"{dur:.3f}", "-ar", "48000", "-ac", "1", str(dst)]
        subprocess.run(cmd, check=True)

    def get_audio_duration(src: str) -> float:
        try:
            p = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries",
                    "format=duration", "-of", "default=nk=1:nw=1", str(src)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )
            return float(p.stdout.decode().strip())
        except Exception:
            return 0.0

# separation wrapper (demucs)
try:
    from models.separation.demucs_wrapper import demucs_separate
except Exception:
    try:
        from diadub.models.separation.demucs_wrapper import demucs_separate
    except Exception:
        demucs_separate = None

# queue manager to manage GPU/CPU heavy tasks
try:
    from tools.queue_manager import QueueManager
except Exception:
    QueueManager = None

# audio cleaning
try:
    from diadub.audio.cleaning import clean_tts_file
except Exception:
    clean_tts_file = None

# viseme mapper
try:
    from diadub.lipsync.viseme_mapper import generate_viseme_timing
except Exception:
    generate_viseme_timing = None

# forced align (MFA wrapper)
try:
    from diadub.lipsync.forced_align import run_mfa_align
except Exception:
    run_mfa_align = None

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

# TTS Engine Manager (the file we will patch)

# optional pydub for assembly
try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None

# checkpoint manager optional
try:
    from diadub.storage.checkpoint import CheckpointManager
except Exception:
    CheckpointManager = None

# registry optional
try:
    from diadub.models.registry import ModelRegistry
except Exception:
    ModelRegistry = None

# prosody mapper optional (keeps compatibility)
try:
    from diadub.prosody.prosody_mapper import ProsodyMapper
except Exception:
    ProsodyMapper = None

# progress bar + concurrency

# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------


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
# Pipeline class (keeps original public API name)
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
        self.device = device
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = Path(
            checkpoint_path) if checkpoint_path else self.temp_dir / "pipeline.checkpoint.json"
        self.resume = resume

        # registry & ckpt
        self.registry = ModelRegistry(
            config_path=model_config, device=device) if ModelRegistry else None
        self.ckpt = CheckpointManager(
            str(self.checkpoint_path)) if CheckpointManager else None
        if resume and self.ckpt and self.checkpoint_path.exists():
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

        # separation cache & queue manager
        self.use_per_segment_separation = True
        self.separation_cache: Dict[str, Any] = {}
        self.queue_mgr = QueueManager(cpu_workers=min(8, max(1, (os.cpu_count(
        ) or 4)//2)), gpu_count=1, gpu_mem_threshold_mb=1200) if QueueManager else None

    # -------------------------
    # Backend initialization
    # -------------------------
    def _ensure_backends(self):
        if self.registry:
            try:
                self.asr = self.registry.get("asr")
                log.info("ASR backend available.")
            except Exception as exc:
                log.warning("ASR backend not available: %s", exc)
            try:
                self.diarizer = self.registry.get("diarization")
                log.info("Diarization backend available.")
            except Exception as exc:
                log.warning("Diarization backend not available: %s", exc)
            try:
                self.translator = self.registry.get("translation")
                log.info("Translation backend available.")
            except Exception as exc:
                log.warning("Translation backend not available: %s", exc)
            try:
                self.emotion_model = self.registry.get("asr_wav2vec_emotion")
                log.info("Emotion model available.")
            except Exception:
                self.emotion_model = None

    def _stage_done(self, name: str) -> bool:
        if self.resume and self.ckpt:
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
    # Core run method (enhanced)
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
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # input selection
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

        # Initialize TTS Engine Manager (improved manager handles NVMe offload)
        self.tts_engine = TTSEngineManager()

        # progress bar for top-level stages
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

        # -------------------------
        # STAGE 1: extract_audio (and mandatory full-file separation)
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
                log.info("Using provided audio %s", audio_path)

            # mandatory global separation (full file)
            if demucs_separate:
                sep_out_root = out_dir / "separated"
                sep_out_root.mkdir(parents=True, exist_ok=True)
                try:
                    sep_info = demucs_separate(
                        str(audio_path), str(sep_out_root), stems="vocals")
                    # demucs_separate expected to return {"vocals": "...", "no_vocals": "..."}
                    if sep_info:
                        vocals = sep_info.get(
                            "vocals") or sep_info.get("vocals.wav")
                        no_vocals = sep_info.get(
                            "no_vocals") or sep_info.get("no_vocals.wav")
                        if vocals:
                            self.ckpt.set_artifact(
                                "dialogue_only", str(vocals))
                        else:
                            self.ckpt.set_artifact(
                                "dialogue_only", str(audio_path))
                        self.ckpt.set_artifact("fx_only", str(
                            no_vocals) if no_vocals else None)
                    else:
                        log.warning(
                            "Demucs separation returned nothing - using full mix as dialogue")
                        self.ckpt.set_artifact(
                            "dialogue_only", str(audio_path))
                        self.ckpt.set_artifact("fx_only", None)
                except Exception as e:
                    log.exception("Full-file demucs separation failed: %s", e)
                    self.ckpt.set_artifact("dialogue_only", str(audio_path))
                    self.ckpt.set_artifact("fx_only", None)
            else:
                log.error(
                    "Demucs separation wrapper missing - separation required but unavailable.")
                self.ckpt.set_artifact("dialogue_only", str(audio_path))
                self.ckpt.set_artifact("fx_only", None)

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

        # -------------------------
        # STAGE 2: ASR (unchanged)
        # -------------------------
        segments: List[Dict] = []
        asr_result = None
        if not self._stage_done("asr"):
            if self.asr:
                try:
                    if hasattr(self.asr, "infer") and callable(getattr(self.asr, "infer")):
                        asr_out = self.asr.infer(
                            str(self.ckpt.get_artifact("dialogue_only") or audio_path))
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
                        raw_segments = self.asr.transcribe(
                            str(self.ckpt.get_artifact("dialogue_only") or audio_path))
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

        if not segments:
            log.warning("No segments available after ASR; aborting.")
            if progress:
                progress.close()
            return {"error": "no_segments"}

        # -------------------------
        # STAGE 3: Diarization (attach speakers if available)
        # -------------------------
        diarize_segs: List[Dict] = []
        if not self._stage_done("diarization"):
            if self.diarizer:
                try:
                    diarize_segs = self.diarizer.diarize(
                        str(self.ckpt.get_artifact("dialogue_only") or audio_path))
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

        # -------------------------
        # STAGE 4: Translation (unchanged)
        # -------------------------
        if not self._stage_done("translation"):
            try:
                bar = tqdm(total=len(segments), desc="Translating", unit="seg")
            except Exception:
                bar = None

            for i, seg in enumerate(segments):
                txt = seg.get("text", "")
                if self.translator and txt.strip():
                    try:
                        seg["translated"] = (self.translator.translate_text(txt) if hasattr(
                            self.translator, "translate_text") else self.translator.translate_lines([seg])[0].get("translated", txt))
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

        # Write translated SRT artifact
        translated_srt_path = out_dir / f"{base_stem}_translated.srt"
        self._write_srt(segments, translated_srt_path)
        if self.ckpt:
            self.ckpt.set_artifact("translated_srt", str(translated_srt_path))

        # -------------------------
        # STAGE 5: generate_script
        # -------------------------
        script_path = out_dir / f"{base_stem}_script.json"
        if not self._stage_done("generate_script"):
            try:
                from diadub.script.script_generator import generate_script
                emotion_model = self.emotion_model
                groq_flag = groq_use
                groq_key = groq_api_key
                script_file = generate_script(
                    out_dir=str(out_dir),
                    base_name=base_stem,
                    audio_path=str(self.ckpt.get_artifact(
                        "dialogue_only") or audio_path),
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

        # -------------------------
        # PREPARATION: Pre-separate per-segment (parallel + cache)
        # -------------------------
        try:
            # Pre-generate segment originals (if not already present)
            tmp_seg_dir = self.temp_dir
            tmp_seg_dir.mkdir(parents=True, exist_ok=True)
            # ensure base segments exist
            for idx, seg in enumerate(segments):
                seg_orig = tmp_seg_dir / f"{base_stem}_seg_{idx:04d}_orig.wav"
                if not seg_orig.exists():
                    try:
                        self._extract_segment(str(self.ckpt.get_artifact("dialogue_only") or audio_path), seg.get(
                            "start", 0.0), seg.get("end", seg.get("start", 0.0) + 0.5), str(seg_orig))
                    except Exception as e:
                        log.debug("Failed to extract seg %d: %s", idx, e)

            # schedule parallel separation for uncached segments
            if self.use_per_segment_separation and demucs_separate:
                sep_out_root = out_dir / "seg_separated"
                sep_out_root.mkdir(parents=True, exist_ok=True)
                to_process = []
                for idx, seg in enumerate(segments):
                    seg_orig = tmp_seg_dir / \
                        f"{base_stem}_seg_{idx:04d}_orig.wav"
                    if not seg_orig.exists():
                        continue
                    checksum = compute_sha256(str(seg_orig))
                    cache_entry = self.separation_cache.get(str(seg_orig))
                    if cache_entry and cache_entry.get("checksum") == checksum:
                        continue
                    to_process.append((idx, seg_orig, sep_out_root, checksum))

                if to_process:
                    workers = min(8, max(1, (os.cpu_count() or 4)//2))
                    log.info("Scheduling %d parallel segment separations (workers=%d)", len(
                        to_process), workers)
                    futures = []
                    with ThreadPoolExecutor(max_workers=workers) as ex:
                        for task in to_process:
                            futures.append(
                                ex.submit(self._parallel_separate_worker, task))
                        for fut in tqdm(as_completed(futures), total=len(futures), desc="Demucs segmentation"):
                            idx, sep_info, err = fut.result()
                            seg_orig = tmp_seg_dir / \
                                f"{base_stem}_seg_{idx:04d}_orig.wav"
                            if sep_info:
                                self.separation_cache[str(seg_orig)] = sep_info
                                log.info("[seg %d] separation cached", idx)
                            else:
                                log.warning(
                                    "[seg %d] separation failed: %s", idx, err)
        except Exception as e:
            log.exception("Parallel per-segment separation step failed: %s", e)

        # -------------------------
        # STAGE 6: TTS (synthesize -> cleaning -> align -> visemes -> collect synthesized_segments)
        # -------------------------
        synthesized_segments: List[Dict] = []
        if not self._stage_done("tts"):
            try:
                bar = tqdm(total=len(segments),
                           desc="Synthesizing voices", unit="seg")
            except Exception:
                bar = None

            prev = self.ckpt.get_progress(
                "tts") or {"current": 0} if self.ckpt else {"current": 0}
            start_index = int(prev.get("current", 0))

            # prosody mapper
            try:
                prosody_mapper = ProsodyMapper() if ProsodyMapper else None
            except Exception:
                prosody_mapper = None

            # parse script to tts_items
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
                        "duration": seg.get("end", seg.get("start", 0.0)) - seg.get("start", 0.0),
                        "prosody": {"rate": 1.0, "pitch_semitones": 0.0, "gain_db": 0.0},
                        "voice": None,
                        "speaker": seg.get("speaker"),
                        "emotion": None,
                    })

            total_segments = len(tts_items)
            tmp_root = out_dir / "tmp_diadub"
            if tmp_root.exists():
                shutil.rmtree(tmp_root)
            tmp_root.mkdir(parents=True, exist_ok=True)

            # iterate TTS items
            for i in range(start_index, total_segments):
                item = tts_items[i]
                text_to_speak = item.get("text", "").strip()
                if not text_to_speak:
                    if bar:
                        bar.update(1)
                    if self.ckpt:
                        self.ckpt.set_progress("tts", i+1, total_segments)
                    continue

                # segment original & cleaned path selection
                seg_orig = self.temp_dir / f"{base_stem}_seg_{i:04d}_orig.wav"
                clean_seg_path = seg_orig
                cache_entry = self.separation_cache.get(str(seg_orig))
                if cache_entry and cache_entry.get("vocals") and Path(cache_entry["vocals"]).exists():
                    # verify checksum
                    current_chk = compute_sha256(str(seg_orig))
                    if cache_entry.get("checksum") == current_chk:
                        clean_seg_path = Path(cache_entry["vocals"])
                        log.debug(
                            "[seg %d] using separated vocals: %s", i, clean_seg_path)
                    else:
                        log.debug(
                            "[seg %d] checksum mismatch - will re-separate on-demand", i)
                else:
                    # try to separate on-demand
                    if demucs_separate:
                        try:
                            sep_out = out_dir / \
                                "seg_separated" / f"seg_{i:04d}"
                            sep_out.mkdir(parents=True, exist_ok=True)
                            sep_info = demucs_separate(
                                str(seg_orig), str(sep_out), stems="vocals")
                            if sep_info and sep_info.get("vocals"):
                                clean_seg_path = Path(sep_info["vocals"])
                                self.separation_cache[str(seg_orig)] = {"vocals": str(
                                    clean_seg_path), "checksum": compute_sha256(str(seg_orig)), "meta": sep_info.get("meta", {})}
                        except Exception as e:
                            log.warning(
                                "[seg %d] on-demand separation failed: %s", i, e)

                # analyze for prosody
                try:
                    audio_meta = analyze_segment(str(clean_seg_path))
                except Exception:
                    audio_meta = {"duration": item.get(
                        "duration", item.get("end", 0.0) - item.get("start", 0.0))}

                # map prosody if mapper available
                prosody_params = {}
                if prosody_mapper:
                    try:
                        prosody_params = prosody_mapper.map(audio_meta)
                        script_prosody = item.get("prosody", {})
                        for k in ("rate", "pitch_semitones", "gain_db"):
                            if k in script_prosody and script_prosody[k] is not None:
                                prosody_params[k] = float(script_prosody[k])
                    except Exception as e:
                        log.debug("Prosody mapping failed: %s", e)
                        prosody_params = {}

                # Prepare tmp filenames
                fname_raw = f"{base_stem}_seg_{i:04d}_tts_raw.wav"
                fname_pitch = f"{base_stem}_seg_{i:04d}_tts_pitch.wav"
                fname_adjusted = f"{base_stem}_seg_{i:04d}_adjusted.wav"
                wav_path_raw = self.temp_dir / fname_raw
                wav_path_pitch = self.temp_dir / fname_pitch
                wav_path_adjusted = self.temp_dir / fname_adjusted

                # TTS scheduling via queue manager and TTSEngineManager
                text_to_synthesize = item.get("text_polished") or text_to_speak
                tts_hint = item.get("tts_hint") or {"voice": item.get(
                    "voice") or item.get("speaker"), "prosody": prosody_params}

                # worker wrapper calls TTSEngineManager.synthesize_text
                def _tts_worker(text, hint, out_dir_local):
                    # TTSEngineManager.synthesize_text returns a path to wav
                    return self.tts_engine.synthesize_text(text=text, hint=hint, out_dir=str(out_dir_local))

                try:
                    if self.queue_mgr:
                        fut = self.queue_mgr.submit_gpu(
                            _tts_worker, text_to_synthesize, tts_hint, tmp_root, gpu_index=0)
                        tts_res = fut.result()
                    else:
                        tts_res = _tts_worker(
                            text_to_synthesize, tts_hint, tmp_root)
                except Exception as e:
                    log.exception("TTS synth error for item %d: %s", i, e)
                    if bar:
                        bar.update(1)
                    if self.ckpt:
                        self.ckpt.set_progress("tts", i+1, total_segments)
                    continue

                # normalize tts_res into wav_path_raw
                try:
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
                        import numpy as _np
                        import soundfile as sf
                        if hasattr(tts_res, "numpy"):
                            arr = tts_res.numpy()
                            sf.write(str(wav_path_raw), arr, 48000)
                        elif isinstance(tts_res, _np.ndarray):
                            sf.write(str(wav_path_raw), tts_res, 48000)
                        else:
                            raise RuntimeError("Unsupported TTS return type")
                except Exception as e:
                    log.exception(
                        "Failed to write TTS output for item %d: %s", i, e)
                    if bar:
                        bar.update(1)
                    if self.ckpt:
                        self.ckpt.set_progress("tts", i+1, total_segments)
                    continue

                # optional post pitch shift (if prosody demands)
                try:
                    pitch_steps = float(prosody_params.get(
                        "pitch_semitones", 0.0)) if prosody_params else 0.0
                except Exception:
                    pitch_steps = 0.0
                base_for_alignment = str(wav_path_raw)
                if abs(pitch_steps) > 0.01:
                    try:
                        if post_tts_pitch_shift:
                            post_tts_pitch_shift(str(wav_path_raw), str(
                                wav_path_pitch), pitch_steps)
                            base_for_alignment = str(wav_path_pitch)
                        else:
                            log.debug("post_tts_pitch_shift not available")
                    except Exception as e:
                        log.debug("Post pitch shift failed: %s", e)
                        base_for_alignment = str(wav_path_raw)

                # Alignment (Prefer MFA -> align_tts_to_target -> naive)
                word_entries = []
                try:
                    if run_mfa_align:
                        try:
                            tmp_lab = tmp_root / f"{base_stem}_seg_{i:04d}.lab"
                            tmp_lab.write_text(
                                text_to_synthesize.strip(), encoding="utf-8")
                            mfa_out = tmp_root / "mfa" / f"seg_{i:04d}"
                            mfa_out.mkdir(parents=True, exist_ok=True)
                            word_entries = run_mfa_align(
                                base_for_alignment, str(tmp_lab), str(mfa_out)) or []
                        except Exception as me:
                            log.debug("MFA failed seg %d: %s", i, me)
                            word_entries = []

                    if not word_entries and align_tts_to_target:
                        try:
                            tts_words = item.get("tts_words") or []
                            if tts_words:
                                word_entries = align_tts_to_target(
                                    tts_audio_path=base_for_alignment,
                                    tts_words=tts_words,
                                    target_words=[w["word"]
                                                  for w in tts_words],
                                    out_path=str(wav_path_adjusted),
                                    options={
                                        "max_stretch": 1.5, "min_stretch": 0.7, "crossfade_ms": 14},
                                ) or []
                                base_for_alignment = str(wav_path_adjusted)
                        except Exception as ae:
                            log.debug(
                                "align_tts_to_target failed seg %d: %s", i, ae)
                            word_entries = []

                    if not word_entries:
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
                    log.debug("Alignment exception seg %d: %s", i, e)
                    word_entries = []

                # Viseme mapping
                try:
                    if generate_viseme_timing and word_entries:
                        vdat = generate_viseme_timing(word_entries)
                        ln["phonemes"] = vdat.get("phonemes")
                        ln["visemes"] = vdat.get("visemes")
                except Exception as e:
                    log.debug("Viseme mapping failed seg %d: %s", i, e)

                # finalize synthesized_segments
                synthesized_segments.append({
                    "wav": base_for_alignment,
                    "start": item.get("start"),
                    "end": item.get("end"),
                    "speaker": item.get("speaker"),
                    "index": item.get("index"),
                    "words": word_entries,
                })

                try:
                    if self.ckpt:
                        self.ckpt.set_artifact(
                            "synthesized_segments", synthesized_segments)
                        self.ckpt.set_progress("tts", i+1, total_segments)
                except Exception:
                    pass

                if bar:
                    bar.update(1)

            if bar:
                bar.close()

            # update script JSON with aligned_tts_path and visemes
            try:
                s_obj = json.loads(script_path.read_text(
                    encoding="utf-8")) if script_path.exists() else None
                if s_obj and isinstance(s_obj.get("lines"), list):
                    for seg in synthesized_segments:
                        idx = int(seg.get("index", -1))
                        if 0 <= idx < len(s_obj.get("lines")):
                            s_obj["lines"][idx]["aligned_tts_path"] = seg.get(
                                "wav")
                            if "phonemes" in s_obj["lines"][idx]:
                                pass
                    final_script_path = out_dir / \
                        f"{base_stem}_script_with_audio.json"
                    final_script_path.write_text(json.dumps(
                        s_obj, indent=2, ensure_ascii=False), encoding="utf-8")
                    if self.ckpt:
                        self.ckpt.set_artifact(
                            "script_with_audio", str(final_script_path))
            except Exception:
                pass

            if self.ckpt:
                self.ckpt.set_artifact(
                    "synthesized_segments", synthesized_segments)
                self.ckpt.set_stage_done(
                    "tts", {"synthesized_segments": synthesized_segments})
            if progress:
                progress.update(1)
        else:
            synthesized_segments = self.ckpt.get_artifact(
                "synthesized_segments", []) if self.ckpt else []
            if progress:
                progress.update(0)

        # -------------------------
        # STAGE 7: assemble_dialogue
        # -------------------------
        dialogue_path = out_dir / f"{base_stem}_dialogue.wav"
        if not self._stage_done("assemble_dialogue"):
            if synthesized_segments and AudioSegment is not None:
                try:
                    self._assemble_and_export_dialogue(str(self.ckpt.get_artifact(
                        "audio_path") or audio_path), synthesized_segments, str(dialogue_path))
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

        # -------------------------
        # STAGE 8: mix (ducking)
        # -------------------------
        mixed_path = out_dir / f"{base_stem}_dialogue_mixed.wav"
        if not self._stage_done("mix"):
            try:
                from diadub.mixing.mixer import Mixer
                mixer = Mixer()
                if dialogue_path.exists():
                    mixer.mix(str(self.ckpt.get_artifact("audio_path") or audio_path), str(
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
                        _ffmpeg_sidechain_duck(str(self.ckpt.get_artifact(
                            "audio_path") or audio_path), str(tts_mix), str(ducked_base))
                        log.info("Applied sidechain ducking to base audio")
                    except Exception as e:
                        log.warning("Sidechain duck failed or unsupported: %s. Falling back to static reduction (gain=%sdB)", e, float(
                            duck_amount_db))
                        _ffmpeg_reduce_base_gain(str(self.ckpt.get_artifact(
                            "audio_path") or audio_path), str(ducked_base), gain_db=float(duck_amount_db))

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

        if progress:
            progress.close()

        result["artifacts"] = self.ckpt.to_dict().get(
            "artifacts", {}) if self.ckpt else {}
        return result

    # ---------------------
    # Helper methods below
    # ---------------------
    def _extract_segment(self, full_audio: str, start_s: float, end_s: float, out_wav: str):
        # robust extraction using ffmpeg
        extract_audio_segment(full_audio, out_wav, start_s, end_s)

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

    # worker for parallel separation (module-level helper could be used too)
    def _parallel_separate_worker(self, args):
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
