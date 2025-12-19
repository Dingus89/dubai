# Script Generator
from pathlib import Path
import json
import os
import logging
import subprocess
from typing import List, Dict, Optional, Any
from datetime import datetime
import tempfile
import numpy as np

# ---------------------------------------------------------------------------
# Optional imports with robust fallbacks
# ---------------------------------------------------------------------------
log = logging.getLogger("diadub.script.script_generator")

# Audio analysis
try:
    from diadub.audio_analysis.audio_features import (
        analyze_audio_features as _analyze_audio_features,
        analyze_segment,
    )

    def analyze_audio_features(path: str) -> Dict[str, Any]:
        return _analyze_audio_features(path)
except Exception:
    def analyze_audio_features(_wav_path: str) -> Dict[str, Any]:
        return {
            "rate": 1.0,
            "loudness_db": -20.0,
            "pitch_shift": 0.0,
            "duration": None,
            "mean_f0": None,
            "pitch_median": None
        }
    analyze_segment = None

# Audio media operations
try:
    from diadub.utils.media import extract_audio_segment, get_audio_duration, load_audio, save_audio
except Exception:
    import shlex

    def extract_audio_segment(src: str, dst: str, start: float, end: float):
        logger = logging.getLogger("diadub.script.script_generator")
        src = str(src)
        dst = str(dst)
        duration = max(0, float(end) - float(start))
        cmd = f"ffmpeg -y -i {shlex.quote(src)} -ss {float(start):.3f} -t {duration:.3f} -ac 1 -ar 16000 {shlex.quote(dst)}"
        ret = subprocess.run(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if ret.returncode != 0:
            logger.error("ffmpeg extract failed: %s",
                         ret.stderr.decode(errors="ignore"))
            raise RuntimeError("extract_audio_segment failed")

    def get_audio_duration(src: str) -> float:
        import subprocess
        src = str(src)
        cmd = f"ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 '{src}'"
        p = subprocess.run(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode == 0:
            try:
                return float(p.stdout.decode().strip())
            except Exception:
                return 0.0
        return 0.0

# Translation
try:
    from diadub.models.translation import translate_text
except Exception:
    def translate_text(text: str, translation_model, language: str) -> str:
        return text

# Emotion model
try:
    from diadub.models.emotion_model import EmotionModel
except Exception:
    EmotionModel = None

# ASR model
try:
    from diadub.models.asr_model import ASRModel
except Exception:
    ASRModel = None

# Speaker matching
try:
    from scripts.match_speakers import match_speaker, load_reference_embeddings
except Exception:
    match_speaker = None
    load_reference_embeddings = None

# Persistent voice assignment
try:
    from scripts.assign_voice_profiles import assign_persistent_voice
except Exception:
    def assign_persistent_voice(speaker_name: str, suggested: Optional[str] = None) -> str:
        return f"voice_{abs(hash(speaker_name)) % 10000}"

# FLAN local polish
try:
    from transformers import pipeline as hf_pipeline
    _FLAN_AVAILABLE = True
except Exception:
    _FLAN_AVAILABLE = False

# Alignment / viseme modules (optional - loaded lazily)
# The script will attempt to import the universal aligner and viseme mapper where needed.

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_srt(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    out = []
    for p in parts:
        lines = [l for l in p.splitlines() if l.strip()]
        if len(lines) < 2:
            continue

        ts_line = next((l for l in lines if "-->" in l), None)
        if not ts_line:
            continue

        start_txt, end_txt = [t.strip() for t in ts_line.split("-->")[:2]]

        def parse_time(s: str) -> float:
            s = s.replace(",", ".").strip()
            parts = s.split(":")
            if len(parts) == 3:
                h, m, sec = parts
                try:
                    return int(h) * 3600 + int(m) * 60 + float(sec)
                except Exception:
                    return 0.0
            elif len(parts) == 2:
                m, sec = parts
                try:
                    return int(m) * 60 + float(sec)
                except Exception:
                    return 0.0
            else:
                try:
                    return float(s)
                except:
                    return 0.0

        start = parse_time(start_txt)
        end = parse_time(end_txt)

        body = "\n".join([l for l in lines if l not in (
            ts_line,) and not l.strip().isdigit()])

        out.append({"start": start, "end": end,
                   "text": body.strip(), "speaker": None})
    return out


def write_script(path: Path, script_obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(script_obj, indent=2,
                    ensure_ascii=False), encoding="utf-8")
    log.info("Wrote script to %s", path)


def compute_mel(wav_path: str, n_mels: int = 80, sample_rate: int = 16000) -> np.ndarray:
    try:
        import torchaudio
        waveform, sr = torchaudio.load(wav_path)
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, sample_rate)
        waveform = waveform.mean(dim=0, keepdim=True)
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=256, n_mels=n_mels)(waveform)
        mel_db = torchaudio.transforms.AmplitudeToDB()(mel)
        return mel_db.squeeze(0).numpy()
    except Exception:
        return np.zeros((n_mels, 1), dtype=np.float32)


def wrap_tts_to_target(text: str, tts_hint: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a deterministic instruction object the TTS module can consume.
    This object includes text, voice id, prosody hints, and the per-word target timings
    (the TTS module should accept this instruction and produce a natural render, plus
    word-level timings if possible).
    """
    return {
        "text": text,
        "voice": tts_hint.get("voice"),
        "rate": float(tts_hint.get("rate", 1.0)),
        "gain_db": float(tts_hint.get("gain_db", -20.0)),
        "pitch_semitones": float(tts_hint.get("pitch_semitones", 0.0)),
        "meta": {k: v for k, v in tts_hint.items() if k not in {"voice", "rate", "gain_db", "pitch_semitones"}},
    }


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _flan_polish_script(lines: List[Dict[str, Any]], model_name="google/flan-t5-large"):
    if not _FLAN_AVAILABLE:
        return lines
    try:
        gen = hf_pipeline("text2text-generation",
                          model=model_name, device=0 if _has_cuda() else -1)
    except Exception:
        return lines

    out = []
    for ln in lines:
        prompt = (
            "Rewrite the following subtitle line into a concise, natural movie-dialogue line keeping the meaning and emotion.\n"
            f"Line: {ln.get('text', '')}\n"
            f"Emotion: {ln.get('emotion', 'neutral')}\n"
        )
        try:
            res = gen(prompt, max_length=256)
            if isinstance(res, list) and res and "generated_text" in res[0]:
                new_text = res[0]["generated_text"].strip()
            else:
                new_text = ln.get("text", "")
            ln = dict(ln)
            ln["text_polished"] = new_text
        except Exception:
            ln["text_polished"] = ln.get("text", "")
        out.append(ln)
    return out

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate_script(
    out_dir: str,
    base_name: str,
    audio_path: str,
    srt_path: Optional[str] = None,
    asr_segments: Optional[List[Dict[str, Any]]] = None,
    emotion_model=None,
    translation_model=None,
    language: str = "en",
    groq_use: bool = False,
    groq_api_key: Optional[str] = None,
    use_local_polish: bool = True,
) -> str:

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_path = str(audio_path)
    _emb_cache = {}

    # Acquire segments
    segments = []
    if srt_path and Path(srt_path).exists():
        segments = read_srt(Path(srt_path))

    elif asr_segments:
        for s in asr_segments:
            start = float(s.get("start", 0.0))
            end = float(s.get("end", start + 0.5))
            if end <= start:
                end = start + 0.5
            segments.append({"start": start, "end": end, "text": s.get(
                "text", ""), "speaker": s.get("speaker")})

    else:
        if ASRModel is not None:
            try:
                asr = ASRModel()
                raw = asr.transcribe(audio_path)
                for seg in raw:
                    start = float(seg.get("start", 0.0))
                    end = float(seg.get("end", start + 0.5))
                    if end <= start:
                        end = start + 0.5
                    segments.append({"start": start, "end": end, "text": seg.get(
                        "text", ""), "speaker": seg.get("speaker")})
            except Exception:
                dur = get_audio_duration(audio_path)
                segments = [{"start": 0.0, "end": dur,
                             "text": "", "speaker": "spk_0"}]
        else:
            dur = get_audio_duration(audio_path)
            segments = [{"start": 0.0, "end": dur,
                         "text": "", "speaker": "spk_0"}]

    # Load speaker embeddings
    speaker_model = speaker_processor = ref_embs = None
    if load_reference_embeddings and match_speaker:
        try:
            speaker_model, speaker_processor, ref_embs = load_reference_embeddings()
        except Exception:
            speaker_model = speaker_processor = ref_embs = None

    # Build lines
    lines = []
    for i, seg in enumerate(segments):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start + 0.5))
        if end <= start:
            end = start + 0.5

        # --- Normalize diarization labels (e.g. 'SPEAKER_00', 'spk1', '0') ---
        raw_spk = seg.get("speaker")
        if not raw_spk or raw_spk in ["", None]:
            speaker_name = f"SPEAKER_{i:02d}"
        else:
            speaker_name = str(raw_spk).replace(" ", "_").replace(":", "_")

        # Normalize patterns
        if speaker_name.lower().startswith("speaker"):
            speaker_name = "SPEAKER_" + speaker_name.split("_")[-1].zfill(2)
        elif speaker_name.lower().startswith("spk"):
            speaker_name = "SPEAKER_" + speaker_name.split("_")[-1].zfill(2)

        text = seg.get("text", "").strip()

        seg_path = out_dir / f"{base_name}_seg_{i:04d}_orig.wav"
        try:
            extract_audio_segment(audio_path, str(seg_path), start, end)
            # Save clean dialogue sample per speaker for voice cloning (VibeVoice compatible)
            try:
                clone_dir = out_dir / "voice_clones" / speaker_name
                clone_dir.mkdir(parents=True, exist_ok=True)
                clone_sample_path = clone_dir / f"sample_{i:04d}.wav"
                if seg_path and seg_path.exists():
                    # copy instead of hardlink to avoid FFmpeg writing issues
                    import shutil
                    shutil.copy2(seg_path, clone_sample_path)
            except Exception as e:
                log.warning(
                    f"Clone sample save failed for {speaker_name}: {e}")
        except Exception:
            seg_path = None

        # Audio features
        try:
            features = analyze_audio_features(
                str(seg_path)) if seg_path else {}
        except Exception:
            features = {}

        # Emotion
        emotion_label = None
        if emotion_model and seg_path:
            try:
                if hasattr(emotion_model, "infer"):
                    emo_res = emotion_model.infer(str(seg_path))
                elif callable(emotion_model):
                    emo_res = emotion_model(str(seg_path))
                else:
                    emo_res = None

                if isinstance(emo_res, list) and emo_res:
                    emotion_label = emo_res[0].get(
                        "label") or emo_res[0].get("emotion")
                elif isinstance(emo_res, dict):
                    emotion_label = emo_res.get(
                        "label") or emo_res.get("emotion")
            except Exception:
                pass

        # Speaker matching
        voice_id = None
        match_score = None
        if speaker_model and ref_embs and match_speaker and seg_path:
            try:
                mv, sim = match_speaker(
                    str(seg_path), speaker_model, speaker_processor, ref_embs)
                try:
                    voice_id = assign_persistent_voice(speaker_name, mv)
                except:
                    voice_id = mv
                match_score = float(sim)
            except Exception:
                try:
                    voice_id = assign_persistent_voice(speaker_name)
                except Exception:
                    voice_id = f"voice_{abs(hash(speaker_name)) % 10000}"
        else:
            try:
                voice_id = assign_persistent_voice(speaker_name)
            except Exception:
                voice_id = f"voice_{abs(hash(speaker_name)) % 10000}"
            # Save embedding vectors if needed for refining future speaker matching
            try:
                if speaker_model and seg_path:
                    spk_emb = match_speaker(
                        str(seg_path), speaker_model, speaker_processor, ref_embs, return_embedding=True
                    )
                    _emb_cache.setdefault(
                        speaker_name, []).append(spk_emb)
            except Exception:
                pass

        # Translation
        translated_text = text
        if translation_model and language != "en":
            try:
                translated_text = translate_text(
                    text, translation_model, language)
            except Exception:
                translated_text = text

        # --- Word-level forced alignment & per-word features ---
        try:
            # use universal wrapper to prefer MFA, fallback to whisper/naive
            from diadub.alignment.align_wrapper import align_words_universal
            from diadub.alignment.word_aligner import _naive_uniform_align
            # align_words_universal returns word times relative to the segment audio (0.0..segment_duration)
            word_entries = []
            try:
                word_entries = align_words_universal(
                    str(seg_path) if seg_path else str(audio_path),
                    text,
                    prefer="mfa",  # use MFA when available
                    mfa_model_path=None,  # optional: specify if you have a custom MFA model path
                    duration=(end - start)  # optional hint for naive aligner
                )
            except Exception as e:
                log.debug("Primary aligner failed, trying naive: %s", e)
                # naive uniform align across the segment length
                word_entries = _naive_uniform_align(
                    max(0.001, end - start), text)
        except Exception as e:
            log.debug("Word alignment step failed for idx %s: %s", i, e)
            word_entries = []

        # convert relative times to absolute timeline and enrich per-word features
        for w in word_entries:
            try:
                rel_start = float(w.get("start", 0.0))
                rel_end = float(
                    w.get("end", rel_start + w.get("duration", 0.0)))
                w["start_abs"] = round(start + rel_start, 6)
                w["end_abs"] = round(start + rel_end, 6)
                w["duration"] = round(rel_end - rel_start, 6)
            except Exception:
                w["start_abs"] = None
                w["end_abs"] = None
                w["duration"] = None

        # enrich with loudness and pitch per word if possible
        try:
            import librosa
            import soundfile as sf
            if seg_path and seg_path.exists():
                y, sr = librosa.load(str(seg_path), sr=None, mono=True)
                for w in word_entries:
                    if w.get("start") is None:
                        continue
                    s_frame = int(max(0, round(w["start"] * sr)))
                    e_frame = int(
                        min(len(y), round((w.get("end", w.get("start", 0.0)) * sr))))
                    if e_frame <= s_frame:
                        w["loudness_db"] = None
                        w["mean_f0"] = None
                    else:
                        chunk = y[s_frame:e_frame]
                        # rms -> dB
                        try:
                            rms = (np.mean(chunk**2))**0.5
                            w["loudness_db"] = 20.0 * np.log10(max(rms, 1e-12))
                        except Exception:
                            w["loudness_db"] = None
                        # mean f0 approximate
                        try:
                            f0 = librosa.yin(chunk, fmin=50, fmax=600, sr=sr)
                            f0 = f0[~np.isnan(f0)]
                            w["mean_f0"] = float(
                                np.mean(f0)) if len(f0) > 0 else None
                        except Exception:
                            w["mean_f0"] = None
        except Exception:
            # librosa not available or failed
            pass

        # Attach word-level data to the line and include summary metrics for tts_hint
        # Compute speech_rate (words per second) for this segment
        try:
            total_words = len([w for w in word_entries if w.get("word")])
            seg_dur = max(0.001, (end - start))
            speech_rate = float(total_words) / seg_dur if total_words else 0.0
        except Exception:
            speech_rate = features.get("speech_rate", 0.0)

        # Attach computed speech_rate to features
        features["word_count"] = total_words
        features["speech_rate"] = speech_rate

        # viseme mapping
        try:
            from diadub.lipsync.viseme_mapper import generate_viseme_timing
            viseme_data = generate_viseme_timing(word_entries)
            phonemes = viseme_data.get("phonemes", [])
            visemes = viseme_data.get("visemes", [])
        except Exception:
            phonemes = []
            visemes = []

        # TTS hint and tts_instruction (TTS module will consume this)
        loud = features.get("loudness_db")
        gain_db = float(loud) if loud is not None else -20.0
        try:
            gain_db = max(-12.0, min(6.0, (-16.0 - float(loud))
                          * 0.3)) if loud is not None else -20.0
        except Exception:
            pass

        tts_hint = {
            "voice": voice_id,
            "rate": float(features.get("rate", features.get("speech_rate", 1.0))),
            "gain_db": gain_db,
            "pitch_semitones": float(features.get("pitch_shift", 0.0)),
            "match_score": match_score,
        }

        # Build a TTS instruction object with per-word target timings for time-warp stage
        tts_instruction = wrap_tts_to_target(
            translated_text if translated_text else text, tts_hint)
        # Add target per-word timings (absolute) so TTS can attempt to produce word timings if capable
        # Additional VibeVoice prosody metadata
        tts_instruction["prosody"] = {
            "emotion": emotion_label or "neutral",
            "average_pitch": features.get("mean_f0"),
            "average_loudness_db": features.get("loudness_db"),
            "speech_rate": features.get("speech_rate", 1.0)
        }
        tts_instruction["target_words"] = [
            {
                "word": w.get("word"),
                "start_abs": w.get("start_abs"),
                "end_abs": w.get("end_abs"),
                "duration": w.get("duration"),
                "loudness_db": w.get("loudness_db"),
                "mean_f0": w.get("mean_f0"),
            } for w in word_entries
        ]

        # Compose final line object (note: aligned_tts_path left for pipeline to produce)
        line = {
            "index": i,
            "start": start,
            "end": end,
            "speaker": speaker_name,
            "text": text,
            "translated": translated_text,
            "emotion": emotion_label,
            "features": features,
            "words": word_entries,
            "phonemes": phonemes,
            "visemes": visemes,
            "tts_hint": tts_hint,
            "tts_instruction": tts_instruction,
            # pipeline/TTS should fill "aligned_tts_path" after synthesizing & time-warping
            "aligned_tts_path": None,
            "speaker_profile": {
                "speaker_name": speaker_name,
                "persistent_voice_id": voice_id,
                "clone_sample_dir": f"voice_clones/{speaker_name}",
                "match_score": match_score,
            }
        }

        if seg_path and features.get("want_mel_preview"):
            try:
                line["mel_preview_shape"] = compute_mel(str(seg_path)).shape
            except Exception:
                pass

        lines.append(line)

    # Local polish (polish textual content only)
    if use_local_polish and _FLAN_AVAILABLE:
        try:
            lines = _flan_polish_script(lines)
        except Exception:
            pass

    # Groq hook
    if groq_use:
        key = groq_api_key or os.getenv("GROQ_API_KEY")
        if key:
            try:
                p = out_dir / f"{base_name}_groq_hook.json"
                payload = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "base_name": base_name,
                    "language": language,
                    "segments": lines,
                }
                p.write_text(json.dumps(payload, indent=2,
                             ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass

    script = {
        "meta": {
            "source": audio_path,
            "base_name": base_name,
            "generated": datetime.utcnow().isoformat(),
            "polished_local": use_local_polish and _FLAN_AVAILABLE,
            "requires_dialogue_separation": True,
            "voice_clone_profiles_dir": "voice_clones",
            "tts_gpu_memory_safety": {
                "force_cpu_fallback": True,
                "max_vram_gb": 6
            },
        },
        "lines": lines,
    }

    out_path = Path(out_dir) / f"{base_name}_script.json"
    write_script(out_path, script)
    return str(out_path)
