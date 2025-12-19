# script_generator.py
from pathlib import Path
import json
import os
import logging
import subprocess
from typing import List, Dict, Optional, Any
from datetime import datetime
import tempfile

# numeric / audio
import numpy as np

# Try optional imports that may exist in your environment
try:
    from diadub.audio_analysis.audio_features import (
        analyze_audio_features,
        analyze_segment,
    )
except Exception:
    # Fallback minimal analyzer placeholder
    def analyze_audio_features(path: str) -> Dict[str, Any]:
        logging.getLogger("diadub.script.script_generator").warning(
            "analyze_audio_features not available; returning defaults for %s", path
        )
        return {"rate": 1.0, "loudness_db": -20.0, "pitch_shift": 0.0}

    analyze_segment = None

try:
    from diadub.utils.media import extract_audio_segment, get_audio_duration
except Exception:
    # fallback helper implementations using ffmpeg if available
    import shlex

    def extract_audio_segment(src: str, dst: str, start: float, end: float):
        """
        Extract a segment using ffmpeg. If ffmpeg not available, raise.
        """
        logger = logging.getLogger("diadub.script.script_generator")
        src = str(src)
        dst = str(dst)
        duration = max(0, float(end) - float(start))
        cmd = f"ffmpeg -y -i {shlex.quote(src)} -ss {float(start):.3f} -t {duration:.3f} -ac 1 -ar 16000 {shlex.quote(dst)}"
        logger.debug("Running extract_audio_segment: %s", cmd)
        ret = subprocess.run(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if ret.returncode != 0:
            logger.error(
                "ffmpeg extract failed: %s", ret.stderr.decode(errors="ignore")
            )
            raise RuntimeError("extract_audio_segment failed")

    def get_audio_duration(src: str) -> float:
        """Attempt to probe duration with ffprobe if available."""
        import shlex
        import subprocess
        import re

        src = str(src)
        cmd = f"ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 {shlex.quote(src)}"
        p = subprocess.run(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if p.returncode == 0:
            try:
                return float(p.stdout.decode().strip())
            except Exception:
                return 0.0
        return 0.0


try:
    from diadub.models.translation import translate_text
except Exception:

    def translate_text(text: str, translation_model, language: str) -> str:
        # fallback no-op
        logging.getLogger("diadub.script.script_generator").warning(
            "translate_text not available, returning original text"
        )
        return text


try:
    from diadub.models.emotion_model import EmotionModel
except Exception:
    EmotionModel = None

try:
    from diadub.models.registry import ModelRegistry
except Exception:
    ModelRegistry = None

# New imports for speaker matching & persistent assignment (optional)
try:
    from scripts.match_speakers import match_speaker, load_reference_embeddings
except Exception:
    match_speaker = None
    load_reference_embeddings = None

try:
    from scripts.assign_voice_profiles import assign_persistent_voice
except Exception:

    def assign_persistent_voice(
        speaker_name: str, suggested: Optional[str] = None
    ) -> str:
        """
        Very small fallback mapping: consistently map speaker_name to a persistent id.
        In real usage you probably want a persistent store (file/db).
        """
        # Use a simple deterministic mapping
        return f"voice_{abs(hash(speaker_name)) % 10000}"


log = logging.getLogger("diadub.script.script_generator")


# ------------------------
# Small utilities
# ------------------------


def read_srt(path: Path) -> List[Dict[str, Any]]:
    """
    Read a simple SRT file and return a list of segments with start, end, text.
    This is intentionally robust against slightly malformed SRTs.
    """
    text = path.read_text(encoding="utf-8")
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    out = []
    for p in parts:
        lines = [l for l in p.splitlines() if l.strip()]
        if len(lines) < 2:
            continue
        # find timestamp line
        ts_line = next((l for l in lines if "-->" in l), None)
        if not ts_line:
            continue
        start_txt, end_txt = [t.strip() for t in ts_line.split("-->")[:2]]

        def parse_time(s: str) -> float:
            # formats: hh:mm:ss,ms  or mm:ss,ms
            s = s.replace(",", ".").strip()
            parts = s.split(":")
            if len(parts) == 3:
                h, m, sec = parts
                secf = float(sec)
                return int(h) * 3600 + int(m) * 60 + secf
            elif len(parts) == 2:
                m, sec = parts
                return int(m) * 60 + float(sec)
            else:
                try:
                    return float(s)
                except:
                    return 0.0

        start = parse_time(start_txt)
        end = parse_time(end_txt)
        # text lines are the rest (excluding index & timestamp)
        text_lines = [
            l for l in lines if "-->" not in l and not l.strip().isdigit()]
        text_join = "\n".join(text_lines).strip()
        out.append(
            {
                "start": float(start),
                "end": float(end),
                "text": text_join,
                "speaker": None,
            }
        )
    return out


def write_script(out_path: Path, script_obj: Any):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(script_obj, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Wrote script to %s", out_path)


# ------------------------
# Audio helper utilities to support missing logic
# ------------------------


def compute_mel(
    wav_path: str, n_mels: int = 80, sample_rate: int = 16000
) -> np.ndarray:
    """
    Compute a mel spectrogram (magnitude) for the audio at wav_path.
    This implementation uses torchaudio if available, otherwise numpy/ffmpeg fallback.
    Returns numpy array (n_mels, frames).
    """
    logger = logging.getLogger("diadub.script.script_generator")
    try:
        import torchaudio

        waveform, sr = torchaudio.load(wav_path)
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, sample_rate)
        waveform = waveform.mean(dim=0, keepdim=True)  # mono
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=256, n_mels=n_mels
        )(waveform)
        mel_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        return mel_db.squeeze(0).numpy()
    except Exception as e:
        logger.warning(
            "torchaudio mel computation failed (%s), returning zeros", e)
        return np.zeros((n_mels, 1), dtype=np.float32)


def wrap_tts_to_target(text: str, tts_hint: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrap text and tts hint into a common structure expected by downstream TTS.
    This is intentionally generic: adjust for a concrete TTS engine as needed.
    """
    data = {
        "text": text,
        "voice": tts_hint.get("voice"),
        "rate": float(tts_hint.get("rate", 1.0)),
        "gain_db": float(tts_hint.get("gain_db", -20.0)),
        "pitch_semitones": float(tts_hint.get("pitch_semitones", 0.0)),
        "meta": {
            k: v
            for k, v in tts_hint.items()
            if k not in {"voice", "rate", "gain_db", "pitch_semitones"}
        },
    }
    return data


# ------------------------
# Main script generation
# ------------------------


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
) -> str:
    """
    Generate a dubbing script JSON with timing, features, TTS hints, speaker mapping, emotion, and optional Groq hook.
    Returns path to generated script JSON as string.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_path = str(audio_path)
    base_name = str(base_name)
    log.info("🎬 Generating script for: %s (audio: %s)", base_name, audio_path)

    # Step 1: obtain segments via SRT or ASR
    segments = []
    if srt_path and Path(srt_path).exists():
        log.info("📜 Using SRT file: %s", srt_path)
        segments = read_srt(Path(srt_path))
    elif asr_segments:
        # expect list of dicts with start,end,text
        segments = [
            {
                "start": float(s.get("start", 0)),
                "end": float(s.get("end", 0)),
                "text": s.get("text", "").strip(),
                "speaker": s.get("speaker"),
            }
            for s in asr_segments
        ]
    else:
        log.info(
            "🗣️ No SRT provided and no ASR segments passed — attempting automatic STT using diadub.models.asr_model.ASRModel if available"
        )
        try:
            from diadub.models.asr_model import ASRModel

            asr = ASRModel()
            segments = asr.transcribe(audio_path)
        except Exception as e:
            log.warning("ASR transcription not available: %s", e)
            # fallback: create single segment for entire audio
            dur = get_audio_duration(audio_path) or 0.0
            segments = [{"start": 0.0, "end": dur,
                         "text": "", "speaker": "spk_0"}]

    if not segments:
        raise RuntimeError("No segments found during script generation")

    # Preload speaker reference embeddings if available
    speaker_model = speaker_processor = ref_embs = None
    try:
        if load_reference_embeddings:
            speaker_model, speaker_processor, ref_embs = load_reference_embeddings()
            log.info(
                "🔊 Speaker reference embeddings loaded (%d)",
                len(ref_embs) if ref_embs else 0,
            )
    except Exception as e:
        log.warning("⚠️ Failed to load speaker reference embeddings: %s", e)
        speaker_model = speaker_processor = ref_embs = None

    # Build the script structure
    script = {
        "meta": {
            "source": str(audio_path),
            "base_name": base_name,
            "generated": datetime.utcnow().isoformat(),
        },
        "lines": [],
    }

    log.info("🧠 Processing %d segments...", len(segments))
    for i, seg in enumerate(segments):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = str(seg.get("text", "")).strip()
        if not text:
            # skip empty text segments (but still include as silence segments optionally)
            continue

        speaker_name = seg.get("speaker") or f"spk_{i}"
        seg_path = out_dir / f"{base_name}_seg_{i:04d}_orig.wav"

        # extract audio for the segment
        try:
            extract_audio_segment(audio_path, str(seg_path), start, end)
        except Exception as e:
            log.warning("⚠️ Failed to extract audio segment %d: %s", i, e)
            # continue without audio features
            seg_path = None

        # audio feature extraction
        features = {}
        try:
            if seg_path:
                features = analyze_audio_features(str(seg_path))
        except Exception as e:
            log.warning(
                "Audio feature analysis failed for segment %d: %s", i, e)
            features = {}

        # emotion detection
        emotion_label = None
        if emotion_model:
            try:
                # emotion_model could be an instance or class providing infer()
                if hasattr(emotion_model, "infer"):
                    emo_res = emotion_model.infer(
                        str(seg_path)) if seg_path else None
                elif callable(emotion_model):
                    emo_res = emotion_model(str(seg_path))
                else:
                    emo_res = None
                if isinstance(emo_res, list) and len(emo_res) > 0:
                    emotion_label = emo_res[0].get("label")
                elif isinstance(emo_res, dict):
                    emotion_label = emo_res.get("label")
            except Exception as e:
                log.warning(
                    "Emotion detection failed for segment %d: %s", i, e)

        # speaker matching and persistent voice assignment
        voice_id = None
        match_score = None
        try:
            if speaker_model and ref_embs and match_speaker:
                match_voice, sim = match_speaker(
                    str(seg_path), speaker_model, speaker_processor, ref_embs
                )
                voice_id = assign_persistent_voice(speaker_name, match_voice)
                match_score = float(sim)
            else:
                # no model or matching code; just assign persistent voice id
                voice_id = assign_persistent_voice(speaker_name)
        except Exception as e:
            log.warning("Voice matching failed for segment %d: %s", i, e)
            voice_id = assign_persistent_voice(speaker_name)

        # translation if requested
        if translation_model and language and language != "en":
            try:
                text = translate_text(text, translation_model, language)
            except Exception as e:
                log.warning("Translation failed for segment %d: %s", i, e)

        # Compose tts hint from features and matching
        tts_hint = {
            "voice": voice_id,
            "rate": float(features.get("rate", 1.0)),
            "gain_db": float(features.get("loudness_db", -20.0)),
            "pitch_semitones": float(features.get("pitch_shift", 0.0)),
            "match_score": match_score,
        }

        # Some heuristics: normalize gain_db into reasonable range
        try:
            loud = features.get("loudness_db")
            if loud is not None:
                # convert to a gain hint in [-6, +6]
                tts_hint["gain_db"] = max(-6.0,
                                          min(6.0, (-16.0 - float(loud)) * 0.3))
        except Exception:
            pass

        # wrap into final per-line structure
        line = {
            "index": i,
            "start": float(start),
            "end": float(end),
            "speaker": speaker_name,
            "text": text,
            "emotion": emotion_label,
            "tts_hint": tts_hint,
            "features": features,
        }

        # optionally compute mel preview if requested in features (lightweight)
        if seg_path and features.get("want_mel_preview"):
            try:
                line["mel_preview_shape"] = compute_mel(str(seg_path)).shape
            except Exception:
                pass

        script["lines"].append(line)

    # optional Groq payload writing for later enhancement via Groq
    if groq_use:
        key = groq_api_key or os.getenv("GROQ_API_KEY")
        groq_payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "source": str(audio_path),
            "base_name": base_name,
            "language": language,
            "segments": script["lines"],
            "meta": script["meta"],
        }
        # If an API key is provided, write a hook file for external use
        try:
            hook_path = Path(out_dir) / f"{base_name}_groq_hook.json"
            hook_path.write_text(
                json.dumps(groq_payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            log.info("💾 Groq payload saved for enhancement: %s", hook_path)
        except Exception as e:
            log.warning("Failed to write groq payload: %s", e)

    # final write
    out_path = Path(out_dir) / f"{base_name}_script.json"
    write_script(out_path, script)

    log.info("✅ Script generation complete: %s", out_path)
    return str(out_path)
