from pathlib import Path
import json
import os
import logging
import subprocess
from datetime import datetime
 from typing import List, Dict, Optional, Any

  log = logging.getLogger("diadub.script.script_generator")

   # Try to import optional helpers used elsewhere in the project.
   # Keep names unchanged so existing pipeline integrations still call the same functions.
   try:
        # audio analysis helper; expected to expose analyze_segment or analyze_audio_features
        from diadub.audio_analysis.audio_features import analyze_segment as analyze_audio_features
    except Exception:
        # fallback stub
        def analyze_audio_features(_wav_path: str) -> Dict[str, Any]:
            return {"duration": None, "loudness_db": None, "mean_f0": None, "pitch_median": None}

    try:
        # helper to extract a segment from a larger wav (ffmpeg wrapper)
        from diadub.utils.media import extract_audio_segment
    except Exception:
        def extract_audio_segment(full_audio: str, out_path: str, start_s: float, end_s: float):
            """
            Minimal ffmpeg-based extractor fallback.
            """
            cmd = [
                "ffmpeg", "-y", "-i", str(full_audio),
                "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}",
                "-ar", "48000", "-ac", "1", "-vn", str(out_path)
            ]
            subprocess.run(
                cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ASR fallback loader name kept if pipeline expects it
    try:
        from diadub.models.asr_model import ASRModel  #  optional project ASR wrapper
    except Exception:
        ASRModel = None

    # Translation helper (if you have a translation wrapper)
    try:
        from diadub.models.translation import translate_text
    except Exception:
        def translate_text(text: str, model=None, target_lang: str = "en") -> str:
            # noop fallback
            return text

    # Emotion model wrapper (optional)
    try:
        from diadub.models.emotion_model import EmotionModel
    except Exception:
        EmotionModel = None

    # Speaker matching & persistent assignment (scripts supplied in repo)
    try:
        from scripts.match_speakers import load_reference_embeddings, match_speaker
        from scripts.assign_voice_profiles import assign_persistent_voice
    except Exception:
        load_reference_embeddings = None
        match_speaker = None
        assign_persistent_voice = None

    # Local polishing (Flan-T5) optional
    _FLAN_AVAILABLE = False
    try:
        from transformers import pipeline as hf_pipeline
        # We'll lazily load the Flan pipeline when needed (to avoid upfront memory usage)
        _FLAN_AVAILABLE = True
    except Exception:
        _FLAN_AVAILABLE = False

    def read_srt(srt_path: Path) -> List[Dict[str, Any]]:
        """
        Very small SRT parser. Returns list of segments with start/end (seconds) and text.
        Keeps SRT parsing logic simple to avoid external deps.
        """
        text = srt_path.read_text(encoding="utf-8", errors="ignore")
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        out = []
        for p in parts:
            lines = [l for l in p.splitlines() if l.strip()]
            if not lines:
                continue
            # find timestamp line
            ts_line = None
            for l in lines:
                if "-->" in l:
                    ts_line = l
                    break
            if not ts_line:
                # fallback: skip block
                continue
            try:
                start_txt, end_txt = [t.strip() for t in ts_line.split("-->")]

                def parse_time(s: str) -> float:
                    # format "HH:MM:SS,ms"
                    h, m, rest = s.split(":")
                    sec, ms = rest.split(",")
                    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000.0
                start = parse_time(start_txt)
                end = parse_time(end_txt)
            except Exception:
                start, end = 0.0, 0.0
            # text lines after timestamp
            idx = lines.index(ts_line)
            body = " ".join(lines[idx+1:]).strip()
            out.append({"start": start, "end": end,
                       "text": body, "speaker": None})
        return out

    def write_script_json(out_path: Path, data: Dict[str, Any]):
        out_path.write_text(json.dumps(
            data, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("Wrote script JSON: %s", out_path)

    def _flan_polish_script(script_lines: List[Dict[str, Any]], model_name: str = "google/flan-t5-large") -> List[Dict[str, Any]]:
        """
        Attempt to polish script lines using an instruction-tuned Flan model locally.
        This function is optional and guarded. It returns a new list of lines with 'text_polished' when successful.
        Keep prompts small to avoid large memory use.
        """
        if not _FLAN_AVAILABLE:
            log.debug("Flan not available locally; skipping polish.")
            return script_lines

        try:
            # Lazily instantiate generator
            gen = hf_pipeline("text2text-generation",
                              model=model_name, device=0 if _has_cuda() else -1)
        except Exception as e:
            log.warning("Failed to load local Flan pipeline: %s", e)
            return script_lines

        polished = []
        for ln in script_lines:
            try:
                prompt = (
                    "Rewrite the following subtitle line into a concise, natural movie-dialogue line "
                    "keeping the meaning and emotion. Return only the rewritten line.\n\n"
                    f"Line: {ln.get('text', '')}\nEmotion: {ln.get('emotion', 'neutral')}\n"
                )
                res = gen(prompt, max_length=256)
                if isinstance(res, list) and res and "generated_text" in res[0]:
                    new_text = res[0]["generated_text"].strip()
                elif isinstance(res, dict) and "generated_text" in res:
                    new_text = res["generated_text"].strip()
                else:
                    new_text = ln.get("text", "")
                ln = dict(ln)
                ln["text_polished"] = new_text
            except Exception as e:
                log.debug("Flan polish failed for line %s: %s",
                          ln.get("index"), e)
            polished.append(ln)
        return polished

    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def generate_script(
        out_dir: str,
        base_name: str,
        audio_path: str,
        srt_path: Optional[str] = None,
        asr_segments: Optional[List[Dict[str, Any]]] = None,
        emotion_model: Optional[Any] = None,
        translation_model: Optional[Any] = None,
        language: str = "en",
        use_local_polish: bool = True,
        groq_use: bool = False,
        groq_api_key: Optional[str] = None,
    ) -> str:
        """
        Primary function that creates a detailed script JSON.

        Parameters:
            out_dir: directory to write outputs
            base_name: base filename used for outputs
            audio_path: path to full audio (wav)
            srt_path: optional srt path (preferred)
            asr_segments: fallback segments from ASR (list of dicts with start/end/text)
            emotion_model: optional emotion model instance exposing infer(audio_path) -> list/dict
            translation_model: optional translation model or model name for translate_text
            language: target language code (for translation)
            use_local_polish: whether to attempt local Flan polishing
            groq_use: whether to write groq hook payload if local polish not used
            groq_api_key: optional groq api key (not used for direct call here)
        Returns:
            path to the generated script json (string)
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        audio_path = str(audio_path)

        # Gather base segments (SRT preferred)
        segments = []
        if srt_path and Path(srt_path).exists():
            log.info("Using SRT as primary source: %s", srt_path)
            segments = read_srt(Path(srt_path))
        elif asr_segments:
            log.info("Using provided ASR segments as fallback")
            # Normalize expected keys
            for i, s in enumerate(asr_segments):
                start = float(s.get("start", 0.0))
                end = float(s.get("end", start + 0.5))
                text = s.get("text", s.get("text", "")).strip()
                segments.append({"start": start, "end": end,
                                "text": text, "speaker": s.get("speaker")})
        else:
            # Try to run a local ASRModel wrapper if available
            if ASRModel is not None:
                try:
                    asr = ASRModel()
                    log.info("No SRT provided; running ASRModel.transcribe()")
                    raw = asr.transcribe(audio_path)
                    # Expect transcribe to return list of dicts with start, end, text
                    for i, seg in enumerate(raw):
                        start = float(seg.get("start", 0.0))
                        end = float(seg.get("end", start + 0.5))
                        text = seg.get("text", "").strip()
                        segments.append(
                            {"start": start, "end": end, "text": text, "speaker": seg.get("speaker")})
                except Exception as e:
                    log.exception("ASRModel transcribe failed: %s", e)
            else:
                raise RuntimeError(
                    "No SRT, no ASR segments provided, and no ASRModel available.")

        if not segments:
            raise RuntimeError("No segments available to build script")

        # Preload speaker references if available
        speaker_model = speaker_processor = ref_embs = None
        if load_reference_embeddings is not None and match_speaker is not None and assign_persistent_voice is not None:
            try:
                speaker_model, speaker_processor, ref_embs = load_reference_embeddings()
                log.info("Loaded %d speaker reference embeddings",
                         len(ref_embs) if ref_embs else 0)
            except Exception as e:
                log.warning("Failed to load reference embeddings: %s", e)
                speaker_model = speaker_processor = ref_embs = None
        else:
            log.debug(
                "Voice matching modules unavailable; skipping speaker-match step")

        # Build script lines
        script_lines: List[Dict[str, Any]] = []
        for idx, seg in enumerate(segments):
            start = float(seg.get("start", 0.0))
            end = float(
                seg.get("end", max(start + 0.01, seg.get("end", start + 0.5))))
            if end <= start:
                end = start + 0.5
            text = seg.get("text", "").strip()
            speaker = seg.get("speaker") or f"spk_{idx}"

            # Extract a small clip for analysis and matching
            seg_wav = out_dir / f"{base_name}_seg_{idx:04d}_orig.wav"
            try:
                # our extract helper signature: (full_audio, out_path, start, end)
                extract_audio_segment(audio_path, str(seg_wav), start, end)
            except Exception as e:
                log.debug("Segment extraction failed for idx %d: %s", idx, e)
                # fallback: write nothing, but continue
                seg_wav = None

            # Audio analysis
            features = {}
            try:
                if seg_wav and seg_wav.exists():
                    features = analyze_audio_features(str(seg_wav))
                else:
                    features = {"duration": end - start}
            except Exception as e:
                log.debug("Audio analysis error for idx %d: %s", idx, e)
                features = {"duration": end - start}

            # Emotion detection
            emotion_label = None
            try:
                if emotion_model and seg_wav:
                    emo_res = emotion_model.infer(str(seg_wav))
                    if isinstance(emo_res, list) and len(emo_res) > 0:
                        emotion_label = emo_res[0].get(
                            "label") or emo_res[0].get("emotion")
                    elif isinstance(emo_res, dict):
                        emotion_label = emo_res.get(
                            "label") or emo_res.get("emotion")
            except Exception as e:
                log.debug("Emotion inference failed for idx %d: %s", idx, e)

            # Speaker / voice matching & persistent assignment
            chosen_voice = None
            match_score = None
            try:
                if speaker_model and ref_embs and seg_wav:
                    mv, sim = match_speaker(
                        str(seg_wav), speaker_model, speaker_processor, ref_embs)
                    try:
                        chosen_voice = assign_persistent_voice(speaker, mv)
                    except Exception:
                        # if persistent mapping unavailable, fall back to mv
                        chosen_voice = mv
                    match_score = float(sim)
                else:
                    # if no matching system, still ensure speaker has a persistent voice
                    if assign_persistent_voice is not None:
                        chosen_voice = assign_persistent_voice(speaker)
            except Exception as e:
                log.debug("Voice match/assign failed for idx %d: %s", idx, e)
                # best-effort fallback
                try:
                    if assign_persistent_voice is not None:
                        chosen_voice = assign_persistent_voice(speaker)
                except Exception:
                    chosen_voice = None

            # Translation (if translation_model present and language mismatch)
            translated_text = text
            if translation_model and language and language != "en":
                try:
                    translated_text = translate_text(
                        text, translation_model, language)
                except Exception as e:
                    log.debug("Translation failed for idx %d: %s", idx, e)
                    translated_text = text

            # Compose TTS hints (voice + prosody)
            tts_hint = {
                "voice": chosen_voice,
                "rate": float(features.get("speech_rate", 1.0)) if features.get("speech_rate") is not None else 1.0,
                "pitch_semitones": float(features.get("pitch_shift", 0.0)) if features.get("pitch_shift") is not None else 0.0,
                "gain_db": float(features.get("loudness_db", -20.0)) if features.get("loudness_db") is not None else -20.0,
                "match_score": match_score
            }

            line = {
                "index": idx,
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": text,
                "translated": translated_text,
                "emotion": emotion_label,
                "analysis": features,
                "tts_hint": tts_hint
            }

            script_lines.append(line)

        # Optionally attempt local polishing of lines using Flan-T5
        polished_lines = script_lines
        if use_local_polish and _FLAN_AVAILABLE:
            try:
                polished_lines = _flan_polish_script(script_lines)
                log.info("Local Flan polish completed for script lines")
            except Exception as e:
                log.warning(
                    "Local Flan polish failed: %s (will keep original lines)", e)
                polished_lines = script_lines

        # If local polish was not used and groq_use is requested, write groq hook payload
        if not use_local_polish and groq_use:
            key = groq_api_key or os.environ.get("GROQ_API_KEY")
            if key:
                try:
                    groq_path = Path(out_dir) / f"{base_name}_groq_hook.json"
                    groq_payload = {
                        "metadata": {"base_name": base_name, "created_at": datetime.utcnow().isoformat() + "Z"},
                        "segments": script_lines
                    }
                    groq_path.write_text(json.dumps(
                        groq_payload, indent=2, ensure_ascii=False), encoding="utf-8")
                    log.info("Wrote Groq hook payload: %s", groq_path)
                except Exception as e:
                    log.warning("Failed to write groq hook payload: %s", e)
            else:
                log.warning(
                    "groq_use requested but GROQ_API_KEY not present; skipping Groq hook")

        # Final output object
        script_obj = {
            "metadata": {
                "base_name": base_name,
                "audio_path": audio_path,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "polished_local": use_local_polish and _FLAN_AVAILABLE,
                "groq_hook_written": groq_use and (os.environ.get("GROQ_API_KEY") is not None)
            },
            "lines": polished_lines
        }

        out_path = Path(out_dir) / f"{base_name}_script.json"
        write_script_json(out_path, script_obj)

        return str(out_path)

