import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

log = logging.getLogger("diadub.tts.manager")

# Required packages: transformers, accelerate, torch, soundfile (or fallback)
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSpeechSeq2Seq
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForSpeechSeq2Seq = None

# optional accelerate helpers
try:
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
except Exception:
    init_empty_weights = None
    load_checkpoint_and_dispatch = None

# optional llama-cpp fallback
try:
    from llama_cpp import Llama
except Exception:
    Llama = None


class TTSEngineManager:
    def __init__(self, vibe_name: str = "microsoft/VibeVoice-1.5B", orpheus_path: Optional[str] = None, cache_dir: str = "data/cache/models"):
        self.vibe_name = vibe_name
        self.orpheus_path = Path(orpheus_path) if orpheus_path else None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vibe = None
        self.vibe_tokenizer = None
        self.vibe_model = None
        self.orpheus = None

    def _ensure_vibe(self):
        if self.vibe_model is not None:
            return
        if AutoTokenizer is None or AutoModelForSpeechSeq2Seq is None:
            log.warning(
                "Transformers/torch not available - cannot load VibeVoice")
            return
        try:
            log.info("Loading VibeVoice tokenizer and model (deferred dispatch)...")
            # basic load of tokenizer & model (attempt low_cpu_mem_usage)
            self.vibe_tokenizer = AutoTokenizer.from_pretrained(
                self.vibe_name, use_fast=True)
            self.vibe_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.vibe_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
            # attempt to dispatch model to device_map auto with offload folder if accelerate available
            if load_checkpoint_and_dispatch:
                try:
                    # Note: load_checkpoint_and_dispatch expects a checkpoint or model object and a dispatch map.
                    # Using a naive call; this may be adjusted for your environment.
                    load_checkpoint_and_dispatch(self.vibe_model, {"": str(
                        self.cache_dir)}, device_map="auto", offload_folder=str(self.cache_dir))
                except Exception as e:
                    log.debug(
                        "Accelerate dispatch attempt failed, proceeding with default model instance: %s", e)
            log.info("VibeVoice model loaded (may be in CPU / offloaded state).")
        except Exception as e:
            log.exception("Failed to load VibeVoice model: %s", e)
            self.vibe_model = None
            self.vibe_tokenizer = None

    def _ensure_orpheus(self):
        if self.orpheus is not None:
            return
        if self.orpheus_path and Llama:
            try:
                log.info("Loading Orpheus (llama-cpp) fallback from %s",
                         self.orpheus_path)
                self.orpheus = Llama(model_path=str(
                    self.orpheus_path), n_ctx=2048, n_threads=max(1, (os.cpu_count() or 2)-1))
            except Exception as e:
                log.exception("Failed to init Orpheus fallback: %s", e)
                self.orpheus = None

    def synthesize_text(self, text: str, hint: Optional[Dict[str, Any]] = None, out_dir: str = ".") -> str:
        """
        Synthesize text and return path to WAV file.
        This method first tries VibeVoice (with safe offload), and falls back to Orpheus/CPU.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"tts_{abs(hash(text)) % 100000}.wav"

        # Try VibeVoice
        self._ensure_vibe()
        if self.vibe_model is not None and self.vibe_tokenizer is not None:
            try:
                # tokenization
                inputs = self.vibe_tokenizer(text, return_tensors="pt")
                # move inputs to model device if possible
                try:
                    device = next(self.vibe_model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                except Exception:
                    pass
                with torch.no_grad():
                    res = self.vibe_model.generate(
                        **inputs, do_sample=True, top_p=0.95, temperature=0.7)
                # res might be audio tensor or similar - attempt to write it
                try:
                    import soundfile as sf
                    arr = res.cpu().numpy().flatten()
                    sf.write(str(out_path), arr, 24000)
                    log.info("Wrote VibeVoice output to %s", out_path)
                    return str(out_path)
                except Exception as e:
                    log.debug(
                        "Failed to write VibeVoice output as array: %s", e)
                    # attempt to decode differently if model returns bytes
                    if isinstance(res, (bytes, bytearray)):
                        out_path.write_bytes(res)
                        return str(out_path)
            except RuntimeError as oom:
                log.warning(
                    "VibeVoice runtime error (OOM?): %s. Clearing cache and falling back.", oom)
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            except Exception as e:
                log.exception("VibeVoice generation error: %s", e)

        # Fallback: Orpheus via llama-cpp
        self._ensure_orpheus()
        if self.orpheus is not None:
            try:
                # llama-cpp may provide audio in different forms; assume 'audio' key or text-to-speech wrapper
                out = self.orpheus(text, max_tokens=2048)
                if isinstance(out, dict) and "audio" in out:
                    arr = out["audio"]
                    import soundfile as sf
                    sf.write(str(out_path), arr, 24000)
                    return str(out_path)
                # else try to interpret text output (non-audio)
                if isinstance(out, str):
                    # no audio returned, write TTS fallback via simple TTS or raise
                    log.warning(
                        "Orpheus returned string instead of audio; no audio fallback available.")
                return str(out_path) if out_path.exists() else None
            except Exception as e:
                log.exception("Orpheus synth failed: %s", e)

        raise RuntimeError(
            "No TTS available: VibeVoice and Orpheus both failed or not installed.")
