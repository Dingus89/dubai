from pathlib import Path
import logging
import tempfile
import gc
from typing import Optional, Dict, Any

from diadub.models.base_model import BaseModel

log = logging.getLogger("diadub.models.tts.vibevoice")

try:
    import torch
    from transformers import pipeline
except Exception:
    torch = None

_HAS_BNB = False
try:
    import bitsandbytes as bnb  # noqa: F401

    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

_HAS_ACCELERATE = False
try:
    import accelerate  # noqa: F401

    _HAS_ACCELERATE = True
except Exception:
    _HAS_ACCELERATE = False


class VibeVoiceTTS(BaseModel):
    def __init__(
        self, model_id: str = "microsoft/VibeVoice-1.5B", device: str = "cuda"
    ):
        super().__init__(device=device)
        self.model_id = model_id
        self.pipe = None
        self.loaded = False

    def load(self, prefer_8bit: bool = True, **kwargs):
        if self.loaded:
            return
        if torch is None:
            raise RuntimeError("torch/transformers not available")

        target_device = "cuda" if (self.device.type == "cuda") else "cpu"
        log.info("Loading VibeVoice TTS %s on %s",
                 self.model_id, target_device)

        last_err = None
        # attempt best-effort load sequence
        try:
            # prefer device_map if accelerate is available
            if _HAS_ACCELERATE:
                log.info("Using device_map='auto' (accelerate available)")
                self.pipe = pipeline(
                    "text-to-speech", model=self.model_id, device_map="auto"
                )
            else:
                device_flag = 0 if target_device == "cuda" else -1
                self.pipe = pipeline(
                    "text-to-speech", model=self.model_id, device=device_flag
                )
            self.loaded = True
            return
        except Exception as e:
            last_err = e
            log.warning("Primary VibeVoice load failed: %s", e)
            self._clear_cuda()

        # attempt 8-bit with bitsandbytes if available
        if prefer_8bit and _HAS_BNB:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, device_map="auto", load_in_8bit=True
                )
                tok = AutoTokenizer.from_pretrained(self.model_id)
                self.pipe = pipeline(
                    "text-to-speech", model=model, tokenizer=tok, device_map="auto"
                )
                self.loaded = True
                return
            except Exception as e:
                last_err = e
                log.warning("8-bit VibeVoice load failed: %s", e)
                self._clear_cuda()

        # final fallback: CPU
        try:
            self.pipe = pipeline(
                "text-to-speech", model=self.model_id, device=-1)
            self.loaded = True
            return
        except Exception as e:
            last_err = e
            log.exception("CPU VibeVoice load failed: %s", e)
            raise RuntimeError(
                f"Failed to load VibeVoice model {self.model_id}: {last_err}"
            )

    def synth_text(
        self,
        text: str,
        voice_id: Optional[str] = None,
        prosody_params: Optional[Dict] = None,
    ):
        if not self.loaded:
            self.load()

        try:
            kwargs = {}
            if prosody_params:
                kwargs.update({"prosody": prosody_params})
            out = self.pipe(text, **kwargs)
            tmp = Path(tempfile.mktemp(suffix=".wav"))
            if isinstance(out, dict) and "audio" in out:
                audio = out["audio"]
                try:
                    import soundfile as sf

                    sf.write(str(tmp), audio, 48000)
                    return str(tmp)
                except Exception:
                    if isinstance(audio, (bytes, bytearray)):
                        tmp.write_bytes(audio)
                        return str(tmp)
                    return audio
            if isinstance(out, bytes):
                tmp.write_bytes(out)
                return str(tmp)
            return out
        except RuntimeError as e:
            if (torch is not None and
                ("out of memory" in str(e).lower() or
                 isinstance(e, torch.cuda.OutOfMemoryError))):
                log.warning(
                    "CUDA OOM during VibeVoice inference; unloading and retrying on CPU"
                )
                self.unload()
                self.load(prefer_8bit=False)
                out = self.pipe(text)
                tmp = Path(tempfile.mktemp(suffix=".wav"))
                if isinstance(out, dict) and "audio" in out:
                    audio = out["audio"]
                    try:
                        import soundfile as sf
                        sf.write(str(tmp), audio, 48000)
                        return str(tmp)
                    except Exception:
                        if isinstance(audio, (bytes, bytearray)):
                            tmp.write_bytes(audio)
                            return str(tmp)
                        return audio
                if isinstance(out, bytes):
                    tmp.write_bytes(out)
                    return str(tmp)
                return out
            raise

    def unload(self):
        try:
            if self.pipe:
                del self.pipe
                self.pipe = None
        finally:
            self.loaded = False
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
