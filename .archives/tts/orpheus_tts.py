from pathlib import Path
import logging
import tempfile
import gc
from typing import Optional, Dict, Any

from diadub.models.base_model import BaseModel

log = logging.getLogger("diadub.models.tts.orpheus")

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


class OrpheusTTS(BaseModel):
    def __init__(self, model_id: str = "EQ4You/Orpheus-TTS", device: str = "cuda"):
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
        log.info("Loading Orpheus TTS %s on %s", self.model_id, target_device)

        last_err = None
        try:
            device_flag = 0 if target_device == "cuda" else -1
            self.pipe = pipeline(
                "text-to-speech", model=self.model_id, device=device_flag
            )
            self.loaded = True
            return
        except Exception as e:
            last_err = e
            log.warning("Primary TTS load failed: %s", e)
            self._clear_cuda()

        if prefer_8bit and _HAS_BNB:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                # heuristic: attempt load_in_8bit with device_map
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
                log.warning("8-bit TTS load failed: %s", e)
                self._clear_cuda()

        # final fallback: CPU
        try:
            self.pipe = pipeline(
                "text-to-speech", model=self.model_id, device=-1)
            self.loaded = True
            return
        except Exception as e:
            last_err = e
            log.exception("CPU TTS load failed: %s", e)
            raise RuntimeError(
                f"Failed to load OrpheusTTS model {self.model_id}: {last_err}"
            )

    def synth_text(
        self,
        text: str,
        voice_id: Optional[str] = None,
        prosody_params: Optional[Dict] = None,
    ) -> Any:
        """
        Attempt to synthesize. Returns either path to wav file or bytes depending on pipeline output.
        Many transformers TTS pipelines return dict with 'audio' = np.ndarray or bytes.
        We write to a temp WAV and return the path for pipeline compatibility.
        """
        if not self.loaded:
            self.load()

        try:
            kwargs = {}
            # pass prosody hints if pipeline supports; many do not and will ignore
            if prosody_params:
                kwargs.update({"prosody": prosody_params})
            out = self.pipe(text, **kwargs)
            # out typically a dict or list of dicts
            # unified handling
            tmp = Path(tempfile.mktemp(suffix=".wav"))
            # If pipeline returns audio array
            if isinstance(out, dict) and "audio" in out:
                audio = out["audio"]
                # audio might be raw bytes or ndarray
                if isinstance(audio, (bytes, bytearray)):
                    tmp.write_bytes(audio)
                    return str(tmp)
                else:
                    # try soundfile to write
                    try:
                        import soundfile as sf

                        sf.write(str(tmp), audio, 48000)
                        return str(tmp)
                    except Exception:
                        # fallback: write bytes if available
                        if hasattr(audio, "tobytes"):
                            tmp.write_bytes(audio.tobytes())
                            return str(tmp)
                        else:
                            # last resort: return raw object
                            return audio
            # sometimes pipeline returns list
            if (
                isinstance(out, list) and
                len(out) > 0 and
                isinstance(out[0], dict) and
                "audio" in out[0]
            ):
                audio = out[0]["audio"]
                try:
                    import soundfile as sf

                    sf.write(str(tmp), audio, 48000)
                    return str(tmp)
                except Exception:
                    if isinstance(audio, (bytes, bytearray)):
                        tmp.write_bytes(audio)
                        return str(tmp)
                    return audio
            # if pipeline returned bytes directly
            if isinstance(out, (bytes, bytearray)):
                tmp.write_bytes(out)
                return str(tmp)
            # otherwise return whatever pipeline returned
            return out
        except RuntimeError as e:
            # handle OOM similarly
            if torch is not None and ("out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError)):
                log.warning(
                    "CUDA OOM during TTS inference; unloading and retrying on CPU")
                self.unload()
                self.load(prefer_8bit=False)
                out = self.pipe(text)
                # attempt same writing logic
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
                if isinstance(out, (bytes, bytearray)):
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
