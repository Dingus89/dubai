import torch
import logging
from pathlib import Path
from typing import Dict, Any

log = logging.getLogger("diadub.tts.vibevoice")


class VibeVoiceTTS:
    """
    Wrapper for Microsoft VibeVoice 1.5B
    """

    def __init__(self, model_name="microsoft/VibeVoice-1.5B", device=None):
        self.model_name = model_name
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")

        log.info(
            f"[VibeVoice] loading model '{self.model_name}' on {self.device}...")
        from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)

        log.info("[VibeVoice] loaded OK.")

    def synthesize(self, text: str, tts_hint: Dict[str, Any], out_path: str) -> str:
        voice = tts_hint.get("voice", "default")
        rate = float(tts_hint.get("rate", 1.0))
        pitch = float(tts_hint.get("pitch_semitones", 0.0))
        gain = float(tts_hint.get("gain_db", 0.0))

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            audio = self.model.generate(
                **inputs,
                do_sample=True,
                top_p=0.95,
                temperature=0.7,
            )

        audio_arr = audio.cpu().numpy().flatten()

        import soundfile as sf
        sf.write(out_path, audio_arr, 24000)

        return out_path
