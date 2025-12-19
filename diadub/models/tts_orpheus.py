import logging
from pathlib import Path
from typing import Dict, Any

log = logging.getLogger("diadub.tts.orpheus")


class OrpheusTTS:
    def __init__(self, model_path="unsloth/orpheus-3b-0.1-ft-UD-Q8_K_XL.gguf"):
        self.model_path = model_path
        log.info(f"[Orpheus] loading GGUF model: {self.model_path}")

        from llama_cpp import Llama
        self.model = Llama(
            model_path=self.model_path,
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=20
        )

        log.info("[Orpheus] loaded OK.")

    def synthesize(self, text: str, tts_hint: Dict[str, Any], out_path: str) -> str:
        prompt = (
            f"<s> [VOICE: {tts_hint.get('voice', 'default')}] "
            f"[GAIN: {tts_hint.get('gain_db', 0)}] "
            f"[PITCH: {tts_hint.get('pitch_semitones', 0)}] "
            f"[RATE: {tts_hint.get('rate', 1.0)}] "
            f"\n{text}\n</s>"
        )

        result = self.model(prompt, max_tokens=300)
        pcm = result["audio"]

        import soundfile as sf
        sf.write(out_path, pcm, 24000)

        return out_path
