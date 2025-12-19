import logging
from typing import List
from transformers import MarianMTModel, MarianTokenizer

log = logging.getLogger("diadub.translation")


class HFTranslator:
    """
    Open-source Hugging Face translation wrapper (MarianMT).
    Loads any model_id like 'Helsinki-NLP/opus-mt-xx-en'.
    """

    def __init__(self, model_id="Helsinki-NLP/opus-mt-mul-en", device="cuda"):
        self.model_id = model_id
        log.info(f"Loading translation model: {model_id}")
        self.tokenizer = MarianTokenizer.from_pretrained(model_id)
        self.model = MarianMTModel.from_pretrained(model_id)
        self.device = device
        if device == "cuda":
            self.model = self.model.to("cuda")

    def translate_text(self, text: str) -> str:
        """Translate a single string."""
        batch = self.tokenizer([text], return_tensors="pt", padding=True)
        if self.device == "cuda":
            batch = {k: v.to("cuda") for k, v in batch.items()}
        translated = self.model.generate(**batch, max_new_tokens=512)
        result = self.tokenizer.decode(translated[0], skip_special_tokens=True)
        return result

    def batch_translate(self, texts: List[str]) -> List[str]:
        """Translate multiple strings in a batch."""
        outputs = []
        for t in texts:
            outputs.append(self.translate_text(t))
        return outputs


def load_model(spec: dict, device="cuda"):
    """Factory used by ModelRegistry."""
    model_id = spec.get("model_id", "Helsinki-NLP/opus-mt-mul-en")
    return HFTranslator(model_id=model_id, device=device)
