import logging
from typing import List, Dict, Any, Optional

log = logging.getLogger("diadub.translation")

try:
    from transformers import pipeline
except Exception:
    pipeline = None

LANG_MODEL_MAP = {
    "en": "facebook/m2m100_418M",
    "es": "facebook/m2m100_418M",
    "fr": "facebook/m2m100_418M",
}


class TranslationManager:
    def __init__(self, model_name=None):
        self.model_name = model_name or "facebook/m2m100_418M"
        self.translator = None
        if pipeline:
            try:
                self.translator = pipeline(
                    "translation", model=self.model_name, device=-1
                )
            except Exception as e:
                log.warning("translation pipeline init failed: %s", e)

    def translate_lines(self, lines: List[Dict[str, Any]], target_lang="en"):
        out = []
        for ln in lines:
            txt = ln.get("text", "")
            translated = txt
            if self.translator:
                try:
                    r = self.translator(txt)
                    translated = r[0]["translation_text"]
                except Exception as e:
                    log.debug("translation failed %s", e)
            ln2 = dict(ln)
            ln2["translated"] = translated
            out.append(ln2)
        return out
