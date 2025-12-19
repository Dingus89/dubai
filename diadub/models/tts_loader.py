"""
Scaffold wrappers for loading TTS engines safely.
Provide small helpers to load VibeVoice and Orpheus models in a VRAM-conscious way.
This file intentionally contains wrappers and strategies, not full model implementations.
"""

import logging
from .model_loader import safe_load_transformers_model, has_cuda, get_gpu_device_info, clear_cuda_cache

log = logging.getLogger("diadub.models.tts_loader")


def load_vibevoice(model_name: str = "microsoft/VibeVoice-1.5B", prefer_cuda: bool = True, max_vram_gb: float = 8.0):
    """
    Try to load VibeVoice checkpoint safely. If fails, return a placeholder object that signals to use remote/inference or CPU.
    """
    device_pref = "cuda" if prefer_cuda and has_cuda() else "cpu"
    res = safe_load_transformers_model(model_name, model_class="AutoModelForCausalLM",
                                       tokenizer_class="AutoTokenizer", device_preference=device_pref, max_vram_gb=max_vram_gb)
    if res["model"] is None:
        log.warning(
            "VibeVoice model failed to load locally; falling back to CPU/inference mode")
    return res


def load_orpheus(model_path: str = "unsloth/orpheus-3b-0.1-ft-UD-Q8_K_XL.gguf", prefer_cuda: bool = False, max_vram_gb: float = 8.0):
    """
    Orpheus quantized models (gguf) are often loaded with GGML runners (e.g. llama.cpp, ggml-based libs).
    This function is a stub showing strategy:
      - prefer local gguf runner (ggml) on CPU for quantized models
      - if you have GPU runner for ggml, adapt accordingly
    """
    # NOTE: actual implementation depends on which runner you will use (ggml/gguf loaders).
    return {"loader": "gguf_stub", "path": model_path, "device": "cpu"}
