"""
Safe model loading helpers with GPU detection and VRAM-aware fallback strategies.

Functions:
 - has_cuda()
 - get_gpu_device_info() -> dict with total/free if available (best-effort)
 - safe_load_transformers_model(model_name, **kwargs)
 - clear_cuda_cache()
Notes:
 - This is a best-effort helper. Actual memory checks are limited by what torch/onnxruntime exposes.
"""

import os
import logging
import importlib
from typing import Optional, Any, Dict

log = logging.getLogger("diadub.models.loader")
log.setLevel(logging.INFO)


def has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def clear_cuda_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def get_gpu_device_info() -> Dict[str, Any]:
    info = {"available": False}
    try:
        import torch
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            prop = torch.cuda.get_device_properties(idx)
            # free mem estimate (best-effort): requires torch.cuda.memory_reserved / allocated
            reserved = torch.cuda.memory_reserved(idx)
            allocated = torch.cuda.memory_allocated(idx)
            free_est = max(0, prop.total_memory - (reserved + allocated))
            info.update({
                "available": True,
                "device_idx": idx,
                "name": prop.name,
                "total_memory": prop.total_memory,
                "free_estimate": free_est
            })
    except Exception:
        pass
    return info


def safe_load_transformers_model(model_name: str, model_class: str = "AutoModelForCausalLM", tokenizer_class: str = "AutoTokenizer", device_preference: str = "cuda", max_vram_gb: Optional[float] = None, **kwargs) -> Dict[str, Any]:
    """
    Attempt to load a HuggingFace Transformers model safely.
    - model_class/tokenizer_class are import paths like "AutoModelForCausalLM"
    - device_preference: "cuda" or "cpu"
    - max_vram_gb: if provided, will avoid loading to GPU if estimated device memory is below threshold
    Returns: {"model": model_or_none, "tokenizer": tokenizer_or_none, "device": device_str}
    """
    result = {"model": None, "tokenizer": None, "device": "cpu"}
    try:
        transformers = importlib.import_module("transformers")
        ModelCls = getattr(transformers, model_class, None)
        TokenizerCls = getattr(transformers, tokenizer_class, None)
        if ModelCls is None or TokenizerCls is None:
            # fallback to Auto classes
            ModelCls = transformers.AutoModel
            TokenizerCls = transformers.AutoTokenizer
    except Exception:
        log.warning("transformers not available; cannot load model")
        return result

    use_cuda = False
    if device_preference == "cuda" and has_cuda():
        info = get_gpu_device_info()
        if info.get("available"):
            free_bytes = info.get("free_estimate", 0)
            if max_vram_gb is not None:
                threshold = int(max_vram_gb * 1024**3)
                if free_bytes < threshold:
                    log.info(
                        "Not enough estimated free VRAM (need %s bytes) to load on GPU; falling back to CPU", threshold)
                    use_cuda = False
                else:
                    use_cuda = True
            else:
                use_cuda = True

    device = "cuda" if use_cuda else "cpu"
    result["device"] = device

    # attempt to load with device_map or low_cpu_mem_usage
    try:
        # prefer the high-level from_pretrained with device_map if available
        load_kwargs = dict(**kwargs)
        # use low_cpu_mem_usage when available
        if "low_cpu_mem_usage" in getattr(ModelCls, "from_pretrained", lambda *a, **k: None).__code__.co_varnames:
            load_kwargs["low_cpu_mem_usage"] = True
        if device == "cuda":
            # device_map="auto" may require accelerate; try to use it
            try:
                model = ModelCls.from_pretrained(
                    model_name, device_map="auto", **load_kwargs)
                tokenizer = TokenizerCls.from_pretrained(model_name)
            except Exception:
                # fallback: load to cpu then move
                model = ModelCls.from_pretrained(model_name, **load_kwargs)
                tokenizer = TokenizerCls.from_pretrained(model_name)
                try:
                    import torch
                    model.to("cuda")
                except Exception:
                    log.warning(
                        "Failed to move model to cuda after loading; keeping on CPU")
        else:
            model = ModelCls.from_pretrained(model_name, **load_kwargs)
            tokenizer = TokenizerCls.from_pretrained(model_name)
        result["model"] = model
        result["tokenizer"] = tokenizer
        log.info("Loaded model %s on %s", model_name, device)
        return result
    except Exception as e:
        log.exception("safe_load_transformers_model failed: %s", e)
        clear_cuda_cache()
        return result
