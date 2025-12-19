"""
diadub/models/base_model.py
Defines abstract base classes and safe loading utilities.
"""

import gc
import torch
from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, device="cuda", dtype=torch.float16):
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.model = None

    @abstractmethod
    def load(self):
        """Load model weights / tokenizer / processor."""
        pass

    def unload(self):
        """Unload from memory and clear cache."""
        if self.model:
            del self.model
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @abstractmethod
    def infer(self, *args, **kwargs):
        pass

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
