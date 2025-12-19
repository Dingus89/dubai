"""
ModelRegistry -- loads models from config and dynamically imports plugin classes.
"""

import importlib
import json
from pathlib import Path


class ModelRegistry:
    def __init__(self, config_path="models.json", device="cuda"):
        self.config_path = Path(config_path)
        self.device = device
        self.models = self._load_config()

    def _load_config(self):
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Model config not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get(self, name):
        entry = self.models.get(name)
        if not entry:
            raise KeyError(f"Model {name} not found in config")
        module_name, class_name = entry["import"].rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls(device=self.device)

    def list_models(self):
        return self.models
