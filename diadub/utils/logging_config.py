import logging
import logging.config
import yaml
from pathlib import Path


def setup_logging(config_path="configs/logging.yaml", default_level=logging.INFO):
    """Configure logging from YAML file."""
    path = Path(config_path)
    if path.exists():
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
