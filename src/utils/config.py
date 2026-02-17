"""Config loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG_PATH = Path("config/config.yaml")


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
