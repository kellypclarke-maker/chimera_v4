from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .paths import META_ROOT


def default_config_path() -> Path:
    return META_ROOT / "config" / "defaults.json"


def load_config(path: Path | None = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path is not None else default_config_path()
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"config must be an object: {cfg_path}")
    return data

