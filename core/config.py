"""Simple persistent user config (last-used folder paths, etc.)."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent.parent / "_config"
_CONFIG_FILE = _CONFIG_DIR / "user_config.json"


def _load() -> dict:
    try:
        if _CONFIG_FILE.exists():
            return json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug(f"Config load failed: {e}")
    return {}


def _save(data: dict):
    try:
        _CONFIG_DIR.mkdir(exist_ok=True)
        _CONFIG_FILE.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.debug(f"Config save failed: {e}")


def get_folder(key: str, default: str = "") -> str:
    return _load().get(f"folder_{key}", default)


def set_folder(key: str, path: str):
    data = _load()
    data[f"folder_{key}"] = path
    _save(data)
