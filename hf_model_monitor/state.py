import json
import logging
import os

from .config import SEEN_MODELS_PATH

logger = logging.getLogger(__name__)


def load_previous_models(path: str | None = None) -> list[str]:
    path = path or SEEN_MODELS_PATH
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.exception("Failed to load seen models from %s", path)
        return []


def save_current_models(model_ids: list[str], path: str | None = None) -> None:
    path = path or SEEN_MODELS_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(model_ids, f, indent=2)


def detect_new_models(
    current_models: list[dict], previous_model_ids: list[str]
) -> list[dict]:
    previous_set = set(previous_model_ids)
    return [m for m in current_models if m["model_id"] not in previous_set]
