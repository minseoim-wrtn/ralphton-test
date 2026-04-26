"""State management for the HuggingFace Model Monitor.

Provides both the legacy JSON-based API (for backward compatibility) and a
new SQLite-backed ModelStore for richer tracking with timestamps.

Legacy functions (still used by main.py and tests):
    - load_previous_models()
    - save_current_models()
    - detect_new_models()

New store-based workflow:
    - get_store()  → shared ModelStore instance
    - detect_new_models_with_store()  → upsert + detect in one step
"""

import json
import logging
import os

from .config import SEEN_MODELS_PATH
from .model_store import ModelStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared ModelStore singleton
# ---------------------------------------------------------------------------
_store: ModelStore | None = None


def get_store(db_path: str | None = None) -> ModelStore:
    """Return a module-level ModelStore singleton (lazy-initialized).

    Args:
        db_path: Optional path to the SQLite database.  Only used on the
                 first call; subsequent calls return the same instance.
    """
    global _store
    if _store is None:
        _store = ModelStore(db_path)
    return _store


def reset_store() -> None:
    """Close and discard the shared store (useful for testing)."""
    global _store
    if _store is not None:
        _store.close()
        _store = None


# ---------------------------------------------------------------------------
# New store-backed detection
# ---------------------------------------------------------------------------
def detect_new_models_with_store(
    current_models: list[dict],
    db_path: str | None = None,
) -> list[dict]:
    """Upsert *current_models* into the SQLite store and return newly seen ones.

    This is the recommended replacement for the legacy three-step workflow
    (load → detect → save).  It handles everything in one call:
    1. Opens/creates the SQLite store
    2. Inserts new models with first_seen_at timestamps
    3. Updates last_checked_at for known models
    4. Returns only the models that were not previously seen

    Args:
        current_models: List of model dicts (must have 'model_id' key).
        db_path: Optional SQLite database path override.

    Returns:
        List of model dicts that are newly detected (first time seen).
    """
    store = get_store(db_path)
    return store.upsert_models(current_models)


# ---------------------------------------------------------------------------
# Legacy JSON-based API (backward compatible)
# ---------------------------------------------------------------------------
def load_previous_models(path: str | None = None) -> list[str]:
    """Load previously seen model IDs from a JSON file.

    Returns an empty list if the file doesn't exist or is corrupted.
    """
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
    """Save current model IDs to a JSON file."""
    path = path or SEEN_MODELS_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(model_ids, f, indent=2)


def detect_new_models(
    current_models: list[dict], previous_model_ids: list[str]
) -> list[dict]:
    """Compare current models against a list of previously seen IDs.

    This is the legacy in-memory comparison.  For timestamp-aware detection,
    use detect_new_models_with_store() instead.
    """
    previous_set = set(previous_model_ids)
    return [m for m in current_models if m["model_id"] not in previous_set]
