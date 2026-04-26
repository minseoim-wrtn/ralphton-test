"""Seed model list loader for the HuggingFace Model Monitor.

Loads a curated list of major existing HuggingFace models from
``config/seed_models.yaml``.  These are pre-populated into the ModelStore
on first run so they appear as "known" baseline entries and don't trigger
false new-model alerts.

Usage:
    from hf_model_monitor.seed_models import load_seed_models, get_all_seed_repo_ids

    # Full structured data by category
    categories = load_seed_models()

    # Flat list of repo IDs (for store pre-population)
    repo_ids = get_all_seed_repo_ids()
"""

import logging
import os
from dataclasses import dataclass, field

import yaml

from .config import SEED_MODELS_PATH

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SeedModel:
    """A single seed model entry."""

    repo_id: str
    name: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        # Default name to the model part of the repo ID
        if not self.name and "/" in self.repo_id:
            self.name = self.repo_id.split("/", 1)[1]

    @property
    def author(self) -> str:
        """Extract the org/author from the repo ID."""
        return self.repo_id.split("/")[0] if "/" in self.repo_id else ""

    def to_dict(self) -> dict:
        """Convert to a plain dict (suitable for ModelStore.upsert_models)."""
        return {
            "model_id": self.repo_id,
            "author": self.author,
            "name": self.name,
            "notes": self.notes,
        }


@dataclass
class SeedCategory:
    """A category grouping of seed models."""

    key: str
    label: str = ""
    description: str = ""
    models: list[SeedModel] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.models)

    @property
    def repo_ids(self) -> list[str]:
        return [m.repo_id for m in self.models]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate_repo_id(repo_id: str) -> bool:
    """Check that a repo ID looks like 'org/model-name'."""
    if not isinstance(repo_id, str) or not repo_id.strip():
        return False
    parts = repo_id.strip().split("/")
    return len(parts) == 2 and all(p.strip() for p in parts)


def _parse_model_entry(entry: dict) -> SeedModel | None:
    """Parse a single model entry from the YAML data.

    Returns None if the entry is invalid (missing or bad repo_id).
    """
    if not isinstance(entry, dict):
        return None
    repo_id = entry.get("repo_id", "")
    if not _validate_repo_id(repo_id):
        logger.warning("Skipping seed model with invalid repo_id: %r", repo_id)
        return None
    return SeedModel(
        repo_id=repo_id.strip(),
        name=str(entry.get("name", "")).strip(),
        notes=str(entry.get("notes", "")).strip(),
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_seed_models(path: str | None = None) -> list[SeedCategory]:
    """Load seed models from the YAML configuration file.

    Args:
        path: Optional path override (defaults to ``config/seed_models.yaml``).

    Returns:
        List of SeedCategory objects, each containing its models.
        Returns an empty list if the file is missing or unparseable.
    """
    filepath = path or SEED_MODELS_PATH

    if not os.path.exists(filepath):
        logger.info("Seed models file not found at %s", filepath)
        return []

    try:
        with open(filepath) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError:
        logger.exception("Failed to parse seed models file %s", filepath)
        return []
    except OSError:
        logger.exception("Failed to read seed models file %s", filepath)
        return []

    if not isinstance(data, dict):
        logger.warning("Seed models file has invalid structure (expected dict)")
        return []

    raw_categories = data.get("categories")
    if not isinstance(raw_categories, dict):
        logger.warning("Seed models file missing 'categories' dict")
        return []

    categories: list[SeedCategory] = []
    seen_repo_ids: set[str] = set()

    for cat_key, cat_data in raw_categories.items():
        if not isinstance(cat_data, dict):
            logger.warning("Skipping invalid category: %s", cat_key)
            continue

        models: list[SeedModel] = []
        for entry in cat_data.get("models", []):
            model = _parse_model_entry(entry)
            if model is None:
                continue
            # Deduplicate across categories (first occurrence wins)
            if model.repo_id in seen_repo_ids:
                logger.debug(
                    "Duplicate seed model %s in category %s, skipping",
                    model.repo_id,
                    cat_key,
                )
                continue
            seen_repo_ids.add(model.repo_id)
            models.append(model)

        category = SeedCategory(
            key=cat_key,
            label=str(cat_data.get("label", cat_key)).strip(),
            description=str(cat_data.get("description", "")).strip(),
            models=models,
        )
        categories.append(category)

    total = sum(c.count for c in categories)
    logger.info(
        "Loaded %d seed models across %d categories from %s",
        total,
        len(categories),
        filepath,
    )
    return categories


def get_all_seed_repo_ids(path: str | None = None) -> list[str]:
    """Return a flat, deduplicated list of all seed model repo IDs.

    Convenience function for store pre-population. Order follows the
    YAML file (categories top-to-bottom, models top-to-bottom within each).
    """
    categories = load_seed_models(path)
    return [
        model.repo_id
        for category in categories
        for model in category.models
    ]


def get_seed_models_by_category(
    path: str | None = None,
) -> dict[str, list[dict]]:
    """Return seed models grouped by category key as plain dicts.

    Useful for dashboard display and API responses.

    Returns:
        Dict mapping category key → list of model dicts, each with keys:
        repo_id, name, notes, author, category, category_label.
    """
    categories = load_seed_models(path)
    result: dict[str, list[dict]] = {}
    for cat in categories:
        result[cat.key] = []
        for model in cat.models:
            d = model.to_dict()
            d["category"] = cat.key
            d["category_label"] = cat.label
            result[cat.key].append(d)
    return result


def get_category_labels(path: str | None = None) -> dict[str, str]:
    """Return a mapping of category key → human-readable label."""
    categories = load_seed_models(path)
    return {c.key: c.label for c in categories}
