"""Seed data loader for pre-populated historical model entries.

Loads structured model data from ``data/seed_data.json`` — a curated set of
well-known models (GPT-4o, Llama 3, DeepSeek-V3, Qwen, etc.) with parameters,
benchmark scores, pricing, release dates, and provider info.

This module is separate from ``seed_models.py`` which handles the simpler
repo-ID-only catalog in ``config/seed_models.yaml`` for new-model detection.
The seed data here provides *rich historical entries* for the dashboard's
comparison and archive features.

Usage:
    from hf_model_monitor.seed_data import load_seed_data, get_seed_model

    # All models
    models = load_seed_data()

    # Single model by ID
    model = get_seed_model("deepseek-ai/DeepSeek-V3-0324")

    # Filtered by category
    llms = get_seed_models_by_category("llm")

    # As reference data dict (for Slack comparison table)
    refs = get_reference_data_from_seed()
"""

import json
import logging
import os

from .config import SEED_DATA_PATH

logger = logging.getLogger(__name__)


def load_seed_data(path: str | None = None) -> list[dict]:
    """Load all seed model entries from the JSON data file.

    Args:
        path: Optional path override (defaults to ``data/seed_data.json``).

    Returns:
        List of model dicts, each with the full fixed schema.
        Returns an empty list if the file is missing or unparseable.
    """
    filepath = path or SEED_DATA_PATH

    if not os.path.exists(filepath):
        logger.info("Seed data file not found at %s", filepath)
        return []

    try:
        with open(filepath) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.exception("Failed to read seed data file %s", filepath)
        return []

    if not isinstance(data, dict):
        logger.warning("Seed data file has invalid structure (expected dict)")
        return []

    models = data.get("models", [])
    if not isinstance(models, list):
        logger.warning("Seed data 'models' field is not a list")
        return []

    # Validate each entry has at minimum a model_id
    valid = [m for m in models if isinstance(m, dict) and m.get("model_id")]
    if len(valid) < len(models):
        logger.warning(
            "Skipped %d invalid entries in seed data", len(models) - len(valid)
        )

    logger.info("Loaded %d seed data entries from %s", len(valid), filepath)
    return valid


def get_seed_model(model_id: str, path: str | None = None) -> dict | None:
    """Look up a single seed model by its model_id.

    Args:
        model_id: Full model ID (e.g. ``"deepseek-ai/DeepSeek-V3-0324"``).
        path: Optional path override for the seed data file.

    Returns:
        The model dict if found, None otherwise.
    """
    for model in load_seed_data(path):
        if model.get("model_id") == model_id:
            return model
    return None


def get_seed_models_by_category(
    category: str, path: str | None = None
) -> list[dict]:
    """Return seed models filtered by category.

    Args:
        category: Category key (e.g. ``"llm"``, ``"code"``, ``"vision"``).
        path: Optional path override for the seed data file.

    Returns:
        List of model dicts matching the category.
    """
    return [
        m for m in load_seed_data(path)
        if m.get("category") == category
    ]


def get_seed_categories(path: str | None = None) -> list[str]:
    """Return sorted list of unique categories present in seed data."""
    categories = {m.get("category") for m in load_seed_data(path)}
    categories.discard(None)
    return sorted(categories)


def get_reference_data_from_seed(path: str | None = None) -> dict[str, dict]:
    """Build a reference data dict compatible with the Slack comparison table.

    Returns a dict mapping display name to a flat dict with keys matching
    the ``REFERENCE_MODELS`` format in ``config.py``: params, mmlu,
    humaneval, license, api_price, context_window, vram.

    Only includes models with ``open_weights: true`` or well-known
    API-only models (GPT-4o, Claude, Gemini) for meaningful comparison.
    """
    # Pick a representative subset for comparison tables
    # (too many reference models makes the table unreadable)
    _REFERENCE_IDS = [
        "openai/gpt-4o",
        "anthropic/claude-sonnet-4",
        "google/gemini-2.5-pro",
        "meta-llama/Llama-3.1-405B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "mistralai/Mistral-Large-Instruct-2411",
        "deepseek-ai/DeepSeek-V3-0324",
    ]

    models = load_seed_data(path)
    index = {m["model_id"]: m for m in models}

    result: dict[str, dict] = {}
    for mid in _REFERENCE_IDS:
        m = index.get(mid)
        if not m:
            continue
        basic = m.get("basic", {})
        perf = m.get("performance", {})
        deploy = m.get("deployment", {})
        practical = m.get("practical", {})
        cost = m.get("cost", {})

        result[m["name"]] = {
            "params": basic.get("params", "N/A"),
            "mmlu": perf.get("mmlu", "N/A"),
            "humaneval": perf.get("humaneval", "N/A"),
            "license": basic.get("license", "N/A"),
            "api_price": cost.get("api_price_input_per_1m", "N/A"),
            "context_window": practical.get("context_window", "N/A"),
            "vram": deploy.get("vram_estimate", "N/A"),
        }

    return result


def get_seed_data_schema_version(path: str | None = None) -> str:
    """Return the schema_version from the seed data file, or 'unknown'."""
    filepath = path or SEED_DATA_PATH
    try:
        with open(filepath) as f:
            data = json.load(f)
        return str(data.get("schema_version", "unknown"))
    except (json.JSONDecodeError, OSError, AttributeError):
        return "unknown"
