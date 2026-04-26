import logging
import os
import re

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "settings.yaml")
SEED_MODELS_PATH = os.path.join(_PROJECT_ROOT, "config", "seed_models.yaml")
SEED_DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "seed_data.json")
SEEN_MODELS_PATH = os.path.join(_PROJECT_ROOT, "data", "seen_models.json")

# ---------------------------------------------------------------------------
# Default watched organizations
# ---------------------------------------------------------------------------
DEFAULT_WATCHED_ORGS: list[str] = [
    "meta-llama",
    "google",
    "mistralai",
    "Qwen",
    "microsoft",
    "deepseek-ai",
    "openai",
    "stabilityai",
    "nvidia",
    "alibaba-nlp",
    "apple",
    "CohereForAI",
    "bigscience",
    "EleutherAI",
    "tiiuae",
]

# HuggingFace org/user IDs: alphanumeric, hyphens, underscores, dots.
# Must be 1-96 characters. Leading/trailing hyphens are not allowed.
_ORG_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9]([a-zA-Z0-9._-]{0,94}[a-zA-Z0-9])?$")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_org_name(name: str) -> bool:
    """Check whether *name* is a valid HuggingFace organization/user ID."""
    if not isinstance(name, str) or not name:
        return False
    return bool(_ORG_NAME_PATTERN.match(name))


def validate_watched_orgs(orgs: list) -> tuple[list[str], list[str]]:
    """Validate and deduplicate a list of organization names.

    Returns:
        A tuple of (valid_orgs, invalid_orgs).
        *valid_orgs* preserves original order with duplicates removed.
    """
    valid: list[str] = []
    invalid: list[str] = []
    seen: set[str] = set()

    for org in orgs:
        if not isinstance(org, str):
            invalid.append(str(org))
            continue
        org_stripped = org.strip()
        if not org_stripped:
            continue
        if org_stripped in seen:
            continue  # skip duplicates silently
        if validate_org_name(org_stripped):
            valid.append(org_stripped)
            seen.add(org_stripped)
        else:
            invalid.append(org_stripped)

    return valid, invalid


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(config_path: str | None = None) -> dict:
    """Load configuration from a YAML file.

    Falls back to built-in defaults when the file is missing or unreadable.
    """
    path = config_path or CONFIG_PATH
    if not os.path.exists(path):
        logger.info("Config file not found at %s, using defaults", path)
        return _default_config()

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError:
        logger.exception("Failed to parse config file %s, using defaults", path)
        return _default_config()
    except OSError:
        logger.exception("Failed to read config file %s, using defaults", path)
        return _default_config()

    if not isinstance(data, dict):
        logger.warning("Config file %s has invalid structure, using defaults", path)
        return _default_config()

    return _merge_with_defaults(data)


def _default_config() -> dict:
    """Return a complete config dict with built-in defaults."""
    return {
        "watched_organizations": list(DEFAULT_WATCHED_ORGS),
        "polling_interval_hours": 12,
        "slack_webhook_url": "",
        "run_on_startup": True,
        "trending_thresholds": _default_trending_thresholds(),
        "schema_fields": _default_schema_fields(),
    }


def _default_trending_thresholds() -> dict:
    """Return default trending/surge detection threshold settings.

    These control when a model is flagged as "trending" or experiencing
    a download surge, independent of the new-model detection pipeline.

    Keys:
        enabled: Whether trending/surge detection is active.
        download_surge_count: Minimum download increase within the time
            window to trigger a surge alert.
        trending_score: Minimum HuggingFace trending score to flag.
        time_window_hours: Rolling window (in hours) over which
            download surges and trending scores are evaluated.
    """
    return {
        "enabled": False,
        "download_surge_count": 10000,
        "trending_score": 50,
        "time_window_hours": 24,
    }


def _merge_trending_thresholds(data: dict) -> dict:
    """Merge user-provided trending threshold values with defaults.

    Each key is validated independently; invalid values are silently
    replaced with their defaults so partial overrides work correctly.
    """
    defaults = _default_trending_thresholds()

    # -- enabled (bool) --
    enabled = data.get("enabled")
    if isinstance(enabled, bool):
        defaults["enabled"] = enabled

    # -- download_surge_count (positive int) --
    surge = data.get("download_surge_count")
    if isinstance(surge, (int, float)) and not isinstance(surge, bool) and surge > 0:
        defaults["download_surge_count"] = int(surge)
    elif surge is not None:
        logger.warning(
            "trending_thresholds.download_surge_count must be a positive number, "
            "using default (%d)",
            defaults["download_surge_count"],
        )

    # -- trending_score (non-negative number) --
    score = data.get("trending_score")
    if isinstance(score, (int, float)) and not isinstance(score, bool) and score >= 0:
        defaults["trending_score"] = int(score)
    elif score is not None:
        logger.warning(
            "trending_thresholds.trending_score must be a non-negative number, "
            "using default (%d)",
            defaults["trending_score"],
        )

    # -- time_window_hours (positive int, 1–168) --
    window = data.get("time_window_hours")
    if isinstance(window, (int, float)) and not isinstance(window, bool) and 1 <= window <= 168:
        defaults["time_window_hours"] = int(window)
    elif window is not None:
        logger.warning(
            "trending_thresholds.time_window_hours must be between 1 and 168, "
            "using default (%d)",
            defaults["time_window_hours"],
        )

    return defaults


# ---------------------------------------------------------------------------
# Schema field definitions
# ---------------------------------------------------------------------------

# All available model schema fields with display metadata.
# The PM can toggle ``visible`` per field via the settings API.
# ``display_name``, ``category``, and ``type`` are read-only metadata
# that the dashboard uses for column headers and sort behavior.
_SCHEMA_FIELD_DEFINITIONS: dict[str, dict] = {
    "name":             {"display_name": "Model Name",        "category": "basic",       "type": "string",  "visible": True},
    "author":           {"display_name": "Organization",      "category": "basic",       "type": "string",  "visible": True},
    "category":         {"display_name": "Category",          "category": "basic",       "type": "string",  "visible": True},
    "release_date":     {"display_name": "Release Date",      "category": "basic",       "type": "date",    "visible": True},
    "params":           {"display_name": "Parameters",        "category": "basic",       "type": "numeric", "visible": True},
    "architecture":     {"display_name": "Architecture",      "category": "basic",       "type": "string",  "visible": False},
    "license":          {"display_name": "License",           "category": "basic",       "type": "string",  "visible": True},
    "mmlu":             {"display_name": "MMLU",              "category": "performance", "type": "numeric", "visible": True},
    "humaneval":        {"display_name": "HumanEval",         "category": "performance", "type": "numeric", "visible": True},
    "gpqa":             {"display_name": "GPQA",              "category": "performance", "type": "numeric", "visible": True},
    "math":             {"display_name": "MATH",              "category": "performance", "type": "numeric", "visible": True},
    "arena_elo":        {"display_name": "Arena ELO",         "category": "performance", "type": "numeric", "visible": True},
    "context_window":   {"display_name": "Context Window",    "category": "practical",   "type": "numeric", "visible": True},
    "output_window":    {"display_name": "Output Window",     "category": "practical",   "type": "string",  "visible": False},
    "multilingual":     {"display_name": "Multilingual",      "category": "practical",   "type": "string",  "visible": False},
    "vram_estimate":    {"display_name": "VRAM Estimate",     "category": "deployment",  "type": "numeric", "visible": True},
    "open_weights":     {"display_name": "Open Weights",      "category": "deployment",  "type": "boolean", "visible": True},
    "api_available":    {"display_name": "API Available",      "category": "deployment",  "type": "string",  "visible": False},
    "downloads":        {"display_name": "Downloads",         "category": "community",   "type": "numeric", "visible": True},
    "likes":            {"display_name": "Likes",             "category": "community",   "type": "numeric", "visible": True},
    "api_price_input":  {"display_name": "API Price (Input)", "category": "cost",        "type": "numeric", "visible": True},
    "api_price_output": {"display_name": "API Price (Output)","category": "cost",        "type": "numeric", "visible": True},
    "provider_name":    {"display_name": "Provider",          "category": "provider",    "type": "string",  "visible": False},
}


def _default_schema_fields() -> dict[str, dict]:
    """Return a deep copy of the default schema field definitions.

    Each field has:
        display_name: Human-readable label for column headers.
        category: Grouping key (basic, performance, practical, etc.).
        type: Data type hint (string, numeric, date, boolean).
        visible: Whether the field is shown in the dashboard table.
    """
    import copy
    return copy.deepcopy(_SCHEMA_FIELD_DEFINITIONS)


def _merge_schema_fields(data: dict) -> dict[str, dict]:
    """Merge user-provided schema field overrides with defaults.

    Users may only change ``visible`` and ``display_name`` per field.
    Unknown field keys are ignored.  Invalid values are silently
    replaced with their defaults.
    """
    defaults = _default_schema_fields()

    for field_key, overrides in data.items():
        if field_key not in defaults:
            logger.debug("Ignoring unknown schema field: %s", field_key)
            continue
        if not isinstance(overrides, dict):
            logger.warning(
                "schema_fields.%s should be a dict, skipping", field_key
            )
            continue

        # -- visible (bool) --
        vis = overrides.get("visible")
        if isinstance(vis, bool):
            defaults[field_key]["visible"] = vis

        # -- display_name (non-empty string) --
        dn = overrides.get("display_name")
        if isinstance(dn, str) and dn.strip():
            defaults[field_key]["display_name"] = dn.strip()

    return defaults


def _merge_with_defaults(data: dict) -> dict:
    """Merge user-provided *data* with defaults.  Validate organizations."""
    defaults = _default_config()

    # -- watched_organizations --
    raw_orgs = data.get("watched_organizations")
    if isinstance(raw_orgs, list) and raw_orgs:
        valid, invalid = validate_watched_orgs(raw_orgs)
        if invalid:
            logger.warning("Skipping invalid organization names: %s", invalid)
        if valid:
            defaults["watched_organizations"] = valid
        else:
            logger.warning(
                "All org names in config are invalid, falling back to defaults"
            )
    elif raw_orgs is not None:
        logger.warning(
            "watched_organizations should be a non-empty list, using defaults"
        )

    # -- polling_interval_hours --
    interval = data.get("polling_interval_hours")
    if isinstance(interval, (int, float)) and interval > 0:
        defaults["polling_interval_hours"] = int(interval)

    # -- slack_webhook_url (env var takes precedence) --
    env_url = os.environ.get("SLACK_WEBHOOK_URL", "")
    yaml_url = data.get("slack_webhook_url", "")
    defaults["slack_webhook_url"] = env_url or (yaml_url if isinstance(yaml_url, str) else "")

    # -- run_on_startup --
    run_on_startup = data.get("run_on_startup")
    if isinstance(run_on_startup, bool):
        defaults["run_on_startup"] = run_on_startup

    # -- trending_thresholds --
    raw_thresholds = data.get("trending_thresholds")
    if isinstance(raw_thresholds, dict):
        defaults["trending_thresholds"] = _merge_trending_thresholds(
            raw_thresholds
        )
    elif raw_thresholds is not None:
        logger.warning(
            "trending_thresholds should be a dict, using defaults"
        )

    # -- schema_fields --
    raw_schema = data.get("schema_fields")
    if isinstance(raw_schema, dict):
        defaults["schema_fields"] = _merge_schema_fields(raw_schema)
    elif raw_schema is not None:
        logger.warning(
            "schema_fields should be a dict, using defaults"
        )

    return defaults


def get_watched_organizations(config_path: str | None = None) -> list[str]:
    """Convenience: load config and return the watched organizations list."""
    cfg = load_config(config_path)
    return cfg["watched_organizations"]


# ---------------------------------------------------------------------------
# Config saving
# ---------------------------------------------------------------------------
def save_config(config: dict, config_path: str | None = None) -> None:
    """Persist *config* to a YAML file.

    Creates parent directories if they don't exist.  Only the keys that
    belong in ``settings.yaml`` are written — transient / env-only values
    (like a resolved Slack URL that came from an env var) are kept as-is
    in the dict but written to the file so the YAML remains self-contained.

    Args:
        config: A full config dict (as returned by :func:`load_config`).
        config_path: Destination path. Defaults to the project's
            ``config/settings.yaml``.

    Raises:
        OSError: If the file cannot be written.
    """
    path = config_path or CONFIG_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Build the YAML-friendly representation.  We intentionally preserve
    # key order to keep the file human-readable and diff-friendly.
    data: dict = {}
    data["watched_organizations"] = list(config.get(
        "watched_organizations", DEFAULT_WATCHED_ORGS
    ))
    data["polling_interval_hours"] = config.get("polling_interval_hours", 12)
    data["run_on_startup"] = config.get("run_on_startup", True)
    data["slack_webhook_url"] = config.get("slack_webhook_url", "")
    data["trending_thresholds"] = dict(config.get(
        "trending_thresholds", _default_trending_thresholds()
    ))

    # Only persist visibility overrides for schema fields (keeps YAML concise).
    # Fields whose visibility matches the built-in default are omitted.
    raw_schema = config.get("schema_fields", {})
    if isinstance(raw_schema, dict) and raw_schema:
        schema_overrides: dict = {}
        builtin = _SCHEMA_FIELD_DEFINITIONS
        for key, field in raw_schema.items():
            if key not in builtin or not isinstance(field, dict):
                continue
            overrides: dict = {}
            if field.get("visible") != builtin[key]["visible"]:
                overrides["visible"] = field["visible"]
            if field.get("display_name") != builtin[key]["display_name"]:
                overrides["display_name"] = field["display_name"]
            if overrides:
                schema_overrides[key] = overrides
        if schema_overrides:
            data["schema_fields"] = schema_overrides

    with open(path, "w") as f:
        # Header comment for human readers.
        f.write(
            "# HuggingFace Model Monitor — Configuration\n"
            "# Edit this file to customize which organizations are monitored.\n\n"
        )
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logger.info("Config saved to %s", path)


# ---------------------------------------------------------------------------
# Organization management (add / remove / list)
# ---------------------------------------------------------------------------
def add_organization(
    org_name: str,
    config_path: str | None = None,
) -> tuple[bool, str]:
    """Add an organization to the watched list and persist the change.

    Args:
        org_name: HuggingFace organization or user ID to add.
        config_path: Optional path to settings.yaml override.

    Returns:
        A ``(success, message)`` tuple.  ``success`` is True when the org
        was actually added (or was already present), False on validation
        failure or I/O error.
    """
    if not isinstance(org_name, str) or not org_name.strip():
        return False, "Organization name must be a non-empty string."

    org_name = org_name.strip()

    if not validate_org_name(org_name):
        return False, (
            f"Invalid organization name '{org_name}'. "
            "Must be 1-96 alphanumeric characters, hyphens, underscores, "
            "or dots. Cannot start or end with a hyphen."
        )

    cfg = load_config(config_path)
    orgs: list[str] = cfg["watched_organizations"]

    # Case-sensitive check — HuggingFace IDs are case-sensitive.
    if org_name in orgs:
        return True, f"'{org_name}' is already in the watched list."

    orgs.append(org_name)
    cfg["watched_organizations"] = orgs

    try:
        save_config(cfg, config_path)
    except OSError as exc:
        logger.exception("Failed to save config after adding '%s'", org_name)
        return False, f"Failed to save config: {exc}"

    logger.info("Added organization '%s' to watched list", org_name)
    return True, f"'{org_name}' added to watched list."


def remove_organization(
    org_name: str,
    config_path: str | None = None,
) -> tuple[bool, str]:
    """Remove an organization from the watched list and persist the change.

    Args:
        org_name: HuggingFace organization or user ID to remove.
        config_path: Optional path to settings.yaml override.

    Returns:
        A ``(success, message)`` tuple.  ``success`` is False when the org
        was not found or an I/O error occurred.
    """
    if not isinstance(org_name, str) or not org_name.strip():
        return False, "Organization name must be a non-empty string."

    org_name = org_name.strip()

    cfg = load_config(config_path)
    orgs: list[str] = cfg["watched_organizations"]

    if org_name not in orgs:
        return False, f"'{org_name}' is not in the watched list."

    orgs.remove(org_name)

    # Guard against emptying the list entirely — at least one org should remain
    # to keep the monitor functional.  If the PM really wants to clear the
    # list, they can edit the YAML directly.
    if not orgs:
        return False, (
            f"Cannot remove '{org_name}' — it is the last organization "
            "in the watched list. At least one organization is required."
        )

    cfg["watched_organizations"] = orgs

    try:
        save_config(cfg, config_path)
    except OSError as exc:
        logger.exception("Failed to save config after removing '%s'", org_name)
        return False, f"Failed to save config: {exc}"

    logger.info("Removed organization '%s' from watched list", org_name)
    return True, f"'{org_name}' removed from watched list."


def list_organizations(config_path: str | None = None) -> list[str]:
    """Return the current watched organizations list.

    This is an alias for :func:`get_watched_organizations` that makes the
    CRUD-style API feel complete (list / add / remove).
    """
    return get_watched_organizations(config_path)


# ---------------------------------------------------------------------------
# Slack — kept as module-level for backward compatibility
# ---------------------------------------------------------------------------
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")

# Slack retry configuration
SLACK_MAX_RETRIES = 3
SLACK_BASE_DELAY_SECONDS = 1.0  # Initial backoff delay
SLACK_MAX_DELAY_SECONDS = 30.0  # Cap for exponential backoff
SLACK_REQUEST_TIMEOUT_SECONDS = 10


def validate_webhook_url(url: str) -> bool:
    """Check whether *url* looks like a valid Slack incoming-webhook URL.

    Accepts the standard ``https://hooks.slack.com/services/...`` format
    as well as custom webhook-proxy URLs (any ``https://`` URL).
    Returns False for empty, non-string, or non-HTTPS URLs.
    """
    if not isinstance(url, str) or not url.strip():
        return False
    url = url.strip()
    return url.startswith("https://")

# ---------------------------------------------------------------------------
# Reference models for comparison tables
# ---------------------------------------------------------------------------
REFERENCE_MODELS = {
    "GPT-4o": {
        "params": "~1.8T (estimated)",
        "mmlu": "88.7",
        "humaneval": "90.2",
        "license": "Proprietary",
        "api_price": "$2.50",
        "context_window": "128K",
        "vram": "N/A (API only)",
    },
    "Claude Sonnet 4": {
        "params": "N/A",
        "mmlu": "88.8",
        "humaneval": "93.0",
        "license": "Proprietary",
        "api_price": "$3.00",
        "context_window": "200K",
        "vram": "N/A (API only)",
    },
    "Gemini 2.5 Pro": {
        "params": "N/A",
        "mmlu": "89.0",
        "humaneval": "84.0",
        "license": "Proprietary",
        "api_price": "$1.25",
        "context_window": "1M",
        "vram": "N/A (API only)",
    },
    "Llama-3.1-405B": {
        "params": "405B",
        "mmlu": "87.3",
        "humaneval": "61.0",
        "license": "Llama 3.1 Community",
        "api_price": "$0.90",
        "context_window": "128K",
        "vram": "~800GB FP16",
    },
    "Qwen-2.5-72B": {
        "params": "72B",
        "mmlu": "85.3",
        "humaneval": "86.4",
        "license": "Apache 2.0",
        "api_price": "$0.30",
        "context_window": "128K",
        "vram": "~144GB FP16",
    },
    "Mistral-Large": {
        "params": "123B",
        "mmlu": "84.0",
        "humaneval": "72.0",
        "license": "Mistral Research",
        "api_price": "$2.00",
        "context_window": "128K",
        "vram": "~246GB FP16",
    },
}
