"""Flask web dashboard for the HuggingFace Model Monitor.

Provides a browser-based UI for:
- Viewing all tracked models in a sortable/filterable table
- Comparing model benchmarks, pricing, and specs side-by-side
- Browsing the archive of seed and detected models
- Managing settings (watched orgs, polling interval, Slack webhook)

Usage:
    # Start the dashboard server
    python -m hf_model_monitor.dashboard

    # Or from code
    from hf_model_monitor.dashboard import create_app
    app = create_app()
    app.run(port=5000)
"""

import logging
import os

from flask import Flask, jsonify, render_template, request

from .config import (
    load_config,
    save_config,
    add_organization,
    remove_organization,
    validate_org_name,
    validate_watched_orgs,
    _default_schema_fields,
    _merge_schema_fields,
    _merge_trending_thresholds,
)
from .model_store import ModelStore
from .seed_data import load_seed_data, get_seed_categories

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


def create_app(config_path: str | None = None) -> Flask:
    """Create and configure the Flask dashboard application.

    Args:
        config_path: Optional path to settings.yaml override.

    Returns:
        Configured Flask app instance.
    """
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
    )
    app.config["CONFIG_PATH"] = config_path

    # ------------------------------------------------------------------
    # Page routes
    # ------------------------------------------------------------------

    @app.route("/")
    def index():
        """Main dashboard page with the model table."""
        return render_template("index.html")

    @app.route("/settings")
    def settings():
        """Settings management page."""
        return render_template("settings.html")

    # ------------------------------------------------------------------
    # API routes — model data
    # ------------------------------------------------------------------

    @app.route("/api/models")
    def api_models():
        """Return all models (seed + detected) as flat JSON rows.

        Flattens the nested seed data schema into table-friendly columns.
        Supports query params:
            ?category=llm       — filter by category
            ?search=deepseek    — search by name/model_id (case-insensitive)
            ?author=meta-llama  — filter by author/org
            ?sort=mmlu&order=desc — sort by column
        """
        rows = _build_model_rows()

        # --- Filtering ---
        category = request.args.get("category", "").strip()
        if category:
            rows = [r for r in rows if r.get("category") == category]

        author = request.args.get("author", "").strip()
        if author:
            rows = [
                r for r in rows
                if r.get("author", "").lower() == author.lower()
            ]

        search = request.args.get("search", "").strip().lower()
        if search:
            rows = [
                r for r in rows
                if search in r.get("name", "").lower()
                or search in r.get("model_id", "").lower()
                or search in r.get("author", "").lower()
            ]

        # --- Sorting ---
        sort_key = request.args.get("sort", "").strip()
        order = request.args.get("order", "desc").strip().lower()

        if sort_key and sort_key in _SORTABLE_COLUMNS:
            rows = _sort_rows(rows, sort_key, reverse=(order == "desc"))

        return jsonify({
            "models": rows,
            "total": len(rows),
        })

    @app.route("/api/categories")
    def api_categories():
        """Return available model categories for filter dropdowns."""
        categories = get_seed_categories()
        # Also pull categories from the model store
        try:
            store = ModelStore()
            db_models = store.get_all_models()
            for m in db_models:
                meta = m.get("metadata_json", {})
                if isinstance(meta, dict):
                    cat = meta.get("category", "")
                    if cat and cat not in categories:
                        categories.append(cat)
            store.close()
        except Exception:
            logger.debug("Could not read categories from model store")

        return jsonify({"categories": sorted(categories)})

    @app.route("/api/filter-options")
    def api_filter_options():
        """Return unique values for filter dropdowns (families, providers).

        Derives values from the combined model rows so dropdowns always
        reflect what's actually in the data.  Also returns the observed
        parameter range (in billions) for the range slider / inputs.
        """
        rows = _build_model_rows()

        families: set[str] = set()
        providers: set[str] = set()
        param_values: list[float] = []

        for row in rows:
            author = row.get("author", "").strip()
            if author:
                families.add(author)

            provider = row.get("provider_name", "").strip()
            if provider and provider != "N/A":
                providers.add(provider)

            params_b = _parse_params_billions(row.get("params", "N/A"))
            if params_b is not None:
                param_values.append(params_b)

        return jsonify({
            "families": sorted(families, key=str.lower),
            "providers": sorted(providers, key=str.lower),
            "params_range": {
                "min": round(min(param_values), 2) if param_values else 0,
                "max": round(max(param_values), 2) if param_values else 0,
            },
        })

    @app.route("/api/models/<path:model_id>")
    def api_model_detail(model_id: str):
        """Return detailed info for a single model."""
        seed_models = load_seed_data()
        for m in seed_models:
            if m.get("model_id") == model_id:
                return jsonify(m)

        # Fall back to model store
        try:
            store = ModelStore()
            record = store.get_model(model_id)
            store.close()
            if record:
                return jsonify(record)
        except Exception:
            pass

        return jsonify({"error": "Model not found"}), 404

    @app.route("/api/stats")
    def api_stats():
        """Return summary statistics for the dashboard header."""
        seed_models = load_seed_data()
        categories = get_seed_categories()

        # Count from DB too
        db_count = 0
        last_poll = None
        try:
            store = ModelStore()
            db_count = store.count_models()
            last_poll = store.get_last_poll_time()
            store.close()
        except Exception:
            pass

        return jsonify({
            "seed_model_count": len(seed_models),
            "db_model_count": db_count,
            "category_count": len(categories),
            "last_poll": last_poll,
        })

    # ------------------------------------------------------------------
    # API routes — config / settings
    # ------------------------------------------------------------------

    @app.route("/api/config", methods=["GET"])
    def api_get_config():
        """Return current configuration."""
        cfg = load_config(app.config.get("CONFIG_PATH"))
        return jsonify(cfg)

    @app.route("/api/config", methods=["POST"])
    def api_update_config():
        """Update configuration from JSON body."""
        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON body"}), 400

        cfg = load_config(app.config.get("CONFIG_PATH"))

        # Only allow updating specific keys
        updatable = {
            "polling_interval_hours",
            "slack_webhook_url",
            "run_on_startup",
        }
        for key in updatable:
            if key in data:
                cfg[key] = data[key]

        try:
            save_config(cfg, app.config.get("CONFIG_PATH"))
        except OSError as exc:
            return jsonify({"error": f"Failed to save config: {exc}"}), 500

        return jsonify({"status": "ok", "config": cfg})

    # ------------------------------------------------------------------
    # API routes — settings (RESTful GET/PUT)
    # ------------------------------------------------------------------

    @app.route("/api/settings", methods=["GET"])
    def api_get_settings():
        """Return all settings: orgs, thresholds, schema fields, and general.

        This is the canonical settings endpoint. It returns the full
        config dict including schema_fields with complete field metadata.
        """
        cfg = load_config(app.config.get("CONFIG_PATH"))
        return jsonify(cfg)

    @app.route("/api/settings", methods=["PUT"])
    def api_put_settings():
        """Update all settings at once.

        Accepts a full or partial settings dict. Only recognized keys
        are applied; unknown keys are ignored. All changes are validated
        and persisted to the config file atomically.

        Accepted top-level keys:
            watched_organizations, polling_interval_hours,
            slack_webhook_url, run_on_startup,
            trending_thresholds, schema_fields
        """
        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Request body must be a JSON object"}), 400

        cfg = load_config(app.config.get("CONFIG_PATH"))
        errors: list[str] = []

        # -- watched_organizations --
        if "watched_organizations" in data:
            raw_orgs = data["watched_organizations"]
            if not isinstance(raw_orgs, list) or not raw_orgs:
                errors.append(
                    "watched_organizations must be a non-empty list"
                )
            else:
                valid, invalid = validate_watched_orgs(raw_orgs)
                if invalid:
                    errors.append(
                        f"Invalid organization names: {invalid}"
                    )
                if valid:
                    cfg["watched_organizations"] = valid
                elif not invalid:
                    errors.append(
                        "watched_organizations list is empty after validation"
                    )

        # -- polling_interval_hours --
        if "polling_interval_hours" in data:
            val = data["polling_interval_hours"]
            if isinstance(val, (int, float)) and not isinstance(val, bool) and val > 0:
                cfg["polling_interval_hours"] = int(val)
            else:
                errors.append(
                    "polling_interval_hours must be a positive number"
                )

        # -- slack_webhook_url --
        if "slack_webhook_url" in data:
            val = data["slack_webhook_url"]
            if isinstance(val, str):
                cfg["slack_webhook_url"] = val
            else:
                errors.append("slack_webhook_url must be a string")

        # -- run_on_startup --
        if "run_on_startup" in data:
            val = data["run_on_startup"]
            if isinstance(val, bool):
                cfg["run_on_startup"] = val
            else:
                errors.append("run_on_startup must be a boolean")

        # -- trending_thresholds --
        if "trending_thresholds" in data:
            val = data["trending_thresholds"]
            if isinstance(val, dict):
                cfg["trending_thresholds"] = _merge_trending_thresholds(val)
            else:
                errors.append("trending_thresholds must be a dict")

        # -- schema_fields --
        if "schema_fields" in data:
            val = data["schema_fields"]
            if isinstance(val, dict):
                cfg["schema_fields"] = _merge_schema_fields(val)
            else:
                errors.append("schema_fields must be a dict")

        try:
            save_config(cfg, app.config.get("CONFIG_PATH"))
        except OSError as exc:
            return jsonify({"error": f"Failed to save: {exc}"}), 500

        result = {"status": "ok", "config": cfg}
        if errors:
            result["warnings"] = errors
        return jsonify(result)

    # --- Organizations sub-resource ---

    @app.route("/api/settings/organizations", methods=["GET"])
    def api_get_organizations():
        """Return the current watched organizations list."""
        cfg = load_config(app.config.get("CONFIG_PATH"))
        orgs = cfg["watched_organizations"]
        return jsonify({"organizations": orgs, "total": len(orgs)})

    @app.route("/api/settings/organizations", methods=["PUT"])
    def api_put_organizations():
        """Replace the entire watched organizations list.

        Expects JSON body: {"organizations": ["meta-llama", "google", ...]}
        Validates all names and rejects the request if the resulting list
        would be empty.
        """
        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Request body must be a JSON object"}), 400

        raw_orgs = data.get("organizations")
        if not isinstance(raw_orgs, list) or not raw_orgs:
            return jsonify({
                "error": "organizations must be a non-empty list of strings"
            }), 400

        valid, invalid = validate_watched_orgs(raw_orgs)
        if not valid:
            return jsonify({
                "error": "No valid organization names provided",
                "invalid": invalid,
            }), 400

        cfg = load_config(app.config.get("CONFIG_PATH"))
        cfg["watched_organizations"] = valid

        try:
            save_config(cfg, app.config.get("CONFIG_PATH"))
        except OSError as exc:
            return jsonify({"error": f"Failed to save: {exc}"}), 500

        result = {"status": "ok", "organizations": valid, "total": len(valid)}
        if invalid:
            result["warnings"] = [f"Skipped invalid names: {invalid}"]
        return jsonify(result)

    @app.route("/api/settings/organizations", methods=["POST"])
    def api_add_organization():
        """Add a single organization to the watchlist.

        Expects JSON body: {"name": "org-name"}
        """
        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Request body must be a JSON object"}), 400

        org_name = data.get("name")
        if not isinstance(org_name, str) or not org_name.strip():
            return jsonify({
                "error": "name must be a non-empty string"
            }), 400

        ok, msg = add_organization(org_name, app.config.get("CONFIG_PATH"))
        if ok:
            cfg = load_config(app.config.get("CONFIG_PATH"))
            return jsonify({
                "status": "ok",
                "message": msg,
                "organizations": cfg["watched_organizations"],
            })
        else:
            return jsonify({"error": msg}), 400

    @app.route(
        "/api/settings/organizations/<path:org_name>", methods=["DELETE"]
    )
    def api_remove_organization(org_name: str):
        """Remove a single organization from the watchlist."""
        ok, msg = remove_organization(
            org_name, app.config.get("CONFIG_PATH")
        )
        if ok:
            cfg = load_config(app.config.get("CONFIG_PATH"))
            return jsonify({
                "status": "ok",
                "message": msg,
                "organizations": cfg["watched_organizations"],
            })
        else:
            return jsonify({"error": msg}), 400

    # --- Thresholds sub-resource ---

    @app.route("/api/settings/thresholds", methods=["GET"])
    def api_get_thresholds():
        """Return current auto-detection / trending thresholds."""
        cfg = load_config(app.config.get("CONFIG_PATH"))
        return jsonify(cfg["trending_thresholds"])

    @app.route("/api/settings/thresholds", methods=["PUT"])
    def api_put_thresholds():
        """Update auto-detection / trending thresholds.

        Accepts a partial or full thresholds dict. Each key is validated
        independently; invalid values are reported as warnings and the
        default is used instead.

        Accepted keys: enabled, download_surge_count, trending_score,
                       time_window_hours
        """
        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Request body must be a JSON object"}), 400

        cfg = load_config(app.config.get("CONFIG_PATH"))
        cfg["trending_thresholds"] = _merge_trending_thresholds(data)

        try:
            save_config(cfg, app.config.get("CONFIG_PATH"))
        except OSError as exc:
            return jsonify({"error": f"Failed to save: {exc}"}), 500

        return jsonify({
            "status": "ok",
            "thresholds": cfg["trending_thresholds"],
        })

    # --- Schema fields sub-resource ---

    @app.route("/api/settings/schema", methods=["GET"])
    def api_get_schema():
        """Return model schema field definitions with visibility settings.

        Each field includes display_name, category, type, and visible.
        The ``visible`` flag controls whether the column appears in the
        dashboard table.
        """
        cfg = load_config(app.config.get("CONFIG_PATH"))
        return jsonify({
            "fields": cfg["schema_fields"],
            "total": len(cfg["schema_fields"]),
        })

    @app.route("/api/settings/schema", methods=["PUT"])
    def api_put_schema():
        """Update schema field settings (visibility and display names).

        Accepts a dict keyed by field name with override values.
        Only ``visible`` (bool) and ``display_name`` (string) can be changed.
        Unknown field names are ignored. Read-only properties (category, type)
        are silently discarded.

        Example body::

            {
                "mmlu": {"visible": false},
                "architecture": {"visible": true, "display_name": "Arch"}
            }
        """
        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Request body must be a JSON object"}), 400

        cfg = load_config(app.config.get("CONFIG_PATH"))
        cfg["schema_fields"] = _merge_schema_fields(data)

        try:
            save_config(cfg, app.config.get("CONFIG_PATH"))
        except OSError as exc:
            return jsonify({"error": f"Failed to save: {exc}"}), 500

        return jsonify({
            "status": "ok",
            "fields": cfg["schema_fields"],
            "total": len(cfg["schema_fields"]),
        })

    return app


# ---------------------------------------------------------------------------
# Helpers — flatten seed data into table rows
# ---------------------------------------------------------------------------

# Column sort-type map — declares whether each sortable column uses
# string, numeric, or date comparison.  Must stay in sync with the
# data-sort-type attributes in templates/index.html.
_COLUMN_SORT_TYPES: dict[str, str] = {
    # String columns
    "name": "string",
    "author": "string",
    "category": "string",
    "license": "string",
    # Numeric columns
    "params": "numeric",
    "mmlu": "numeric",
    "humaneval": "numeric",
    "gpqa": "numeric",
    "math": "numeric",
    "arena_elo": "numeric",
    "context_window": "numeric",
    "vram_estimate": "numeric",
    "api_price_input": "numeric",
    "api_price_output": "numeric",
    "downloads": "numeric",
    "likes": "numeric",
    "open_weights": "numeric",
    # Date columns
    "release_date": "date",
}

# Set of sortable column keys (for quick membership checks in the API)
_SORTABLE_COLUMNS = set(_COLUMN_SORT_TYPES.keys())


def _flatten_seed_model(m: dict) -> dict:
    """Flatten a nested seed data model into a flat table row."""
    basic = m.get("basic", {})
    perf = m.get("performance", {})
    practical = m.get("practical", {})
    deploy = m.get("deployment", {})
    community = m.get("community", {})
    cost = m.get("cost", {})
    provider = m.get("provider", {})

    return {
        "model_id": m.get("model_id", ""),
        "name": m.get("name", basic.get("name", "")),
        "author": m.get("author", basic.get("org", "")),
        "category": m.get("category", ""),
        "release_date": m.get("release_date", "N/A"),
        "source": m.get("source", "seed"),
        # Basic
        "params": basic.get("params", "N/A"),
        "architecture": basic.get("architecture", "N/A"),
        "license": basic.get("license", "N/A"),
        # Performance benchmarks
        "mmlu": perf.get("mmlu", "N/A"),
        "humaneval": perf.get("humaneval", "N/A"),
        "gpqa": perf.get("gpqa", "N/A"),
        "math": perf.get("math", "N/A"),
        "arena_elo": perf.get("arena_elo", "N/A"),
        # Practical
        "context_window": practical.get("context_window", "N/A"),
        "output_window": practical.get("output_window", "N/A"),
        "multilingual": practical.get("multilingual", "N/A"),
        # Deployment
        "vram_estimate": deploy.get("vram_estimate", "N/A"),
        "open_weights": deploy.get("open_weights", False),
        "api_available": deploy.get("api_available", "N/A"),
        # Community
        "downloads": community.get("downloads", 0),
        "likes": community.get("likes", 0),
        # Cost
        "api_price_input": cost.get("api_price_input_per_1m", "N/A"),
        "api_price_output": cost.get("api_price_output_per_1m", "N/A"),
        # Provider
        "provider_name": provider.get("name", "N/A"),
        "hf_url": (
            f"https://huggingface.co/{m.get('model_id', '')}"
            if m.get("model_id") and provider.get("hf_repo_id", "N/A") != "N/A"
            else provider.get("website", "")
        ),
    }


def _flatten_db_model(m: dict) -> dict:
    """Flatten a ModelStore record into a table row."""
    meta = m.get("metadata_json", {})
    if not isinstance(meta, dict):
        meta = {}

    return {
        "model_id": m.get("model_id", ""),
        "name": m.get("model_id", "").split("/")[-1] if "/" in m.get("model_id", "") else m.get("model_id", ""),
        "author": m.get("author", ""),
        "category": meta.get("category", "N/A"),
        "release_date": m.get("created_at", "N/A"),
        "source": "detected",
        "params": "N/A",
        "architecture": "N/A",
        "license": "N/A",
        "mmlu": "N/A",
        "humaneval": "N/A",
        "gpqa": "N/A",
        "math": "N/A",
        "arena_elo": "N/A",
        "context_window": "N/A",
        "output_window": "N/A",
        "multilingual": "N/A",
        "vram_estimate": "N/A",
        "open_weights": True,  # HF models are generally open
        "api_available": "N/A",
        "downloads": m.get("downloads", 0),
        "likes": m.get("likes", 0),
        "api_price_input": "N/A",
        "api_price_output": "N/A",
        "provider_name": m.get("author", "N/A"),
        "hf_url": f"https://huggingface.co/{m.get('model_id', '')}",
    }


def _build_model_rows() -> list[dict]:
    """Build a combined list of model rows from seed data and the DB.

    Seed data entries take priority (richer data). DB-only models are
    appended with their available fields.
    """
    rows: list[dict] = []
    seen_ids: set[str] = set()

    # 1. Seed data (rich, curated entries)
    for m in load_seed_data():
        mid = m.get("model_id", "")
        if mid and mid not in seen_ids:
            rows.append(_flatten_seed_model(m))
            seen_ids.add(mid)

    # 2. DB models (detected via polling)
    try:
        store = ModelStore()
        for m in store.get_all_models():
            mid = m.get("model_id", "")
            if mid and mid not in seen_ids:
                rows.append(_flatten_db_model(m))
                seen_ids.add(mid)
        store.close()
    except Exception:
        logger.debug("Could not read models from store for dashboard")

    return rows


def _parse_params_billions(value: str) -> float | None:
    """Parse a params string into a number in billions.

    Returns None for N/A or unparseable values.

    Examples:
        "70B"                       -> 70.0
        "~1.8T (estimated)"         -> 1800.0
        "671B total (37B active)"   -> 671.0
        "809M"                      -> 0.809
        "N/A"                       -> None
    """
    import re as _re

    if not isinstance(value, str):
        try:
            return float(value) / 1e9
        except (TypeError, ValueError):
            return None

    s = value.strip()
    if not s or s == "N/A":
        return None

    s = s.replace("~", "").replace(",", "").strip()

    match = _re.match(r"([0-9]*\.?[0-9]+)\s*([TBMK])?", s)
    if not match:
        return None

    num = float(match.group(1))
    suffix = match.group(2)

    if suffix == "T":
        return num * 1000.0
    elif suffix == "B":
        return num
    elif suffix == "M":
        return num / 1000.0
    elif suffix == "K":
        return num / 1000000.0
    else:
        return num  # assume raw number


def _is_na(value) -> bool:
    """Check if a value should be treated as N/A (missing data)."""
    if value is None:
        return True
    if isinstance(value, str):
        s = value.strip()
        return s == "" or s.upper() == "N/A" or s.startswith("N/A")
    return False


def _parse_numeric(value) -> float | None:
    """Extract a numeric value from a string for sorting.

    Handles formats like "88.7", "$2.50", "~1.8T", "128K", "405B", etc.
    Returns None for N/A or unparseable values.
    """
    import re

    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    s = value.strip()
    if not s or _is_na(s):
        return None

    # Strip common prefixes/suffixes
    s = s.replace("$", "").replace(",", "").replace("~", "").strip()

    # Handle suffixes: T (trillion), B (billion), M (million), K (thousand)
    multipliers = {"T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3}

    match = re.match(r"([0-9]*\.?[0-9]+)\s*([TBMK])?", s)
    if match:
        num = float(match.group(1))
        suffix = match.group(2)
        if suffix and suffix in multipliers:
            num *= multipliers[suffix]
        return num

    return None


def _parse_date(value) -> float | None:
    """Parse a date string into a timestamp for sorting.

    Handles "YYYY-MM-DD" and full ISO 8601 strings.
    Returns None for N/A or unparseable values.
    """
    import re
    from datetime import datetime

    if not isinstance(value, str):
        return None

    s = value.strip()
    if not s or _is_na(s):
        return None

    date_match = re.match(r"^(\d{4}-\d{2}-\d{2})", s)
    if date_match:
        try:
            dt = datetime.fromisoformat(date_match.group(1))
            return dt.timestamp()
        except ValueError:
            pass

    return None


def _sort_rows(rows: list[dict], key: str, reverse: bool = True) -> list[dict]:
    """Sort rows by column key using the column's declared sort type.

    Dispatches to string, numeric, or date comparator based on
    _COLUMN_SORT_TYPES.  N/A values always sort to the bottom regardless
    of sort direction.
    """
    sort_type = _COLUMN_SORT_TYPES.get(key, "string")

    # Sentinel values that sort to the bottom:
    #   For reverse=True (desc):  _NA_BOTTOM sorts below real values
    #   For reverse=False (asc):  _NA_BOTTOM sorts below real values
    # We achieve this by returning a tuple: (is_na_flag, value)
    # where is_na_flag=1 always pushes N/A entries after real data.

    def sort_val(row):
        val = row.get(key, "N/A")

        if _is_na(val):
            # N/A → always at bottom: (1, ...) sorts after (0, ...)
            # When reverse=True, we need (1, ...) to NOT be reversed to top,
            # so we negate: use a tuple that stays at bottom regardless.
            return (1, "")

        if sort_type == "numeric":
            parsed = _parse_numeric(val)
            if parsed is not None:
                return (0, parsed)
            # Unparseable → treat as N/A
            return (1, "")

        elif sort_type == "date":
            parsed = _parse_date(val)
            if parsed is not None:
                return (0, parsed)
            return (1, "")

        else:  # string
            return (0, str(val).lower())

    # For N/A-to-bottom behavior with reverse sort, we need a custom approach:
    # Split into real values and N/A values, sort real values, append N/A.
    real = []
    na_rows = []
    for row in rows:
        val = row.get(key, "N/A")
        if _is_na(val):
            na_rows.append(row)
        else:
            real.append(row)

    def real_sort_val(row):
        val = row.get(key, "N/A")
        if sort_type == "numeric":
            parsed = _parse_numeric(val)
            return parsed if parsed is not None else 0.0
        elif sort_type == "date":
            parsed = _parse_date(val)
            return parsed if parsed is not None else 0.0
        else:
            return str(val).lower()

    real.sort(key=real_sort_val, reverse=reverse)
    return real + na_rows


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Start the dashboard server from the command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="HF Model Monitor — Web Dashboard",
    )
    parser.add_argument(
        "--port", type=int, default=5000,
        help="Port to run the dashboard on (default: 5000)",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Run in debug mode with auto-reload",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to settings.yaml config file",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    app = create_app(args.config)
    logger.info("Starting dashboard at http://%s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
