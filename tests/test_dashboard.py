"""Tests for the dashboard Flask app and table component.

Covers:
- API endpoints return correct JSON structure
- Seed data loads and flattens into table-friendly rows
- Filtering by category, search, and weights works
- Sorting works with numeric awareness
- Model detail endpoint works
- Stats endpoint returns expected fields
- Edge cases: empty data, N/A values, missing fields
"""

import json
import os
import tempfile

import pytest

from hf_model_monitor.dashboard import (
    create_app,
    _flatten_seed_model,
    _flatten_db_model,
    _is_na,
    _parse_numeric,
    _parse_date,
    _parse_params_billions,
    _sort_rows,
    _build_model_rows,
    _COLUMN_SORT_TYPES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def seed_data_path(tmp_path):
    """Create a minimal seed_data.json for testing."""
    data = {
        "schema_version": "1.0",
        "description": "Test seed data",
        "last_updated": "2026-01-01",
        "models": [
            {
                "model_id": "test-org/model-a",
                "name": "Model A",
                "author": "test-org",
                "release_date": "2025-01-15",
                "category": "llm",
                "source": "seed",
                "basic": {
                    "name": "Model A",
                    "org": "test-org",
                    "params": "70B",
                    "architecture": "Transformer",
                    "license": "Apache 2.0",
                },
                "performance": {
                    "mmlu": "87.3",
                    "humaneval": "82.0",
                    "gpqa": "55.0",
                    "math": "75.0",
                    "arena_elo": "1250",
                    "arena_rank": "N/A",
                },
                "practical": {
                    "context_window": "128K",
                    "output_window": "8K",
                    "multilingual": "Yes",
                    "fine_tuning_support": "Yes",
                },
                "deployment": {
                    "vram_estimate": "~140GB FP16",
                    "quantization_options": "GPTQ, AWQ",
                    "api_available": "HF Inference API",
                    "open_weights": True,
                },
                "community": {
                    "downloads": 5000000,
                    "likes": 3000,
                    "trending_rank": "N/A",
                },
                "cost": {
                    "api_price_input_per_1m": "$0.50",
                    "api_price_output_per_1m": "$1.00",
                    "hosting_cost_estimate": "~$2/hr",
                },
                "provider": {
                    "name": "TestOrg",
                    "hf_repo_id": "test-org/model-a",
                    "api_providers": ["TestProvider"],
                    "website": "https://test.org",
                },
            },
            {
                "model_id": "another-org/model-b",
                "name": "Model B",
                "author": "another-org",
                "release_date": "2025-03-10",
                "category": "code",
                "source": "seed",
                "basic": {
                    "name": "Model B",
                    "org": "another-org",
                    "params": "32B",
                    "architecture": "Transformer (code-tuned)",
                    "license": "MIT",
                },
                "performance": {
                    "mmlu": "N/A",
                    "humaneval": "92.0",
                    "gpqa": "N/A",
                    "math": "N/A",
                    "arena_elo": "N/A",
                    "arena_rank": "N/A",
                },
                "practical": {
                    "context_window": "64K",
                    "output_window": "N/A",
                    "multilingual": "Yes",
                    "fine_tuning_support": "Yes",
                },
                "deployment": {
                    "vram_estimate": "~65GB FP16",
                    "quantization_options": "GPTQ",
                    "api_available": "HF Inference API",
                    "open_weights": True,
                },
                "community": {
                    "downloads": 2000000,
                    "likes": 1500,
                    "trending_rank": "N/A",
                },
                "cost": {
                    "api_price_input_per_1m": "$0.15",
                    "api_price_output_per_1m": "$0.30",
                    "hosting_cost_estimate": "~$1/hr",
                },
                "provider": {
                    "name": "AnotherOrg",
                    "hf_repo_id": "another-org/model-b",
                    "api_providers": ["AnotherProvider"],
                    "website": "https://another.org",
                },
            },
            {
                "model_id": "closed-co/proprietary-model",
                "name": "Proprietary Model",
                "author": "closed-co",
                "release_date": "2025-02-20",
                "category": "llm",
                "source": "seed",
                "basic": {
                    "name": "Proprietary Model",
                    "org": "closed-co",
                    "params": "N/A",
                    "architecture": "Transformer",
                    "license": "Proprietary",
                },
                "performance": {
                    "mmlu": "90.0",
                    "humaneval": "95.0",
                    "gpqa": "70.0",
                    "math": "88.0",
                    "arena_elo": "1350",
                    "arena_rank": "N/A",
                },
                "practical": {
                    "context_window": "200K",
                    "output_window": "64K",
                    "multilingual": "Yes",
                    "fine_tuning_support": "No",
                },
                "deployment": {
                    "vram_estimate": "N/A (API only)",
                    "quantization_options": "N/A",
                    "api_available": "Proprietary API",
                    "open_weights": False,
                },
                "community": {
                    "downloads": 0,
                    "likes": 0,
                    "trending_rank": "N/A",
                },
                "cost": {
                    "api_price_input_per_1m": "$3.00",
                    "api_price_output_per_1m": "$15.00",
                    "hosting_cost_estimate": "N/A",
                },
                "provider": {
                    "name": "ClosedCo",
                    "hf_repo_id": "N/A",
                    "api_providers": ["ClosedCo API"],
                    "website": "https://closed.co",
                },
            },
        ],
    }
    filepath = tmp_path / "seed_data.json"
    filepath.write_text(json.dumps(data))
    return str(filepath)


@pytest.fixture
def app(seed_data_path, monkeypatch):
    """Create a test Flask app with patched seed data path."""
    monkeypatch.setattr(
        "hf_model_monitor.seed_data.SEED_DATA_PATH", seed_data_path
    )
    monkeypatch.setattr(
        "hf_model_monitor.dashboard.load_seed_data",
        lambda path=None: _load_test_seed(seed_data_path),
    )
    monkeypatch.setattr(
        "hf_model_monitor.dashboard.get_seed_categories",
        lambda path=None: ["code", "llm"],
    )
    # Patch ModelStore to avoid needing a real DB
    monkeypatch.setattr(
        "hf_model_monitor.dashboard.ModelStore",
        _MockModelStore,
    )
    application = create_app()
    application.config["TESTING"] = True
    return application


@pytest.fixture
def client(app):
    """Flask test client."""
    return app.test_client()


def _load_test_seed(path):
    """Load seed data from the test fixture file."""
    with open(path) as f:
        data = json.load(f)
    return data.get("models", [])


class _MockModelStore:
    """Minimal mock for ModelStore to avoid SQLite in tests."""

    def __init__(self, *args, **kwargs):
        pass

    def get_all_models(self):
        return []

    def count_models(self):
        return 0

    def get_last_poll_time(self):
        return None

    def get_model(self, model_id):
        return None

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Tests: _flatten_seed_model
# ---------------------------------------------------------------------------

class TestFlattenSeedModel:
    """Tests for flattening nested seed data into table rows."""

    def test_flatten_basic_fields(self):
        model = {
            "model_id": "org/model",
            "name": "Test Model",
            "author": "org",
            "category": "llm",
            "release_date": "2025-01-01",
            "source": "seed",
            "basic": {"params": "70B", "architecture": "Transformer", "license": "MIT"},
            "performance": {"mmlu": "85.0", "humaneval": "80.0", "gpqa": "50.0", "math": "70.0", "arena_elo": "1200"},
            "practical": {"context_window": "128K", "output_window": "8K", "multilingual": "Yes"},
            "deployment": {"vram_estimate": "~140GB FP16", "open_weights": True, "api_available": "HF API"},
            "community": {"downloads": 1000, "likes": 500},
            "cost": {"api_price_input_per_1m": "$0.50", "api_price_output_per_1m": "$1.00"},
            "provider": {"name": "TestOrg", "hf_repo_id": "org/model", "website": "https://test.org"},
        }
        row = _flatten_seed_model(model)

        assert row["model_id"] == "org/model"
        assert row["name"] == "Test Model"
        assert row["author"] == "org"
        assert row["category"] == "llm"
        assert row["release_date"] == "2025-01-01"
        assert row["params"] == "70B"
        assert row["mmlu"] == "85.0"
        assert row["humaneval"] == "80.0"
        assert row["gpqa"] == "50.0"
        assert row["math"] == "70.0"
        assert row["context_window"] == "128K"
        assert row["open_weights"] is True
        assert row["downloads"] == 1000
        assert row["likes"] == 500
        assert row["api_price_input"] == "$0.50"
        assert row["api_price_output"] == "$1.00"

    def test_flatten_missing_sections(self):
        """Model with missing sections should default to N/A."""
        model = {
            "model_id": "org/minimal",
            "name": "Minimal",
            "author": "org",
        }
        row = _flatten_seed_model(model)

        assert row["model_id"] == "org/minimal"
        assert row["name"] == "Minimal"
        assert row["params"] == "N/A"
        assert row["mmlu"] == "N/A"
        assert row["context_window"] == "N/A"
        assert row["api_price_input"] == "N/A"
        assert row["downloads"] == 0


class TestFlattenDbModel:
    """Tests for flattening ModelStore records into table rows."""

    def test_flatten_db_record(self):
        record = {
            "model_id": "org/detected-model",
            "author": "org",
            "downloads": 5000,
            "likes": 200,
            "created_at": "2025-06-01",
            "metadata_json": {"category": "llm"},
        }
        row = _flatten_db_model(record)

        assert row["model_id"] == "org/detected-model"
        assert row["name"] == "detected-model"
        assert row["author"] == "org"
        assert row["category"] == "llm"
        assert row["downloads"] == 5000
        assert row["source"] == "detected"
        assert row["hf_url"] == "https://huggingface.co/org/detected-model"

    def test_flatten_db_record_no_metadata(self):
        record = {
            "model_id": "org/bare-model",
            "author": "org",
            "metadata_json": {},
        }
        row = _flatten_db_model(record)
        assert row["category"] == "N/A"
        assert row["params"] == "N/A"


# ---------------------------------------------------------------------------
# Tests: _parse_numeric
# ---------------------------------------------------------------------------

class TestIsNA:
    """Tests for the N/A detection helper."""

    def test_none_is_na(self):
        assert _is_na(None) is True

    def test_empty_string_is_na(self):
        assert _is_na("") is True
        assert _is_na("   ") is True

    def test_na_string_is_na(self):
        assert _is_na("N/A") is True
        assert _is_na("n/a") is True

    def test_na_prefix_is_na(self):
        assert _is_na("N/A (API only)") is True

    def test_real_value_is_not_na(self):
        assert _is_na("70B") is False
        assert _is_na(42) is False
        assert _is_na(0) is False
        assert _is_na(False) is False


class TestParseNumeric:
    """Tests for numeric parsing used in sorting."""

    def test_plain_number(self):
        assert _parse_numeric("88.7") == 88.7

    def test_dollar_prefix(self):
        assert _parse_numeric("$2.50") == 2.50

    def test_tilde_prefix(self):
        assert _parse_numeric("~1.8T") == pytest.approx(1.8e12)

    def test_billions(self):
        assert _parse_numeric("405B") == pytest.approx(405e9)

    def test_millions(self):
        assert _parse_numeric("1M") == pytest.approx(1e6)

    def test_thousands(self):
        assert _parse_numeric("128K") == pytest.approx(128e3)

    def test_na_returns_none(self):
        assert _parse_numeric("N/A") is None

    def test_empty_returns_none(self):
        assert _parse_numeric("") is None

    def test_api_only_returns_none(self):
        assert _parse_numeric("N/A (API only)") is None

    def test_already_numeric(self):
        assert _parse_numeric(42) == 42.0
        assert _parse_numeric(3.14) == 3.14

    def test_none_returns_none(self):
        assert _parse_numeric(None) is None

    def test_boolean_values(self):
        assert _parse_numeric(True) == 1.0
        assert _parse_numeric(False) == 0.0

    def test_complex_string_with_number(self):
        assert _parse_numeric("~140GB FP16") == pytest.approx(140.0)


class TestParseDate:
    """Tests for date parsing used in date-type sorting."""

    def test_iso_date(self):
        result = _parse_date("2025-01-15")
        assert result is not None
        assert isinstance(result, float)

    def test_iso_datetime(self):
        result = _parse_date("2025-01-15T10:30:00")
        assert result is not None

    def test_na_returns_none(self):
        assert _parse_date("N/A") is None

    def test_empty_returns_none(self):
        assert _parse_date("") is None

    def test_none_returns_none(self):
        assert _parse_date(None) is None

    def test_non_date_string(self):
        assert _parse_date("hello") is None

    def test_numeric_returns_none(self):
        assert _parse_date(42) is None

    def test_date_ordering(self):
        """Earlier dates should have smaller timestamps."""
        early = _parse_date("2025-01-01")
        late = _parse_date("2025-12-31")
        assert early < late


# ---------------------------------------------------------------------------
# Tests: _sort_rows
# ---------------------------------------------------------------------------

class TestSortRows:
    """Tests for multi-type sorting of model rows.

    _sort_rows dispatches to string, numeric, or date comparator based
    on _COLUMN_SORT_TYPES. N/A values always sort to the bottom.
    """

    # --- Numeric sorting ---

    def test_sort_by_numeric_desc(self):
        rows = [
            {"name": "A", "mmlu": "85.0"},
            {"name": "B", "mmlu": "90.0"},
            {"name": "C", "mmlu": "80.0"},
        ]
        result = _sort_rows(rows, "mmlu", reverse=True)
        assert [r["name"] for r in result] == ["B", "A", "C"]

    def test_sort_by_numeric_asc(self):
        rows = [
            {"name": "A", "mmlu": "85.0"},
            {"name": "B", "mmlu": "90.0"},
        ]
        result = _sort_rows(rows, "mmlu", reverse=False)
        assert [r["name"] for r in result] == ["A", "B"]

    def test_sort_numeric_na_to_bottom_desc(self):
        """N/A goes to bottom even when sorting descending."""
        rows = [
            {"name": "A", "mmlu": "N/A"},
            {"name": "B", "mmlu": "90.0"},
            {"name": "C", "mmlu": "85.0"},
        ]
        result = _sort_rows(rows, "mmlu", reverse=True)
        assert result[0]["name"] == "B"
        assert result[1]["name"] == "C"
        assert result[2]["name"] == "A"  # N/A at bottom

    def test_sort_numeric_na_to_bottom_asc(self):
        """N/A goes to bottom even when sorting ascending."""
        rows = [
            {"name": "A", "mmlu": "N/A"},
            {"name": "B", "mmlu": "90.0"},
            {"name": "C", "mmlu": "85.0"},
        ]
        result = _sort_rows(rows, "mmlu", reverse=False)
        assert result[0]["name"] == "C"   # 85 first ascending
        assert result[1]["name"] == "B"   # 90 second
        assert result[2]["name"] == "A"   # N/A still at bottom

    def test_sort_numeric_all_na(self):
        """All N/A values — order is stable, no crash."""
        rows = [
            {"name": "A", "mmlu": "N/A"},
            {"name": "B", "mmlu": "N/A"},
        ]
        result = _sort_rows(rows, "mmlu", reverse=True)
        assert len(result) == 2

    def test_sort_by_downloads(self):
        rows = [
            {"name": "A", "downloads": 100},
            {"name": "B", "downloads": 5000},
            {"name": "C", "downloads": 500},
        ]
        result = _sort_rows(rows, "downloads", reverse=True)
        assert [r["name"] for r in result] == ["B", "C", "A"]

    def test_sort_by_params_with_suffixes(self):
        """Numeric sort handles B/M/T suffixes correctly."""
        rows = [
            {"name": "Small", "params": "7B"},
            {"name": "Big", "params": "70B"},
            {"name": "Tiny", "params": "809M"},
        ]
        result = _sort_rows(rows, "params", reverse=True)
        assert result[0]["name"] == "Big"    # 70B
        assert result[1]["name"] == "Small"  # 7B
        assert result[2]["name"] == "Tiny"   # 809M = 0.809B

    def test_sort_by_price_with_dollar(self):
        """Numeric sort handles $-prefixed values."""
        rows = [
            {"name": "Cheap", "api_price_input": "$0.15"},
            {"name": "Expensive", "api_price_input": "$3.00"},
            {"name": "Mid", "api_price_input": "$0.50"},
        ]
        result = _sort_rows(rows, "api_price_input", reverse=False)
        assert result[0]["name"] == "Cheap"
        assert result[1]["name"] == "Mid"
        assert result[2]["name"] == "Expensive"

    # --- String sorting ---

    def test_sort_by_string_asc(self):
        rows = [
            {"name": "Zebra", "author": "z"},
            {"name": "Alpha", "author": "a"},
        ]
        result = _sort_rows(rows, "name", reverse=False)
        assert result[0]["name"] == "Alpha"
        assert result[1]["name"] == "Zebra"

    def test_sort_by_string_desc(self):
        rows = [
            {"name": "Alpha"},
            {"name": "Zebra"},
        ]
        result = _sort_rows(rows, "name", reverse=True)
        assert result[0]["name"] == "Zebra"

    def test_sort_string_case_insensitive(self):
        rows = [
            {"name": "banana"},
            {"name": "Apple"},
            {"name": "cherry"},
        ]
        result = _sort_rows(rows, "name", reverse=False)
        assert [r["name"] for r in result] == ["Apple", "banana", "cherry"]

    def test_sort_string_na_to_bottom(self):
        rows = [
            {"name": "B", "license": "N/A"},
            {"name": "A", "license": "MIT"},
            {"name": "C", "license": "Apache 2.0"},
        ]
        result = _sort_rows(rows, "license", reverse=False)
        assert result[0]["name"] == "C"   # Apache first
        assert result[1]["name"] == "A"   # MIT second
        assert result[2]["name"] == "B"   # N/A at bottom

    # --- Date sorting ---

    def test_sort_by_date_desc(self):
        rows = [
            {"name": "Old", "release_date": "2025-01-01"},
            {"name": "New", "release_date": "2025-06-15"},
            {"name": "Mid", "release_date": "2025-03-10"},
        ]
        result = _sort_rows(rows, "release_date", reverse=True)
        assert result[0]["name"] == "New"
        assert result[1]["name"] == "Mid"
        assert result[2]["name"] == "Old"

    def test_sort_by_date_asc(self):
        rows = [
            {"name": "Old", "release_date": "2025-01-01"},
            {"name": "New", "release_date": "2025-06-15"},
        ]
        result = _sort_rows(rows, "release_date", reverse=False)
        assert result[0]["name"] == "Old"
        assert result[1]["name"] == "New"

    def test_sort_date_na_to_bottom(self):
        """N/A dates go to bottom regardless of direction."""
        rows = [
            {"name": "Unknown", "release_date": "N/A"},
            {"name": "New", "release_date": "2025-06-15"},
            {"name": "Old", "release_date": "2025-01-01"},
        ]
        # Descending
        result_desc = _sort_rows(rows, "release_date", reverse=True)
        assert result_desc[-1]["name"] == "Unknown"
        # Ascending
        result_asc = _sort_rows(rows, "release_date", reverse=False)
        assert result_asc[-1]["name"] == "Unknown"

    # --- Boolean sorting ---

    def test_sort_by_boolean(self):
        rows = [
            {"name": "A", "open_weights": False},
            {"name": "B", "open_weights": True},
        ]
        result = _sort_rows(rows, "open_weights", reverse=True)
        assert result[0]["name"] == "B"

    # --- Column type map consistency ---

    def test_column_sort_types_has_all_expected_keys(self):
        """Verify all sortable columns have a declared type."""
        expected = {
            "name", "author", "category", "release_date", "params",
            "mmlu", "humaneval", "gpqa", "math", "arena_elo",
            "context_window", "vram_estimate",
            "api_price_input", "api_price_output",
            "downloads", "likes", "open_weights", "license",
        }
        for key in expected:
            assert key in _COLUMN_SORT_TYPES, f"Missing sort type for: {key}"

    def test_column_sort_types_valid_values(self):
        """All sort types must be one of the three valid types."""
        valid = {"string", "numeric", "date"}
        for key, sort_type in _COLUMN_SORT_TYPES.items():
            assert sort_type in valid, f"Invalid sort type '{sort_type}' for {key}"


# ---------------------------------------------------------------------------
# Tests: API endpoints
# ---------------------------------------------------------------------------

class TestApiModels:
    """Tests for the /api/models endpoint."""

    def test_get_all_models(self, client):
        resp = client.get("/api/models")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "models" in data
        assert "total" in data
        assert data["total"] == 3  # 3 seed models
        assert len(data["models"]) == 3

    def test_models_have_required_columns(self, client):
        resp = client.get("/api/models")
        data = resp.get_json()
        model = data["models"][0]

        required_keys = [
            "model_id", "name", "author", "category", "release_date",
            "params", "mmlu", "humaneval", "gpqa", "math",
            "api_price_input", "api_price_output", "context_window",
            "license", "downloads", "likes", "open_weights",
        ]
        for key in required_keys:
            assert key in model, f"Missing key: {key}"

    def test_filter_by_category(self, client):
        resp = client.get("/api/models?category=code")
        data = resp.get_json()
        assert data["total"] == 1
        assert data["models"][0]["category"] == "code"

    def test_filter_by_search(self, client):
        resp = client.get("/api/models?search=model%20a")
        data = resp.get_json()
        assert data["total"] == 1
        assert data["models"][0]["name"] == "Model A"

    def test_search_by_org(self, client):
        resp = client.get("/api/models?search=another-org")
        data = resp.get_json()
        assert data["total"] == 1

    def test_filter_no_results(self, client):
        resp = client.get("/api/models?category=nonexistent")
        data = resp.get_json()
        assert data["total"] == 0
        assert data["models"] == []

    def test_sort_by_mmlu_desc(self, client):
        resp = client.get("/api/models?sort=mmlu&order=desc")
        data = resp.get_json()
        # Proprietary (90.0) > Model A (87.3) > Model B (N/A)
        assert data["models"][0]["name"] == "Proprietary Model"
        assert data["models"][1]["name"] == "Model A"

    def test_sort_by_release_date_asc(self, client):
        resp = client.get("/api/models?sort=release_date&order=asc")
        data = resp.get_json()
        # 2025-01-15 < 2025-02-20 < 2025-03-10
        assert data["models"][0]["name"] == "Model A"
        assert data["models"][-1]["name"] == "Model B"


class TestApiCategories:
    """Tests for the /api/categories endpoint."""

    def test_get_categories(self, client):
        resp = client.get("/api/categories")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "categories" in data
        assert "llm" in data["categories"]
        assert "code" in data["categories"]


class TestApiStats:
    """Tests for the /api/stats endpoint."""

    def test_get_stats(self, client):
        resp = client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "seed_model_count" in data
        assert "db_model_count" in data
        assert "category_count" in data
        assert "last_poll" in data
        assert data["seed_model_count"] == 3
        assert data["category_count"] == 2


class TestApiModelDetail:
    """Tests for the /api/models/<id> endpoint."""

    def test_get_existing_model(self, client):
        resp = client.get("/api/models/test-org/model-a")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["model_id"] == "test-org/model-a"
        assert data["name"] == "Model A"

    def test_get_nonexistent_model(self, client):
        resp = client.get("/api/models/nonexistent/model")
        assert resp.status_code == 404


class TestIndexPage:
    """Tests for the main dashboard HTML page."""

    def test_index_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"HF Model Monitor" in resp.data
        assert b"model-table" in resp.data
        assert b"search-input" in resp.data


class TestApiConfig:
    """Tests for the /api/config endpoints."""

    def test_get_config(self, client):
        resp = client.get("/api/config")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "watched_organizations" in data
        assert "polling_interval_hours" in data

    def test_update_config_invalid_body(self, client):
        resp = client.post(
            "/api/config",
            data="not json",
            content_type="text/plain",
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Tests: _parse_params_billions
# ---------------------------------------------------------------------------

class TestParseParamsBillions:
    """Tests for parsing parameter strings into billions."""

    def test_billions_suffix(self):
        assert _parse_params_billions("70B") == 70.0

    def test_billions_decimal(self):
        assert _parse_params_billions("72.7B") == 72.7

    def test_trillions_suffix(self):
        assert _parse_params_billions("1.8T") == pytest.approx(1800.0)

    def test_tilde_prefix(self):
        assert _parse_params_billions("~1.8T (estimated)") == pytest.approx(1800.0)

    def test_millions_suffix(self):
        assert _parse_params_billions("809M") == pytest.approx(0.809)

    def test_thousands_suffix(self):
        assert _parse_params_billions("128K") == pytest.approx(0.000128)

    def test_complex_string(self):
        # "671B total (37B active)" — should parse the first number
        assert _parse_params_billions("671B total (37B active)") == 671.0

    def test_na_returns_none(self):
        assert _parse_params_billions("N/A") is None

    def test_empty_returns_none(self):
        assert _parse_params_billions("") is None

    def test_none_returns_none(self):
        assert _parse_params_billions(None) is None

    def test_no_suffix(self):
        # Raw number — returned as-is
        assert _parse_params_billions("405") == 405.0


# ---------------------------------------------------------------------------
# Tests: /api/filter-options endpoint
# ---------------------------------------------------------------------------

class TestApiFilterOptions:
    """Tests for the /api/filter-options endpoint."""

    def test_returns_families(self, client):
        resp = client.get("/api/filter-options")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "families" in data
        families = data["families"]
        assert isinstance(families, list)
        # Seed data has test-org, another-org, closed-co
        assert "test-org" in families
        assert "another-org" in families
        assert "closed-co" in families

    def test_returns_providers(self, client):
        resp = client.get("/api/filter-options")
        data = resp.get_json()
        assert "providers" in data
        providers = data["providers"]
        assert isinstance(providers, list)
        # Seed data has TestOrg, AnotherOrg, ClosedCo
        assert "TestOrg" in providers
        assert "AnotherOrg" in providers
        assert "ClosedCo" in providers

    def test_returns_params_range(self, client):
        resp = client.get("/api/filter-options")
        data = resp.get_json()
        assert "params_range" in data
        pr = data["params_range"]
        assert "min" in pr
        assert "max" in pr
        # Model A = 70B, Model B = 32B, Proprietary = N/A
        # So min = 32, max = 70
        assert pr["min"] == 32.0
        assert pr["max"] == 70.0

    def test_families_sorted_case_insensitive(self, client):
        resp = client.get("/api/filter-options")
        data = resp.get_json()
        families = data["families"]
        assert families == sorted(families, key=str.lower)

    def test_providers_exclude_na(self, client):
        resp = client.get("/api/filter-options")
        data = resp.get_json()
        assert "N/A" not in data["providers"]


# ---------------------------------------------------------------------------
# Tests: HTML page includes new filter elements
# ---------------------------------------------------------------------------

class TestIndexPageFilters:
    """Tests that the dashboard HTML includes the new filter controls."""

    def test_has_family_filter(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"filter-family" in resp.data

    def test_has_provider_filter(self, client):
        resp = client.get("/")
        assert b"filter-provider" in resp.data

    def test_has_params_range_inputs(self, client):
        resp = client.get("/")
        assert b"filter-params-min" in resp.data
        assert b"filter-params-max" in resp.data

    def test_has_active_filters_badge(self, client):
        resp = client.get("/")
        assert b"active-filters" in resp.data


class TestIndexPageSortAttributes:
    """Tests that column headers have data-sort-type attributes for multi-type sorting."""

    def test_has_sort_type_attributes(self, client):
        """Every sortable column should declare a data-sort-type."""
        resp = client.get("/")
        html = resp.data.decode()
        assert 'data-sort-type="string"' in html
        assert 'data-sort-type="numeric"' in html
        assert 'data-sort-type="date"' in html

    def test_name_column_is_string(self, client):
        resp = client.get("/")
        html = resp.data.decode()
        assert 'data-sort="name" data-sort-type="string"' in html

    def test_params_column_is_numeric(self, client):
        resp = client.get("/")
        html = resp.data.decode()
        assert 'data-sort="params" data-sort-type="numeric"' in html

    def test_release_date_column_is_date(self, client):
        resp = client.get("/")
        html = resp.data.decode()
        assert 'data-sort="release_date" data-sort-type="date"' in html

    def test_benchmark_columns_are_numeric(self, client):
        resp = client.get("/")
        html = resp.data.decode()
        for col in ["mmlu", "humaneval", "gpqa", "math"]:
            assert f'data-sort="{col}" data-sort-type="numeric"' in html, \
                f"Benchmark column {col} should be numeric"

    def test_price_columns_are_numeric(self, client):
        resp = client.get("/")
        html = resp.data.decode()
        assert 'data-sort="api_price_input" data-sort-type="numeric"' in html
        assert 'data-sort="api_price_output" data-sort-type="numeric"' in html

    def test_license_column_is_string(self, client):
        resp = client.get("/")
        html = resp.data.decode()
        assert 'data-sort="license" data-sort-type="string"' in html


# ---------------------------------------------------------------------------
# Tests: Client-side filtering logic (via /api/models query params)
# ---------------------------------------------------------------------------

class TestApiModelsAuthorFilter:
    """Tests for the author query-param filter on /api/models."""

    def test_filter_by_author(self, client):
        resp = client.get("/api/models?author=test-org")
        data = resp.get_json()
        assert data["total"] == 1
        assert data["models"][0]["author"] == "test-org"

    def test_filter_by_author_case_insensitive(self, client):
        resp = client.get("/api/models?author=Test-Org")
        data = resp.get_json()
        assert data["total"] == 1

    def test_filter_by_author_no_match(self, client):
        resp = client.get("/api/models?author=nonexistent")
        data = resp.get_json()
        assert data["total"] == 0


# ---------------------------------------------------------------------------
# Fixtures for settings API tests (need writable config)
# ---------------------------------------------------------------------------

@pytest.fixture
def settings_app(seed_data_path, monkeypatch, tmp_path):
    """Create a test Flask app with a writable temp config file."""
    import yaml

    config_path = str(tmp_path / "settings.yaml")
    initial_config = {
        "watched_organizations": ["meta-llama", "google", "mistralai"],
        "polling_interval_hours": 12,
        "run_on_startup": True,
        "slack_webhook_url": "",
        "trending_thresholds": {
            "enabled": False,
            "download_surge_count": 10000,
            "trending_score": 50,
            "time_window_hours": 24,
        },
    }
    with open(config_path, "w") as f:
        yaml.dump(initial_config, f)

    monkeypatch.setattr(
        "hf_model_monitor.seed_data.SEED_DATA_PATH", seed_data_path
    )
    monkeypatch.setattr(
        "hf_model_monitor.dashboard.load_seed_data",
        lambda path=None: _load_test_seed(seed_data_path),
    )
    monkeypatch.setattr(
        "hf_model_monitor.dashboard.get_seed_categories",
        lambda path=None: ["code", "llm"],
    )
    monkeypatch.setattr(
        "hf_model_monitor.dashboard.ModelStore",
        _MockModelStore,
    )
    # Clear env var to avoid interference
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)

    application = create_app(config_path)
    application.config["TESTING"] = True
    return application


@pytest.fixture
def settings_client(settings_app):
    """Flask test client with writable config."""
    return settings_app.test_client()


# ---------------------------------------------------------------------------
# Tests: GET /api/settings
# ---------------------------------------------------------------------------

class TestApiGetSettings:
    """Tests for the GET /api/settings endpoint."""

    def test_returns_full_config(self, settings_client):
        resp = settings_client.get("/api/settings")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "watched_organizations" in data
        assert "polling_interval_hours" in data
        assert "slack_webhook_url" in data
        assert "run_on_startup" in data
        assert "trending_thresholds" in data
        assert "schema_fields" in data

    def test_schema_fields_have_metadata(self, settings_client):
        resp = settings_client.get("/api/settings")
        data = resp.get_json()
        fields = data["schema_fields"]
        assert "mmlu" in fields
        mmlu = fields["mmlu"]
        assert mmlu["display_name"] == "MMLU"
        assert mmlu["category"] == "performance"
        assert mmlu["type"] == "numeric"
        assert "visible" in mmlu

    def test_all_expected_schema_fields_present(self, settings_client):
        resp = settings_client.get("/api/settings")
        data = resp.get_json()
        fields = data["schema_fields"]
        expected = [
            "name", "author", "category", "release_date", "params",
            "architecture", "license", "mmlu", "humaneval", "gpqa",
            "math", "arena_elo", "context_window", "output_window",
            "multilingual", "vram_estimate", "open_weights",
            "api_available", "downloads", "likes",
            "api_price_input", "api_price_output", "provider_name",
        ]
        for key in expected:
            assert key in fields, f"Missing schema field: {key}"


# ---------------------------------------------------------------------------
# Tests: PUT /api/settings
# ---------------------------------------------------------------------------

class TestApiPutSettings:
    """Tests for the PUT /api/settings endpoint."""

    def test_update_polling_interval(self, settings_client):
        resp = settings_client.put(
            "/api/settings",
            json={"polling_interval_hours": 6},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert data["config"]["polling_interval_hours"] == 6

    def test_update_multiple_fields(self, settings_client):
        resp = settings_client.put(
            "/api/settings",
            json={
                "polling_interval_hours": 8,
                "run_on_startup": False,
                "slack_webhook_url": "https://hooks.slack.com/test",
            },
        )
        assert resp.status_code == 200
        cfg = resp.get_json()["config"]
        assert cfg["polling_interval_hours"] == 8
        assert cfg["run_on_startup"] is False
        assert cfg["slack_webhook_url"] == "https://hooks.slack.com/test"

    def test_update_persists(self, settings_client):
        """Changes survive a fresh GET."""
        settings_client.put(
            "/api/settings",
            json={"polling_interval_hours": 4},
        )
        resp = settings_client.get("/api/settings")
        assert resp.get_json()["polling_interval_hours"] == 4

    def test_invalid_body_returns_400(self, settings_client):
        resp = settings_client.put(
            "/api/settings",
            data="not json",
            content_type="text/plain",
        )
        assert resp.status_code == 400

    def test_invalid_polling_interval_returns_warning(self, settings_client):
        resp = settings_client.put(
            "/api/settings",
            json={"polling_interval_hours": -5},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "warnings" in data
        # Original value should be preserved
        assert data["config"]["polling_interval_hours"] == 12

    def test_invalid_run_on_startup_returns_warning(self, settings_client):
        resp = settings_client.put(
            "/api/settings",
            json={"run_on_startup": "yes"},
        )
        data = resp.get_json()
        assert "warnings" in data
        assert data["config"]["run_on_startup"] is True  # unchanged

    def test_update_orgs_via_settings(self, settings_client):
        resp = settings_client.put(
            "/api/settings",
            json={"watched_organizations": ["openai", "anthropic"]},
        )
        assert resp.status_code == 200
        cfg = resp.get_json()["config"]
        assert cfg["watched_organizations"] == ["openai", "anthropic"]

    def test_update_thresholds_via_settings(self, settings_client):
        resp = settings_client.put(
            "/api/settings",
            json={"trending_thresholds": {"enabled": True, "trending_score": 80}},
        )
        assert resp.status_code == 200
        th = resp.get_json()["config"]["trending_thresholds"]
        assert th["enabled"] is True
        assert th["trending_score"] == 80
        # Unchanged defaults preserved
        assert th["download_surge_count"] == 10000

    def test_update_schema_via_settings(self, settings_client):
        resp = settings_client.put(
            "/api/settings",
            json={"schema_fields": {"architecture": {"visible": True}}},
        )
        assert resp.status_code == 200
        fields = resp.get_json()["config"]["schema_fields"]
        assert fields["architecture"]["visible"] is True

    def test_empty_orgs_returns_warning(self, settings_client):
        resp = settings_client.put(
            "/api/settings",
            json={"watched_organizations": []},
        )
        data = resp.get_json()
        assert "warnings" in data


# ---------------------------------------------------------------------------
# Tests: GET /api/settings/organizations
# ---------------------------------------------------------------------------

class TestApiGetOrganizations:
    """Tests for the GET /api/settings/organizations endpoint."""

    def test_returns_org_list(self, settings_client):
        resp = settings_client.get("/api/settings/organizations")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "organizations" in data
        assert "total" in data
        assert data["total"] == 3
        assert "meta-llama" in data["organizations"]

    def test_reflects_current_config(self, settings_client):
        """Should match what's in the config file."""
        resp = settings_client.get("/api/settings/organizations")
        orgs = resp.get_json()["organizations"]
        assert orgs == ["meta-llama", "google", "mistralai"]


# ---------------------------------------------------------------------------
# Tests: PUT /api/settings/organizations
# ---------------------------------------------------------------------------

class TestApiPutOrganizations:
    """Tests for the PUT /api/settings/organizations endpoint."""

    def test_replace_org_list(self, settings_client):
        resp = settings_client.put(
            "/api/settings/organizations",
            json={"organizations": ["openai", "anthropic", "deepseek-ai"]},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert data["organizations"] == ["openai", "anthropic", "deepseek-ai"]
        assert data["total"] == 3

    def test_replace_persists(self, settings_client):
        settings_client.put(
            "/api/settings/organizations",
            json={"organizations": ["nvidia"]},
        )
        resp = settings_client.get("/api/settings/organizations")
        assert resp.get_json()["organizations"] == ["nvidia"]

    def test_invalid_body_returns_400(self, settings_client):
        resp = settings_client.put(
            "/api/settings/organizations",
            data="not json",
            content_type="text/plain",
        )
        assert resp.status_code == 400

    def test_empty_list_returns_400(self, settings_client):
        resp = settings_client.put(
            "/api/settings/organizations",
            json={"organizations": []},
        )
        assert resp.status_code == 400

    def test_all_invalid_returns_400(self, settings_client):
        resp = settings_client.put(
            "/api/settings/organizations",
            json={"organizations": ["-bad-", "also-bad-"]},
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "invalid" in data

    def test_mixed_valid_invalid_skips_invalid(self, settings_client):
        resp = settings_client.put(
            "/api/settings/organizations",
            json={"organizations": ["openai", "-invalid-", "google"]},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["organizations"] == ["openai", "google"]
        assert "warnings" in data

    def test_deduplicates(self, settings_client):
        resp = settings_client.put(
            "/api/settings/organizations",
            json={"organizations": ["openai", "google", "openai"]},
        )
        assert resp.status_code == 200
        assert resp.get_json()["organizations"] == ["openai", "google"]

    def test_missing_key_returns_400(self, settings_client):
        resp = settings_client.put(
            "/api/settings/organizations",
            json={"orgs": ["openai"]},  # wrong key
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Tests: POST /api/settings/organizations
# ---------------------------------------------------------------------------

class TestApiAddOrganization:
    """Tests for the POST /api/settings/organizations endpoint."""

    def test_add_new_org(self, settings_client):
        resp = settings_client.post(
            "/api/settings/organizations",
            json={"name": "deepseek-ai"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert "deepseek-ai" in data["organizations"]

    def test_add_duplicate_is_ok(self, settings_client):
        resp = settings_client.post(
            "/api/settings/organizations",
            json={"name": "google"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert "already" in data["message"].lower()

    def test_add_invalid_name_returns_400(self, settings_client):
        resp = settings_client.post(
            "/api/settings/organizations",
            json={"name": "-invalid-"},
        )
        assert resp.status_code == 400

    def test_add_empty_name_returns_400(self, settings_client):
        resp = settings_client.post(
            "/api/settings/organizations",
            json={"name": ""},
        )
        assert resp.status_code == 400

    def test_add_missing_name_returns_400(self, settings_client):
        resp = settings_client.post(
            "/api/settings/organizations",
            json={"org": "openai"},  # wrong key
        )
        assert resp.status_code == 400

    def test_add_invalid_body_returns_400(self, settings_client):
        resp = settings_client.post(
            "/api/settings/organizations",
            data="not json",
            content_type="text/plain",
        )
        assert resp.status_code == 400

    def test_add_persists(self, settings_client):
        settings_client.post(
            "/api/settings/organizations",
            json={"name": "nvidia"},
        )
        resp = settings_client.get("/api/settings/organizations")
        assert "nvidia" in resp.get_json()["organizations"]


# ---------------------------------------------------------------------------
# Tests: DELETE /api/settings/organizations/<name>
# ---------------------------------------------------------------------------

class TestApiRemoveOrganization:
    """Tests for the DELETE /api/settings/organizations/<name> endpoint."""

    def test_remove_existing_org(self, settings_client):
        resp = settings_client.delete("/api/settings/organizations/google")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert "google" not in data["organizations"]

    def test_remove_nonexistent_returns_400(self, settings_client):
        resp = settings_client.delete(
            "/api/settings/organizations/nonexistent"
        )
        assert resp.status_code == 400

    def test_remove_last_org_returns_400(self, settings_client):
        """Cannot remove the last org — at least one must remain."""
        # Remove two of three first
        settings_client.delete("/api/settings/organizations/google")
        settings_client.delete("/api/settings/organizations/mistralai")
        # Now try to remove the last one
        resp = settings_client.delete(
            "/api/settings/organizations/meta-llama"
        )
        assert resp.status_code == 400
        assert "last" in resp.get_json()["error"].lower()

    def test_remove_persists(self, settings_client):
        settings_client.delete("/api/settings/organizations/mistralai")
        resp = settings_client.get("/api/settings/organizations")
        assert "mistralai" not in resp.get_json()["organizations"]

    def test_remove_case_sensitive(self, settings_client):
        """HF IDs are case-sensitive: 'Google' != 'google'."""
        resp = settings_client.delete("/api/settings/organizations/Google")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Tests: GET /api/settings/thresholds
# ---------------------------------------------------------------------------

class TestApiGetThresholds:
    """Tests for the GET /api/settings/thresholds endpoint."""

    def test_returns_threshold_fields(self, settings_client):
        resp = settings_client.get("/api/settings/thresholds")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "enabled" in data
        assert "download_surge_count" in data
        assert "trending_score" in data
        assert "time_window_hours" in data

    def test_returns_current_values(self, settings_client):
        resp = settings_client.get("/api/settings/thresholds")
        data = resp.get_json()
        assert data["enabled"] is False
        assert data["download_surge_count"] == 10000
        assert data["trending_score"] == 50
        assert data["time_window_hours"] == 24


# ---------------------------------------------------------------------------
# Tests: PUT /api/settings/thresholds
# ---------------------------------------------------------------------------

class TestApiPutThresholds:
    """Tests for the PUT /api/settings/thresholds endpoint."""

    def test_update_all_thresholds(self, settings_client):
        resp = settings_client.put(
            "/api/settings/thresholds",
            json={
                "enabled": True,
                "download_surge_count": 5000,
                "trending_score": 75,
                "time_window_hours": 48,
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        th = data["thresholds"]
        assert th["enabled"] is True
        assert th["download_surge_count"] == 5000
        assert th["trending_score"] == 75
        assert th["time_window_hours"] == 48

    def test_partial_update_preserves_defaults(self, settings_client):
        resp = settings_client.put(
            "/api/settings/thresholds",
            json={"enabled": True},
        )
        assert resp.status_code == 200
        th = resp.get_json()["thresholds"]
        assert th["enabled"] is True
        # Others unchanged
        assert th["download_surge_count"] == 10000
        assert th["trending_score"] == 50
        assert th["time_window_hours"] == 24

    def test_invalid_body_returns_400(self, settings_client):
        resp = settings_client.put(
            "/api/settings/thresholds",
            data="not json",
            content_type="text/plain",
        )
        assert resp.status_code == 400

    def test_invalid_values_use_defaults(self, settings_client):
        """Invalid threshold values are silently replaced with defaults."""
        resp = settings_client.put(
            "/api/settings/thresholds",
            json={
                "download_surge_count": -100,
                "time_window_hours": 999,
            },
        )
        assert resp.status_code == 200
        th = resp.get_json()["thresholds"]
        assert th["download_surge_count"] == 10000  # default
        assert th["time_window_hours"] == 24  # default

    def test_update_persists(self, settings_client):
        settings_client.put(
            "/api/settings/thresholds",
            json={"enabled": True, "trending_score": 90},
        )
        resp = settings_client.get("/api/settings/thresholds")
        data = resp.get_json()
        assert data["enabled"] is True
        assert data["trending_score"] == 90


# ---------------------------------------------------------------------------
# Tests: GET /api/settings/schema
# ---------------------------------------------------------------------------

class TestApiGetSchema:
    """Tests for the GET /api/settings/schema endpoint."""

    def test_returns_fields_dict(self, settings_client):
        resp = settings_client.get("/api/settings/schema")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "fields" in data
        assert "total" in data
        assert data["total"] == 23  # all schema fields

    def test_each_field_has_metadata(self, settings_client):
        resp = settings_client.get("/api/settings/schema")
        fields = resp.get_json()["fields"]
        for key, field in fields.items():
            assert "display_name" in field, f"{key} missing display_name"
            assert "category" in field, f"{key} missing category"
            assert "type" in field, f"{key} missing type"
            assert "visible" in field, f"{key} missing visible"

    def test_field_types_are_valid(self, settings_client):
        resp = settings_client.get("/api/settings/schema")
        fields = resp.get_json()["fields"]
        valid_types = {"string", "numeric", "date", "boolean"}
        for key, field in fields.items():
            assert field["type"] in valid_types, \
                f"{key} has invalid type: {field['type']}"

    def test_field_categories_are_valid(self, settings_client):
        resp = settings_client.get("/api/settings/schema")
        fields = resp.get_json()["fields"]
        valid_cats = {
            "basic", "performance", "practical",
            "deployment", "community", "cost", "provider",
        }
        for key, field in fields.items():
            assert field["category"] in valid_cats, \
                f"{key} has invalid category: {field['category']}"


# ---------------------------------------------------------------------------
# Tests: PUT /api/settings/schema
# ---------------------------------------------------------------------------

class TestApiPutSchema:
    """Tests for the PUT /api/settings/schema endpoint."""

    def test_toggle_visibility(self, settings_client):
        resp = settings_client.put(
            "/api/settings/schema",
            json={"architecture": {"visible": True}},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert data["fields"]["architecture"]["visible"] is True

    def test_hide_field(self, settings_client):
        resp = settings_client.put(
            "/api/settings/schema",
            json={"mmlu": {"visible": False}},
        )
        assert resp.status_code == 200
        assert resp.get_json()["fields"]["mmlu"]["visible"] is False

    def test_update_display_name(self, settings_client):
        resp = settings_client.put(
            "/api/settings/schema",
            json={"arena_elo": {"display_name": "Chatbot Arena"}},
        )
        assert resp.status_code == 200
        assert resp.get_json()["fields"]["arena_elo"]["display_name"] == "Chatbot Arena"

    def test_multiple_field_updates(self, settings_client):
        resp = settings_client.put(
            "/api/settings/schema",
            json={
                "architecture": {"visible": True},
                "output_window": {"visible": True},
                "mmlu": {"display_name": "MMLU Score"},
            },
        )
        assert resp.status_code == 200
        fields = resp.get_json()["fields"]
        assert fields["architecture"]["visible"] is True
        assert fields["output_window"]["visible"] is True
        assert fields["mmlu"]["display_name"] == "MMLU Score"

    def test_unknown_fields_ignored(self, settings_client):
        resp = settings_client.put(
            "/api/settings/schema",
            json={"nonexistent_field": {"visible": True}},
        )
        assert resp.status_code == 200
        assert "nonexistent_field" not in resp.get_json()["fields"]

    def test_readonly_properties_not_changed(self, settings_client):
        """category and type are read-only; attempts to change them are ignored."""
        resp = settings_client.put(
            "/api/settings/schema",
            json={
                "mmlu": {"category": "basic", "type": "string"},
            },
        )
        assert resp.status_code == 200
        mmlu = resp.get_json()["fields"]["mmlu"]
        assert mmlu["category"] == "performance"  # unchanged
        assert mmlu["type"] == "numeric"  # unchanged

    def test_invalid_body_returns_400(self, settings_client):
        resp = settings_client.put(
            "/api/settings/schema",
            data="not json",
            content_type="text/plain",
        )
        assert resp.status_code == 400

    def test_invalid_visible_value_ignored(self, settings_client):
        """Non-bool visible values are silently ignored."""
        resp = settings_client.put(
            "/api/settings/schema",
            json={"mmlu": {"visible": "yes"}},
        )
        assert resp.status_code == 200
        # Original value preserved
        assert resp.get_json()["fields"]["mmlu"]["visible"] is True

    def test_empty_display_name_ignored(self, settings_client):
        resp = settings_client.put(
            "/api/settings/schema",
            json={"mmlu": {"display_name": ""}},
        )
        assert resp.status_code == 200
        assert resp.get_json()["fields"]["mmlu"]["display_name"] == "MMLU"

    def test_update_persists(self, settings_client):
        settings_client.put(
            "/api/settings/schema",
            json={"architecture": {"visible": True}},
        )
        resp = settings_client.get("/api/settings/schema")
        assert resp.get_json()["fields"]["architecture"]["visible"] is True

    def test_total_count_unchanged(self, settings_client):
        """Schema field count should remain 23 regardless of updates."""
        resp = settings_client.put(
            "/api/settings/schema",
            json={"mmlu": {"visible": False}},
        )
        assert resp.get_json()["total"] == 23
