"""Tests for hf_model_monitor.model_store — SQLite-based state store."""

import json
import os
import tempfile

import pytest

from hf_model_monitor.model_store import ModelStore, _now_iso


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_db(tmp_path):
    """Yield a ModelStore backed by a temporary SQLite file."""
    db_path = str(tmp_path / "test_models.db")
    store = ModelStore(db_path)
    yield store
    store.close()


@pytest.fixture
def sample_models():
    """Return a list of sample model dicts (matches ModelRecord.to_dict())."""
    return [
        {
            "model_id": "meta-llama/Llama-4-Scout-17B-16E",
            "author": "meta-llama",
            "created_at": "2025-04-05T10:00:00Z",
            "last_modified": "2025-04-06T12:00:00Z",
            "pipeline_tag": "text-generation",
            "tags": ["transformers", "llama"],
            "downloads": 50000,
            "likes": 1200,
            "library_name": "transformers",
        },
        {
            "model_id": "google/gemma-3-27b-it",
            "author": "google",
            "created_at": "2025-03-15T08:00:00Z",
            "last_modified": "2025-03-20T09:00:00Z",
            "pipeline_tag": "text-generation",
            "tags": ["transformers", "gemma"],
            "downloads": 80000,
            "likes": 2000,
            "library_name": "transformers",
        },
        {
            "model_id": "mistralai/Mistral-Large-2",
            "author": "mistralai",
            "created_at": "2025-02-01T06:00:00Z",
            "last_modified": "2025-02-10T07:00:00Z",
            "pipeline_tag": "text-generation",
            "tags": ["transformers"],
            "downloads": 30000,
            "likes": 800,
            "library_name": "transformers",
        },
    ]


# ---------------------------------------------------------------------------
# Schema & initialization
# ---------------------------------------------------------------------------
class TestModelStoreInit:
    def test_creates_database_file(self, tmp_path):
        db_path = str(tmp_path / "sub" / "dir" / "models.db")
        store = ModelStore(db_path)
        assert os.path.exists(db_path)
        store.close()

    def test_creates_tables(self, tmp_db):
        cursor = tmp_db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row["name"] for row in cursor.fetchall()}
        assert "known_models" in tables
        assert "poll_history" in tables

    def test_known_models_schema(self, tmp_db):
        cursor = tmp_db._conn.execute("PRAGMA table_info(known_models)")
        columns = {row[1] for row in cursor.fetchall()}
        expected = {
            "model_id",
            "author",
            "first_seen_at",
            "last_checked_at",
            "created_at",
            "last_modified",
            "pipeline_tag",
            "library_name",
            "downloads",
            "likes",
            "tags",
            "status",
            "metadata_json",
            "notified",
        }
        assert expected.issubset(columns)

    def test_reopen_existing_db(self, tmp_path):
        db_path = str(tmp_path / "models.db")
        store1 = ModelStore(db_path)
        store1.upsert_model({"model_id": "org/test-model", "author": "org"})
        store1.close()

        store2 = ModelStore(db_path)
        assert store2.is_known("org/test-model")
        store2.close()


# ---------------------------------------------------------------------------
# is_known
# ---------------------------------------------------------------------------
class TestIsKnown:
    def test_unknown_model(self, tmp_db):
        assert tmp_db.is_known("nonexistent/model") is False

    def test_known_model(self, tmp_db, sample_models):
        tmp_db.upsert_models(sample_models)
        assert tmp_db.is_known("meta-llama/Llama-4-Scout-17B-16E") is True

    def test_after_deletion(self, tmp_db, sample_models):
        tmp_db.upsert_models(sample_models)
        tmp_db.delete_model("meta-llama/Llama-4-Scout-17B-16E")
        assert tmp_db.is_known("meta-llama/Llama-4-Scout-17B-16E") is False


# ---------------------------------------------------------------------------
# upsert_models
# ---------------------------------------------------------------------------
class TestUpsertModels:
    def test_first_insert_returns_all_new(self, tmp_db, sample_models):
        new = tmp_db.upsert_models(sample_models)
        assert len(new) == 3
        assert {m["model_id"] for m in new} == {
            "meta-llama/Llama-4-Scout-17B-16E",
            "google/gemma-3-27b-it",
            "mistralai/Mistral-Large-2",
        }

    def test_second_insert_returns_empty(self, tmp_db, sample_models):
        tmp_db.upsert_models(sample_models)
        new = tmp_db.upsert_models(sample_models)
        assert new == []

    def test_mixed_new_and_existing(self, tmp_db, sample_models):
        tmp_db.upsert_models(sample_models[:2])
        new = tmp_db.upsert_models(sample_models)
        assert len(new) == 1
        assert new[0]["model_id"] == "mistralai/Mistral-Large-2"

    def test_updates_last_checked_at(self, tmp_db, sample_models):
        tmp_db.upsert_models(sample_models[:1])
        before = tmp_db.get_model("meta-llama/Llama-4-Scout-17B-16E")
        first_checked = before["last_checked_at"]

        # Re-upsert to update last_checked_at
        tmp_db.upsert_models(sample_models[:1])
        after = tmp_db.get_model("meta-llama/Llama-4-Scout-17B-16E")
        assert after["last_checked_at"] >= first_checked

    def test_preserves_first_seen_at(self, tmp_db, sample_models):
        tmp_db.upsert_models(sample_models[:1])
        first = tmp_db.get_model("meta-llama/Llama-4-Scout-17B-16E")
        original_first_seen = first["first_seen_at"]

        tmp_db.upsert_models(sample_models[:1])
        after = tmp_db.get_model("meta-llama/Llama-4-Scout-17B-16E")
        assert after["first_seen_at"] == original_first_seen

    def test_updates_downloads_and_likes(self, tmp_db):
        model = {"model_id": "org/model", "author": "org", "downloads": 100, "likes": 10}
        tmp_db.upsert_models([model])

        updated = {"model_id": "org/model", "author": "org", "downloads": 200, "likes": 20}
        tmp_db.upsert_models([updated])

        result = tmp_db.get_model("org/model")
        assert result["downloads"] == 200
        assert result["likes"] == 20

    def test_skips_empty_model_id(self, tmp_db):
        models = [{"model_id": "", "author": "x"}, {"author": "y"}]
        new = tmp_db.upsert_models(models)
        assert new == []
        assert tmp_db.count_models() == 0

    def test_auto_extracts_author_from_model_id(self, tmp_db):
        model = {"model_id": "deepseek-ai/DeepSeek-V4"}
        tmp_db.upsert_models([model])
        result = tmp_db.get_model("deepseek-ai/DeepSeek-V4")
        assert result["author"] == "deepseek-ai"

    def test_new_model_has_status_new(self, tmp_db, sample_models):
        tmp_db.upsert_models(sample_models[:1])
        result = tmp_db.get_model("meta-llama/Llama-4-Scout-17B-16E")
        assert result["status"] == "new"

    def test_new_model_has_notified_false(self, tmp_db, sample_models):
        tmp_db.upsert_models(sample_models[:1])
        result = tmp_db.get_model("meta-llama/Llama-4-Scout-17B-16E")
        assert result["notified"] == 0


# ---------------------------------------------------------------------------
# upsert_model (single)
# ---------------------------------------------------------------------------
class TestUpsertModel:
    def test_returns_true_for_new(self, tmp_db):
        assert tmp_db.upsert_model({"model_id": "org/new"}) is True

    def test_returns_false_for_existing(self, tmp_db):
        tmp_db.upsert_model({"model_id": "org/existing"})
        assert tmp_db.upsert_model({"model_id": "org/existing"}) is False


# ---------------------------------------------------------------------------
# get_model
# ---------------------------------------------------------------------------
class TestGetModel:
    def test_returns_none_for_unknown(self, tmp_db):
        assert tmp_db.get_model("nonexistent") is None

    def test_returns_dict_with_all_fields(self, tmp_db, sample_models):
        tmp_db.upsert_models(sample_models[:1])
        result = tmp_db.get_model("meta-llama/Llama-4-Scout-17B-16E")
        assert result is not None
        assert result["model_id"] == "meta-llama/Llama-4-Scout-17B-16E"
        assert result["author"] == "meta-llama"
        assert result["pipeline_tag"] == "text-generation"
        assert result["downloads"] == 50000
        assert result["likes"] == 1200

    def test_tags_parsed_as_list(self, tmp_db, sample_models):
        tmp_db.upsert_models(sample_models[:1])
        result = tmp_db.get_model("meta-llama/Llama-4-Scout-17B-16E")
        assert isinstance(result["tags"], list)
        assert "transformers" in result["tags"]

    def test_metadata_json_parsed(self, tmp_db, sample_models):
        tmp_db.upsert_models(sample_models[:1])
        result = tmp_db.get_model("meta-llama/Llama-4-Scout-17B-16E")
        assert isinstance(result["metadata_json"], dict)


# ---------------------------------------------------------------------------
# get_all_models
# ---------------------------------------------------------------------------
class TestGetAllModels:
    def test_empty_store(self, tmp_db):
        assert tmp_db.get_all_models() == []

    def test_returns_all(self, tmp_db, sample_models):
        tmp_db.upsert_models(sample_models)
        result = tmp_db.get_all_models()
        assert len(result) == 3

    def test_ordered_by_first_seen_desc(self, tmp_db):
        # Insert one at a time to get distinct timestamps
        tmp_db.upsert_model({"model_id": "org/first"})
        tmp_db.upsert_model({"model_id": "org/second"})
        tmp_db.upsert_model({"model_id": "org/third"})
        result = tmp_db.get_all_models()
        assert result[0]["model_id"] == "org/third"
        assert result[2]["model_id"] == "org/first"


# ---------------------------------------------------------------------------
# get_models_since
# ---------------------------------------------------------------------------
class TestGetModelsSince:
    def test_filters_by_timestamp(self, tmp_db):
        tmp_db.upsert_model({"model_id": "org/old"})
        cutoff = _now_iso()
        tmp_db.upsert_model({"model_id": "org/new"})

        result = tmp_db.get_models_since(cutoff)
        ids = {m["model_id"] for m in result}
        assert "org/new" in ids
        # org/old may or may not be included depending on timing precision
        # but org/new should definitely be there

    def test_empty_result_for_future(self, tmp_db, sample_models):
        tmp_db.upsert_models(sample_models)
        result = tmp_db.get_models_since("2099-01-01T00:00:00")
        assert result == []


# ---------------------------------------------------------------------------
# Status management
# ---------------------------------------------------------------------------
class TestStatusManagement:
    def test_mark_as_known(self, tmp_db):
        tmp_db.upsert_model({"model_id": "org/test"})
        assert tmp_db.get_model("org/test")["status"] == "new"

        tmp_db.mark_as_known("org/test")
        assert tmp_db.get_model("org/test")["status"] == "known"

    def test_get_new_models(self, tmp_db):
        tmp_db.upsert_model({"model_id": "org/a"})
        tmp_db.upsert_model({"model_id": "org/b"})
        tmp_db.mark_as_known("org/a")

        new = tmp_db.get_new_models()
        assert len(new) == 1
        assert new[0]["model_id"] == "org/b"

    def test_mark_as_notified(self, tmp_db):
        tmp_db.upsert_model({"model_id": "org/test"})
        assert tmp_db.get_model("org/test")["notified"] == 0

        tmp_db.mark_as_notified("org/test")
        assert tmp_db.get_model("org/test")["notified"] == 1

    def test_get_unnotified_models(self, tmp_db):
        tmp_db.upsert_model({"model_id": "org/a"})
        tmp_db.upsert_model({"model_id": "org/b"})
        tmp_db.mark_as_notified("org/a")

        unnotified = tmp_db.get_unnotified_models()
        assert len(unnotified) == 1
        assert unnotified[0]["model_id"] == "org/b"

    def test_get_models_by_status(self, tmp_db):
        tmp_db.upsert_model({"model_id": "org/a"})
        tmp_db.upsert_model({"model_id": "org/b"})
        tmp_db.mark_as_known("org/a")

        known = tmp_db.get_models_by_status("known")
        assert len(known) == 1
        assert known[0]["model_id"] == "org/a"


# ---------------------------------------------------------------------------
# Poll history
# ---------------------------------------------------------------------------
class TestPollHistory:
    def test_record_and_retrieve(self, tmp_db):
        tmp_db.record_poll("meta-llama", models_found=10, new_models=2)
        history = tmp_db.get_poll_history()
        assert len(history) == 1
        assert history[0]["source"] == "meta-llama"
        assert history[0]["models_found"] == 10
        assert history[0]["new_models"] == 2

    def test_get_last_poll_time(self, tmp_db):
        assert tmp_db.get_last_poll_time() is None
        tmp_db.record_poll("trending", models_found=5, new_models=1)
        last = tmp_db.get_last_poll_time()
        assert last is not None
        assert "T" in last  # ISO-8601 format

    def test_history_ordered_desc(self, tmp_db):
        tmp_db.record_poll("first", models_found=1, new_models=0)
        tmp_db.record_poll("second", models_found=2, new_models=1)
        history = tmp_db.get_poll_history()
        assert history[0]["source"] == "second"
        assert history[1]["source"] == "first"

    def test_history_limit(self, tmp_db):
        for i in range(10):
            tmp_db.record_poll(f"poll-{i}", models_found=i, new_models=0)
        history = tmp_db.get_poll_history(limit=3)
        assert len(history) == 3

    def test_record_with_errors(self, tmp_db):
        tmp_db.record_poll("org", models_found=0, new_models=0, errors="timeout")
        history = tmp_db.get_poll_history()
        assert history[0]["errors"] == "timeout"


# ---------------------------------------------------------------------------
# JSON import/export
# ---------------------------------------------------------------------------
class TestJsonImportExport:
    def test_import_from_json(self, tmp_db, tmp_path):
        json_path = str(tmp_path / "seen.json")
        with open(json_path, "w") as f:
            json.dump(["org/model-a", "org/model-b"], f)

        imported = tmp_db.import_from_json(json_path)
        assert imported == 2
        assert tmp_db.is_known("org/model-a")
        assert tmp_db.is_known("org/model-b")

    def test_imported_models_are_known_status(self, tmp_db, tmp_path):
        json_path = str(tmp_path / "seen.json")
        with open(json_path, "w") as f:
            json.dump(["org/model-a"], f)

        tmp_db.import_from_json(json_path)
        model = tmp_db.get_model("org/model-a")
        assert model["status"] == "known"
        assert model["notified"] == 1

    def test_import_skips_already_known(self, tmp_db, tmp_path):
        tmp_db.upsert_model({"model_id": "org/existing"})
        json_path = str(tmp_path / "seen.json")
        with open(json_path, "w") as f:
            json.dump(["org/existing", "org/new-one"], f)

        imported = tmp_db.import_from_json(json_path)
        assert imported == 1  # only new-one

    def test_import_nonexistent_file(self, tmp_db):
        imported = tmp_db.import_from_json("/nonexistent/path.json")
        assert imported == 0

    def test_import_corrupt_json(self, tmp_db, tmp_path):
        json_path = str(tmp_path / "bad.json")
        with open(json_path, "w") as f:
            f.write("not json")
        imported = tmp_db.import_from_json(json_path)
        assert imported == 0

    def test_import_extracts_author(self, tmp_db, tmp_path):
        json_path = str(tmp_path / "seen.json")
        with open(json_path, "w") as f:
            json.dump(["deepseek-ai/DeepSeek-V4"], f)

        tmp_db.import_from_json(json_path)
        model = tmp_db.get_model("deepseek-ai/DeepSeek-V4")
        assert model["author"] == "deepseek-ai"

    def test_export_model_ids(self, tmp_db, sample_models):
        tmp_db.upsert_models(sample_models)
        ids = tmp_db.export_model_ids()
        assert len(ids) == 3
        assert all(isinstance(i, str) for i in ids)

    def test_export_to_json(self, tmp_db, sample_models, tmp_path):
        tmp_db.upsert_models(sample_models)
        json_path = str(tmp_path / "export.json")
        tmp_db.export_to_json(json_path)

        with open(json_path) as f:
            data = json.load(f)
        assert len(data) == 3

    def test_roundtrip_import_export(self, tmp_db, tmp_path):
        """Import from JSON, export back, verify contents match."""
        original_ids = ["org/a", "org/b", "org/c"]
        json_in = str(tmp_path / "in.json")
        with open(json_in, "w") as f:
            json.dump(original_ids, f)

        tmp_db.import_from_json(json_in)
        exported = tmp_db.export_model_ids()
        assert set(exported) == set(original_ids)


# ---------------------------------------------------------------------------
# Search & filtering
# ---------------------------------------------------------------------------
class TestSearchAndFiltering:
    def test_get_models_by_author(self, tmp_db, sample_models):
        tmp_db.upsert_models(sample_models)
        result = tmp_db.get_models_by_author("google")
        assert len(result) == 1
        assert result[0]["model_id"] == "google/gemma-3-27b-it"

    def test_get_models_by_author_empty(self, tmp_db, sample_models):
        tmp_db.upsert_models(sample_models)
        assert tmp_db.get_models_by_author("nonexistent") == []

    def test_search_models(self, tmp_db, sample_models):
        tmp_db.upsert_models(sample_models)
        result = tmp_db.search_models("llama")
        assert len(result) == 1
        assert "llama" in result[0]["model_id"].lower()

    def test_search_case_insensitive(self, tmp_db, sample_models):
        tmp_db.upsert_models(sample_models)
        result = tmp_db.search_models("GEMMA")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------
class TestDelete:
    def test_delete_existing(self, tmp_db):
        tmp_db.upsert_model({"model_id": "org/to-delete"})
        assert tmp_db.delete_model("org/to-delete") is True
        assert tmp_db.is_known("org/to-delete") is False

    def test_delete_nonexistent(self, tmp_db):
        assert tmp_db.delete_model("org/nope") is False


# ---------------------------------------------------------------------------
# count_models
# ---------------------------------------------------------------------------
class TestCountModels:
    def test_empty(self, tmp_db):
        assert tmp_db.count_models() == 0

    def test_after_inserts(self, tmp_db, sample_models):
        tmp_db.upsert_models(sample_models)
        assert tmp_db.count_models() == 3


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------
class TestContextManager:
    def test_context_manager(self, tmp_path):
        db_path = str(tmp_path / "ctx.db")
        with ModelStore(db_path) as store:
            store.upsert_model({"model_id": "org/test"})
            assert store.is_known("org/test")
        # After context exit, store is closed — reopen to verify persistence
        store2 = ModelStore(db_path)
        assert store2.is_known("org/test")
        store2.close()


# ---------------------------------------------------------------------------
# N/A tolerance (fields not provided)
# ---------------------------------------------------------------------------
class TestNATolerance:
    def test_minimal_model_dict(self, tmp_db):
        """A model dict with only model_id should work with N/A defaults."""
        tmp_db.upsert_model({"model_id": "org/minimal"})
        result = tmp_db.get_model("org/minimal")
        assert result is not None
        assert result["pipeline_tag"] == "N/A"
        assert result["library_name"] == "N/A"
        assert result["created_at"] == "N/A"
        assert result["downloads"] == 0
        assert result["likes"] == 0

    def test_tags_default_to_empty_list(self, tmp_db):
        tmp_db.upsert_model({"model_id": "org/no-tags"})
        result = tmp_db.get_model("org/no-tags")
        assert result["tags"] == []
