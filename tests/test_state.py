import json
import os
import tempfile

from hf_model_monitor.state import (
    detect_new_models,
    detect_new_models_with_store,
    get_store,
    load_previous_models,
    reset_store,
    save_current_models,
)


class TestLoadPreviousModels:
    def test_returns_empty_when_no_file(self):
        result = load_previous_models("/nonexistent/path.json")
        assert result == []

    def test_loads_existing_state(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(["model-a", "model-b"], f)
            path = f.name
        try:
            result = load_previous_models(path)
            assert result == ["model-a", "model-b"]
        finally:
            os.unlink(path)

    def test_handles_corrupt_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not json")
            path = f.name
        try:
            result = load_previous_models(path)
            assert result == []
        finally:
            os.unlink(path)


class TestSaveCurrentModels:
    def test_saves_and_creates_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "seen.json")
            save_current_models(["m1", "m2"], path)
            with open(path) as f:
                data = json.load(f)
            assert data == ["m1", "m2"]


class TestDetectNewModels:
    def test_detects_new_models(self):
        current = [
            {"model_id": "a", "author": "x"},
            {"model_id": "b", "author": "y"},
            {"model_id": "c", "author": "z"},
        ]
        previous = ["a", "b"]
        result = detect_new_models(current, previous)
        assert len(result) == 1
        assert result[0]["model_id"] == "c"

    def test_first_run_all_new(self):
        current = [
            {"model_id": "a", "author": "x"},
            {"model_id": "b", "author": "y"},
        ]
        result = detect_new_models(current, [])
        assert len(result) == 2

    def test_no_new_models(self):
        current = [{"model_id": "a", "author": "x"}]
        result = detect_new_models(current, ["a"])
        assert result == []


# ---------------------------------------------------------------------------
# Store-backed detection (new API)
# ---------------------------------------------------------------------------
class TestDetectNewModelsWithStore:
    def setup_method(self):
        reset_store()

    def teardown_method(self):
        reset_store()

    def test_first_run_all_new(self, tmp_path):
        db = str(tmp_path / "test.db")
        current = [
            {"model_id": "org/a", "author": "org"},
            {"model_id": "org/b", "author": "org"},
        ]
        result = detect_new_models_with_store(current, db_path=db)
        assert len(result) == 2

    def test_second_run_returns_empty(self, tmp_path):
        db = str(tmp_path / "test.db")
        current = [{"model_id": "org/a", "author": "org"}]
        detect_new_models_with_store(current, db_path=db)
        reset_store()
        result = detect_new_models_with_store(current, db_path=db)
        assert result == []

    def test_mixed_new_and_known(self, tmp_path):
        db = str(tmp_path / "test.db")
        first = [{"model_id": "org/a", "author": "org"}]
        detect_new_models_with_store(first, db_path=db)
        reset_store()

        second = [
            {"model_id": "org/a", "author": "org"},
            {"model_id": "org/b", "author": "org"},
        ]
        result = detect_new_models_with_store(second, db_path=db)
        assert len(result) == 1
        assert result[0]["model_id"] == "org/b"


class TestGetStore:
    def setup_method(self):
        reset_store()

    def teardown_method(self):
        reset_store()

    def test_returns_same_instance(self, tmp_path):
        db = str(tmp_path / "test.db")
        s1 = get_store(db)
        s2 = get_store(db)
        assert s1 is s2

    def test_reset_clears_singleton(self, tmp_path):
        db = str(tmp_path / "test.db")
        s1 = get_store(db)
        reset_store()
        s2 = get_store(db)
        assert s1 is not s2
