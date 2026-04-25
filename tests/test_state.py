import json
import os
import tempfile

from hf_model_monitor.state import load_previous_models, save_current_models, detect_new_models


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
