from hf_model_monitor.reference_models import get_reference_data

REQUIRED_FIELDS = {"params", "mmlu", "humaneval", "license", "api_price", "context_window", "vram"}
EXPECTED_MODELS = {
    "GPT-4o",
    "Claude Sonnet 4",
    "Gemini 2.5 Pro",
    "Llama-3.1-405B",
    "Qwen-2.5-72B",
    "Mistral-Large",
}


class TestGetReferenceData:
    def test_has_all_expected_models(self):
        data = get_reference_data()
        assert set(data.keys()) == EXPECTED_MODELS

    def test_each_model_has_required_fields(self):
        data = get_reference_data()
        for model_name, model_data in data.items():
            for field in REQUIRED_FIELDS:
                assert field in model_data, f"{model_name} missing field: {field}"

    def test_returns_new_dict(self):
        a = get_reference_data()
        b = get_reference_data()
        assert a is not b
