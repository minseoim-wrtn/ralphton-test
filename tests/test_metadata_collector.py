from unittest.mock import patch

from hf_model_monitor.metadata_collector import collect_model_metadata


MOCK_MODEL_INFO = {
    "id": "test-org/test-model",
    "author": "test-org",
    "downloads": 5000,
    "likes": 300,
    "tags": ["transformers", "license:apache-2.0", "multilingual"],
    "pipeline_tag": "text-generation",
    "library_name": "transformers",
    "inference": True,
    "config": {
        "model_type": "llama",
        "max_position_embeddings": 131072,
    },
    "safetensors": {"total": 7000000000},
    "cardData": {
        "license": "apache-2.0",
        "model-index": [
            {
                "results": [
                    {
                        "dataset": {"name": "MMLU"},
                        "metrics": [{"name": "accuracy", "value": 82.5}],
                    },
                    {
                        "dataset": {"name": "HumanEval"},
                        "metrics": [{"name": "pass@1", "value": 71.0}],
                    },
                ]
            }
        ],
    },
}


class TestCollectModelMetadata:
    @patch("hf_model_monitor.metadata_collector.fetch_model_info")
    def test_collects_all_six_categories(self, mock_fetch):
        mock_fetch.return_value = MOCK_MODEL_INFO
        result = collect_model_metadata("test-org/test-model")

        assert "basic" in result
        assert "performance" in result
        assert "practical" in result
        assert "deployment" in result
        assert "community" in result
        assert "cost" in result

    @patch("hf_model_monitor.metadata_collector.fetch_model_info")
    def test_basic_fields(self, mock_fetch):
        mock_fetch.return_value = MOCK_MODEL_INFO
        result = collect_model_metadata("test-org/test-model")
        basic = result["basic"]

        assert basic["name"] == "test-model"
        assert basic["org"] == "test-org"
        assert "7.0B" in basic["params"]
        assert basic["architecture"] == "llama"
        assert basic["license"] == "apache-2.0"

    @patch("hf_model_monitor.metadata_collector.fetch_model_info")
    def test_performance_fields(self, mock_fetch):
        mock_fetch.return_value = MOCK_MODEL_INFO
        result = collect_model_metadata("test-org/test-model")
        perf = result["performance"]

        assert perf["mmlu"] == "82.5"
        assert perf["humaneval"] == "71.0"

    @patch("hf_model_monitor.metadata_collector.fetch_model_info")
    def test_practical_fields(self, mock_fetch):
        mock_fetch.return_value = MOCK_MODEL_INFO
        result = collect_model_metadata("test-org/test-model")
        pract = result["practical"]

        assert "128K" in pract["context_window"]
        assert pract["multilingual"] == "Yes"
        assert pract["fine_tuning_support"] == "Yes"

    @patch("hf_model_monitor.metadata_collector.fetch_model_info")
    def test_deployment_fields(self, mock_fetch):
        mock_fetch.return_value = MOCK_MODEL_INFO
        result = collect_model_metadata("test-org/test-model")
        deploy = result["deployment"]

        assert "GB" in deploy["vram_estimate"]
        assert deploy["api_available"] == "HF Inference API"

    @patch("hf_model_monitor.metadata_collector.fetch_model_info")
    def test_community_with_trending_data(self, mock_fetch):
        mock_fetch.return_value = MOCK_MODEL_INFO
        trending = {"downloads": 9999, "likes": 888, "trending_rank": 3}
        result = collect_model_metadata("test-org/test-model", trending_data=trending)
        comm = result["community"]

        assert comm["downloads"] == 9999
        assert comm["likes"] == 888
        assert comm["trending_rank"] == 3

    @patch("hf_model_monitor.metadata_collector.fetch_model_info")
    def test_missing_data_returns_na(self, mock_fetch):
        mock_fetch.return_value = {}
        result = collect_model_metadata("unknown/model")
        assert result["basic"]["params"] == "N/A"
        assert result["performance"]["mmlu"] == "N/A"
        assert result["cost"]["api_price_per_million_tokens"] == "N/A"

    @patch("hf_model_monitor.metadata_collector.fetch_model_info")
    def test_api_failure_returns_empty_metadata(self, mock_fetch):
        mock_fetch.return_value = {}
        result = collect_model_metadata("fail/model", trending_data=None)
        assert result["basic"]["name"] == "model"
        assert result["basic"]["org"] == "fail"
