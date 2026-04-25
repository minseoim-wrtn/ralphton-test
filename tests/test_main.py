from unittest.mock import patch

from hf_model_monitor.main import run


MOCK_TRENDING = [
    {"model_id": "org/new-model", "author": "org", "downloads": 1000, "likes": 50},
    {"model_id": "org/old-model", "author": "org", "downloads": 2000, "likes": 100},
]

MOCK_METADATA = {
    "basic": {"name": "new-model", "org": "org", "params": "7B", "architecture": "llama", "license": "apache-2.0"},
    "performance": {"mmlu": "80.0", "humaneval": "70.0", "arena_rank": "N/A"},
    "practical": {"context_window": "128K", "multilingual": "N/A", "fine_tuning_support": "Yes"},
    "deployment": {"vram_estimate": "~14GB", "quantization_options": "N/A", "api_available": "N/A"},
    "community": {"downloads": 1000, "likes": 50, "trending_rank": "N/A"},
    "cost": {"api_price_per_million_tokens": "N/A", "hosting_cost_estimate": "N/A"},
}

MOCK_ANALYSIS = {
    "summary": "A new 7B model.",
    "differentiation_points": ["Open source"],
    "b2b_assessment": {"pros": ["Free"], "warnings": ["Small"]},
    "takeaway": "Good for testing.",
}


class TestRun:
    @patch("hf_model_monitor.main.send_to_slack", return_value=True)
    @patch("hf_model_monitor.main.format_report", return_value="report text")
    @patch("hf_model_monitor.main.analyze_model", return_value=MOCK_ANALYSIS)
    @patch("hf_model_monitor.main.collect_model_metadata", return_value=MOCK_METADATA)
    @patch("hf_model_monitor.main.get_reference_data", return_value={"GPT-4o": {}})
    @patch("hf_model_monitor.main.save_current_models")
    @patch("hf_model_monitor.main.detect_new_models", return_value=[MOCK_TRENDING[0]])
    @patch("hf_model_monitor.main.load_previous_models", return_value=["org/old-model"])
    @patch("hf_model_monitor.main.fetch_trending_models", return_value=MOCK_TRENDING)
    def test_full_pipeline_with_new_model(self, mock_fetch, mock_load, mock_detect,
                                          mock_save, mock_ref, mock_collect,
                                          mock_analyze, mock_format, mock_send):
        result = run()

        assert result["new_models_found"] == 1
        assert result["reports_sent"] == 1
        assert result["errors"] == []
        mock_collect.assert_called_once()
        mock_analyze.assert_called_once()
        mock_send.assert_called_once_with("report text")
        mock_save.assert_called_once()

    @patch("hf_model_monitor.main.save_current_models")
    @patch("hf_model_monitor.main.detect_new_models", return_value=[])
    @patch("hf_model_monitor.main.load_previous_models", return_value=["org/old-model"])
    @patch("hf_model_monitor.main.fetch_trending_models", return_value=MOCK_TRENDING)
    def test_no_new_models_no_slack(self, mock_fetch, mock_load, mock_detect, mock_save):
        result = run()

        assert result["new_models_found"] == 0
        assert result["reports_sent"] == 0
        mock_save.assert_called_once()

    @patch("hf_model_monitor.main.fetch_trending_models", return_value=[])
    def test_no_trending_data(self, mock_fetch):
        result = run()

        assert result["new_models_found"] == 0
        assert result["reports_sent"] == 0
        assert result["errors"] == []

    @patch("hf_model_monitor.main.send_to_slack", return_value=True)
    @patch("hf_model_monitor.main.format_report", return_value="report")
    @patch("hf_model_monitor.main.analyze_model", return_value=MOCK_ANALYSIS)
    @patch("hf_model_monitor.main.collect_model_metadata", side_effect=[Exception("API fail"), MOCK_METADATA])
    @patch("hf_model_monitor.main.get_reference_data", return_value={"GPT-4o": {}})
    @patch("hf_model_monitor.main.save_current_models")
    @patch("hf_model_monitor.main.detect_new_models", return_value=MOCK_TRENDING)
    @patch("hf_model_monitor.main.load_previous_models", return_value=[])
    @patch("hf_model_monitor.main.fetch_trending_models", return_value=MOCK_TRENDING)
    def test_error_in_one_model_continues(self, mock_fetch, mock_load, mock_detect,
                                           mock_save, mock_ref, mock_collect,
                                           mock_analyze, mock_format, mock_send):
        result = run()

        assert result["new_models_found"] == 2
        assert result["reports_sent"] == 1
        assert len(result["errors"]) == 1
