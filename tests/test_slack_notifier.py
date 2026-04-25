from unittest.mock import patch, MagicMock

from hf_model_monitor.slack_notifier import format_report, send_to_slack


SAMPLE_METADATA = {
    "basic": {"name": "TestModel", "org": "test-org", "params": "7B", "architecture": "llama", "license": "apache-2.0"},
    "performance": {"mmlu": "82.5", "humaneval": "71.0", "arena_rank": "N/A"},
    "practical": {"context_window": "128K", "multilingual": "Yes", "fine_tuning_support": "Yes"},
    "deployment": {"vram_estimate": "~14GB FP16", "quantization_options": "N/A", "api_available": "HF Inference API"},
    "community": {"downloads": 5000, "likes": 300, "trending_rank": 5},
    "cost": {"api_price_per_million_tokens": "N/A", "hosting_cost_estimate": "N/A"},
}

SAMPLE_REFERENCE = {
    "GPT-4o": {"params": "N/A", "mmlu": "88.7", "humaneval": "90.2", "license": "Proprietary", "api_price": "$2.50", "context_window": "128K", "vram": "N/A"},
    "Llama-3.1-405B": {"params": "405B", "mmlu": "87.3", "humaneval": "61.0", "license": "Llama 3.1", "api_price": "$0.90", "context_window": "128K", "vram": "~800GB FP16"},
}


class TestFormatReport:
    def test_contains_model_name(self):
        report = format_report(SAMPLE_METADATA, SAMPLE_REFERENCE)
        assert "TestModel" in report

    def test_contains_comparison_table(self):
        report = format_report(SAMPLE_METADATA, SAMPLE_REFERENCE)
        assert "Params" in report
        assert "MMLU" in report
        assert "HumanEval" in report
        assert "License" in report
        assert "API$/1M" in report
        assert "Context" in report
        assert "VRAM" in report

    def test_contains_reference_models(self):
        report = format_report(SAMPLE_METADATA, SAMPLE_REFERENCE)
        assert "GPT-4o" in report
        assert "Llama-3.1-405B" in report

    def test_contains_license_and_community(self):
        report = format_report(SAMPLE_METADATA, SAMPLE_REFERENCE)
        assert "apache-2.0" in report
        assert "5,000" in report


class TestSendToSlack:
    @patch("hf_model_monitor.slack_notifier.requests.post")
    def test_sends_successfully(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = send_to_slack("test message", webhook_url="https://hooks.slack.com/test")
        assert result is True
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["json"]["text"] == "test message"

    @patch("hf_model_monitor.slack_notifier.requests.post")
    def test_handles_error(self, mock_post):
        mock_post.side_effect = Exception("Network error")
        result = send_to_slack("msg", webhook_url="https://hooks.slack.com/test")
        assert result is False

    def test_no_webhook_url_returns_false(self):
        result = send_to_slack("msg", webhook_url="")
        assert result is False

    def test_empty_message_returns_false(self):
        result = send_to_slack("", webhook_url="https://hooks.slack.com/test")
        assert result is False
