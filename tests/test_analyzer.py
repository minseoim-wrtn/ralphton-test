import json
from unittest.mock import patch, MagicMock

from hf_model_monitor.analyzer import analyze_model, _fallback_analysis, _parse_analysis


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
}

MOCK_CLAUDE_RESPONSE = json.dumps({
    "summary": "TestModel is a 7B parameter model with strong coding performance.",
    "differentiation_points": ["Open source with Apache 2.0", "Strong multilingual support"],
    "b2b_assessment": {"pros": ["Free to use commercially"], "warnings": ["Smaller than GPT-4o"]},
    "takeaway": "Good candidate for cost-sensitive B2B POCs.",
})


class TestAnalyzeModel:
    @patch("hf_model_monitor.analyzer.ANTHROPIC_API_KEY", "test-key")
    @patch("hf_model_monitor.analyzer.anthropic.Anthropic")
    def test_successful_analysis(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text=MOCK_CLAUDE_RESPONSE)]
        mock_client.messages.create.return_value = mock_msg

        result = analyze_model(SAMPLE_METADATA, SAMPLE_REFERENCE)

        assert "summary" in result
        assert "differentiation_points" in result
        assert "b2b_assessment" in result
        assert "takeaway" in result
        assert "7B" in result["summary"]

    @patch("hf_model_monitor.analyzer.ANTHROPIC_API_KEY", "test-key")
    @patch("hf_model_monitor.analyzer.anthropic.Anthropic")
    def test_api_error_returns_fallback(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API down")

        result = analyze_model(SAMPLE_METADATA, SAMPLE_REFERENCE)

        assert "summary" in result
        assert "TestModel" in result["summary"]
        assert "Manual review" in result["summary"]

    @patch("hf_model_monitor.analyzer.ANTHROPIC_API_KEY", "")
    def test_no_api_key_returns_fallback(self):
        result = analyze_model(SAMPLE_METADATA, SAMPLE_REFERENCE)
        assert "summary" in result
        assert "Manual review" in result["summary"]


class TestParseAnalysis:
    def test_parses_valid_json(self):
        result = _parse_analysis(MOCK_CLAUDE_RESPONSE)
        assert result["summary"] == "TestModel is a 7B parameter model with strong coding performance."

    def test_parses_json_with_markdown_fence(self):
        fenced = f"```json\n{MOCK_CLAUDE_RESPONSE}\n```"
        result = _parse_analysis(fenced)
        assert "TestModel" in result["summary"]

    def test_invalid_json_returns_fallback(self):
        result = _parse_analysis("not json at all")
        assert "summary" in result


class TestFallbackAnalysis:
    def test_uses_model_name(self):
        result = _fallback_analysis(SAMPLE_METADATA)
        assert "TestModel" in result["summary"]

    def test_empty_metadata(self):
        result = _fallback_analysis({})
        assert "Unknown" in result["summary"]
