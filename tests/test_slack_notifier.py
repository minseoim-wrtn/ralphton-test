from unittest.mock import patch, MagicMock

import pytest
import requests

from hf_model_monitor.config import validate_webhook_url
from hf_model_monitor.slack_notifier import (
    SlackNotifier,
    _build_benchmark_blocks,
    _build_pricing_blocks,
    _build_tags_block,
    _is_available,
    classify_crawler_error,
    format_block_kit_report,
    format_crawler_errors_summary,
    format_error_alert,
    format_pipeline_error_alert,
    format_report,
    send_to_slack,
)


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
        mock_post.side_effect = requests.exceptions.ConnectionError("Network error")
        result = send_to_slack("msg", webhook_url="https://hooks.slack.com/test")
        assert result is False

    def test_no_webhook_url_returns_false(self):
        result = send_to_slack("msg", webhook_url="")
        assert result is False

    def test_empty_message_returns_false(self):
        result = send_to_slack("", webhook_url="https://hooks.slack.com/test")
        assert result is False


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------
class TestClassifyCrawlerError:
    """Test keyword-based error classification."""

    @pytest.mark.parametrize(
        "msg, expected",
        [
            ("Request timed out after 30s", "Timeout"),
            ("Connection timed out", "Timeout"),
            ("Read timeout on request", "Timeout"),
            ("Rate limited by HuggingFace API (retry after 60s)", "Rate Limited"),
            ("HF API error — 429 Too Many Requests", "Rate Limited"),
            ("Authentication error — check HF_TOKEN", "Authentication"),
            ("HTTP 401 Unauthorized", "Authentication"),
            ("HTTP 403 Forbidden", "Authentication"),
            ("Connection error: DNS resolution failed", "Network"),
            ("Network unreachable", "Network"),
            ("Failed to resolve hostname", "Network"),
            ("Server error 500", "Server Error"),
            ("HTTP 502 Bad Gateway", "Server Error"),
            ("HTTP 503 Service Unavailable", "Server Error"),
            ("HTTP 504 Gateway Timeout", "Server Error"),  # "504" status code matches first
            ("Resource not found: /api/models/xyz", "Not Found"),
            ("HTTP 404 Not Found", "Not Found"),
            ("Failed to parse JSON response", "Structure Change"),
            ("Unexpected response format from API", "Structure Change"),
            ("JSON decode error", "Structure Change"),
            ("Something completely unknown happened", "Unknown"),
            ("", "Unknown"),
        ],
    )
    def test_classifies_error_types(self, msg, expected):
        assert classify_crawler_error(msg) == expected

    def test_case_insensitive(self):
        assert classify_crawler_error("CONNECTION ERROR") == "Network"
        assert classify_crawler_error("TIMEOUT") == "Timeout"

    def test_first_match_wins(self):
        # "server error" matches before "timeout" (status codes take priority)
        msg = "Server error: request timed out"
        assert classify_crawler_error(msg) == "Server Error"


# ---------------------------------------------------------------------------
# Single error alert formatting
# ---------------------------------------------------------------------------
class TestFormatErrorAlert:
    """Test format_error_alert() for single crawler failures."""

    def test_contains_source_and_error(self):
        alert = format_error_alert(
            source="meta-llama",
            error_msg="Connection timed out",
        )
        assert "meta-llama" in alert
        assert "Connection timed out" in alert

    def test_auto_classifies_error_type(self):
        alert = format_error_alert(
            source="google",
            error_msg="Request timed out after 30s",
        )
        assert "Timeout" in alert
        assert ":hourglass:" in alert

    def test_manual_error_type_override(self):
        alert = format_error_alert(
            source="google",
            error_msg="some error",
            error_type="Custom Type",
        )
        assert "Custom Type" in alert

    def test_contains_consecutive_failures(self):
        alert = format_error_alert(
            source="openai",
            error_msg="Network error",
            consecutive_failures=3,
        )
        assert "Consecutive failures: 3" in alert

    def test_contains_next_retry(self):
        alert = format_error_alert(
            source="openai",
            error_msg="Network error",
            next_retry_hours=6,
        )
        assert "Next retry in 6h" in alert

    def test_contains_timestamp(self):
        alert = format_error_alert(
            source="test",
            error_msg="error",
        )
        assert "UTC" in alert

    def test_network_error_gets_plug_emoji(self):
        alert = format_error_alert(
            source="test",
            error_msg="Connection error: DNS resolution failed",
        )
        assert ":electric_plug:" in alert

    def test_rate_limit_gets_snail_emoji(self):
        alert = format_error_alert(
            source="test",
            error_msg="Rate limited by API (429)",
        )
        assert ":snail:" in alert


# ---------------------------------------------------------------------------
# Multi-org error summary formatting
# ---------------------------------------------------------------------------
class TestFormatCrawlerErrorsSummary:
    """Test format_crawler_errors_summary() for batch error reports."""

    def test_single_org_failure(self):
        errors = [{"source": "meta-llama", "error": "Connection timed out"}]
        alert = format_crawler_errors_summary(errors, total_orgs=5)
        assert "1/5" in alert
        assert "4 succeeded" in alert
        assert "meta-llama" in alert
        assert "Timeout" in alert

    def test_multiple_org_failures(self):
        errors = [
            {"source": "meta-llama", "error": "Connection timed out"},
            {"source": "google", "error": "Server error 500"},
            {"source": "openai", "error": "Rate limited (429)"},
        ]
        alert = format_crawler_errors_summary(errors, total_orgs=10)
        assert "3/10" in alert
        assert "7 succeeded" in alert
        assert "meta-llama" in alert
        assert "google" in alert
        assert "openai" in alert
        assert "Timeout" in alert
        assert "Server Error" in alert
        assert "Rate Limited" in alert

    def test_all_orgs_failed(self):
        errors = [
            {"source": "org1", "error": "Network down"},
            {"source": "org2", "error": "Network down"},
        ]
        alert = format_crawler_errors_summary(errors, total_orgs=2)
        assert "2/2" in alert
        assert "0 succeeded" in alert

    def test_contains_next_retry(self):
        errors = [{"source": "org", "error": "timeout"}]
        alert = format_crawler_errors_summary(errors, total_orgs=1, next_retry_hours=8)
        assert "Next retry in 8h" in alert

    def test_contains_rotating_light(self):
        errors = [{"source": "org", "error": "error"}]
        alert = format_crawler_errors_summary(errors, total_orgs=1)
        assert ":rotating_light:" in alert

    def test_contains_timestamp(self):
        errors = [{"source": "org", "error": "error"}]
        alert = format_crawler_errors_summary(errors, total_orgs=1)
        assert "UTC" in alert

    def test_error_messages_in_code_blocks(self):
        errors = [{"source": "org", "error": "Something broke badly"}]
        alert = format_crawler_errors_summary(errors, total_orgs=1)
        assert "`Something broke badly`" in alert


# ---------------------------------------------------------------------------
# Pipeline error alert formatting
# ---------------------------------------------------------------------------
class TestFormatPipelineErrorAlert:
    """Test format_pipeline_error_alert() for catastrophic failures."""

    def test_contains_error_message(self):
        alert = format_pipeline_error_alert("Detection pipeline failed: DB locked")
        assert "Detection pipeline failed: DB locked" in alert

    def test_contains_pipeline_error_header(self):
        alert = format_pipeline_error_alert("error")
        assert "Pipeline Error" in alert

    def test_classifies_error_type(self):
        alert = format_pipeline_error_alert("Connection error: network unreachable")
        assert "Network" in alert
        assert ":electric_plug:" in alert

    def test_normal_severity_below_3_failures(self):
        alert = format_pipeline_error_alert("error", consecutive_failures=1)
        assert "Normal" in alert
        assert ":large_orange_circle:" in alert

    def test_high_severity_at_3_plus_failures(self):
        alert = format_pipeline_error_alert("error", consecutive_failures=3)
        assert "HIGH" in alert
        assert ":red_circle:" in alert

    def test_high_severity_at_5_failures(self):
        alert = format_pipeline_error_alert("error", consecutive_failures=5)
        assert "HIGH" in alert
        assert "multiple consecutive failures" in alert

    def test_contains_consecutive_count(self):
        alert = format_pipeline_error_alert("error", consecutive_failures=7)
        assert "Consecutive failures: 7" in alert

    def test_contains_next_retry(self):
        alert = format_pipeline_error_alert("error", next_retry_hours=4)
        assert "Next retry in 4h" in alert

    def test_contains_timestamp(self):
        alert = format_pipeline_error_alert("error")
        assert "UTC" in alert


# ---------------------------------------------------------------------------
# Webhook URL validation (config.py)
# ---------------------------------------------------------------------------
class TestValidateWebhookUrl:
    """Test validate_webhook_url from config module."""

    def test_valid_slack_url(self):
        assert validate_webhook_url("https://hooks.slack.com/services/T00/B00/xxx") is True

    def test_valid_custom_https_url(self):
        assert validate_webhook_url("https://my-proxy.example.com/webhook") is True

    def test_rejects_http_url(self):
        assert validate_webhook_url("http://hooks.slack.com/services/T00/B00/xxx") is False

    def test_rejects_empty_string(self):
        assert validate_webhook_url("") is False

    def test_rejects_whitespace_only(self):
        assert validate_webhook_url("   ") is False

    def test_rejects_none(self):
        assert validate_webhook_url(None) is False

    def test_rejects_non_string(self):
        assert validate_webhook_url(12345) is False

    def test_rejects_random_string(self):
        assert validate_webhook_url("not-a-url") is False

    def test_accepts_url_with_whitespace(self):
        assert validate_webhook_url("  https://hooks.slack.com/test  ") is True


# ---------------------------------------------------------------------------
# SlackNotifier class
# ---------------------------------------------------------------------------
VALID_WEBHOOK = "https://hooks.slack.com/services/T00/B00/xxx"


class TestSlackNotifierInit:
    """Test SlackNotifier construction and properties."""

    def test_default_construction(self):
        notifier = SlackNotifier()
        assert notifier.webhook_url == ""
        assert notifier.max_retries == 3
        assert notifier.base_delay == 1.0
        assert notifier.max_delay == 30.0
        assert notifier.timeout == 10
        assert notifier.dashboard_base_url == ""

    def test_custom_construction(self):
        notifier = SlackNotifier(
            webhook_url=VALID_WEBHOOK,
            max_retries=5,
            base_delay=2.0,
            max_delay=60.0,
            timeout=15,
            dashboard_base_url="http://localhost:8080",
        )
        assert notifier.webhook_url == VALID_WEBHOOK
        assert notifier.max_retries == 5
        assert notifier.base_delay == 2.0
        assert notifier.max_delay == 60.0
        assert notifier.timeout == 15
        assert notifier.dashboard_base_url == "http://localhost:8080"

    def test_is_configured_true(self):
        notifier = SlackNotifier(webhook_url=VALID_WEBHOOK)
        assert notifier.is_configured is True

    def test_is_configured_false_empty(self):
        notifier = SlackNotifier(webhook_url="")
        assert notifier.is_configured is False

    def test_is_configured_false_http(self):
        notifier = SlackNotifier(webhook_url="http://not-secure.com")
        assert notifier.is_configured is False

    def test_from_config(self):
        config = {
            "slack_webhook_url": VALID_WEBHOOK,
            "dashboard_base_url": "http://localhost:3000",
        }
        notifier = SlackNotifier.from_config(config)
        assert notifier.webhook_url == VALID_WEBHOOK
        assert notifier.dashboard_base_url == "http://localhost:3000"

    def test_from_config_empty(self):
        notifier = SlackNotifier.from_config({})
        assert notifier.webhook_url == ""
        assert notifier.dashboard_base_url == ""


class TestSlackNotifierSend:
    """Test SlackNotifier.send() — success and input validation."""

    @patch("hf_model_monitor.slack_notifier.requests.post")
    def test_send_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_post.return_value = mock_resp

        notifier = SlackNotifier(webhook_url=VALID_WEBHOOK)
        assert notifier.send("hello") is True

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["json"] == {"text": "hello"}
        assert call_kwargs[1]["timeout"] == 10

    def test_send_empty_message(self):
        notifier = SlackNotifier(webhook_url=VALID_WEBHOOK)
        assert notifier.send("") is False

    def test_send_no_webhook(self):
        notifier = SlackNotifier(webhook_url="")
        assert notifier.send("hello") is False

    def test_send_invalid_webhook(self):
        notifier = SlackNotifier(webhook_url="http://not-https.com")
        assert notifier.send("hello") is False


class TestSlackNotifierRetry:
    """Test exponential-backoff retry logic in SlackNotifier."""

    @patch("hf_model_monitor.slack_notifier.time.sleep")
    @patch("hf_model_monitor.slack_notifier.requests.post")
    def test_retries_on_500_then_succeeds(self, mock_post, mock_sleep):
        """Server error on first attempt, success on retry."""
        fail_resp = MagicMock()
        fail_resp.ok = False
        fail_resp.status_code = 500
        fail_resp.text = "Internal Server Error"

        ok_resp = MagicMock()
        ok_resp.ok = True

        mock_post.side_effect = [fail_resp, ok_resp]

        notifier = SlackNotifier(
            webhook_url=VALID_WEBHOOK, max_retries=3, base_delay=1.0,
        )
        assert notifier.send("test") is True
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once_with(1.0)  # 1.0 * 2^0

    @patch("hf_model_monitor.slack_notifier.time.sleep")
    @patch("hf_model_monitor.slack_notifier.requests.post")
    def test_retries_exhausted_on_500(self, mock_post, mock_sleep):
        """All retries fail with 500 — returns False."""
        fail_resp = MagicMock()
        fail_resp.ok = False
        fail_resp.status_code = 500
        fail_resp.text = "Internal Server Error"

        mock_post.return_value = fail_resp

        notifier = SlackNotifier(
            webhook_url=VALID_WEBHOOK, max_retries=2, base_delay=1.0,
        )
        assert notifier.send("test") is False
        assert mock_post.call_count == 3  # 1 initial + 2 retries
        assert mock_sleep.call_count == 2

    @patch("hf_model_monitor.slack_notifier.time.sleep")
    @patch("hf_model_monitor.slack_notifier.requests.post")
    def test_no_retry_on_400(self, mock_post, mock_sleep):
        """Client error (400) — no retry, immediate failure."""
        fail_resp = MagicMock()
        fail_resp.ok = False
        fail_resp.status_code = 400
        fail_resp.text = "Bad Request"

        mock_post.return_value = fail_resp

        notifier = SlackNotifier(webhook_url=VALID_WEBHOOK, max_retries=3)
        assert notifier.send("test") is False
        assert mock_post.call_count == 1
        mock_sleep.assert_not_called()

    @patch("hf_model_monitor.slack_notifier.time.sleep")
    @patch("hf_model_monitor.slack_notifier.requests.post")
    def test_no_retry_on_403(self, mock_post, mock_sleep):
        """Client error (403) — no retry."""
        fail_resp = MagicMock()
        fail_resp.ok = False
        fail_resp.status_code = 403
        fail_resp.text = "Forbidden"

        mock_post.return_value = fail_resp

        notifier = SlackNotifier(webhook_url=VALID_WEBHOOK, max_retries=3)
        assert notifier.send("test") is False
        assert mock_post.call_count == 1

    @patch("hf_model_monitor.slack_notifier.time.sleep")
    @patch("hf_model_monitor.slack_notifier.requests.post")
    def test_retries_on_429_with_retry_after(self, mock_post, mock_sleep):
        """Rate-limited (429) with Retry-After header."""
        rate_resp = MagicMock()
        rate_resp.ok = False
        rate_resp.status_code = 429
        rate_resp.headers = {"Retry-After": "5"}

        ok_resp = MagicMock()
        ok_resp.ok = True

        mock_post.side_effect = [rate_resp, ok_resp]

        notifier = SlackNotifier(
            webhook_url=VALID_WEBHOOK, max_retries=3, base_delay=1.0,
        )
        assert notifier.send("test") is True
        mock_sleep.assert_called_once_with(5.0)  # Uses Retry-After header

    @patch("hf_model_monitor.slack_notifier.time.sleep")
    @patch("hf_model_monitor.slack_notifier.requests.post")
    def test_retries_on_429_without_retry_after(self, mock_post, mock_sleep):
        """Rate-limited (429) without Retry-After — uses backoff."""
        rate_resp = MagicMock()
        rate_resp.ok = False
        rate_resp.status_code = 429
        rate_resp.headers = {}

        ok_resp = MagicMock()
        ok_resp.ok = True

        mock_post.side_effect = [rate_resp, ok_resp]

        notifier = SlackNotifier(
            webhook_url=VALID_WEBHOOK, max_retries=3, base_delay=2.0,
        )
        assert notifier.send("test") is True
        mock_sleep.assert_called_once_with(2.0)  # 2.0 * 2^0

    @patch("hf_model_monitor.slack_notifier.time.sleep")
    @patch("hf_model_monitor.slack_notifier.requests.post")
    def test_retries_on_timeout(self, mock_post, mock_sleep):
        """Timeout exception triggers retry."""
        mock_post.side_effect = [
            requests.exceptions.Timeout("timed out"),
            MagicMock(ok=True),
        ]

        notifier = SlackNotifier(
            webhook_url=VALID_WEBHOOK, max_retries=3, base_delay=1.0,
        )
        assert notifier.send("test") is True
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once_with(1.0)

    @patch("hf_model_monitor.slack_notifier.time.sleep")
    @patch("hf_model_monitor.slack_notifier.requests.post")
    def test_retries_on_connection_error(self, mock_post, mock_sleep):
        """ConnectionError triggers retry."""
        mock_post.side_effect = [
            requests.exceptions.ConnectionError("refused"),
            MagicMock(ok=True),
        ]

        notifier = SlackNotifier(
            webhook_url=VALID_WEBHOOK, max_retries=3, base_delay=1.0,
        )
        assert notifier.send("test") is True
        assert mock_post.call_count == 2

    @patch("hf_model_monitor.slack_notifier.requests.post")
    def test_no_retry_on_generic_request_exception(self, mock_post):
        """Non-retryable RequestException — immediate failure."""
        mock_post.side_effect = requests.exceptions.RequestException("bad")

        notifier = SlackNotifier(webhook_url=VALID_WEBHOOK, max_retries=3)
        assert notifier.send("test") is False
        assert mock_post.call_count == 1


class TestSlackNotifierBackoff:
    """Test the exponential backoff delay calculation."""

    def test_backoff_sequence(self):
        notifier = SlackNotifier(
            webhook_url=VALID_WEBHOOK, base_delay=1.0, max_delay=30.0,
        )
        assert notifier._backoff_delay(0) == 1.0   # 1 * 2^0
        assert notifier._backoff_delay(1) == 2.0   # 1 * 2^1
        assert notifier._backoff_delay(2) == 4.0   # 1 * 2^2
        assert notifier._backoff_delay(3) == 8.0   # 1 * 2^3
        assert notifier._backoff_delay(4) == 16.0  # 1 * 2^4
        assert notifier._backoff_delay(5) == 30.0  # capped at max_delay

    def test_backoff_respects_max_delay(self):
        notifier = SlackNotifier(
            webhook_url=VALID_WEBHOOK, base_delay=10.0, max_delay=15.0,
        )
        assert notifier._backoff_delay(0) == 10.0
        assert notifier._backoff_delay(1) == 15.0  # 20 capped to 15
        assert notifier._backoff_delay(2) == 15.0  # 40 capped to 15

    def test_backoff_with_custom_base(self):
        notifier = SlackNotifier(
            webhook_url=VALID_WEBHOOK, base_delay=0.5, max_delay=30.0,
        )
        assert notifier._backoff_delay(0) == 0.5
        assert notifier._backoff_delay(1) == 1.0
        assert notifier._backoff_delay(2) == 2.0


class TestSlackNotifierParseRetryAfter:
    """Test Retry-After header parsing."""

    def test_valid_numeric_header(self):
        resp = MagicMock()
        resp.headers = {"Retry-After": "10"}
        assert SlackNotifier._parse_retry_after(resp) == 10.0

    def test_valid_float_header(self):
        resp = MagicMock()
        resp.headers = {"Retry-After": "2.5"}
        assert SlackNotifier._parse_retry_after(resp) == 2.5

    def test_missing_header(self):
        resp = MagicMock()
        resp.headers = {}
        assert SlackNotifier._parse_retry_after(resp) is None

    def test_invalid_header(self):
        resp = MagicMock()
        resp.headers = {"Retry-After": "not-a-number"}
        assert SlackNotifier._parse_retry_after(resp) is None

    def test_negative_clamped_to_zero(self):
        resp = MagicMock()
        resp.headers = {"Retry-After": "-5"}
        assert SlackNotifier._parse_retry_after(resp) == 0.0


class TestSlackNotifierSendReport:
    """Test send_report with dashboard link injection."""

    @patch("hf_model_monitor.slack_notifier.requests.post")
    def test_send_report_without_dashboard(self, mock_post):
        mock_post.return_value = MagicMock(ok=True)

        notifier = SlackNotifier(webhook_url=VALID_WEBHOOK)
        result = notifier.send_report(SAMPLE_METADATA, SAMPLE_REFERENCE)

        assert result is True
        payload = mock_post.call_args[1]["json"]["text"]
        assert "TestModel" in payload
        assert "View on Dashboard" not in payload

    @patch("hf_model_monitor.slack_notifier.requests.post")
    def test_send_report_with_dashboard(self, mock_post):
        mock_post.return_value = MagicMock(ok=True)

        notifier = SlackNotifier(
            webhook_url=VALID_WEBHOOK,
            dashboard_base_url="http://localhost:8080",
        )
        result = notifier.send_report(SAMPLE_METADATA, SAMPLE_REFERENCE)

        assert result is True
        payload = mock_post.call_args[1]["json"]["text"]
        assert "TestModel" in payload
        assert "View on Dashboard" in payload
        assert "http://localhost:8080/models/TestModel" in payload


class TestSlackNotifierSendErrorAlert:
    """Test send_error_alert delegates to format_pipeline_error_alert."""

    @patch("hf_model_monitor.slack_notifier.requests.post")
    def test_send_error_alert(self, mock_post):
        mock_post.return_value = MagicMock(ok=True)

        notifier = SlackNotifier(webhook_url=VALID_WEBHOOK)
        result = notifier.send_error_alert("Pipeline crashed", consecutive_failures=3)

        assert result is True
        payload = mock_post.call_args[1]["json"]["text"]
        assert "Pipeline Error" in payload
        assert "Pipeline crashed" in payload
        assert "Consecutive failures: 3" in payload
        assert "HIGH" in payload  # 3+ failures = HIGH severity


class TestSendToSlackBackwardCompat:
    """Verify send_to_slack still works and now uses retry logic internally."""

    @patch("hf_model_monitor.slack_notifier.requests.post")
    def test_delegates_to_notifier(self, mock_post):
        mock_post.return_value = MagicMock(ok=True)

        result = send_to_slack("test msg", webhook_url=VALID_WEBHOOK)
        assert result is True
        mock_post.assert_called_once()

    @patch("hf_model_monitor.slack_notifier.time.sleep")
    @patch("hf_model_monitor.slack_notifier.requests.post")
    def test_retries_via_notifier(self, mock_post, mock_sleep):
        """send_to_slack now benefits from retry logic."""
        fail_resp = MagicMock()
        fail_resp.ok = False
        fail_resp.status_code = 502
        fail_resp.text = "Bad Gateway"

        ok_resp = MagicMock()
        ok_resp.ok = True

        mock_post.side_effect = [fail_resp, ok_resp]

        result = send_to_slack("test", webhook_url=VALID_WEBHOOK)
        assert result is True
        assert mock_post.call_count == 2


# ---------------------------------------------------------------------------
# _is_available helper
# ---------------------------------------------------------------------------
class TestIsAvailable:
    """Test the _is_available helper used by optional section builders."""

    def test_real_value(self):
        assert _is_available("82.5") is True

    def test_na(self):
        assert _is_available("N/A") is False

    def test_na_lowercase(self):
        assert _is_available("n/a") is False

    def test_unknown(self):
        assert _is_available("Unknown") is False

    def test_none_string(self):
        assert _is_available("None") is False

    def test_empty_string(self):
        assert _is_available("") is False

    def test_whitespace_only(self):
        assert _is_available("   ") is False

    def test_real_price(self):
        assert _is_available("$2.50") is True

    def test_real_category(self):
        assert _is_available("llm") is True


# ---------------------------------------------------------------------------
# Benchmark blocks builder
# ---------------------------------------------------------------------------

# Metadata with all benchmarks populated
_META_FULL_BENCHMARKS = {
    "basic": {"name": "BenchModel", "org": "test-org", "params": "70B",
              "architecture": "transformer", "license": "apache-2.0"},
    "performance": {
        "mmlu": "88.7", "humaneval": "90.2", "gpqa": "53.6",
        "math": "76.6", "arena_elo": "1286",
    },
    "community": {"downloads": 100, "likes": 10},
    "cost": {},
}

# Metadata with partial benchmarks
_META_PARTIAL_BENCHMARKS = {
    "basic": {"name": "PartialModel", "org": "test-org", "params": "7B",
              "architecture": "llama", "license": "mit"},
    "performance": {"mmlu": "82.5", "humaneval": "N/A", "gpqa": "N/A",
                    "math": "N/A", "arena_elo": "N/A"},
    "community": {"downloads": 0, "likes": 0},
    "cost": {},
}

# Metadata with no benchmarks
_META_NO_BENCHMARKS = {
    "basic": {"name": "NoBench", "org": "test-org", "params": "1B",
              "architecture": "gpt", "license": "mit"},
    "performance": {"mmlu": "N/A", "humaneval": "N/A"},
    "community": {"downloads": 0, "likes": 0},
    "cost": {},
}


class TestBuildBenchmarkBlocks:
    """Test _build_benchmark_blocks() optional section."""

    def test_all_benchmarks_present(self):
        blocks = _build_benchmark_blocks(_META_FULL_BENCHMARKS)
        assert len(blocks) > 0
        # Should have divider, header section, and fields section
        assert blocks[0]["type"] == "divider"
        assert "Benchmark Scores" in blocks[1]["text"]["text"]

        # Find the fields section
        fields_block = blocks[2]
        assert fields_block["type"] == "section"
        field_texts = [f["text"] for f in fields_block["fields"]]
        assert any("MMLU" in t and "88.7" in t for t in field_texts)
        assert any("HumanEval" in t and "90.2" in t for t in field_texts)
        assert any("GPQA" in t and "53.6" in t for t in field_texts)
        assert any("MATH" in t and "76.6" in t for t in field_texts)
        assert any("Arena ELO" in t and "1286" in t for t in field_texts)

    def test_partial_benchmarks(self):
        blocks = _build_benchmark_blocks(_META_PARTIAL_BENCHMARKS)
        assert len(blocks) > 0
        # Only MMLU should appear (others are N/A)
        fields_block = blocks[2]
        assert len(fields_block["fields"]) == 1
        assert "MMLU" in fields_block["fields"][0]["text"]
        assert "82.5" in fields_block["fields"][0]["text"]

    def test_no_benchmarks_returns_empty(self):
        blocks = _build_benchmark_blocks(_META_NO_BENCHMARKS)
        assert blocks == []

    def test_empty_performance_section(self):
        meta = {"basic": {"name": "X"}, "performance": {}, "community": {}}
        blocks = _build_benchmark_blocks(meta)
        assert blocks == []

    def test_missing_performance_section(self):
        meta = {"basic": {"name": "X"}, "community": {}}
        blocks = _build_benchmark_blocks(meta)
        assert blocks == []


# ---------------------------------------------------------------------------
# Pricing blocks builder
# ---------------------------------------------------------------------------

# Seed-format pricing (input/output separate)
_META_SEED_PRICING = {
    "basic": {"name": "PricedModel", "org": "openai", "params": "N/A",
              "architecture": "transformer", "license": "Proprietary"},
    "performance": {},
    "community": {"downloads": 0, "likes": 0},
    "cost": {
        "api_price_input_per_1m": "$2.50",
        "api_price_output_per_1m": "$10.00",
    },
    "provider": {
        "name": "OpenAI",
        "api_providers": ["OpenAI", "Azure OpenAI"],
    },
}

# Legacy pricing format
_META_LEGACY_PRICING = {
    "basic": {"name": "LegacyPrice", "org": "test", "params": "7B",
              "architecture": "llama", "license": "apache-2.0"},
    "performance": {},
    "community": {"downloads": 0, "likes": 0},
    "cost": {"api_price_per_million_tokens": "$0.50"},
}

# No pricing data
_META_NO_PRICING = {
    "basic": {"name": "FreeModel", "org": "test", "params": "7B",
              "architecture": "llama", "license": "apache-2.0"},
    "performance": {},
    "community": {"downloads": 0, "likes": 0},
    "cost": {"api_price_per_million_tokens": "N/A"},
}


class TestBuildPricingBlocks:
    """Test _build_pricing_blocks() optional section."""

    def test_seed_format_pricing(self):
        blocks = _build_pricing_blocks(_META_SEED_PRICING)
        assert len(blocks) > 0
        assert blocks[0]["type"] == "divider"
        assert "Pricing" in blocks[1]["text"]["text"]

        fields_block = blocks[2]
        field_texts = [f["text"] for f in fields_block["fields"]]
        assert any("Input Price" in t and "$2.50" in t for t in field_texts)
        assert any("Output Price" in t and "$10.00" in t for t in field_texts)
        assert any("Provider" in t and "OpenAI" in t for t in field_texts)

    def test_provider_with_api_providers_list(self):
        blocks = _build_pricing_blocks(_META_SEED_PRICING)
        fields_block = blocks[2]
        field_texts = [f["text"] for f in fields_block["fields"]]
        # Should show "OpenAI (OpenAI, Azure OpenAI)"
        provider_field = [t for t in field_texts if "Provider" in t][0]
        assert "Azure OpenAI" in provider_field

    def test_legacy_pricing_format(self):
        blocks = _build_pricing_blocks(_META_LEGACY_PRICING)
        assert len(blocks) > 0
        fields_block = blocks[2]
        field_texts = [f["text"] for f in fields_block["fields"]]
        assert any("API Price" in t and "$0.50" in t for t in field_texts)
        # Should NOT show "Input Price" since legacy format
        assert not any("Input Price" in t for t in field_texts)

    def test_no_pricing_returns_empty(self):
        blocks = _build_pricing_blocks(_META_NO_PRICING)
        assert blocks == []

    def test_missing_cost_section(self):
        meta = {"basic": {"name": "X"}, "community": {}}
        blocks = _build_pricing_blocks(meta)
        assert blocks == []

    def test_provider_only_no_price(self):
        meta = {
            "basic": {"name": "X"},
            "cost": {"api_price_per_million_tokens": "N/A"},
            "provider": {"name": "SomeProvider"},
            "community": {},
        }
        blocks = _build_pricing_blocks(meta)
        assert len(blocks) > 0
        fields_block = blocks[2]
        field_texts = [f["text"] for f in fields_block["fields"]]
        assert any("Provider" in t and "SomeProvider" in t for t in field_texts)


# ---------------------------------------------------------------------------
# Tags block builder
# ---------------------------------------------------------------------------
class TestBuildTagsBlock:
    """Test _build_tags_block() optional section."""

    def test_category_only(self):
        meta = {"category": "llm", "basic": {"name": "X"}}
        blocks = _build_tags_block(meta)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "context"
        text = blocks[0]["elements"][0]["text"]
        assert "`llm`" in text
        assert "Tags" in text

    def test_tags_list(self):
        meta = {"tags": ["text-generation", "pytorch", "transformers"]}
        blocks = _build_tags_block(meta)
        assert len(blocks) == 1
        text = blocks[0]["elements"][0]["text"]
        assert "`text-generation`" in text
        assert "`pytorch`" in text
        assert "`transformers`" in text

    def test_category_and_tags(self):
        meta = {"category": "code", "tags": ["python", "code-generation"]}
        blocks = _build_tags_block(meta)
        text = blocks[0]["elements"][0]["text"]
        assert "`code`" in text
        assert "`python`" in text
        assert "`code-generation`" in text

    def test_no_tags_returns_empty(self):
        meta = {"basic": {"name": "X"}}
        blocks = _build_tags_block(meta)
        assert blocks == []

    def test_na_category_ignored(self):
        meta = {"category": "N/A", "tags": []}
        blocks = _build_tags_block(meta)
        assert blocks == []

    def test_empty_tags_list(self):
        meta = {"tags": []}
        blocks = _build_tags_block(meta)
        assert blocks == []

    def test_tags_capped_at_8(self):
        meta = {"tags": [f"tag-{i}" for i in range(15)]}
        blocks = _build_tags_block(meta)
        text = blocks[0]["elements"][0]["text"]
        assert "`tag-7`" in text       # 8th tag (0-indexed) should be present
        assert "`tag-8`" not in text    # 9th tag should be excluded

    def test_basic_category_fallback(self):
        meta = {"basic": {"category": "vision"}}
        blocks = _build_tags_block(meta)
        assert len(blocks) == 1
        text = blocks[0]["elements"][0]["text"]
        assert "`vision`" in text


# ---------------------------------------------------------------------------
# Full Block Kit report — integration tests
# ---------------------------------------------------------------------------

# Rich metadata with all optional fields populated (like seed data)
_META_FULL_REPORT = {
    "model_id": "openai/gpt-4o",
    "name": "GPT-4o",
    "category": "llm",
    "tags": ["text-generation", "conversational"],
    "basic": {
        "name": "GPT-4o",
        "org": "openai",
        "model_id": "openai/gpt-4o",
        "params": "~1.8T (estimated)",
        "architecture": "Transformer (MoE, estimated)",
        "license": "Proprietary",
        "release_date": "2024-05-13",
    },
    "performance": {
        "mmlu": "88.7",
        "humaneval": "90.2",
        "gpqa": "53.6",
        "math": "76.6",
        "arena_elo": "1286",
    },
    "practical": {"context_window": "128K"},
    "deployment": {"vram_estimate": "N/A (API only)"},
    "community": {"downloads": 50000, "likes": 1200, "trending_rank": 3},
    "cost": {
        "api_price_input_per_1m": "$2.50",
        "api_price_output_per_1m": "$10.00",
    },
    "provider": {
        "name": "OpenAI",
        "api_providers": ["OpenAI", "Azure OpenAI"],
    },
}

# Minimal metadata — only mandatory fields, no optional data
_META_MINIMAL_REPORT = {
    "basic": {
        "name": "MinimalModel",
        "org": "unknown-org",
        "params": "N/A",
        "architecture": "N/A",
        "license": "N/A",
        "release_date": "N/A",
    },
    "performance": {"mmlu": "N/A", "humaneval": "N/A"},
    "practical": {},
    "deployment": {},
    "community": {"downloads": 0, "likes": 0},
    "cost": {"api_price_per_million_tokens": "N/A"},
}


class TestFormatBlockKitReport:
    """Integration tests for the full Block Kit report template."""

    def test_mandatory_fields_always_present(self):
        """All 6 mandatory fields + HF URL appear even with minimal data."""
        blocks = format_block_kit_report(_META_MINIMAL_REPORT, {})
        all_text = _extract_all_block_text(blocks)
        # Mandatory labels
        assert "Model Name:" in all_text
        assert "Author:" in all_text
        assert "Release Date:" in all_text
        assert "Model Size:" in all_text
        assert "Architecture:" in all_text
        assert "License:" in all_text
        assert "HF URL:" in all_text

    def test_mandatory_fields_show_na_when_missing(self):
        """N/A is rendered (not omitted) for unavailable mandatory fields."""
        blocks = format_block_kit_report(_META_MINIMAL_REPORT, {})
        # Find the mandatory fields section(s)
        field_sections = [b for b in blocks if b.get("type") == "section" and "fields" in b]
        field_texts = []
        for s in field_sections:
            field_texts.extend(f["text"] for f in s.get("fields", []))
        # Architecture should show N/A
        arch_field = [t for t in field_texts if "Architecture:" in t]
        assert len(arch_field) == 1
        assert "N/A" in arch_field[0]

    def test_full_report_includes_benchmarks(self):
        """Benchmark section appears when performance data exists."""
        blocks = format_block_kit_report(_META_FULL_REPORT, {})
        all_text = _extract_all_block_text(blocks)
        assert "Benchmark Scores" in all_text
        assert "MMLU" in all_text
        assert "88.7" in all_text
        assert "HumanEval" in all_text
        assert "90.2" in all_text

    def test_full_report_includes_pricing(self):
        """Pricing section appears when cost data exists."""
        blocks = format_block_kit_report(_META_FULL_REPORT, {})
        all_text = _extract_all_block_text(blocks)
        assert "Pricing" in all_text
        assert "$2.50" in all_text
        assert "$10.00" in all_text
        assert "OpenAI" in all_text

    def test_full_report_includes_tags(self):
        """Tags context appears when tags/category data exists."""
        blocks = format_block_kit_report(_META_FULL_REPORT, {})
        all_text = _extract_all_block_text(blocks)
        assert "Tags" in all_text
        assert "`llm`" in all_text
        assert "`text-generation`" in all_text

    def test_minimal_report_omits_optional_sections(self):
        """No benchmark/pricing/tags sections when data is unavailable."""
        blocks = format_block_kit_report(_META_MINIMAL_REPORT, {})
        all_text = _extract_all_block_text(blocks)
        assert "Benchmark Scores" not in all_text
        assert "Pricing" not in all_text
        assert "Tags" not in all_text

    def test_header_contains_model_name(self):
        blocks = format_block_kit_report(_META_FULL_REPORT, {})
        header = blocks[0]
        assert header["type"] == "header"
        assert "GPT-4o" in header["text"]["text"]

    def test_context_line_has_author_and_date(self):
        blocks = format_block_kit_report(_META_FULL_REPORT, {})
        context = blocks[1]
        assert context["type"] == "context"
        texts = [e["text"] for e in context["elements"]]
        assert any("openai" in t for t in texts)
        assert any("2024-05-13" in t for t in texts)

    def test_community_stats_always_present(self):
        blocks = format_block_kit_report(_META_FULL_REPORT, {})
        all_text = _extract_all_block_text(blocks)
        assert "Downloads" in all_text
        assert "50,000" in all_text
        assert "Likes" in all_text
        assert "1,200" in all_text
        assert "Trending Rank" in all_text

    def test_comparison_table_with_references(self):
        blocks = format_block_kit_report(_META_FULL_REPORT, SAMPLE_REFERENCE)
        all_text = _extract_all_block_text(blocks)
        assert "Comparison with Reference Models" in all_text
        assert "GPT-4o" in all_text
        assert "Llama-3.1-405B" in all_text

    def test_comparison_table_omitted_without_references(self):
        blocks = format_block_kit_report(_META_FULL_REPORT, {})
        all_text = _extract_all_block_text(blocks)
        assert "Comparison with Reference Models" not in all_text

    def test_dashboard_link_button(self):
        blocks = format_block_kit_report(
            _META_FULL_REPORT, {}, dashboard_url="http://localhost:8080/models/GPT-4o"
        )
        actions = [b for b in blocks if b.get("type") == "actions"]
        assert len(actions) == 1
        button = actions[0]["elements"][0]
        assert button["url"] == "http://localhost:8080/models/GPT-4o"

    def test_no_dashboard_link_without_url(self):
        blocks = format_block_kit_report(_META_FULL_REPORT, {})
        actions = [b for b in blocks if b.get("type") == "actions"]
        assert len(actions) == 0

    def test_footer_has_timestamp(self):
        blocks = format_block_kit_report(_META_FULL_REPORT, {})
        footer = blocks[-1]
        assert footer["type"] == "context"
        assert "UTC" in footer["elements"][0]["text"]
        assert "HF Model Monitor" in footer["elements"][0]["text"]

    def test_hf_url_rendered_as_link(self):
        blocks = format_block_kit_report(_META_FULL_REPORT, {})
        all_text = _extract_all_block_text(blocks)
        assert "huggingface.co/openai/gpt-4o" in all_text

    def test_block_order_mandatory_before_optional(self):
        """Verify mandatory fields come before optional benchmark/pricing."""
        blocks = format_block_kit_report(_META_FULL_REPORT, SAMPLE_REFERENCE)
        all_text_blocks = []
        for b in blocks:
            all_text_blocks.append(_extract_block_text(b))

        # Find positions of key sections
        mandatory_idx = None
        benchmark_idx = None
        pricing_idx = None
        community_idx = None
        for i, text in enumerate(all_text_blocks):
            if "Model Name:" in text and mandatory_idx is None:
                mandatory_idx = i
            if "Benchmark Scores" in text:
                benchmark_idx = i
            if "Pricing" in text:
                pricing_idx = i
            if "Downloads:" in text and "Likes:" in text:
                community_idx = i

        assert mandatory_idx is not None
        assert benchmark_idx is not None
        assert pricing_idx is not None
        assert community_idx is not None
        # Order: mandatory < benchmarks < pricing < community
        assert mandatory_idx < benchmark_idx < pricing_idx < community_idx

    def test_slack_block_limit_not_exceeded(self):
        """Slack allows max 50 blocks per message."""
        blocks = format_block_kit_report(
            _META_FULL_REPORT, SAMPLE_REFERENCE,
            dashboard_url="http://localhost:8080/models/GPT-4o"
        )
        assert len(blocks) <= 50


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _extract_block_text(block: dict) -> str:
    """Extract all text content from a single Block Kit block."""
    parts = []
    if "text" in block:
        t = block["text"]
        if isinstance(t, dict):
            parts.append(t.get("text", ""))
        elif isinstance(t, str):
            parts.append(t)
    for field in block.get("fields", []):
        if isinstance(field, dict):
            parts.append(field.get("text", ""))
    for elem in block.get("elements", []):
        if isinstance(elem, dict):
            parts.append(elem.get("text", ""))
            if "text" in elem and isinstance(elem["text"], dict):
                parts.append(elem["text"].get("text", ""))
    return " ".join(parts)


def _extract_all_block_text(blocks: list[dict]) -> str:
    """Extract all text content from a list of Block Kit blocks."""
    return " ".join(_extract_block_text(b) for b in blocks)
