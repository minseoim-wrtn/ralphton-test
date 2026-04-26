"""Tests for hf_model_monitor.errors — crawler error types and classification.

Covers:
- Exception hierarchy (CrawlerError subclasses)
- classify_error() with typed exceptions, requests exceptions, builtins
- classify_error_message() string-based fallback
- ClassifiedError dataclass serialization
- category_to_label / label_to_category round-trip
- wrap_exception() normalization
- Backward compatibility with slack_notifier.classify_crawler_error()
"""

from unittest.mock import MagicMock

import pytest
import requests

from hf_model_monitor.errors import (
    AuthenticationError,
    ClassifiedError,
    ConfigurationError,
    CrawlerError,
    CrawlerErrorCategory,
    CrawlerTimeoutError,
    ErrorSeverity,
    NetworkError,
    NotFoundError,
    RateLimitError,
    ServerError,
    StructureChangeError,
    category_to_label,
    classify_error,
    classify_error_message,
    label_to_category,
    wrap_exception,
)
from hf_model_monitor.hf_client import (
    HFApiError,
    HFAuthError,
    HFRateLimitError,
)


# ---------------------------------------------------------------------------
# Exception hierarchy tests
# ---------------------------------------------------------------------------
class TestExceptionHierarchy:
    """Verify the CrawlerError class hierarchy and default attributes."""

    def test_crawler_error_is_base(self):
        exc = CrawlerError("base error")
        assert isinstance(exc, Exception)
        assert str(exc) == "base error"
        assert exc.category == CrawlerErrorCategory.UNKNOWN
        assert exc.retryable is False
        assert exc.source == ""
        assert exc.status_code is None

    def test_network_error(self):
        exc = NetworkError("DNS failed", source="hf_api")
        assert isinstance(exc, CrawlerError)
        assert exc.category == CrawlerErrorCategory.NETWORK
        assert exc.retryable is True
        assert exc.source == "hf_api"
        assert str(exc) == "DNS failed"

    def test_timeout_error(self):
        exc = CrawlerTimeoutError("timed out after 30s", source="reddit")
        assert isinstance(exc, CrawlerError)
        assert exc.category == CrawlerErrorCategory.TIMEOUT
        assert exc.retryable is True
        assert exc.source == "reddit"

    def test_rate_limit_error(self):
        exc = RateLimitError("slow down", source="hf_api", retry_after=60.0)
        assert isinstance(exc, CrawlerError)
        assert exc.category == CrawlerErrorCategory.RATE_LIMIT
        assert exc.retryable is True
        assert exc.retry_after == 60.0
        assert exc.status_code == 429

    def test_rate_limit_error_default_status(self):
        exc = RateLimitError("slow down")
        assert exc.status_code == 429

    def test_auth_error(self):
        exc = AuthenticationError("invalid token", status_code=401)
        assert isinstance(exc, CrawlerError)
        assert exc.category == CrawlerErrorCategory.AUTH
        assert exc.retryable is False
        assert exc.status_code == 401

    def test_not_found_error(self):
        exc = NotFoundError("org not found", status_code=404)
        assert isinstance(exc, CrawlerError)
        assert exc.category == CrawlerErrorCategory.NOT_FOUND
        assert exc.retryable is False

    def test_server_error(self):
        exc = ServerError("internal server error", status_code=500)
        assert isinstance(exc, CrawlerError)
        assert exc.category == CrawlerErrorCategory.SERVER_ERROR
        assert exc.retryable is True

    def test_structure_change_error(self):
        exc = StructureChangeError("unexpected JSON key 'new_field'")
        assert isinstance(exc, CrawlerError)
        assert exc.category == CrawlerErrorCategory.STRUCTURE_CHANGE
        assert exc.retryable is False
        assert exc.default_severity == ErrorSeverity.CRITICAL

    def test_configuration_error(self):
        exc = ConfigurationError("missing webhook URL")
        assert isinstance(exc, CrawlerError)
        assert exc.category == CrawlerErrorCategory.CONFIGURATION
        assert exc.retryable is False

    def test_all_subclasses_inherit_from_crawler_error(self):
        subclasses = [
            NetworkError, CrawlerTimeoutError, RateLimitError,
            AuthenticationError, NotFoundError, ServerError,
            StructureChangeError, ConfigurationError,
        ]
        for cls in subclasses:
            assert issubclass(cls, CrawlerError)
            assert issubclass(cls, Exception)


# ---------------------------------------------------------------------------
# classify_error() — typed exceptions
# ---------------------------------------------------------------------------
class TestClassifyErrorTyped:
    """Test classify_error() with our own CrawlerError subclasses."""

    def test_network_error(self):
        exc = NetworkError("connection refused", source="hf_api")
        result = classify_error(exc)
        assert result.category == CrawlerErrorCategory.NETWORK
        assert result.retryable is True
        assert result.severity == ErrorSeverity.MEDIUM
        assert result.source_exception is exc
        assert "network" in result.suggested_action.lower() or "connectivity" in result.suggested_action.lower()

    def test_timeout_error(self):
        exc = CrawlerTimeoutError("timed out after 30s")
        result = classify_error(exc)
        assert result.category == CrawlerErrorCategory.TIMEOUT
        assert result.retryable is True
        assert result.severity == ErrorSeverity.LOW

    def test_rate_limit_error(self):
        exc = RateLimitError("429 Too Many Requests", retry_after=60.0)
        result = classify_error(exc)
        assert result.category == CrawlerErrorCategory.RATE_LIMIT
        assert result.retryable is True

    def test_auth_error(self):
        exc = AuthenticationError("401 Unauthorized")
        result = classify_error(exc)
        assert result.category == CrawlerErrorCategory.AUTH
        assert result.retryable is False
        assert result.severity == ErrorSeverity.HIGH

    def test_not_found_error(self):
        exc = NotFoundError("org 'nonexistent' not found")
        result = classify_error(exc)
        assert result.category == CrawlerErrorCategory.NOT_FOUND
        assert result.retryable is False
        assert result.severity == ErrorSeverity.LOW

    def test_server_error(self):
        exc = ServerError("502 Bad Gateway")
        result = classify_error(exc)
        assert result.category == CrawlerErrorCategory.SERVER_ERROR
        assert result.retryable is True

    def test_structure_change_error(self):
        exc = StructureChangeError("API response missing 'models' key")
        result = classify_error(exc)
        assert result.category == CrawlerErrorCategory.STRUCTURE_CHANGE
        assert result.retryable is False
        assert result.severity == ErrorSeverity.CRITICAL

    def test_configuration_error(self):
        exc = ConfigurationError("watched_organizations is empty")
        result = classify_error(exc)
        assert result.category == CrawlerErrorCategory.CONFIGURATION
        assert result.retryable is False
        assert result.severity == ErrorSeverity.HIGH

    def test_source_from_exception(self):
        exc = NetworkError("failed", source="reddit")
        result = classify_error(exc)
        assert result.source_exception is exc
        assert exc.source == "reddit"

    def test_source_override(self):
        exc = NetworkError("failed", source="reddit")
        result = classify_error(exc, source="override_source")
        # Explicit source arg takes precedence
        assert result.source_exception is exc

    def test_generic_crawler_error(self):
        exc = CrawlerError("something went wrong")
        result = classify_error(exc)
        assert result.category == CrawlerErrorCategory.UNKNOWN
        assert result.retryable is False


# ---------------------------------------------------------------------------
# classify_error() — HF client exceptions
# ---------------------------------------------------------------------------
class TestClassifyErrorHFClient:
    """Test classify_error() with hf_client.py exception types."""

    def test_hf_rate_limit_error(self):
        exc = HFRateLimitError(retry_after=30.0)
        result = classify_error(exc)
        assert result.category == CrawlerErrorCategory.RATE_LIMIT
        assert result.retryable is True

    def test_hf_auth_error(self):
        exc = HFAuthError(status_code=403)
        result = classify_error(exc)
        assert result.category == CrawlerErrorCategory.AUTH
        assert result.retryable is False

    def test_hf_api_error_server(self):
        exc = HFApiError("Server error 502", status_code=502)
        result = classify_error(exc)
        assert result.category == CrawlerErrorCategory.SERVER_ERROR
        assert result.retryable is True

    def test_hf_api_error_generic(self):
        exc = HFApiError("Something broke", status_code=400)
        result = classify_error(exc)
        # Generic HFApiError without 5xx → falls through to NETWORK
        assert result.category == CrawlerErrorCategory.NETWORK
        assert result.retryable is True


# ---------------------------------------------------------------------------
# classify_error() — requests exceptions
# ---------------------------------------------------------------------------
class TestClassifyErrorRequests:
    """Test classify_error() with requests library exceptions."""

    def test_requests_timeout(self):
        exc = requests.exceptions.Timeout("Connection timed out")
        result = classify_error(exc)
        assert result.category == CrawlerErrorCategory.TIMEOUT
        assert result.retryable is True

    def test_requests_connection_error(self):
        exc = requests.exceptions.ConnectionError("DNS resolution failed")
        result = classify_error(exc)
        assert result.category == CrawlerErrorCategory.NETWORK
        assert result.retryable is True

    def test_requests_http_error_429(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        exc = requests.exceptions.HTTPError(response=mock_resp)
        result = classify_error(exc)
        assert result.category == CrawlerErrorCategory.RATE_LIMIT
        assert result.retryable is True

    def test_requests_http_error_401(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        exc = requests.exceptions.HTTPError(response=mock_resp)
        result = classify_error(exc)
        assert result.category == CrawlerErrorCategory.AUTH
        assert result.retryable is False

    def test_requests_http_error_403(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        exc = requests.exceptions.HTTPError(response=mock_resp)
        result = classify_error(exc)
        assert result.category == CrawlerErrorCategory.AUTH

    def test_requests_http_error_404(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        exc = requests.exceptions.HTTPError(response=mock_resp)
        result = classify_error(exc)
        assert result.category == CrawlerErrorCategory.NOT_FOUND

    def test_requests_http_error_500(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        exc = requests.exceptions.HTTPError(response=mock_resp)
        result = classify_error(exc)
        assert result.category == CrawlerErrorCategory.SERVER_ERROR
        assert result.retryable is True

    def test_requests_http_error_503(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        exc = requests.exceptions.HTTPError(response=mock_resp)
        result = classify_error(exc)
        assert result.category == CrawlerErrorCategory.SERVER_ERROR

    def test_requests_generic(self):
        exc = requests.exceptions.RequestException("something")
        result = classify_error(exc)
        assert result.category == CrawlerErrorCategory.NETWORK
        assert result.retryable is True


# ---------------------------------------------------------------------------
# classify_error() — Python built-in exceptions
# ---------------------------------------------------------------------------
class TestClassifyErrorBuiltins:
    """Test classify_error() with standard Python exceptions."""

    def test_builtin_timeout_error(self):
        exc = TimeoutError("operation timed out")
        result = classify_error(exc)
        assert result.category == CrawlerErrorCategory.TIMEOUT
        assert result.retryable is True

    def test_builtin_connection_error(self):
        exc = ConnectionError("connection refused")
        result = classify_error(exc)
        assert result.category == CrawlerErrorCategory.NETWORK
        assert result.retryable is True

    def test_builtin_os_error(self):
        exc = OSError("network unreachable")
        result = classify_error(exc)
        assert result.category == CrawlerErrorCategory.NETWORK
        assert result.retryable is True

    def test_value_error_as_structure_change(self):
        exc = ValueError("unexpected value in JSON field")
        result = classify_error(exc)
        assert result.category == CrawlerErrorCategory.STRUCTURE_CHANGE
        assert result.retryable is False
        assert result.severity == ErrorSeverity.CRITICAL

    def test_key_error_as_structure_change(self):
        exc = KeyError("missing_field")
        result = classify_error(exc)
        assert result.category == CrawlerErrorCategory.STRUCTURE_CHANGE
        assert result.retryable is False

    def test_type_error_as_structure_change(self):
        exc = TypeError("expected dict, got list")
        result = classify_error(exc)
        assert result.category == CrawlerErrorCategory.STRUCTURE_CHANGE

    def test_attribute_error_as_structure_change(self):
        exc = AttributeError("'NoneType' has no attribute 'get'")
        result = classify_error(exc)
        assert result.category == CrawlerErrorCategory.STRUCTURE_CHANGE

    def test_json_decode_error(self):
        import json
        exc = json.JSONDecodeError("Expecting value", "", 0)
        result = classify_error(exc)
        assert result.category == CrawlerErrorCategory.STRUCTURE_CHANGE
        assert result.retryable is False

    def test_permission_error(self):
        exc = PermissionError("access denied")
        result = classify_error(exc)
        assert result.category == CrawlerErrorCategory.AUTH
        assert result.retryable is False

    def test_file_not_found_error(self):
        exc = FileNotFoundError("settings.yaml not found")
        result = classify_error(exc)
        assert result.category == CrawlerErrorCategory.CONFIGURATION
        assert result.retryable is False

    def test_unknown_exception(self):
        exc = RuntimeError("something truly unexpected")
        result = classify_error(exc)
        # RuntimeError doesn't match any rule → falls to string matching
        # "unexpected" keyword → STRUCTURE_CHANGE
        # If message doesn't match either → UNKNOWN
        assert result.category in (
            CrawlerErrorCategory.STRUCTURE_CHANGE,
            CrawlerErrorCategory.UNKNOWN,
        )


# ---------------------------------------------------------------------------
# classify_error_message() — string-based fallback
# ---------------------------------------------------------------------------
class TestClassifyErrorMessage:
    """Test the string-based keyword matcher."""

    @pytest.mark.parametrize(
        "msg, expected_category",
        [
            ("Connection timed out after 30s", CrawlerErrorCategory.TIMEOUT),
            ("Request timed out", CrawlerErrorCategory.TIMEOUT),
            ("Rate limit exceeded", CrawlerErrorCategory.RATE_LIMIT),
            ("HTTP 429 Too Many Requests", CrawlerErrorCategory.RATE_LIMIT),
            ("401 Unauthorized", CrawlerErrorCategory.AUTH),
            ("403 Forbidden", CrawlerErrorCategory.AUTH),
            ("Authentication required", CrawlerErrorCategory.AUTH),
            ("Connection refused", CrawlerErrorCategory.NETWORK),
            ("DNS resolution failed", CrawlerErrorCategory.NETWORK),
            ("Network unreachable", CrawlerErrorCategory.NETWORK),
            ("500 Internal Server Error", CrawlerErrorCategory.SERVER_ERROR),
            ("502 Bad Gateway", CrawlerErrorCategory.SERVER_ERROR),
            ("503 Service Unavailable", CrawlerErrorCategory.SERVER_ERROR),
            ("504 Gateway Timeout", CrawlerErrorCategory.SERVER_ERROR),
            ("404 Not Found", CrawlerErrorCategory.NOT_FOUND),
            ("Resource not found", CrawlerErrorCategory.NOT_FOUND),
            ("JSON parse error", CrawlerErrorCategory.STRUCTURE_CHANGE),
            ("Unexpected format in response", CrawlerErrorCategory.STRUCTURE_CHANGE),
            ("Failed to decode response", CrawlerErrorCategory.STRUCTURE_CHANGE),
            ("Config setting missing", CrawlerErrorCategory.CONFIGURATION),
            ("Invalid value in config", CrawlerErrorCategory.CONFIGURATION),
        ],
    )
    def test_keyword_patterns(self, msg, expected_category):
        result = classify_error_message(msg)
        assert result.category == expected_category

    def test_unknown_message(self):
        result = classify_error_message("something completely random happened")
        assert result.category == CrawlerErrorCategory.UNKNOWN
        assert result.retryable is False
        assert result.severity == ErrorSeverity.MEDIUM

    def test_empty_message(self):
        result = classify_error_message("")
        assert result.category == CrawlerErrorCategory.UNKNOWN
        assert result.message == "Unknown error"

    def test_case_insensitive(self):
        result = classify_error_message("TIMEOUT occurred")
        assert result.category == CrawlerErrorCategory.TIMEOUT

    def test_preserves_source_exception(self):
        original = ValueError("oops")
        result = classify_error_message(
            "some error", source_exception=original
        )
        assert result.source_exception is original

    def test_first_match_wins(self):
        # "server error" comes before "timeout" in patterns (status codes first)
        result = classify_error_message("timeout from server error")
        assert result.category == CrawlerErrorCategory.SERVER_ERROR


# ---------------------------------------------------------------------------
# ClassifiedError
# ---------------------------------------------------------------------------
class TestClassifiedError:
    """Test the ClassifiedError dataclass."""

    def test_to_dict(self):
        result = ClassifiedError(
            category=CrawlerErrorCategory.NETWORK,
            message="DNS failed",
            retryable=True,
            severity=ErrorSeverity.MEDIUM,
            suggested_action="Check network",
        )
        d = result.to_dict()
        assert d == {
            "category": "network",
            "message": "DNS failed",
            "retryable": True,
            "severity": "medium",
            "suggested_action": "Check network",
        }

    def test_severity_value_property(self):
        result = ClassifiedError(
            category=CrawlerErrorCategory.AUTH,
            message="unauthorized",
            retryable=False,
            severity=ErrorSeverity.HIGH,
            suggested_action="Fix token",
        )
        assert result.severity_value == "high"

    def test_category_value_property(self):
        result = ClassifiedError(
            category=CrawlerErrorCategory.RATE_LIMIT,
            message="429",
            retryable=True,
            severity=ErrorSeverity.MEDIUM,
            suggested_action="Wait",
        )
        assert result.category_value == "rate_limit"

    def test_frozen_dataclass(self):
        result = ClassifiedError(
            category=CrawlerErrorCategory.UNKNOWN,
            message="oops",
            retryable=False,
            severity=ErrorSeverity.LOW,
            suggested_action="Check logs",
        )
        with pytest.raises(AttributeError):
            result.category = CrawlerErrorCategory.NETWORK


# ---------------------------------------------------------------------------
# Label conversion
# ---------------------------------------------------------------------------
class TestLabelConversion:
    """Test category_to_label and label_to_category."""

    @pytest.mark.parametrize(
        "category, label",
        [
            (CrawlerErrorCategory.NETWORK, "Network"),
            (CrawlerErrorCategory.TIMEOUT, "Timeout"),
            (CrawlerErrorCategory.RATE_LIMIT, "Rate Limited"),
            (CrawlerErrorCategory.AUTH, "Authentication"),
            (CrawlerErrorCategory.NOT_FOUND, "Not Found"),
            (CrawlerErrorCategory.SERVER_ERROR, "Server Error"),
            (CrawlerErrorCategory.STRUCTURE_CHANGE, "Structure Change"),
            (CrawlerErrorCategory.CONFIGURATION, "Configuration"),
            (CrawlerErrorCategory.UNKNOWN, "Unknown"),
        ],
    )
    def test_round_trip(self, category, label):
        assert category_to_label(category) == label
        assert label_to_category(label) == category

    def test_unknown_label(self):
        assert label_to_category("NonexistentLabel") == CrawlerErrorCategory.UNKNOWN


# ---------------------------------------------------------------------------
# wrap_exception()
# ---------------------------------------------------------------------------
class TestWrapException:
    """Test wrapping arbitrary exceptions into CrawlerError subclasses."""

    def test_already_crawler_error(self):
        exc = NetworkError("DNS failed", source="hf_api")
        wrapped = wrap_exception(exc)
        assert wrapped is exc  # Returns the same instance

    def test_already_crawler_error_updates_source(self):
        exc = NetworkError("DNS failed")
        assert exc.source == ""
        wrapped = wrap_exception(exc, source="hf_api")
        assert wrapped is exc
        assert wrapped.source == "hf_api"

    def test_wrap_requests_timeout(self):
        exc = requests.exceptions.Timeout("timed out")
        wrapped = wrap_exception(exc, source="hf_api")
        assert isinstance(wrapped, CrawlerTimeoutError)
        assert isinstance(wrapped, CrawlerError)
        assert wrapped.source == "hf_api"
        assert wrapped.__cause__ is exc

    def test_wrap_requests_connection_error(self):
        exc = requests.exceptions.ConnectionError("refused")
        wrapped = wrap_exception(exc, source="reddit")
        assert isinstance(wrapped, NetworkError)
        assert wrapped.__cause__ is exc

    def test_wrap_value_error(self):
        exc = ValueError("unexpected JSON")
        wrapped = wrap_exception(exc)
        assert isinstance(wrapped, StructureChangeError)
        assert wrapped.__cause__ is exc

    def test_wrap_unknown(self):
        exc = RuntimeError("weird")
        wrapped = wrap_exception(exc)
        assert isinstance(wrapped, CrawlerError)
        assert wrapped.__cause__ is exc

    def test_wrap_json_decode_error(self):
        import json
        exc = json.JSONDecodeError("oops", "", 0)
        wrapped = wrap_exception(exc, source="leaderboard")
        assert isinstance(wrapped, StructureChangeError)
        assert wrapped.source == "leaderboard"


# ---------------------------------------------------------------------------
# Suggested actions coverage
# ---------------------------------------------------------------------------
class TestSuggestedActions:
    """Verify every category has a meaningful suggested action."""

    def test_all_categories_have_actions(self):
        for category in CrawlerErrorCategory:
            exc_map = {
                CrawlerErrorCategory.NETWORK: NetworkError("test"),
                CrawlerErrorCategory.TIMEOUT: CrawlerTimeoutError("test"),
                CrawlerErrorCategory.RATE_LIMIT: RateLimitError("test"),
                CrawlerErrorCategory.AUTH: AuthenticationError("test"),
                CrawlerErrorCategory.NOT_FOUND: NotFoundError("test"),
                CrawlerErrorCategory.SERVER_ERROR: ServerError("test"),
                CrawlerErrorCategory.STRUCTURE_CHANGE: StructureChangeError("test"),
                CrawlerErrorCategory.CONFIGURATION: ConfigurationError("test"),
                CrawlerErrorCategory.UNKNOWN: CrawlerError("test"),
            }
            exc = exc_map[category]
            result = classify_error(exc)
            assert result.suggested_action, f"No action for {category}"
            assert len(result.suggested_action) > 10, f"Action too short for {category}"


# ---------------------------------------------------------------------------
# Backward compatibility with slack_notifier
# ---------------------------------------------------------------------------
class TestBackwardCompatibility:
    """Verify the new module is compatible with slack_notifier's classify_crawler_error."""

    def test_slack_classify_still_works(self):
        """The existing classify_crawler_error in slack_notifier should still return
        the same label strings after we update it to delegate to errors.py."""
        from hf_model_monitor.slack_notifier import classify_crawler_error

        assert classify_crawler_error("Connection timed out") == "Timeout"
        assert classify_crawler_error("rate limit exceeded") == "Rate Limited"
        assert classify_crawler_error("401 unauthorized") == "Authentication"
        assert classify_crawler_error("connection refused") == "Network"
        assert classify_crawler_error("500 server error") == "Server Error"
        assert classify_crawler_error("404 not found") == "Not Found"
        assert classify_crawler_error("JSON parse error") == "Structure Change"
        assert classify_crawler_error("something random") == "Unknown"

    def test_new_classifier_produces_same_labels(self):
        """classify_error_message categories should map to the same labels."""
        test_cases = [
            ("Connection timed out", "Timeout"),
            ("rate limit exceeded", "Rate Limited"),
            ("401 unauthorized", "Authentication"),
            ("connection refused", "Network"),
            ("500 server error", "Server Error"),
            ("404 not found", "Not Found"),
            ("JSON parse error", "Structure Change"),
            ("something random", "Unknown"),
        ]
        for msg, expected_label in test_cases:
            result = classify_error_message(msg)
            actual_label = category_to_label(result.category)
            assert actual_label == expected_label, (
                f"Message '{msg}': expected label '{expected_label}', "
                f"got '{actual_label}'"
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Edge cases and robustness checks."""

    def test_exception_with_no_message(self):
        exc = NetworkError("")
        result = classify_error(exc)
        assert result.category == CrawlerErrorCategory.NETWORK
        # Empty message should get a fallback
        assert result.message  # Non-empty

    def test_classify_error_with_source(self):
        exc = requests.exceptions.Timeout("timed out")
        result = classify_error(exc, source="open_llm_leaderboard")
        assert result.category == CrawlerErrorCategory.TIMEOUT
        assert result.source_exception is exc

    def test_nested_exception_chain(self):
        """Classify the outer exception even if it wraps an inner one."""
        inner = ConnectionError("refused")
        outer = requests.exceptions.ConnectionError("wrap")
        outer.__cause__ = inner
        result = classify_error(outer)
        assert result.category == CrawlerErrorCategory.NETWORK

    def test_enum_values_are_strings(self):
        """Category and severity enum values should be serializable strings."""
        for cat in CrawlerErrorCategory:
            assert isinstance(cat.value, str)
        for sev in ErrorSeverity:
            assert isinstance(sev.value, str)
