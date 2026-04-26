"""Crawler error types and classification for the HuggingFace Model Monitor.

Provides:
- ``CrawlerErrorCategory`` — enum of actionable error categories
- ``CrawlerError`` hierarchy — typed exceptions for each failure mode
- ``ClassifiedError`` — structured classification result with metadata
- ``classify_error()`` — classifies any exception into an actionable category
- ``classify_error_message()`` — string-based fallback classifier

Every crawler/fetcher in the pipeline (HF API, Open LLM Leaderboard,
Reddit, HF Discussions, API pricing) can raise or be wrapped into these
error types.  The scheduler and Slack notifier then use the classification
to decide retry strategy, alert severity, and suggested actions.

Design principles:
- Each error type knows whether it's retryable (transient vs permanent)
- Severity levels guide the PM on urgency (info → critical)
- Suggested actions are concrete ("check HF_TOKEN", "wait and retry")
- Backward-compatible with the string-based classifier in slack_notifier

Usage::

    from hf_model_monitor.errors import classify_error, CrawlerError, NetworkError

    try:
        response = requests.get(url, timeout=30)
    except requests.exceptions.ConnectionError as exc:
        classified = classify_error(exc)
        # classified.category    → CrawlerErrorCategory.NETWORK
        # classified.retryable   → True
        # classified.severity    → "medium"
        # classified.suggested_action → "Check network connectivity..."

    # Or raise typed errors directly:
    raise NetworkError("DNS resolution failed for huggingface.co", source="hf_api")
"""

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error categories
# ---------------------------------------------------------------------------
class CrawlerErrorCategory(Enum):
    """Actionable categories for crawler failures.

    Each category maps to a distinct failure mode with its own retry
    strategy and suggested resolution.  The PM can glance at the category
    in a Slack alert and know immediately what kind of problem occurred.
    """

    NETWORK = "network"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    AUTH = "auth"
    NOT_FOUND = "not_found"
    SERVER_ERROR = "server_error"
    STRUCTURE_CHANGE = "structure_change"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Severity levels
# ---------------------------------------------------------------------------
class ErrorSeverity(Enum):
    """Severity levels for PM alerting.

    - LOW: informational, self-resolving (e.g., single timeout)
    - MEDIUM: warrants attention but not urgent (e.g., rate limit)
    - HIGH: action needed soon (e.g., auth failure, repeated errors)
    - CRITICAL: immediate action needed (e.g., structure change suggesting
      API broke, or configuration error preventing all crawling)
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------
class CrawlerError(Exception):
    """Base exception for all crawler failures.

    Attributes:
        category: The error category for classification.
        retryable: Whether the operation should be retried.
        source: Name of the failing crawler/source (e.g., "hf_api", "reddit").
        status_code: HTTP status code if applicable, or None.
    """

    category: CrawlerErrorCategory = CrawlerErrorCategory.UNKNOWN
    retryable: bool = False
    default_severity: ErrorSeverity = ErrorSeverity.MEDIUM

    def __init__(
        self,
        message: str = "",
        *,
        source: str = "",
        status_code: int | None = None,
    ):
        super().__init__(message)
        self.source = source
        self.status_code = status_code


class NetworkError(CrawlerError):
    """Connection failures: DNS resolution, TCP connect, SSL errors.

    Typically transient — retry after a delay.
    """

    category = CrawlerErrorCategory.NETWORK
    retryable = True
    default_severity = ErrorSeverity.MEDIUM


class CrawlerTimeoutError(CrawlerError):
    """Request timed out waiting for a response.

    Transient — retry with possible timeout increase.
    Named ``CrawlerTimeoutError`` to avoid shadowing the builtin
    ``TimeoutError``.
    """

    category = CrawlerErrorCategory.TIMEOUT
    retryable = True
    default_severity = ErrorSeverity.LOW


class RateLimitError(CrawlerError):
    """HTTP 429 Too Many Requests or equivalent rate-limit signal.

    Retryable after honouring the backoff/Retry-After period.

    Attributes:
        retry_after: Suggested wait time in seconds (from Retry-After header).
    """

    category = CrawlerErrorCategory.RATE_LIMIT
    retryable = True
    default_severity = ErrorSeverity.MEDIUM

    def __init__(
        self,
        message: str = "",
        *,
        source: str = "",
        status_code: int | None = 429,
        retry_after: float | None = None,
    ):
        super().__init__(message, source=source, status_code=status_code)
        self.retry_after = retry_after


class AuthenticationError(CrawlerError):
    """HTTP 401/403 — invalid or missing credentials.

    Permanent until the PM fixes the token/credentials.
    """

    category = CrawlerErrorCategory.AUTH
    retryable = False
    default_severity = ErrorSeverity.HIGH


class NotFoundError(CrawlerError):
    """HTTP 404 — resource (org, model, endpoint) not found.

    Usually permanent — the resource was deleted, renamed, or the URL
    is wrong.  May be transient if HuggingFace is having routing issues.
    """

    category = CrawlerErrorCategory.NOT_FOUND
    retryable = False
    default_severity = ErrorSeverity.LOW


class ServerError(CrawlerError):
    """HTTP 5xx — upstream server error.

    Transient — the server is having issues, retry after a delay.
    """

    category = CrawlerErrorCategory.SERVER_ERROR
    retryable = True
    default_severity = ErrorSeverity.MEDIUM


class StructureChangeError(CrawlerError):
    """Response parsing failed — API structure changed or unexpected format.

    This is the most important error for long-term maintenance: it means
    the upstream API changed its response format, and the crawler code
    needs to be updated.

    Permanent until the crawler is fixed by the PM.
    """

    category = CrawlerErrorCategory.STRUCTURE_CHANGE
    retryable = False
    default_severity = ErrorSeverity.CRITICAL


class ConfigurationError(CrawlerError):
    """Invalid or missing configuration preventing crawling.

    Examples: missing webhook URL, empty org list, invalid API token format.
    Permanent until the PM fixes the configuration.
    """

    category = CrawlerErrorCategory.CONFIGURATION
    retryable = False
    default_severity = ErrorSeverity.HIGH


# ---------------------------------------------------------------------------
# Classification result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClassifiedError:
    """Structured result from error classification.

    Provides all the context the scheduler and Slack notifier need to
    decide how to handle a failure: retry or skip, what severity to
    assign the alert, and what the PM should do about it.

    Attributes:
        category: Actionable error category.
        message: Human-readable error description.
        retryable: Whether the operation should be retried.
        severity: Alert severity level for the PM.
        suggested_action: Concrete guidance for the PM.
        source_exception: The original exception (for logging/debugging).
    """

    category: CrawlerErrorCategory
    message: str
    retryable: bool
    severity: ErrorSeverity
    suggested_action: str
    source_exception: Exception | None = None

    @property
    def severity_value(self) -> str:
        """Return severity as a plain string (for JSON serialization)."""
        return self.severity.value

    @property
    def category_value(self) -> str:
        """Return category as a plain string (for JSON serialization)."""
        return self.category.value

    def to_dict(self) -> dict:
        """Serialize to a plain dict (for API responses / logging)."""
        return {
            "category": self.category.value,
            "message": self.message,
            "retryable": self.retryable,
            "severity": self.severity.value,
            "suggested_action": self.suggested_action,
        }


# ---------------------------------------------------------------------------
# Suggested actions per category
# ---------------------------------------------------------------------------
_SUGGESTED_ACTIONS: dict[CrawlerErrorCategory, str] = {
    CrawlerErrorCategory.NETWORK: (
        "Check network connectivity and DNS resolution. "
        "If persistent, verify that the target service is reachable."
    ),
    CrawlerErrorCategory.TIMEOUT: (
        "The request timed out. This is usually transient. "
        "If persistent, consider increasing the timeout setting."
    ),
    CrawlerErrorCategory.RATE_LIMIT: (
        "Rate limited by the upstream API. The system will "
        "automatically back off and retry. If frequent, consider "
        "reducing polling frequency or adding an API token."
    ),
    CrawlerErrorCategory.AUTH: (
        "Authentication failed. Check that HF_TOKEN (or other API "
        "credentials) are set correctly in your environment or config."
    ),
    CrawlerErrorCategory.NOT_FOUND: (
        "The requested resource was not found. Verify the organization "
        "name or model ID is correct and still exists on HuggingFace."
    ),
    CrawlerErrorCategory.SERVER_ERROR: (
        "The upstream server returned an error. This is usually "
        "transient. The system will retry automatically."
    ),
    CrawlerErrorCategory.STRUCTURE_CHANGE: (
        "The response format has changed unexpectedly. This likely "
        "means the upstream API was updated. The crawler code may "
        "need to be updated to handle the new format."
    ),
    CrawlerErrorCategory.CONFIGURATION: (
        "There is a configuration problem preventing this crawler "
        "from running. Check settings.yaml and environment variables."
    ),
    CrawlerErrorCategory.UNKNOWN: (
        "An unexpected error occurred. Check the logs for the full "
        "stack trace and investigate the root cause."
    ),
}


# ---------------------------------------------------------------------------
# Exception → category mapping rules
# ---------------------------------------------------------------------------
# Each rule: (exception_check_callable, category, retryable, severity)
# Checked in order; first match wins.  The check function receives the
# exception instance and returns True/False.

def _is_requests_available() -> bool:
    """Check if the requests library is importable (it should be)."""
    try:
        import requests  # noqa: F401
        return True
    except ImportError:
        return False


def _build_exception_rules() -> list[tuple]:
    """Build the ordered exception classification rules.

    Separated into a builder so we can gracefully handle the case
    where ``requests`` isn't installed (unlikely but defensive).
    """
    rules: list[tuple] = []

    # 1. Our own typed exceptions — highest priority (most specific)
    rules.append((
        lambda exc: isinstance(exc, RateLimitError),
        CrawlerErrorCategory.RATE_LIMIT,
        True,
        ErrorSeverity.MEDIUM,
    ))
    rules.append((
        lambda exc: isinstance(exc, AuthenticationError),
        CrawlerErrorCategory.AUTH,
        False,
        ErrorSeverity.HIGH,
    ))
    rules.append((
        lambda exc: isinstance(exc, CrawlerTimeoutError),
        CrawlerErrorCategory.TIMEOUT,
        True,
        ErrorSeverity.LOW,
    ))
    rules.append((
        lambda exc: isinstance(exc, NetworkError),
        CrawlerErrorCategory.NETWORK,
        True,
        ErrorSeverity.MEDIUM,
    ))
    rules.append((
        lambda exc: isinstance(exc, NotFoundError),
        CrawlerErrorCategory.NOT_FOUND,
        False,
        ErrorSeverity.LOW,
    ))
    rules.append((
        lambda exc: isinstance(exc, ServerError),
        CrawlerErrorCategory.SERVER_ERROR,
        True,
        ErrorSeverity.MEDIUM,
    ))
    rules.append((
        lambda exc: isinstance(exc, StructureChangeError),
        CrawlerErrorCategory.STRUCTURE_CHANGE,
        False,
        ErrorSeverity.CRITICAL,
    ))
    rules.append((
        lambda exc: isinstance(exc, ConfigurationError),
        CrawlerErrorCategory.CONFIGURATION,
        False,
        ErrorSeverity.HIGH,
    ))
    # Generic CrawlerError fallback (if someone raises CrawlerError directly)
    rules.append((
        lambda exc: isinstance(exc, CrawlerError),
        None,  # Use the instance's own category
        None,  # Use the instance's own retryable
        None,  # Use the instance's own severity
    ))

    # 2. HF client exceptions (from hf_client.py) — bridge to our hierarchy
    try:
        from .hf_client import HFRateLimitError as _HFRate, HFAuthError as _HFAuth, HFApiError as _HFApi

        rules.append((
            lambda exc: isinstance(exc, _HFRate),
            CrawlerErrorCategory.RATE_LIMIT,
            True,
            ErrorSeverity.MEDIUM,
        ))
        rules.append((
            lambda exc: isinstance(exc, _HFAuth),
            CrawlerErrorCategory.AUTH,
            False,
            ErrorSeverity.HIGH,
        ))
        rules.append((
            lambda exc: isinstance(exc, _HFApi) and getattr(exc, "status_code", None) and getattr(exc, "status_code", 0) >= 500,
            CrawlerErrorCategory.SERVER_ERROR,
            True,
            ErrorSeverity.MEDIUM,
        ))
        rules.append((
            lambda exc: isinstance(exc, _HFApi),
            CrawlerErrorCategory.NETWORK,
            True,
            ErrorSeverity.MEDIUM,
        ))
    except ImportError:
        pass  # hf_client not available — skip these rules

    # 3. Standard library / requests exceptions — lowest priority
    if _is_requests_available():
        import requests as _req

        rules.append((
            lambda exc: isinstance(exc, _req.exceptions.Timeout),
            CrawlerErrorCategory.TIMEOUT,
            True,
            ErrorSeverity.LOW,
        ))
        rules.append((
            lambda exc: isinstance(exc, _req.exceptions.ConnectionError),
            CrawlerErrorCategory.NETWORK,
            True,
            ErrorSeverity.MEDIUM,
        ))
        rules.append((
            lambda exc: (
                isinstance(exc, _req.exceptions.HTTPError)
                and hasattr(exc, "response")
                and exc.response is not None
                and exc.response.status_code == 429
            ),
            CrawlerErrorCategory.RATE_LIMIT,
            True,
            ErrorSeverity.MEDIUM,
        ))
        rules.append((
            lambda exc: (
                isinstance(exc, _req.exceptions.HTTPError)
                and hasattr(exc, "response")
                and exc.response is not None
                and exc.response.status_code in (401, 403)
            ),
            CrawlerErrorCategory.AUTH,
            False,
            ErrorSeverity.HIGH,
        ))
        rules.append((
            lambda exc: (
                isinstance(exc, _req.exceptions.HTTPError)
                and hasattr(exc, "response")
                and exc.response is not None
                and exc.response.status_code == 404
            ),
            CrawlerErrorCategory.NOT_FOUND,
            False,
            ErrorSeverity.LOW,
        ))
        rules.append((
            lambda exc: (
                isinstance(exc, _req.exceptions.HTTPError)
                and hasattr(exc, "response")
                and exc.response is not None
                and exc.response.status_code >= 500
            ),
            CrawlerErrorCategory.SERVER_ERROR,
            True,
            ErrorSeverity.MEDIUM,
        ))
        rules.append((
            lambda exc: isinstance(exc, _req.exceptions.RequestException),
            CrawlerErrorCategory.NETWORK,
            True,
            ErrorSeverity.MEDIUM,
        ))

    # 4. Built-in Python exceptions
    #    Order matters: more specific subclasses BEFORE their parents.
    #    PermissionError and FileNotFoundError are subclasses of OSError,
    #    so they must be checked before the generic OSError/ConnectionError rule.
    rules.append((
        lambda exc: isinstance(exc, PermissionError),
        CrawlerErrorCategory.AUTH,
        False,
        ErrorSeverity.HIGH,
    ))
    rules.append((
        lambda exc: isinstance(exc, FileNotFoundError),
        CrawlerErrorCategory.CONFIGURATION,
        False,
        ErrorSeverity.HIGH,
    ))
    rules.append((
        lambda exc: isinstance(exc, TimeoutError),
        CrawlerErrorCategory.TIMEOUT,
        True,
        ErrorSeverity.LOW,
    ))
    rules.append((
        lambda exc: isinstance(exc, (ConnectionError, OSError)),
        CrawlerErrorCategory.NETWORK,
        True,
        ErrorSeverity.MEDIUM,
    ))
    rules.append((
        lambda exc: isinstance(exc, (ValueError, KeyError, TypeError, AttributeError)),
        CrawlerErrorCategory.STRUCTURE_CHANGE,
        False,
        ErrorSeverity.CRITICAL,
    ))
    rules.append((
        lambda exc: isinstance(exc, (
            json_module.JSONDecodeError if (json_module := __import__("json")) else type(None)
        )),
        CrawlerErrorCategory.STRUCTURE_CHANGE,
        False,
        ErrorSeverity.CRITICAL,
    ))

    return rules


# Build rules once at import time
_EXCEPTION_RULES = _build_exception_rules()


# ---------------------------------------------------------------------------
# String-based classification (keyword matching)
# ---------------------------------------------------------------------------
# Order matters: first match wins.
_MESSAGE_PATTERNS: list[tuple[list[str], CrawlerErrorCategory]] = [
    # HTTP status codes first — they're unambiguous identifiers.
    # "504 Gateway Timeout" should be SERVER_ERROR, not TIMEOUT.
    (["500", "502", "503", "504", "server error", "internal server"], CrawlerErrorCategory.SERVER_ERROR),
    (["rate limit", "429", "too many requests"], CrawlerErrorCategory.RATE_LIMIT),
    (["401", "403", "auth", "unauthorized", "forbidden"], CrawlerErrorCategory.AUTH),
    (["404", "not found"], CrawlerErrorCategory.NOT_FOUND),
    # Generic keywords after status codes
    (["timeout", "timed out"], CrawlerErrorCategory.TIMEOUT),
    (["connection", "network", "dns", "resolve", "unreachable"], CrawlerErrorCategory.NETWORK),
    (["parse", "json", "structure", "format", "unexpected", "decode"], CrawlerErrorCategory.STRUCTURE_CHANGE),
    (["config", "setting", "missing key", "invalid value"], CrawlerErrorCategory.CONFIGURATION),
]

# Default metadata per category (for string-based classification)
_CATEGORY_DEFAULTS: dict[CrawlerErrorCategory, tuple[bool, ErrorSeverity]] = {
    CrawlerErrorCategory.NETWORK: (True, ErrorSeverity.MEDIUM),
    CrawlerErrorCategory.TIMEOUT: (True, ErrorSeverity.LOW),
    CrawlerErrorCategory.RATE_LIMIT: (True, ErrorSeverity.MEDIUM),
    CrawlerErrorCategory.AUTH: (False, ErrorSeverity.HIGH),
    CrawlerErrorCategory.NOT_FOUND: (False, ErrorSeverity.LOW),
    CrawlerErrorCategory.SERVER_ERROR: (True, ErrorSeverity.MEDIUM),
    CrawlerErrorCategory.STRUCTURE_CHANGE: (False, ErrorSeverity.CRITICAL),
    CrawlerErrorCategory.CONFIGURATION: (False, ErrorSeverity.HIGH),
    CrawlerErrorCategory.UNKNOWN: (False, ErrorSeverity.MEDIUM),
}


# ---------------------------------------------------------------------------
# Public API: classify_error
# ---------------------------------------------------------------------------
def classify_error(
    exc: Exception,
    *,
    source: str = "",
) -> ClassifiedError:
    """Classify any exception into an actionable error category.

    Checks the exception type against a prioritized rule list:
    1. Our own CrawlerError subclasses (most specific)
    2. HF client exceptions (HFApiError hierarchy)
    3. ``requests`` library exceptions
    4. Python built-in exceptions (ValueError, ConnectionError, etc.)
    5. Falls back to string-based keyword matching on the message

    Args:
        exc: The exception to classify.
        source: Name of the failing crawler/source (e.g., "hf_api").

    Returns:
        A ClassifiedError with category, retryable flag, severity,
        and a suggested action for the PM.

    Examples::

        >>> import requests
        >>> classify_error(requests.exceptions.Timeout("timed out"))
        ClassifiedError(category=TIMEOUT, retryable=True, severity=low, ...)

        >>> classify_error(NetworkError("DNS failed", source="hf_api"))
        ClassifiedError(category=NETWORK, retryable=True, severity=medium, ...)
    """
    # Try exception-type-based rules first
    for check_fn, category, retryable, severity in _EXCEPTION_RULES:
        try:
            if check_fn(exc):
                # For generic CrawlerError, use instance attributes
                if category is None and isinstance(exc, CrawlerError):
                    category = exc.category
                    retryable = exc.retryable
                    severity = exc.default_severity

                # Resolve source: prefer explicit arg, then exception attr
                resolved_source = source or getattr(exc, "source", "") or ""

                return ClassifiedError(
                    category=category,
                    message=str(exc) or f"{type(exc).__name__} (no message)",
                    retryable=retryable,
                    severity=severity,
                    suggested_action=_SUGGESTED_ACTIONS.get(
                        category, _SUGGESTED_ACTIONS[CrawlerErrorCategory.UNKNOWN]
                    ),
                    source_exception=exc,
                )
        except Exception:
            # If a rule check itself fails, skip it silently
            continue

    # Fall back to string-based classification on the error message
    return classify_error_message(str(exc), source=source, source_exception=exc)


def classify_error_message(
    error_msg: str,
    *,
    source: str = "",
    source_exception: Exception | None = None,
) -> ClassifiedError:
    """Classify an error based on its message string (keyword matching).

    This is the fallback classifier used when the exception type doesn't
    match any known rule.  Also useful for classifying error messages
    stored in logs or databases.

    Args:
        error_msg: The raw error message to classify.
        source: Name of the failing source.
        source_exception: The original exception, if available.

    Returns:
        A ClassifiedError with the best-guess category.
    """
    lower = error_msg.lower() if error_msg else ""

    for keywords, category in _MESSAGE_PATTERNS:
        if any(kw in lower for kw in keywords):
            retryable, severity = _CATEGORY_DEFAULTS[category]
            return ClassifiedError(
                category=category,
                message=error_msg or "Unknown error",
                retryable=retryable,
                severity=severity,
                suggested_action=_SUGGESTED_ACTIONS.get(
                    category, _SUGGESTED_ACTIONS[CrawlerErrorCategory.UNKNOWN]
                ),
                source_exception=source_exception,
            )

    # No pattern matched — unknown
    retryable, severity = _CATEGORY_DEFAULTS[CrawlerErrorCategory.UNKNOWN]
    return ClassifiedError(
        category=CrawlerErrorCategory.UNKNOWN,
        message=error_msg or "Unknown error",
        retryable=retryable,
        severity=severity,
        suggested_action=_SUGGESTED_ACTIONS[CrawlerErrorCategory.UNKNOWN],
        source_exception=source_exception,
    )


# ---------------------------------------------------------------------------
# Backward-compatible label mapping
# ---------------------------------------------------------------------------
# Maps CrawlerErrorCategory → the short label strings used by the existing
# classify_crawler_error() in slack_notifier.py, so we can update that
# function to delegate here without changing its return type.

_CATEGORY_TO_LABEL: dict[CrawlerErrorCategory, str] = {
    CrawlerErrorCategory.NETWORK: "Network",
    CrawlerErrorCategory.TIMEOUT: "Timeout",
    CrawlerErrorCategory.RATE_LIMIT: "Rate Limited",
    CrawlerErrorCategory.AUTH: "Authentication",
    CrawlerErrorCategory.NOT_FOUND: "Not Found",
    CrawlerErrorCategory.SERVER_ERROR: "Server Error",
    CrawlerErrorCategory.STRUCTURE_CHANGE: "Structure Change",
    CrawlerErrorCategory.CONFIGURATION: "Configuration",
    CrawlerErrorCategory.UNKNOWN: "Unknown",
}

_LABEL_TO_CATEGORY: dict[str, CrawlerErrorCategory] = {
    v: k for k, v in _CATEGORY_TO_LABEL.items()
}


def category_to_label(category: CrawlerErrorCategory) -> str:
    """Convert a CrawlerErrorCategory to the short label string.

    Used by the Slack notifier for display.

    Args:
        category: The error category enum value.

    Returns:
        Human-readable label string (e.g., "Network", "Rate Limited").
    """
    return _CATEGORY_TO_LABEL.get(category, "Unknown")


def label_to_category(label: str) -> CrawlerErrorCategory:
    """Convert a short label string back to a CrawlerErrorCategory.

    Args:
        label: Label string (e.g., "Network", "Rate Limited").

    Returns:
        The corresponding CrawlerErrorCategory, or UNKNOWN if not found.
    """
    return _LABEL_TO_CATEGORY.get(label, CrawlerErrorCategory.UNKNOWN)


# ---------------------------------------------------------------------------
# Convenience: wrap a raw exception into the appropriate CrawlerError
# ---------------------------------------------------------------------------
def wrap_exception(
    exc: Exception,
    *,
    source: str = "",
) -> CrawlerError:
    """Wrap an arbitrary exception into the appropriate CrawlerError subclass.

    If the exception is already a CrawlerError, returns it unchanged
    (with source updated if not already set).

    This is useful in crawler code that wants to normalize all exceptions
    into the CrawlerError hierarchy before propagating.

    Args:
        exc: The exception to wrap.
        source: Name of the failing crawler/source.

    Returns:
        A CrawlerError subclass instance wrapping the original exception.
    """
    # Already one of ours — just update source if needed
    if isinstance(exc, CrawlerError):
        if source and not exc.source:
            exc.source = source
        return exc

    # Classify to determine the right subclass
    classified = classify_error(exc, source=source)

    # Map category → exception class
    exc_class = _CATEGORY_TO_EXCEPTION.get(classified.category, CrawlerError)

    wrapped = exc_class(
        str(exc),
        source=source,
        status_code=getattr(exc, "status_code", None),
    )
    wrapped.__cause__ = exc
    return wrapped


# Category → exception class mapping for wrap_exception
_CATEGORY_TO_EXCEPTION: dict[CrawlerErrorCategory, type[CrawlerError]] = {
    CrawlerErrorCategory.NETWORK: NetworkError,
    CrawlerErrorCategory.TIMEOUT: CrawlerTimeoutError,
    CrawlerErrorCategory.RATE_LIMIT: RateLimitError,
    CrawlerErrorCategory.AUTH: AuthenticationError,
    CrawlerErrorCategory.NOT_FOUND: NotFoundError,
    CrawlerErrorCategory.SERVER_ERROR: ServerError,
    CrawlerErrorCategory.STRUCTURE_CHANGE: StructureChangeError,
    CrawlerErrorCategory.CONFIGURATION: ConfigurationError,
    CrawlerErrorCategory.UNKNOWN: CrawlerError,
}
