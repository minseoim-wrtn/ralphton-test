"""Slack notification infrastructure for the HuggingFace Model Monitor.

Provides:
- ``SlackNotifier`` — class-based notifier with exponential-backoff retry,
  structured error handling, and dashboard-link support.
- ``format_block_kit_report`` — Slack Block Kit formatter for rich model
  reports with all mandatory fields (name, author, release date, model size,
  architecture, license, HF URL).
- ``format_report`` / ``send_to_slack`` — backward-compatible module-level
  functions used by the legacy pipeline and scheduler.
- Error classification helpers for crawler/pipeline failure alerts.

Usage (class-based, Block Kit)::

    notifier = SlackNotifier(webhook_url="https://hooks.slack.com/services/...")
    success = notifier.send_report(model_metadata, reference_data)
    # → sends a structured Block Kit message with all mandatory fields

Usage (function-based, legacy)::

    from hf_model_monitor.slack_notifier import format_report, send_to_slack

    report = format_report(metadata, references)
    send_to_slack(report, webhook_url="https://hooks.slack.com/services/...")
"""

import logging
import time
from datetime import datetime, timezone

import requests

from .config import (
    SLACK_BASE_DELAY_SECONDS,
    SLACK_MAX_DELAY_SECONDS,
    SLACK_MAX_RETRIES,
    SLACK_REQUEST_TIMEOUT_SECONDS,
    SLACK_WEBHOOK_URL,
    validate_webhook_url,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SlackNotifier class
# ---------------------------------------------------------------------------
class SlackNotifier:
    """Sends messages to Slack via incoming webhooks with retry logic.

    Features:
    - Exponential-backoff retry for transient failures (5xx, timeout, connection)
    - Distinguishes retryable from permanent errors (4xx except 429)
    - Honours ``Retry-After`` header on 429 rate-limit responses
    - Dashboard link injection into model reports
    - Factory method for config-dict construction

    Args:
        webhook_url: Slack incoming webhook URL (``https://hooks.slack.com/...``).
        max_retries: Maximum number of retry attempts for transient failures.
        base_delay: Initial delay in seconds before the first retry.
        max_delay: Maximum delay cap in seconds (backoff won't exceed this).
        timeout: HTTP request timeout in seconds.
        dashboard_base_url: Base URL for dashboard links in reports (optional).
    """

    def __init__(
        self,
        webhook_url: str = "",
        *,
        max_retries: int = SLACK_MAX_RETRIES,
        base_delay: float = SLACK_BASE_DELAY_SECONDS,
        max_delay: float = SLACK_MAX_DELAY_SECONDS,
        timeout: float = SLACK_REQUEST_TIMEOUT_SECONDS,
        dashboard_base_url: str = "",
    ):
        self.webhook_url = webhook_url
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.dashboard_base_url = dashboard_base_url

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_configured(self) -> bool:
        """Return True if a valid webhook URL is set."""
        return validate_webhook_url(self.webhook_url)

    def send(self, message: str) -> bool:
        """Send a plain-text message to Slack.

        Args:
            message: The message text to send. Empty messages are rejected.

        Returns:
            True if the message was delivered successfully, False otherwise.
        """
        if not message:
            logger.warning("Empty message, skipping Slack send")
            return False
        if not self.is_configured:
            logger.warning(
                "Slack webhook URL not configured or invalid, skipping send"
            )
            return False

        payload = {"text": message}
        return self._post_with_retry(payload)

    def send_blocks(self, blocks: list[dict], fallback_text: str = "") -> bool:
        """Send a Slack Block Kit message.

        Args:
            blocks: List of Block Kit block dicts (sections, dividers, etc.).
            fallback_text: Plain-text fallback for notifications/accessibility.
                          If empty, a generic fallback is used.

        Returns:
            True if the message was delivered successfully, False otherwise.
        """
        if not blocks:
            logger.warning("Empty blocks list, skipping Slack send")
            return False
        if not self.is_configured:
            logger.warning(
                "Slack webhook URL not configured or invalid, skipping send"
            )
            return False

        payload = {
            "blocks": blocks,
            "text": fallback_text or "HF Model Monitor — New Model Report",
        }
        return self._post_with_retry(payload)

    def send_report(
        self, model_metadata: dict, reference_data: dict
    ) -> bool:
        """Format a model report as Block Kit and send it to Slack.

        Builds a structured Block Kit message with all mandatory fields
        (name, author, release date, model size, architecture, license,
        HF URL) plus comparison table and dashboard link.

        Falls back to plain-text ``format_report`` if Block Kit construction
        fails for any reason.

        Args:
            model_metadata: Collected metadata dict for the new model.
            reference_data: Reference model comparison data.

        Returns:
            True if the report was delivered successfully.
        """
        dashboard_url = ""
        if self.dashboard_base_url:
            model_name = model_metadata.get("basic", {}).get("name", "")
            if model_name:
                dashboard_url = (
                    f"{self.dashboard_base_url.rstrip('/')}/models/{model_name}"
                )

        blocks = format_block_kit_report(
            model_metadata, reference_data, dashboard_url=dashboard_url
        )
        fallback = format_report(model_metadata, reference_data)

        # Append dashboard link to fallback text so plain-text clients
        # (and Slack notifications) also surface the dashboard URL.
        if dashboard_url:
            fallback += f"\n\n:link: View on Dashboard: {dashboard_url}"

        return self.send_blocks(blocks, fallback_text=fallback)

    def send_error_alert(
        self, error_msg: str, consecutive_failures: int = 1
    ) -> bool:
        """Send a pipeline-error alert to Slack.

        Uses ``format_pipeline_error_alert`` for consistent formatting.

        Args:
            error_msg: Description of the error.
            consecutive_failures: Number of consecutive failures (for severity).

        Returns:
            True if the alert was delivered successfully.
        """
        alert = format_pipeline_error_alert(
            error_msg, consecutive_failures=consecutive_failures
        )
        return self.send(alert)

    # ------------------------------------------------------------------
    # Retry internals
    # ------------------------------------------------------------------

    def _post_with_retry(self, payload: dict) -> bool:
        """POST *payload* to the webhook URL with exponential-backoff retry.

        Retries on:
        - HTTP 429 (rate limited) — honours ``Retry-After`` header when present
        - HTTP 5xx (server errors)
        - Connection / timeout errors

        Does NOT retry on:
        - HTTP 4xx (except 429) — these indicate a permanent problem
        - Empty or invalid webhook URL (checked before calling this method)

        Returns:
            True if the message was accepted (HTTP 2xx), False otherwise.
        """
        last_exception: Exception | None = None

        for attempt in range(1 + self.max_retries):
            try:
                resp = requests.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout,
                )

                # Success
                if resp.ok:
                    if attempt > 0:
                        logger.info(
                            "Slack message delivered on retry attempt %d",
                            attempt,
                        )
                    return True

                # Rate-limited — always retry with Retry-After if available
                if resp.status_code == 429:
                    retry_after = self._parse_retry_after(resp)
                    delay = retry_after or self._backoff_delay(attempt)
                    logger.warning(
                        "Slack rate-limited (429). Retrying in %.1fs "
                        "(attempt %d/%d)",
                        delay,
                        attempt + 1,
                        self.max_retries,
                    )
                    if attempt < self.max_retries:
                        time.sleep(delay)
                        continue
                    logger.error(
                        "Slack rate-limited — exhausted all %d retries",
                        self.max_retries,
                    )
                    return False

                # Server error — retryable
                if resp.status_code >= 500:
                    delay = self._backoff_delay(attempt)
                    logger.warning(
                        "Slack server error (%d). Retrying in %.1fs "
                        "(attempt %d/%d)",
                        resp.status_code,
                        delay,
                        attempt + 1,
                        self.max_retries,
                    )
                    if attempt < self.max_retries:
                        time.sleep(delay)
                        continue
                    logger.error(
                        "Slack server error (%d) — exhausted all %d retries",
                        resp.status_code,
                        self.max_retries,
                    )
                    return False

                # Client error (4xx, not 429) — permanent, don't retry
                logger.error(
                    "Slack returned client error %d: %s. "
                    "Check webhook URL configuration.",
                    resp.status_code,
                    resp.text[:200],
                )
                return False

            except requests.exceptions.Timeout as exc:
                last_exception = exc
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "Slack request timed out (%.0fs). Retrying in %.1fs "
                    "(attempt %d/%d)",
                    self.timeout,
                    delay,
                    attempt + 1,
                    self.max_retries,
                )
                if attempt < self.max_retries:
                    time.sleep(delay)
                    continue

            except requests.exceptions.ConnectionError as exc:
                last_exception = exc
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "Slack connection error: %s. Retrying in %.1fs "
                    "(attempt %d/%d)",
                    exc,
                    delay,
                    attempt + 1,
                    self.max_retries,
                )
                if attempt < self.max_retries:
                    time.sleep(delay)
                    continue

            except requests.exceptions.RequestException as exc:
                # Catch-all for other request issues — don't retry
                logger.error(
                    "Slack request failed with non-retryable error: %s", exc
                )
                return False

        # All retries exhausted
        logger.error(
            "Failed to send Slack message after %d retries. Last error: %s",
            self.max_retries,
            last_exception,
        )
        return False

    def _backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay for a given attempt number.

        Formula: ``min(base_delay * 2^attempt, max_delay)``

        No jitter is added to keep behavior deterministic and testable.
        For a solo-PM system with low message volume, thundering-herd
        is not a concern.

        Args:
            attempt: Zero-based attempt number.

        Returns:
            Delay in seconds.
        """
        delay = self.base_delay * (2 ** attempt)
        return min(delay, self.max_delay)

    @staticmethod
    def _parse_retry_after(resp: requests.Response) -> float | None:
        """Extract ``Retry-After`` header value as seconds, if present.

        Returns None when the header is missing or unparseable.
        """
        header = resp.headers.get("Retry-After")
        if header is None:
            return None
        try:
            value = float(header)
            return max(value, 0.0)
        except (ValueError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict) -> "SlackNotifier":
        """Create a SlackNotifier from a config dict (as from load_config).

        Args:
            config: Configuration dict containing ``slack_webhook_url``
                    and optionally ``dashboard_base_url``.

        Returns:
            Configured SlackNotifier instance.
        """
        return cls(
            webhook_url=config.get("slack_webhook_url", ""),
            dashboard_base_url=config.get("dashboard_base_url", ""),
        )


# ---------------------------------------------------------------------------
# Block Kit report formatting
# ---------------------------------------------------------------------------

# Mandatory fields that must appear in every Block Kit report.
# Each tuple: (display_label, metadata_section, metadata_key, fallback_value)
_MANDATORY_FIELDS: list[tuple[str, str, str, str]] = [
    ("Model Name", "basic", "name", "Unknown"),
    ("Author", "basic", "org", "N/A"),
    ("Release Date", "basic", "release_date", "N/A"),
    ("Model Size", "basic", "params", "N/A"),
    ("Architecture", "basic", "architecture", "N/A"),
    ("License", "basic", "license", "N/A"),
]


def _get_field(metadata: dict, section: str, key: str, fallback: str) -> str:
    """Safely extract a field value from nested metadata.

    Returns the fallback value when the key is missing, None, or empty string.
    """
    value = metadata.get(section, {}).get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        return fallback
    return str(value)


def _is_available(value: str) -> bool:
    """Return True when *value* contains real data (not a placeholder).

    Treats ``"N/A"``, empty strings, and None-like sentinels as unavailable.
    """
    if not value:
        return False
    return value.strip().upper() not in ("N/A", "UNKNOWN", "NONE", "")


def _build_hf_url(model_metadata: dict) -> str:
    """Construct the HuggingFace model URL from metadata.

    Tries ``basic.model_id`` first (full ``org/name``), then falls back to
    combining ``basic.org`` and ``basic.name``.
    """
    model_id = model_metadata.get("model_id", "")
    basic = model_metadata.get("basic", {})
    if not model_id:
        model_id = basic.get("model_id", "")
    if model_id:
        return f"https://huggingface.co/{model_id}"

    org = basic.get("org", "") or model_metadata.get("org", "")
    name = basic.get("name", "") or model_metadata.get("name", "")
    if org and org != "N/A" and name and name != "Unknown":
        return f"https://huggingface.co/{org}/{name}"
    if name and name != "Unknown":
        return f"https://huggingface.co/{name}"
    return ""


# ---------------------------------------------------------------------------
# Optional Block Kit section builders
# ---------------------------------------------------------------------------

# Benchmark fields: (display_label, metadata_section, metadata_key)
_BENCHMARK_FIELDS: list[tuple[str, str, str]] = [
    ("MMLU", "performance", "mmlu"),
    ("HumanEval", "performance", "humaneval"),
    ("GPQA", "performance", "gpqa"),
    ("MATH", "performance", "math"),
    ("Arena ELO", "performance", "arena_elo"),
]


def _build_benchmark_blocks(model_metadata: dict) -> list[dict]:
    """Build Block Kit blocks for benchmark scores (optional section).

    Only returns blocks when at least one benchmark field has a real value
    (not N/A). Renders available scores as two-column fields with a section
    header.

    Args:
        model_metadata: Collected metadata dict.

    Returns:
        List of Block Kit block dicts. Empty list if no benchmarks available.
    """
    fields = []
    for label, section, key in _BENCHMARK_FIELDS:
        value = _get_field(model_metadata, section, key, "N/A")
        if _is_available(value):
            fields.append({
                "type": "mrkdwn",
                "text": f"*{label}:*\n{value}",
            })

    if not fields:
        return []

    blocks: list[dict] = [
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": ":trophy: *Benchmark Scores*",
            },
        },
    ]

    # Slack limits sections to 10 fields; split if needed
    for i in range(0, len(fields), 10):
        blocks.append({
            "type": "section",
            "fields": fields[i : i + 10],
        })

    return blocks


def _build_pricing_blocks(model_metadata: dict) -> list[dict]:
    """Build Block Kit blocks for API pricing information (optional section).

    Checks multiple pricing key conventions used across the codebase:
    - ``cost.api_price_input_per_1m`` / ``cost.api_price_output_per_1m``
      (seed data format)
    - ``cost.api_price_per_million_tokens`` (legacy metadata collector format)
    - ``provider.name`` / ``provider.api_providers`` for provider info

    Only returns blocks when at least one pricing field has a real value.

    Args:
        model_metadata: Collected metadata dict.

    Returns:
        List of Block Kit block dicts. Empty list if no pricing available.
    """
    cost = model_metadata.get("cost", {})
    provider = model_metadata.get("provider", {})

    # Collect pricing fields from both naming conventions
    price_input = str(cost.get("api_price_input_per_1m", "N/A"))
    price_output = str(cost.get("api_price_output_per_1m", "N/A"))
    price_legacy = str(cost.get("api_price_per_million_tokens", "N/A"))

    # Provider info
    provider_name = str(provider.get("name", "N/A"))
    api_providers = provider.get("api_providers", [])
    if isinstance(api_providers, list) and api_providers:
        provider_list = ", ".join(str(p) for p in api_providers)
    else:
        provider_list = ""

    # Check if ANY pricing/provider data is available
    has_data = any(
        _is_available(v)
        for v in [price_input, price_output, price_legacy, provider_name]
    )
    if not has_data:
        return []

    fields = []

    # Input/output pricing (preferred format)
    if _is_available(price_input):
        fields.append({
            "type": "mrkdwn",
            "text": f"*Input Price (per 1M tokens):*\n{price_input}",
        })
    if _is_available(price_output):
        fields.append({
            "type": "mrkdwn",
            "text": f"*Output Price (per 1M tokens):*\n{price_output}",
        })

    # Legacy single-price field (only if input/output not present)
    if not _is_available(price_input) and _is_available(price_legacy):
        fields.append({
            "type": "mrkdwn",
            "text": f"*API Price (per 1M tokens):*\n{price_legacy}",
        })

    # Provider name
    if _is_available(provider_name):
        provider_text = provider_name
        if provider_list and provider_list != provider_name:
            provider_text += f" ({provider_list})"
        fields.append({
            "type": "mrkdwn",
            "text": f"*Provider:*\n{provider_text}",
        })

    blocks: list[dict] = [
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": ":moneybag: *Pricing & Availability*",
            },
        },
    ]

    for i in range(0, len(fields), 10):
        blocks.append({
            "type": "section",
            "fields": fields[i : i + 10],
        })

    return blocks


def _build_community_links_blocks(
    model_metadata: dict,
    *,
    dashboard_url: str = "",
) -> list[dict]:
    """Build Block Kit blocks for community discussion and dashboard links.

    Renders a section with clickable links to:
    - **HF Discussions** — always available when a model_id/HF URL can be
      constructed.  URL pattern: ``https://huggingface.co/{model_id}/discussions``
    - **Reddit threads** — shown when ``community.reddit_urls`` (list) or
      ``community.reddit_url`` (string) is present in metadata.
    - **Dashboard** — shown when *dashboard_url* is provided.

    All links are rendered as Slack mrkdwn ``<url|label>`` inline links
    inside a single section block, keeping the message compact.

    Only returns blocks when at least one link is available.

    Args:
        model_metadata: Collected metadata dict.
        dashboard_url: Full URL to the model's dashboard page (optional).

    Returns:
        List of Block Kit block dicts. Empty list if no links available.
    """
    link_parts: list[str] = []

    # ── HF Discussions URL ───────────────────────────────────────────
    hf_url = _build_hf_url(model_metadata)
    if hf_url:
        discussions_url = f"{hf_url.rstrip('/')}/discussions"
        link_parts.append(
            f":speech_balloon: *HF Discussions:* <{discussions_url}|View Discussions>"
        )

    # ── Reddit thread URLs ───────────────────────────────────────────
    community = model_metadata.get("community", {})
    reddit_urls: list[str] = []

    # Support both list and single-string conventions
    raw_urls = community.get("reddit_urls", [])
    if isinstance(raw_urls, list):
        reddit_urls.extend(
            u for u in raw_urls if isinstance(u, str) and u.strip()
        )
    raw_url = community.get("reddit_url", "")
    if isinstance(raw_url, str) and raw_url.strip() and raw_url not in reddit_urls:
        reddit_urls.append(raw_url.strip())

    if reddit_urls:
        if len(reddit_urls) == 1:
            link_parts.append(
                f":reddit: *Reddit:* <{reddit_urls[0]}|View Thread>"
            )
        else:
            thread_links = ", ".join(
                f"<{url}|Thread {i + 1}>"
                for i, url in enumerate(reddit_urls[:5])  # Cap at 5
            )
            link_parts.append(f":reddit: *Reddit:* {thread_links}")

    # ── Dashboard link ───────────────────────────────────────────────
    if dashboard_url:
        link_parts.append(
            f":bar_chart: *Dashboard:* <{dashboard_url}|View on Dashboard>"
        )

    if not link_parts:
        return []

    blocks: list[dict] = [
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": ":link: *Community & Links*",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "\n".join(link_parts),
            },
        },
    ]

    return blocks


def _build_tags_block(model_metadata: dict) -> list[dict]:
    """Build a Block Kit context block for model tags/category (optional).

    Sources tags from:
    - Top-level ``category`` string (e.g. ``"llm"``, ``"code"``, ``"vision"``)
    - Top-level ``tags`` list (HuggingFace model tags)
    - ``basic.pipeline_tag`` or ``basic.category`` as fallbacks

    Only returns blocks when at least one tag or category is present.

    Args:
        model_metadata: Collected metadata dict.

    Returns:
        List of Block Kit block dicts. Empty list if no tags available.
    """
    tag_parts: list[str] = []

    # Category (top-level or from basic section)
    category = model_metadata.get("category", "")
    if not category or not _is_available(str(category)):
        category = _get_field(model_metadata, "basic", "category", "")
    if category and _is_available(str(category)):
        tag_parts.append(f"`{category}`")

    # Tags list (top-level)
    tags = model_metadata.get("tags", [])
    if isinstance(tags, list):
        for tag in tags[:8]:  # Cap at 8 tags to avoid message overflow
            tag_str = str(tag).strip()
            if tag_str and _is_available(tag_str):
                tag_parts.append(f"`{tag_str}`")

    if not tag_parts:
        return []

    return [
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f":label: *Tags:* {' '.join(tag_parts)}",
                },
            ],
        },
    ]


def format_block_kit_report(
    model_metadata: dict,
    reference_data: dict,
    *,
    dashboard_url: str = "",
) -> list[dict]:
    """Format a full model report as Slack Block Kit blocks.

    Produces a structured Slack message with clearly separated mandatory
    and optional sections:

    **Mandatory fields** (always shown, N/A when unavailable):
    - Model Name, Author, Release Date, Model Size (parameters),
      Architecture, License, HF URL

    **Optional fields** (shown only when real data exists):
    - Benchmark scores: MMLU, HumanEval, GPQA, MATH, Arena ELO
    - Pricing info: API input/output pricing, provider name
    - Tags: model category and HuggingFace tags

    **Layout order:**
    1. Header block with model name
    2. Author + release date context line
    3. Mandatory fields section (two-column key-value pairs)
    4. Tags context (optional)
    5. Benchmark scores section (optional)
    6. Pricing & availability section (optional)
    7. Community stats (downloads, likes, trending rank)
    8. Comparison table against reference models
    9. Dashboard link button (optional)
    10. Footer with timestamp

    Mandatory fields that are unavailable are rendered as "N/A" rather
    than omitted, ensuring the PM always sees the full schema.  Optional
    sections are entirely omitted when no data exists, keeping the
    message compact.

    Args:
        model_metadata: Collected metadata dict with sections:
            basic, performance, practical, deployment, community, cost,
            provider (optional), plus top-level category/tags.
        reference_data: Dict of reference models for comparison.
        dashboard_url: Full URL to the model's dashboard page (optional).

    Returns:
        List of Block Kit block dicts ready for Slack's ``blocks`` payload.
    """
    blocks: list[dict] = []
    basic = model_metadata.get("basic", {})
    community = model_metadata.get("community", {})

    model_name = _get_field(model_metadata, "basic", "name", "Unknown")
    author = _get_field(model_metadata, "basic", "org", "N/A")
    hf_url = _build_hf_url(model_metadata)

    # ── Header ────────────────────────────────────────────────────────
    blocks.append({
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": f":sparkles: New Model: {model_name}",
            "emoji": True,
        },
    })

    # ── Author context line ───────────────────────────────────────────
    context_elements = [
        {
            "type": "mrkdwn",
            "text": f":building_construction: *{author}*",
        },
    ]
    release_date = _get_field(model_metadata, "basic", "release_date", "N/A")
    context_elements.append({
        "type": "mrkdwn",
        "text": f":calendar: {release_date}",
    })
    blocks.append({"type": "context", "elements": context_elements})

    # ── Divider ───────────────────────────────────────────────────────
    blocks.append({"type": "divider"})

    # ── Mandatory fields (two-column layout) ──────────────────────────
    fields = []
    for label, section, key, fallback in _MANDATORY_FIELDS:
        value = _get_field(model_metadata, section, key, fallback)
        fields.append({
            "type": "mrkdwn",
            "text": f"*{label}:*\n{value}",
        })

    # HF URL as a mandatory field (always present)
    if hf_url:
        fields.append({
            "type": "mrkdwn",
            "text": f"*HF URL:*\n<{hf_url}|{hf_url}>",
        })
    else:
        fields.append({
            "type": "mrkdwn",
            "text": "*HF URL:*\nN/A",
        })

    # Slack limits sections to 10 fields; split if needed
    for i in range(0, len(fields), 10):
        blocks.append({
            "type": "section",
            "fields": fields[i : i + 10],
        })

    # ── Tags (optional) ──────────────────────────────────────────────
    blocks.extend(_build_tags_block(model_metadata))

    # ── Benchmark scores (optional) ──────────────────────────────────
    blocks.extend(_build_benchmark_blocks(model_metadata))

    # ── Pricing & availability (optional) ────────────────────────────
    blocks.extend(_build_pricing_blocks(model_metadata))

    # ── Divider ───────────────────────────────────────────────────────
    blocks.append({"type": "divider"})

    # ── Community stats ───────────────────────────────────────────────
    downloads = community.get("downloads", 0)
    likes = community.get("likes", 0)
    trending_rank = community.get("trending_rank", "N/A")

    stats_text = (
        f":chart_with_upwards_trend: *Downloads:* {downloads:,}  |  "
        f":heart: *Likes:* {likes:,}"
    )
    if trending_rank != "N/A":
        stats_text += f"  |  :fire: *Trending Rank:* #{trending_rank}"

    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": stats_text},
    })

    # ── Comparison table ──────────────────────────────────────────────
    if reference_data:
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*:bar_chart: Comparison with Reference Models*",
            },
        })
        table_text = _build_comparison_table(model_metadata, reference_data)
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": table_text},
        })

    # ── Dashboard link button (optional) ──────────────────────────────
    if dashboard_url:
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": ":link: View on Dashboard",
                        "emoji": True,
                    },
                    "url": dashboard_url,
                    "action_id": "view_dashboard",
                },
            ],
        })

    # ── Footer context ────────────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    blocks.append({
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": f"HF Model Monitor | {ts}",
            },
        ],
    })

    return blocks


# ---------------------------------------------------------------------------
# Plain-text report formatting (backward-compatible)
# ---------------------------------------------------------------------------
def format_report(model_metadata: dict, reference_data: dict) -> str:
    """Format a model comparison report as a Slack-friendly text block.

    Args:
        model_metadata: Collected metadata dict with sections:
            basic, performance, practical, deployment, community, cost.
        reference_data: Dict of reference models for comparison.

    Returns:
        Formatted report string with header, stats, and comparison table.
    """
    basic = model_metadata.get("basic", {})
    community = model_metadata.get("community", {})

    header = (
        f":bar_chart: *New Trending: {basic.get('name', 'Unknown')}* "
        f"({basic.get('org', 'N/A')})"
    )
    info = (
        f":mag: License: {basic.get('license', 'N/A')} | "
        f"Downloads: {community.get('downloads', 0):,} | "
        f"Likes: {community.get('likes', 0):,}"
    )

    table = _build_comparison_table(model_metadata, reference_data)

    parts = [header, "", info, "", table]
    return "\n".join(parts)


def _build_comparison_table(
    model_metadata: dict, reference_data: dict
) -> str:
    """Build a fixed-width comparison table wrapped in Slack code block."""
    basic = model_metadata.get("basic", {})
    perf = model_metadata.get("performance", {})
    deploy = model_metadata.get("deployment", {})
    cost = model_metadata.get("cost", {})
    practical = model_metadata.get("practical", {})

    col_names = ["", basic.get("name", "New")] + list(reference_data.keys())

    metrics = [
        ("Params", basic.get("params", "N/A"), "params"),
        ("MMLU", perf.get("mmlu", "N/A"), "mmlu"),
        ("HumanEval", perf.get("humaneval", "N/A"), "humaneval"),
        ("License", basic.get("license", "N/A"), "license"),
        ("API$/1M", cost.get("api_price_per_million_tokens", "N/A"), "api_price"),
        ("Context", practical.get("context_window", "N/A"), "context_window"),
        ("VRAM", deploy.get("vram_estimate", "N/A"), "vram"),
    ]

    rows = []
    for label, new_val, ref_key in metrics:
        ref_vals = [
            str(ref.get(ref_key, "N/A")) for ref in reference_data.values()
        ]
        rows.append([label, str(new_val)] + ref_vals)

    all_rows = [col_names] + rows
    col_widths = [
        max(len(row[i]) for row in all_rows) for i in range(len(col_names))
    ]

    def fmt(row: list[str]) -> str:
        return "  ".join(
            cell.ljust(col_widths[i]) for i, cell in enumerate(row)
        )

    sep = "  ".join("-" * w for w in col_widths)
    lines = [fmt(col_names), sep] + [fmt(r) for r in rows]
    return "```\n" + "\n".join(lines) + "\n```"


# ---------------------------------------------------------------------------
# Backward-compatible module-level function
# ---------------------------------------------------------------------------
def send_to_slack(message: str, webhook_url: str | None = None) -> bool:
    """Send a message to Slack (legacy function interface).

    Delegates to ``SlackNotifier`` for retry logic. Kept for backward
    compatibility with the scheduler and main pipeline.

    Args:
        message: Text to send.
        webhook_url: Webhook URL override; falls back to env var / config.

    Returns:
        True on success, False on failure.
    """
    url = webhook_url or SLACK_WEBHOOK_URL
    if not url:
        logger.warning("No SLACK_WEBHOOK_URL configured, skipping send")
        return False
    if not message:
        return False

    notifier = SlackNotifier(webhook_url=url)
    return notifier.send(message)


# ---------------------------------------------------------------------------
# Error classification — delegates to errors.py
# ---------------------------------------------------------------------------

from .errors import (
    CrawlerErrorCategory,
    classify_error_message as _classify_msg,
    category_to_label as _cat_to_label,
)

_ERROR_EMOJIS: dict[str, str] = {
    "Timeout": ":hourglass:",
    "Rate Limited": ":snail:",
    "Authentication": ":lock:",
    "Network": ":electric_plug:",
    "Server Error": ":fire:",
    "Not Found": ":mag:",
    "Structure Change": ":warning:",
    "Configuration": ":gear:",
    "Unknown": ":x:",
}


def classify_crawler_error(error_msg: str) -> str:
    """Classify a crawler error message into a human-readable error type.

    Delegates to :func:`~hf_model_monitor.errors.classify_error_message`
    and converts the category enum to the legacy label string.

    Args:
        error_msg: The raw error message string.

    Returns:
        A short label like ``"Network"``, ``"Timeout"``, ``"Rate Limited"``, etc.
    """
    classified = _classify_msg(error_msg)
    return _cat_to_label(classified.category)


def _error_type_emoji(error_type: str) -> str:
    """Return an appropriate Slack emoji for the given error type."""
    return _ERROR_EMOJIS.get(error_type, ":x:")


# ---------------------------------------------------------------------------
# Error alert formatting
# ---------------------------------------------------------------------------

def format_error_alert(
    source: str,
    error_msg: str,
    *,
    error_type: str | None = None,
    consecutive_failures: int = 0,
    next_retry_hours: int = 12,
) -> str:
    """Format a Slack error alert for a single crawler failure.

    Args:
        source: Name of the failing source (e.g. org name, "HF API").
        error_msg: The raw error message.
        error_type: Override for auto-classified error type.
        consecutive_failures: How many consecutive runs have failed.
        next_retry_hours: Hours until the next scheduled retry.

    Returns:
        Formatted Slack message string.
    """
    if error_type is None:
        error_type = classify_crawler_error(error_msg)

    emoji = _error_type_emoji(error_type)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return (
        f"{emoji} *HF Model Monitor — Crawler Error*\n"
        f"*Source:* {source}\n"
        f"*Type:* {error_type}\n"
        f"*Time:* {ts}\n"
        f"```{error_msg}```\n"
        f"Consecutive failures: {consecutive_failures}\n"
        f"Next retry in {next_retry_hours}h"
    )


def format_crawler_errors_summary(
    org_errors: list[dict],
    total_orgs: int,
    next_retry_hours: int = 12,
) -> str:
    """Format a single Slack alert summarizing multiple crawler failures.

    Sent when a detection run completes but some orgs encountered errors.
    Aggregates all failures into one message for readability.

    Args:
        org_errors: List of dicts with ``"source"`` and ``"error"`` keys.
        total_orgs: Total number of organizations polled in the run.
        next_retry_hours: Hours until the next scheduled retry.

    Returns:
        Formatted Slack message string.
    """
    failed = len(org_errors)
    succeeded = total_orgs - failed
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        ":rotating_light: *HF Model Monitor — Crawler Errors Detected*",
        f"*{failed}/{total_orgs}* org(s) failed | {succeeded} succeeded",
        f"*Time:* {ts}",
        "",
    ]

    for entry in org_errors:
        source = entry.get("source", "unknown")
        error = entry.get("error", "unknown error")
        error_type = classify_crawler_error(error)
        emoji = _error_type_emoji(error_type)
        lines.append(f"{emoji} *{source}* — {error_type}")
        lines.append(f"    `{error}`")

    lines.append("")
    lines.append(f"Next retry in {next_retry_hours}h")

    return "\n".join(lines)


def format_pipeline_error_alert(
    error_msg: str,
    consecutive_failures: int = 0,
    next_retry_hours: int = 12,
) -> str:
    """Format a Slack alert for a catastrophic pipeline failure.

    This replaces the inline alert string previously built in the scheduler,
    providing a consistent format with error classification.

    Args:
        error_msg: The exception message from the pipeline crash.
        consecutive_failures: How many consecutive pipeline runs have failed.
        next_retry_hours: Hours until the next scheduled retry.

    Returns:
        Formatted Slack message string.
    """
    error_type = classify_crawler_error(error_msg)
    emoji = _error_type_emoji(error_type)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    severity = (
        ":red_circle:"
        if consecutive_failures >= 3
        else ":large_orange_circle:"
    )

    return (
        f":rotating_light: *HF Model Monitor — Pipeline Error*\n"
        f"{severity} *Severity:* "
        f"{'HIGH — multiple consecutive failures' if consecutive_failures >= 3 else 'Normal'}\n"
        f"{emoji} *Error type:* {error_type}\n"
        f"*Time:* {ts}\n"
        f"```{error_msg}```\n"
        f"Consecutive failures: {consecutive_failures}\n"
        f"Next retry in {next_retry_hours}h"
    )
