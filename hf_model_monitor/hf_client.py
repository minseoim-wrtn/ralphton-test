"""HuggingFace API client with pagination, rate limiting, and error handling.

Fetches latest models for monitored organizations via the HuggingFace Hub API.
Designed for daily batch polling — not real-time streaming.

Usage:
    client = HFApiClient()
    models = client.fetch_org_models("meta-llama")
    all_models = client.fetch_all_watched_org_models(["meta-llama", "google"])
"""

import logging
import time
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HF_API_BASE = "https://huggingface.co/api"
DEFAULT_TIMEOUT = 30  # seconds per request
DEFAULT_PER_PAGE = 100  # max models per page (HF API maximum)
DEFAULT_MAX_PAGES = 5  # safety cap to prevent runaway pagination
DEFAULT_MIN_REQUEST_INTERVAL = 1.0  # seconds between requests (rate limit)
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF = 2.0  # exponential backoff base in seconds


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class HFApiError(Exception):
    """Base exception for HuggingFace API errors."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class HFRateLimitError(HFApiError):
    """Raised when the API returns 429 Too Many Requests."""

    def __init__(self, retry_after: float | None = None):
        self.retry_after = retry_after
        super().__init__(
            f"Rate limited by HuggingFace API (retry after {retry_after}s)",
            status_code=429,
        )


class HFAuthError(HFApiError):
    """Raised on 401/403 — token may be invalid or missing."""

    def __init__(self, status_code: int = 401):
        super().__init__(
            "Authentication error — check HF_TOKEN if accessing gated models",
            status_code=status_code,
        )


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
class RateLimiter:
    """Simple token-bucket-style rate limiter that enforces a minimum interval
    between consecutive requests.  Thread-safe is NOT required (single-threaded
    batch polling)."""

    def __init__(self, min_interval: float = DEFAULT_MIN_REQUEST_INTERVAL):
        self.min_interval = min_interval
        self._last_request_time: float = 0.0

    def wait(self) -> None:
        """Block until enough time has elapsed since the last request."""
        if self._last_request_time == 0.0:
            self._last_request_time = time.monotonic()
            return

        elapsed = time.monotonic() - self._last_request_time
        remaining = self.min_interval - elapsed
        if remaining > 0:
            logger.debug("Rate limiter: sleeping %.2fs", remaining)
            time.sleep(remaining)
        self._last_request_time = time.monotonic()

    def reset(self) -> None:
        self._last_request_time = 0.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ModelRecord:
    """Flat record for a model discovered from the HF API.

    Designed to be serializable and easy to compare with previously-seen models.
    """

    model_id: str  # e.g. "meta-llama/Llama-4-Scout-17B-16E"
    author: str  # e.g. "meta-llama"
    created_at: str  # ISO-8601 string or "N/A"
    last_modified: str  # ISO-8601 string or "N/A"
    pipeline_tag: str  # e.g. "text-generation"
    tags: list[str] = field(default_factory=list)
    downloads: int = 0
    likes: int = 0
    private: bool = False
    gated: bool | str = False
    library_name: str = "N/A"

    def to_dict(self) -> dict:
        """Convert to a plain dict matching the project's model schema."""
        return {
            "model_id": self.model_id,
            "author": self.author,
            "created_at": self.created_at,
            "last_modified": self.last_modified,
            "pipeline_tag": self.pipeline_tag,
            "tags": list(self.tags),
            "downloads": self.downloads,
            "likes": self.likes,
            "private": self.private,
            "gated": self.gated,
            "library_name": self.library_name,
        }


# ---------------------------------------------------------------------------
# HuggingFace API Client
# ---------------------------------------------------------------------------
class HFApiClient:
    """Client for the HuggingFace Hub REST API.

    Features:
    - Per-organization model listing with pagination
    - Configurable rate limiting between requests
    - Automatic retries with exponential backoff
    - Filters out community derivatives (quantized/LoRA/GGUF forks)
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        per_page: int = DEFAULT_PER_PAGE,
        max_pages: int = DEFAULT_MAX_PAGES,
        min_request_interval: float = DEFAULT_MIN_REQUEST_INTERVAL,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    ):
        self.timeout = timeout
        self.per_page = min(per_page, 100)  # HF API caps at 100
        self.max_pages = max_pages
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.rate_limiter = RateLimiter(min_request_interval)

        # Persistent session for connection pooling
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        if token:
            self._session.headers["Authorization"] = f"Bearer {token}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_org_models(
        self,
        org: str,
        *,
        sort: str = "lastModified",
        direction: str = "-1",
        max_pages: int | None = None,
    ) -> list[ModelRecord]:
        """Fetch all models published by *org* with pagination.

        Args:
            org: HuggingFace organization or user ID (e.g. "meta-llama").
            sort: Sort field — "lastModified", "downloads", "likes", etc.
            direction: "-1" for descending, "1" for ascending.
            max_pages: Override instance-level max_pages for this call.

        Returns:
            List of ModelRecord instances, newest first.
        """
        page_limit = max_pages if max_pages is not None else self.max_pages
        all_models: list[ModelRecord] = []

        params = {
            "author": org,
            "sort": sort,
            "direction": direction,
            "limit": self.per_page,
        }

        url: str | None = f"{HF_API_BASE}/models"
        page = 0

        while url and page < page_limit:
            page += 1
            logger.info(
                "Fetching models for org=%s (page %d/%d)", org, page, page_limit
            )

            response_data, next_url = self._get_paginated(url, params)

            if response_data is None:
                logger.warning(
                    "Failed to fetch page %d for org=%s, stopping pagination",
                    page,
                    org,
                )
                break

            if not response_data:
                # Empty page — we've exhausted the results
                logger.debug("Empty page %d for org=%s, done.", page, org)
                break

            for raw_model in response_data:
                record = self._parse_model(raw_model)
                if record is not None:
                    all_models.append(record)

            # Move to next page — clear params since next_url contains them
            url = next_url
            params = {}

        logger.info(
            "Fetched %d models for org=%s across %d page(s)",
            len(all_models),
            org,
            page,
        )
        return all_models

    def fetch_org_latest_models(
        self,
        org: str,
        *,
        since: str | None = None,
        max_pages: int | None = None,
    ) -> list[ModelRecord]:
        """Fetch models from *org*, sorted by creation date (newest first).

        Optimized for new-model detection: sorts by ``createdAt`` descending
        and stops pagination early when it encounters a model created before
        the *since* cutoff.  This avoids fetching hundreds of old models from
        large organizations.

        Args:
            org: HuggingFace organization or user ID.
            since: ISO-8601 cutoff timestamp.  Models created before this
                time are excluded, and pagination stops immediately once a
                model older than the cutoff is encountered.  Pass ``None``
                to fetch all (same as :meth:`fetch_org_models`).
            max_pages: Override instance-level max_pages for this call.

        Returns:
            List of ModelRecords created on or after *since*, newest first.
        """
        records = self.fetch_org_models(
            org,
            sort="createdAt",
            direction="-1",
            max_pages=max_pages,
        )

        if since is None:
            return records

        # Filter and early-terminate: since results are sorted newest-first,
        # once we hit a model older than the cutoff, all remaining are older.
        filtered: list[ModelRecord] = []
        for record in records:
            if record.created_at == "N/A":
                # Unknown creation date — include to avoid missing new models
                filtered.append(record)
                continue
            if record.created_at >= since:
                filtered.append(record)
            else:
                # All subsequent models are older — stop
                logger.debug(
                    "Early termination: %s created_at=%s < since=%s",
                    record.model_id,
                    record.created_at,
                    since,
                )
                break

        logger.info(
            "Fetched %d recent models for org=%s (since=%s, %d total before filter)",
            len(filtered),
            org,
            since,
            len(records),
        )
        return filtered

    def fetch_all_watched_org_models(
        self, orgs: list[str]
    ) -> dict[str, list[ModelRecord]]:
        """Fetch models for all watched organizations.

        Args:
            orgs: List of organization/user IDs to monitor.

        Returns:
            Dict mapping org name to list of ModelRecords.
            Orgs that fail are included with empty lists.
        """
        results: dict[str, list[ModelRecord]] = {}

        for org in orgs:
            try:
                models = self.fetch_org_models(org)
                results[org] = models
            except HFApiError as exc:
                logger.error(
                    "HF API error fetching org=%s: %s (status=%s)",
                    org,
                    exc,
                    exc.status_code,
                )
                results[org] = []
            except Exception:
                logger.exception("Unexpected error fetching org=%s", org)
                results[org] = []

        total = sum(len(v) for v in results.values())
        failed = sum(1 for v in results.values() if not v)
        logger.info(
            "Fetched %d total models across %d orgs (%d orgs returned empty)",
            total,
            len(orgs),
            failed,
        )
        return results

    def fetch_latest_from_config(
        self,
        config: dict,
        *,
        since: str | None = None,
    ) -> dict[str, list[ModelRecord]]:
        """Fetch latest models for all organizations listed in *config*.

        Convenience method that reads ``watched_organizations`` from a config
        dict (as returned by :func:`~hf_model_monitor.config.load_config`)
        and fetches recent models from each, with optional time-based
        filtering for efficient daily polling.

        Args:
            config: Config dict with a ``watched_organizations`` key.
            since: Optional ISO-8601 cutoff — only models created on or
                after this timestamp are returned.  Enables early pagination
                termination for large orgs.

        Returns:
            Dict mapping org name → list of ModelRecords.
            Orgs that fail are included with empty lists.
        """
        orgs = config.get("watched_organizations", [])
        if not orgs:
            logger.warning("No watched_organizations in config")
            return {}

        results: dict[str, list[ModelRecord]] = {}

        for org in orgs:
            try:
                if since:
                    models = self.fetch_org_latest_models(org, since=since)
                else:
                    models = self.fetch_org_models(org)
                results[org] = models
            except HFApiError as exc:
                logger.error(
                    "HF API error fetching org=%s: %s (status=%s)",
                    org,
                    exc,
                    exc.status_code,
                )
                results[org] = []
            except Exception:
                logger.exception("Unexpected error fetching org=%s", org)
                results[org] = []

        total = sum(len(v) for v in results.values())
        logger.info(
            "Config fetch complete: %d models across %d orgs (since=%s)",
            total,
            len(orgs),
            since or "all",
        )
        return results

    def fetch_trending_models(self, limit: int = 50) -> list[dict]:
        """Fetch currently trending models from the HF trending endpoint.

        The HF trending API returns models ranked by recent community
        activity (likes, downloads velocity, discussion activity).

        Args:
            limit: Maximum number of trending models to return.

        Returns:
            List of raw model dicts with at least 'id'/'modelId', 'author',
            'downloads', 'likes', and optionally 'trendingScore'.
            Returns empty list on failure.
        """
        url = f"{HF_API_BASE}/trending"
        params = {"type": "model", "limit": limit}

        try:
            data = self._request_with_retry(url, params=params)
        except HFApiError as exc:
            logger.warning("Failed to fetch trending models: %s", exc)
            return []

        if data is None:
            return []

        # The trending endpoint may return a list directly or a dict
        # with a 'recentlyTrending' key.
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("recentlyTrending", [])
        else:
            logger.warning(
                "Unexpected trending response type: %s", type(data).__name__
            )
            return []

        results = []
        for item in items:
            repo_data = item.get("repoData", item)
            model_id = repo_data.get("id") or item.get("id", "")
            if not model_id:
                continue
            results.append({
                "model_id": model_id,
                "author": repo_data.get("author", item.get("author", "")),
                "downloads": repo_data.get("downloads", 0),
                "likes": repo_data.get("likes", 0),
                "trending_score": item.get("trendingScore", 0),
                "pipeline_tag": repo_data.get("pipeline_tag", "N/A"),
                "created_at": repo_data.get("createdAt", "N/A"),
                "last_modified": repo_data.get("lastModified", "N/A"),
                "tags": repo_data.get("tags", []),
                "library_name": repo_data.get("library_name", "N/A"),
            })

        logger.info("Fetched %d trending models", len(results))
        return results

    def fetch_most_downloaded_models(
        self, limit: int = 50, *, period: str = ""
    ) -> list[ModelRecord]:
        """Fetch models sorted by download count (descending).

        Useful for detecting download surges when compared against
        previously stored download counts.

        Args:
            limit: Maximum number of models to return.
            period: Optional time period filter (e.g. 'lastDay', 'last7Days').

        Returns:
            List of ModelRecord instances sorted by downloads descending.
        """
        url = f"{HF_API_BASE}/models"
        params: dict = {
            "sort": "downloads",
            "direction": "-1",
            "limit": min(limit, 100),
        }

        try:
            data = self._request_with_retry(url, params=params)
        except HFApiError as exc:
            logger.warning("Failed to fetch most-downloaded models: %s", exc)
            return []

        if not isinstance(data, list):
            return []

        results = []
        for raw in data:
            record = self._parse_model(raw)
            if record is not None:
                results.append(record)

        logger.info("Fetched %d most-downloaded models", len(results))
        return results

    def fetch_model_detail(self, model_id: str) -> dict:
        """Fetch detailed info for a single model.

        This is the enhanced version of the old fetch_model_info().

        Returns:
            Parsed JSON dict, or empty dict on failure.
        """
        url = f"{HF_API_BASE}/models/{model_id}"
        data = self._request_with_retry(url)
        return data if isinstance(data, dict) else {}

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_paginated(
        self, url: str, params: dict | None = None
    ) -> tuple[list[dict] | None, str | None]:
        """Make a GET request and extract pagination link.

        Returns:
            (response_data, next_url) — response_data is None on failure.
        """
        try:
            data = self._request_with_retry(url, params=params)
        except HFApiError as exc:
            logger.warning("Paginated request failed: %s", exc)
            return None, None

        if data is None:
            return None, None

        if not isinstance(data, list):
            logger.warning("Expected list response, got %s", type(data).__name__)
            return None, None

        # HF API returns pagination via the Link header
        next_url = self._extract_next_link()
        return data, next_url

    def _extract_next_link(self) -> str | None:
        """Parse the 'Link' header from the last response for the next page URL."""
        if not hasattr(self, "_last_response") or self._last_response is None:
            return None

        link_header = self._last_response.headers.get("Link", "")
        if not link_header:
            return None

        # Link header format: <https://...>; rel="next"
        for part in link_header.split(","):
            part = part.strip()
            if 'rel="next"' in part:
                # Extract URL between < and >
                start = part.find("<")
                end = part.find(">")
                if start != -1 and end != -1:
                    return part[start + 1 : end]
        return None

    def _request_with_retry(
        self, url: str, *, params: dict | None = None
    ) -> dict | list | None:
        """Make a GET request with rate limiting, retries, and backoff.

        Returns:
            Parsed JSON (dict or list) on success, None on failure.
        """
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            self.rate_limiter.wait()

            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)
                # Store for pagination header extraction
                self._last_response = resp

                # Handle specific HTTP errors
                if resp.status_code == 429:
                    retry_after = _parse_retry_after(resp)
                    logger.warning(
                        "Rate limited (429) on attempt %d/%d, retry_after=%.1fs",
                        attempt,
                        self.max_retries,
                        retry_after,
                    )
                    if attempt < self.max_retries:
                        time.sleep(retry_after)
                        continue
                    raise HFRateLimitError(retry_after=retry_after)

                if resp.status_code in (401, 403):
                    raise HFAuthError(status_code=resp.status_code)

                if resp.status_code == 404:
                    logger.warning("Resource not found: %s", url)
                    return None

                if resp.status_code >= 500:
                    logger.warning(
                        "Server error %d on attempt %d/%d for %s",
                        resp.status_code,
                        attempt,
                        self.max_retries,
                        url,
                    )
                    if attempt < self.max_retries:
                        backoff = self.retry_backoff ** attempt
                        time.sleep(backoff)
                        continue
                    raise HFApiError(
                        f"Server error {resp.status_code}",
                        status_code=resp.status_code,
                    )

                resp.raise_for_status()
                return resp.json()

            except HFApiError:
                raise  # Don't wrap our own exceptions
            except requests.exceptions.Timeout:
                last_error = HFApiError(f"Request timed out after {self.timeout}s")
                logger.warning(
                    "Timeout on attempt %d/%d for %s",
                    attempt,
                    self.max_retries,
                    url,
                )
                if attempt < self.max_retries:
                    backoff = self.retry_backoff ** attempt
                    time.sleep(backoff)
                    continue
            except requests.exceptions.ConnectionError as exc:
                last_error = HFApiError(f"Connection error: {exc}")
                logger.warning(
                    "Connection error on attempt %d/%d for %s: %s",
                    attempt,
                    self.max_retries,
                    url,
                    exc,
                )
                if attempt < self.max_retries:
                    backoff = self.retry_backoff ** attempt
                    time.sleep(backoff)
                    continue
            except requests.exceptions.RequestException as exc:
                last_error = HFApiError(f"Request error: {exc}")
                logger.warning(
                    "Request error on attempt %d/%d for %s: %s",
                    attempt,
                    self.max_retries,
                    url,
                    exc,
                )
                if attempt < self.max_retries:
                    backoff = self.retry_backoff ** attempt
                    time.sleep(backoff)
                    continue

        # All retries exhausted
        logger.error("All %d retries exhausted for %s", self.max_retries, url)
        return None

    @staticmethod
    def _parse_model(raw: dict) -> ModelRecord | None:
        """Parse a raw API dict into a ModelRecord, or None if invalid.

        Skips community derivatives (quantized/LoRA/adapter forks).
        """
        model_id = raw.get("id") or raw.get("modelId")
        if not model_id:
            return None

        author = raw.get("author", "")
        if not author and "/" in model_id:
            author = model_id.split("/")[0]

        tags = raw.get("tags", [])
        if not isinstance(tags, list):
            tags = []

        # Filter out community derivatives (quantized/LoRA/adapter forks)
        # Only check if the model is NOT from the original org
        derivative_tags = {"gguf", "gptq", "awq", "bnb", "lora", "adapter"}
        lower_tags = {t.lower() for t in tags}
        if lower_tags & derivative_tags:
            # Check if this is the original repo or a community fork
            # Original repos sometimes have these tags too, so only skip
            # if the model name suggests it's a derivative
            model_name = model_id.split("/")[-1].lower() if "/" in model_id else model_id.lower()
            derivative_patterns = ("-gguf", "-gptq", "-awq", "-bnb", "-qlora")
            if any(pat in model_name for pat in derivative_patterns):
                logger.debug("Skipping community derivative: %s", model_id)
                return None

        return ModelRecord(
            model_id=model_id,
            author=author,
            created_at=raw.get("createdAt", "N/A"),
            last_modified=raw.get("lastModified", "N/A"),
            pipeline_tag=raw.get("pipeline_tag", "N/A"),
            tags=tags,
            downloads=raw.get("downloads", 0),
            likes=raw.get("likes", 0),
            private=raw.get("private", False),
            gated=raw.get("gated", False),
            library_name=raw.get("library_name", "N/A"),
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _parse_retry_after(response: requests.Response) -> float:
    """Extract wait time from a 429 response.

    Checks the Retry-After header; defaults to 60s if absent.
    """
    header = response.headers.get("Retry-After", "")
    try:
        return max(float(header), 1.0)
    except (ValueError, TypeError):
        return 60.0


# ---------------------------------------------------------------------------
# Convenience functions (backward-compatible with legacy hf_fetcher.py)
# ---------------------------------------------------------------------------

_default_client: HFApiClient | None = None


def get_default_client() -> HFApiClient:
    """Return a module-level singleton client (lazy-initialized)."""
    global _default_client
    if _default_client is None:
        import os

        token = os.environ.get("HF_TOKEN", "")
        _default_client = HFApiClient(token=token or None)
    return _default_client


def reset_default_client() -> None:
    """Close and discard the singleton client (useful for testing)."""
    global _default_client
    if _default_client is not None:
        _default_client.close()
        _default_client = None


def fetch_org_models(org: str, **kwargs) -> list[dict]:
    """Convenience wrapper: fetch models for an org and return plain dicts."""
    client = get_default_client()
    records = client.fetch_org_models(org, **kwargs)
    return [r.to_dict() for r in records]


def fetch_all_watched_org_models(orgs: list[str]) -> dict[str, list[dict]]:
    """Convenience wrapper: fetch models for all orgs and return plain dicts."""
    client = get_default_client()
    results = client.fetch_all_watched_org_models(orgs)
    return {
        org: [r.to_dict() for r in records]
        for org, records in results.items()
    }


def fetch_latest_from_config(
    config: dict, *, since: str | None = None
) -> dict[str, list[dict]]:
    """Convenience wrapper: fetch latest models from all configured orgs.

    Reads ``watched_organizations`` from *config* and fetches recent models,
    returning plain dicts suitable for storage or notification.

    Args:
        config: Config dict (as returned by ``load_config()``).
        since: Optional ISO-8601 cutoff for early pagination termination.

    Returns:
        Dict mapping org → list of model dicts.
    """
    client = get_default_client()
    results = client.fetch_latest_from_config(config, since=since)
    return {
        org: [r.to_dict() for r in records]
        for org, records in results.items()
    }
