"""Trending and download-surge detection engine for the HF Model Monitor.

Detects models experiencing unusual popularity surges, independent of the
core org-based new-model detection pipeline.  Works by:

1. Querying the HF trending API for currently hot models
2. Fetching high-download models and comparing against stored snapshots
3. Computing surge metrics (download delta, trending score)
4. Filtering out models from already-watched core organizations
5. Returning candidates that exceed configurable thresholds

This module does NOT use any AI/LLM API calls — all metrics are computed
from raw download counts and HF's built-in trending scores.

Usage:
    from hf_model_monitor.trending_detector import TrendingDetector

    detector = TrendingDetector.from_config(config)
    result = detector.run()
    # result.candidates → list of TrendingCandidate
    # result.summary()  → human-readable summary string
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .hf_client import HFApiClient, HFApiError
from .model_store import ModelStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TrendingCandidate:
    """A model flagged by the trending/surge detection engine.

    Attributes:
        model_id: Full HuggingFace model identifier (e.g. "org/model-name").
        author: Organization or user who published the model.
        downloads: Current total download count.
        likes: Current like count.
        trending_score: HF trending score (0 if unavailable).
        download_delta: Increase in downloads since last snapshot (0 on first
            observation — a surge is only flagged after two data points).
        surge_triggered: True if download_delta exceeds the threshold.
        trending_triggered: True if trending_score exceeds the threshold.
        pipeline_tag: Model pipeline type (e.g. "text-generation").
        created_at: ISO-8601 creation timestamp or "N/A".
        last_modified: ISO-8601 last-modified timestamp or "N/A".
        tags: List of HF tags.
        library_name: Primary ML library (e.g. "transformers").
    """

    model_id: str
    author: str = ""
    downloads: int = 0
    likes: int = 0
    trending_score: int = 0
    download_delta: int = 0
    surge_triggered: bool = False
    trending_triggered: bool = False
    pipeline_tag: str = "N/A"
    created_at: str = "N/A"
    last_modified: str = "N/A"
    tags: list[str] = field(default_factory=list)
    library_name: str = "N/A"

    @property
    def reason(self) -> str:
        """Human-readable explanation of why this candidate was flagged."""
        reasons = []
        if self.surge_triggered:
            reasons.append(f"download surge (+{self.download_delta:,})")
        if self.trending_triggered:
            reasons.append(f"trending score ({self.trending_score})")
        return "; ".join(reasons) if reasons else "unknown"

    def to_dict(self) -> dict:
        """Convert to a plain dict for serialization."""
        return {
            "model_id": self.model_id,
            "author": self.author,
            "downloads": self.downloads,
            "likes": self.likes,
            "trending_score": self.trending_score,
            "download_delta": self.download_delta,
            "surge_triggered": self.surge_triggered,
            "trending_triggered": self.trending_triggered,
            "pipeline_tag": self.pipeline_tag,
            "created_at": self.created_at,
            "last_modified": self.last_modified,
            "tags": list(self.tags),
            "library_name": self.library_name,
            "reason": self.reason,
        }


@dataclass
class TrendingDetectionResult:
    """Aggregated result from a trending/surge detection run."""

    candidates: list[TrendingCandidate] = field(default_factory=list)
    trending_fetched: int = 0
    downloaded_fetched: int = 0
    filtered_core_org: int = 0
    below_threshold: int = 0
    poll_timestamp: str = ""
    errors: list[str] = field(default_factory=list)
    enabled: bool = True

    @property
    def total_candidates(self) -> int:
        return len(self.candidates)

    @property
    def surge_candidates(self) -> list[TrendingCandidate]:
        return [c for c in self.candidates if c.surge_triggered]

    @property
    def trending_candidates(self) -> list[TrendingCandidate]:
        return [c for c in self.candidates if c.trending_triggered]

    def summary(self) -> str:
        """Return a human-readable summary of the trending detection run."""
        if not self.enabled:
            return "Trending/surge detection is disabled"

        lines = [
            f"Trending detection run at {self.poll_timestamp}",
            f"  Trending models fetched: {self.trending_fetched}",
            f"  High-download models fetched: {self.downloaded_fetched}",
            f"  Filtered (core org): {self.filtered_core_org}",
            f"  Below threshold: {self.below_threshold}",
            f"  Candidates flagged: {self.total_candidates}",
        ]
        if self.candidates:
            lines.append("  Candidates:")
            for c in self.candidates:
                lines.append(f"    - {c.model_id} ({c.reason})")
        if self.errors:
            lines.append("  Errors:")
            for err in self.errors:
                lines.append(f"    - {err}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Surge metric computation
# ---------------------------------------------------------------------------
def compute_download_delta(
    model_id: str,
    current_downloads: int,
    store: ModelStore,
) -> int:
    """Compute the download increase since the model was last observed.

    Returns 0 if the model has never been observed before (we need two
    data points to measure a delta).  Also returns 0 if current downloads
    are somehow lower than the stored value (data inconsistency).

    Args:
        model_id: HF model identifier.
        current_downloads: Current download count from the API.
        store: ModelStore to look up previous download snapshot.

    Returns:
        Non-negative integer representing the download increase.
    """
    stored = store.get_model(model_id)
    if stored is None:
        # First observation — no delta to compute
        return 0

    previous_downloads = stored.get("downloads", 0)
    delta = current_downloads - previous_downloads
    return max(delta, 0)


# ---------------------------------------------------------------------------
# Core org filtering
# ---------------------------------------------------------------------------
def is_core_org_model(model_id: str, watched_orgs: list[str]) -> bool:
    """Check if a model belongs to one of the core watched organizations.

    Core org models are already handled by the primary detection pipeline,
    so we filter them from trending/surge results to avoid duplicates.

    Args:
        model_id: Full model ID like "meta-llama/Llama-4-Scout".
        watched_orgs: List of org/user IDs to exclude.

    Returns:
        True if the model's author matches a watched org (case-insensitive).
    """
    if not model_id or not watched_orgs:
        return False

    # Extract author from model_id (e.g. "meta-llama" from "meta-llama/Llama-4")
    if "/" in model_id:
        author = model_id.split("/")[0]
    else:
        return False  # No org prefix — can't determine

    watched_lower = {org.lower() for org in watched_orgs}
    return author.lower() in watched_lower


# ---------------------------------------------------------------------------
# TrendingDetector
# ---------------------------------------------------------------------------
class TrendingDetector:
    """Detects trending models and download surges from HuggingFace.

    Coordinates the HFApiClient (fetch trending/downloads) and ModelStore
    (historical download snapshots) to identify models experiencing unusual
    popularity that aren't already covered by core org monitoring.

    Args:
        watched_orgs: Core org list — models from these orgs are filtered out.
        download_surge_threshold: Minimum download increase to flag a surge.
        trending_score_threshold: Minimum HF trending score to flag.
        time_window_hours: Rolling window for evaluating surges.
        enabled: Whether trending detection is active.
        store: ModelStore instance for download snapshots.
        client: HFApiClient instance for API calls.
        trending_fetch_limit: Max models to fetch from trending endpoint.
        downloaded_fetch_limit: Max models to fetch from downloads endpoint.
    """

    def __init__(
        self,
        watched_orgs: list[str],
        *,
        download_surge_threshold: int = 10000,
        trending_score_threshold: int = 50,
        time_window_hours: int = 24,
        enabled: bool = False,
        store: ModelStore | None = None,
        client: HFApiClient | None = None,
        trending_fetch_limit: int = 50,
        downloaded_fetch_limit: int = 50,
    ):
        self.watched_orgs = watched_orgs
        self.download_surge_threshold = download_surge_threshold
        self.trending_score_threshold = trending_score_threshold
        self.time_window_hours = time_window_hours
        self.enabled = enabled
        self._store = store or ModelStore()
        self._client = client or HFApiClient()
        self.trending_fetch_limit = trending_fetch_limit
        self.downloaded_fetch_limit = downloaded_fetch_limit

    @classmethod
    def from_config(
        cls,
        config: dict,
        *,
        store: ModelStore | None = None,
        client: HFApiClient | None = None,
    ) -> "TrendingDetector":
        """Create a TrendingDetector from a config dict.

        Reads the 'trending_thresholds' section and 'watched_organizations'
        from the config (as returned by load_config()).

        Args:
            config: Full config dict.
            store: Optional ModelStore override.
            client: Optional HFApiClient override.
        """
        thresholds = config.get("trending_thresholds", {})
        return cls(
            watched_orgs=config.get("watched_organizations", []),
            download_surge_threshold=thresholds.get("download_surge_count", 10000),
            trending_score_threshold=thresholds.get("trending_score", 50),
            time_window_hours=thresholds.get("time_window_hours", 24),
            enabled=thresholds.get("enabled", False),
            store=store,
            client=client,
        )

    # ------------------------------------------------------------------
    # Main detection entry point
    # ------------------------------------------------------------------

    def run(self) -> TrendingDetectionResult:
        """Execute a full trending/surge detection run.

        Steps:
        1. Check if trending detection is enabled
        2. Fetch trending models from HF API
        3. Fetch most-downloaded models from HF API
        4. Merge and deduplicate the two sources
        5. Filter out core-org models
        6. Compute surge metrics (download delta)
        7. Apply thresholds to find candidates
        8. Snapshot download counts for future delta computation
        9. Record poll history
        10. Return aggregated results

        Returns:
            TrendingDetectionResult with candidates and stats.
        """
        now = datetime.now(timezone.utc)
        result = TrendingDetectionResult(
            poll_timestamp=now.isoformat(),
            enabled=self.enabled,
        )

        if not self.enabled:
            logger.info("Trending/surge detection is disabled, skipping")
            return result

        logger.info(
            "Starting trending detection: surge_threshold=%d, "
            "trending_threshold=%d, window=%dh",
            self.download_surge_threshold,
            self.trending_score_threshold,
            self.time_window_hours,
        )

        # Step 1: Fetch trending models
        trending_models = self._fetch_trending(result)

        # Step 2: Fetch most-downloaded models
        downloaded_models = self._fetch_most_downloaded(result)

        # Step 3: Merge and deduplicate
        merged = self._merge_sources(trending_models, downloaded_models)

        # Step 4: Filter out core-org models
        filtered = self._filter_core_orgs(merged, result)

        # Step 5: Compute metrics and apply thresholds
        candidates = self._evaluate_candidates(filtered, result)
        result.candidates = candidates

        # Step 6: Snapshot download counts for future runs
        self._snapshot_downloads(merged)

        # Step 7: Record in poll history
        error_summary = "; ".join(result.errors) if result.errors else ""
        self._store.record_poll(
            source="trending_detection",
            models_found=len(merged),
            new_models=len(candidates),
            errors=error_summary,
        )

        logger.info(
            "Trending detection complete: %d fetched, %d after filter, "
            "%d candidates flagged",
            len(merged),
            len(filtered),
            len(candidates),
        )

        return result

    # ------------------------------------------------------------------
    # Fetch steps
    # ------------------------------------------------------------------

    def _fetch_trending(
        self, result: TrendingDetectionResult
    ) -> list[dict]:
        """Fetch trending models from the HF API.

        Updates result.trending_fetched and appends errors on failure.
        """
        try:
            models = self._client.fetch_trending_models(
                limit=self.trending_fetch_limit
            )
            result.trending_fetched = len(models)
            return models
        except Exception as exc:
            error_msg = f"Failed to fetch trending models: {exc}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            return []

    def _fetch_most_downloaded(
        self, result: TrendingDetectionResult
    ) -> list[dict]:
        """Fetch most-downloaded models from the HF API.

        Converts ModelRecord instances to dicts for uniform processing.
        Updates result.downloaded_fetched and appends errors on failure.
        """
        try:
            records = self._client.fetch_most_downloaded_models(
                limit=self.downloaded_fetch_limit
            )
            models = []
            for r in records:
                d = r.to_dict()
                d["trending_score"] = 0  # Not available from this endpoint
                models.append(d)
            result.downloaded_fetched = len(models)
            return models
        except Exception as exc:
            error_msg = f"Failed to fetch most-downloaded models: {exc}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            return []

    # ------------------------------------------------------------------
    # Merge and filter
    # ------------------------------------------------------------------

    def _merge_sources(
        self,
        trending: list[dict],
        downloaded: list[dict],
    ) -> list[dict]:
        """Merge trending and downloaded model lists, deduplicating by model_id.

        When a model appears in both sources, the trending version takes
        precedence (it has the trending_score populated).

        Returns:
            Deduplicated list of model dicts.
        """
        seen: dict[str, dict] = {}

        # Trending models first (they have trending_score)
        for model in trending:
            mid = model.get("model_id", "")
            if mid:
                seen[mid] = model

        # Downloaded models fill in gaps
        for model in downloaded:
            mid = model.get("model_id", "")
            if mid and mid not in seen:
                seen[mid] = model

        return list(seen.values())

    def _filter_core_orgs(
        self,
        models: list[dict],
        result: TrendingDetectionResult,
    ) -> list[dict]:
        """Remove models from core watched organizations.

        These are already monitored by the primary ModelDetector,
        so including them would create duplicate notifications.
        """
        filtered = []
        for model in models:
            mid = model.get("model_id", "")
            if is_core_org_model(mid, self.watched_orgs):
                result.filtered_core_org += 1
                logger.debug("Filtered core-org model: %s", mid)
            else:
                filtered.append(model)
        return filtered

    # ------------------------------------------------------------------
    # Metric evaluation
    # ------------------------------------------------------------------

    def _evaluate_candidates(
        self,
        models: list[dict],
        result: TrendingDetectionResult,
    ) -> list[TrendingCandidate]:
        """Evaluate each model against surge and trending thresholds.

        A model becomes a candidate if EITHER:
        - Its download delta exceeds download_surge_threshold, OR
        - Its trending score exceeds trending_score_threshold

        Models that meet neither threshold are counted in
        result.below_threshold.
        """
        candidates = []

        for model in models:
            mid = model.get("model_id", "")
            if not mid:
                continue

            current_downloads = model.get("downloads", 0)
            trending_score = model.get("trending_score", 0)

            # Compute download surge delta
            download_delta = compute_download_delta(
                mid, current_downloads, self._store
            )

            # Check thresholds
            surge_triggered = download_delta >= self.download_surge_threshold
            trending_triggered = trending_score >= self.trending_score_threshold

            if surge_triggered or trending_triggered:
                candidate = TrendingCandidate(
                    model_id=mid,
                    author=model.get("author", ""),
                    downloads=current_downloads,
                    likes=model.get("likes", 0),
                    trending_score=trending_score,
                    download_delta=download_delta,
                    surge_triggered=surge_triggered,
                    trending_triggered=trending_triggered,
                    pipeline_tag=model.get("pipeline_tag", "N/A"),
                    created_at=model.get("created_at", "N/A"),
                    last_modified=model.get("last_modified", "N/A"),
                    tags=model.get("tags", []),
                    library_name=model.get("library_name", "N/A"),
                )
                candidates.append(candidate)
                logger.info(
                    "Trending candidate: %s — %s",
                    mid,
                    candidate.reason,
                )
            else:
                result.below_threshold += 1

        return candidates

    # ------------------------------------------------------------------
    # Download snapshot
    # ------------------------------------------------------------------

    def _snapshot_downloads(self, models: list[dict]) -> None:
        """Save current download counts into the ModelStore.

        This creates the baseline for future download-delta computations.
        Uses ModelStore.upsert_models which will insert new entries or
        update the downloads field for existing ones.
        """
        if not models:
            return

        # Ensure each model dict has the minimum required fields
        snapshot_records = []
        for model in models:
            mid = model.get("model_id", "")
            if not mid:
                continue
            snapshot_records.append({
                "model_id": mid,
                "author": model.get("author", ""),
                "created_at": model.get("created_at", "N/A"),
                "last_modified": model.get("last_modified", "N/A"),
                "pipeline_tag": model.get("pipeline_tag", "N/A"),
                "library_name": model.get("library_name", "N/A"),
                "downloads": model.get("downloads", 0),
                "likes": model.get("likes", 0),
                "tags": model.get("tags", []),
            })

        self._store.upsert_models(snapshot_records)
        logger.debug("Snapshotted downloads for %d models", len(snapshot_records))

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close underlying resources."""
        self._client.close()
        self._store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
