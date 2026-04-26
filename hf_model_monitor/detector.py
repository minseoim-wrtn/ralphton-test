"""New model detection logic for the HuggingFace Model Monitor.

Compares API results from watched organizations against stored state
(SQLite via ModelStore) and flags models released within the polling window.

This module is the core detection engine:
1. Fetches current models from watched orgs via HFApiClient
2. Compares against previously-seen models in the ModelStore
3. Applies time-based filtering (only flags models within polling window)
4. Records poll history for auditing
5. Returns structured DetectionResult with new models, stats, and errors

First-run bootstrap:
    On the first run (empty store), seed models from ``config/seed_models.yaml``
    are pre-loaded as "known" so they don't trigger false new-model alerts.
    Call ``detector.initialize_store()`` before ``detector.run()`` to ensure
    the store is properly bootstrapped.

Usage:
    from hf_model_monitor.detector import ModelDetector

    detector = ModelDetector(config)
    detector.initialize_store()        # bootstrap seed models on first run
    result = detector.run()
    # result.new_models → list of newly detected model dicts
    # result.summary()  → human-readable summary string
    detector.mark_models_processed(result)  # complete the lifecycle
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .hf_client import HFApiClient, HFApiError
from .model_store import ModelStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class OrgPollResult:
    """Result of polling a single organization."""

    org: str
    models_fetched: int = 0
    new_models: list[dict] = field(default_factory=list)
    error: str = ""

    @property
    def success(self) -> bool:
        return self.error == ""

    @property
    def new_count(self) -> int:
        return len(self.new_models)


@dataclass
class DetectionResult:
    """Aggregated result from a full detection run across all watched orgs."""

    new_models: list[dict] = field(default_factory=list)
    org_results: list[OrgPollResult] = field(default_factory=list)
    poll_timestamp: str = ""
    polling_window_hours: int = 24

    @property
    def total_fetched(self) -> int:
        return sum(r.models_fetched for r in self.org_results)

    @property
    def total_new(self) -> int:
        return len(self.new_models)

    @property
    def orgs_polled(self) -> int:
        return len(self.org_results)

    @property
    def orgs_succeeded(self) -> int:
        return sum(1 for r in self.org_results if r.success)

    @property
    def orgs_failed(self) -> int:
        return sum(1 for r in self.org_results if not r.success)

    @property
    def errors(self) -> list[str]:
        return [r.error for r in self.org_results if r.error]

    def summary(self) -> str:
        """Return a human-readable summary of the detection run."""
        lines = [
            f"Detection run at {self.poll_timestamp}",
            f"  Polling window: {self.polling_window_hours}h",
            f"  Organizations polled: {self.orgs_polled} "
            f"({self.orgs_succeeded} OK, {self.orgs_failed} failed)",
            f"  Total models fetched: {self.total_fetched}",
            f"  New models detected: {self.total_new}",
        ]
        if self.new_models:
            lines.append("  New models:")
            for m in self.new_models:
                lines.append(f"    - {m.get('model_id', 'unknown')}")
        if self.errors:
            lines.append("  Errors:")
            for err in self.errors:
                lines.append(f"    - {err}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------
def _now_utc() -> datetime:
    """Return the current UTC datetime (mockable in tests)."""
    return datetime.now(timezone.utc)


def _parse_iso_datetime(iso_str: str) -> datetime | None:
    """Parse an ISO-8601 datetime string to a timezone-aware datetime.

    Returns None if the string is empty, 'N/A', or unparseable.
    """
    if not iso_str or iso_str == "N/A":
        return None
    try:
        dt = datetime.fromisoformat(iso_str)
        # Ensure timezone-aware (assume UTC if naive)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        logger.debug("Could not parse datetime: %s", iso_str)
        return None


def is_within_polling_window(
    created_at: str,
    polling_window_hours: int,
    *,
    now: datetime | None = None,
) -> bool:
    """Check if a model's created_at timestamp falls within the polling window.

    Args:
        created_at: ISO-8601 timestamp string from the HF API.
        polling_window_hours: Size of the window in hours (e.g. 24 for daily).
        now: Override for current time (for testing).

    Returns:
        True if the model was created within the window, or if created_at
        is unparseable (benefit of the doubt — don't miss new models).
    """
    if now is None:
        now = _now_utc()

    dt = _parse_iso_datetime(created_at)
    if dt is None:
        # Cannot determine creation time — flag it to avoid missing new models
        return True

    cutoff = now - timedelta(hours=polling_window_hours)
    return dt >= cutoff


# ---------------------------------------------------------------------------
# ModelDetector
# ---------------------------------------------------------------------------
class ModelDetector:
    """Detects new models from watched HuggingFace organizations.

    Coordinates the HFApiClient (fetch) and ModelStore (state comparison)
    to identify models that are both:
    1. Not previously seen in the store
    2. Released within the configured polling window

    Args:
        watched_orgs: List of HF organization/user IDs to monitor.
        polling_interval_hours: Polling window size in hours (default 24).
        store: ModelStore instance for state tracking (creates default if None).
        client: HFApiClient instance for API calls (creates default if None).
    """

    def __init__(
        self,
        watched_orgs: list[str],
        polling_interval_hours: int = 24,
        *,
        store: ModelStore | None = None,
        client: HFApiClient | None = None,
    ):
        self.watched_orgs = watched_orgs
        self.polling_interval_hours = polling_interval_hours
        self._store = store or ModelStore()
        self._client = client or HFApiClient()

    @classmethod
    def from_config(
        cls,
        config: dict,
        *,
        store: ModelStore | None = None,
        client: HFApiClient | None = None,
    ) -> "ModelDetector":
        """Create a detector from a config dict (as returned by load_config()).

        Args:
            config: Dict with 'watched_organizations' and 'polling_interval_hours'.
            store: Optional ModelStore override.
            client: Optional HFApiClient override.
        """
        return cls(
            watched_orgs=config.get("watched_organizations", []),
            polling_interval_hours=config.get("polling_interval_hours", 24),
            store=store,
            client=client,
        )

    # ------------------------------------------------------------------
    # Store initialization (first-run bootstrap)
    # ------------------------------------------------------------------

    def is_first_run(self) -> bool:
        """Check whether the store has any known models.

        Returns True if the store is empty (no models have ever been tracked).
        Used to decide whether to bootstrap with seed models.
        """
        return self._store.count_models() == 0

    def initialize_store(
        self,
        *,
        seed_models_path: str | None = None,
        legacy_json_path: str | None = None,
    ) -> dict:
        """Bootstrap the ModelStore on first run to prevent false positives.

        On the very first run, every model fetched from watched orgs would
        appear as "new".  This method pre-populates the store with:
        1. Seed models from ``config/seed_models.yaml`` (curated baseline)
        2. Legacy ``seen_models.json`` (if migrating from JSON-based tracking)

        This is idempotent — calling it on an already-populated store is safe
        (existing models are silently skipped).

        Args:
            seed_models_path: Override path to ``seed_models.yaml``.
            legacy_json_path: Override path to ``seen_models.json``.

        Returns:
            Dict with bootstrap statistics:
            - ``was_first_run``: Whether the store was empty before bootstrap
            - ``seed_models_loaded``: Number of seed models pre-populated
            - ``legacy_models_imported``: Number of models imported from JSON
            - ``total_known_after``: Total known models after bootstrap
        """
        was_first_run = self.is_first_run()
        stats = {
            "was_first_run": was_first_run,
            "seed_models_loaded": 0,
            "legacy_models_imported": 0,
            "total_known_after": 0,
        }

        if not was_first_run:
            stats["total_known_after"] = self._store.count_models()
            logger.debug(
                "Store already has %d models, skipping bootstrap",
                stats["total_known_after"],
            )
            return stats

        logger.info("First run detected — bootstrapping store with seed models")

        # 1. Load seed models from YAML catalog
        seed_count = self._load_seed_models(seed_models_path)
        stats["seed_models_loaded"] = seed_count

        # 2. Import legacy JSON state (migration path)
        legacy_count = self._import_legacy_json(legacy_json_path)
        stats["legacy_models_imported"] = legacy_count

        stats["total_known_after"] = self._store.count_models()

        logger.info(
            "Store bootstrap complete: %d seed + %d legacy = %d total known",
            seed_count,
            legacy_count,
            stats["total_known_after"],
        )

        # Record bootstrap in poll history for audit trail
        self._store.record_poll(
            source="bootstrap",
            models_found=stats["total_known_after"],
            new_models=0,
            errors="",
        )

        return stats

    def _load_seed_models(self, path: str | None = None) -> int:
        """Pre-populate the store with seed models from the YAML catalog.

        Seed models are marked as status='known' and notified=1 so they
        don't trigger new-model alerts or Slack notifications.

        Returns:
            Number of seed models actually inserted (excludes duplicates).
        """
        try:
            from .seed_models import get_all_seed_repo_ids
        except ImportError:
            logger.warning("seed_models module not available, skipping seed load")
            return 0

        repo_ids = get_all_seed_repo_ids(path)
        if not repo_ids:
            logger.info("No seed models found to pre-load")
            return 0

        loaded = 0
        for repo_id in repo_ids:
            if self._store.is_known(repo_id):
                continue

            author = repo_id.split("/")[0] if "/" in repo_id else ""
            # Insert directly as 'known' (not 'new') to avoid false alerts
            model_dict = {
                "model_id": repo_id,
                "author": author,
            }
            # Use upsert_model which returns True if new
            was_new = self._store.upsert_model(model_dict)
            if was_new:
                # Immediately mark as known + notified (not a genuine detection)
                self._store.mark_as_known(repo_id)
                self._store.mark_as_notified(repo_id)
                loaded += 1

        logger.info("Pre-loaded %d seed models (from %d in catalog)", loaded, len(repo_ids))
        return loaded

    def _import_legacy_json(self, path: str | None = None) -> int:
        """Import model IDs from the legacy seen_models.json file.

        Returns:
            Number of models imported.
        """
        if path is None:
            from .config import SEEN_MODELS_PATH
            path = SEEN_MODELS_PATH

        import os
        if not os.path.exists(path):
            logger.debug("No legacy JSON at %s, skipping import", path)
            return 0

        imported = self._store.import_from_json(path)
        if imported > 0:
            logger.info("Imported %d models from legacy JSON %s", imported, path)
        return imported

    # ------------------------------------------------------------------
    # Main detection entry point
    # ------------------------------------------------------------------

    def run(self, *, now: datetime | None = None) -> DetectionResult:
        """Execute a full detection run across all watched organizations.

        Steps:
        1. Fetch models from each watched org via the HF API
        2. Compare against the ModelStore to find new models
        3. Filter new models by polling window (created_at check)
        4. Record poll history in the store
        5. Return aggregated results

        Args:
            now: Override for current time (for testing).

        Returns:
            DetectionResult with all new models, per-org stats, and errors.
        """
        if now is None:
            now = _now_utc()

        result = DetectionResult(
            poll_timestamp=now.isoformat(),
            polling_window_hours=self.polling_interval_hours,
        )

        if not self.watched_orgs:
            logger.warning("No watched organizations configured — nothing to poll")
            return result

        logger.info(
            "Starting detection run: %d orgs, %dh window",
            len(self.watched_orgs),
            self.polling_interval_hours,
        )

        for org in self.watched_orgs:
            org_result = self._poll_org(org, now=now)
            result.org_results.append(org_result)
            result.new_models.extend(org_result.new_models)

        # Record overall poll in history
        error_summary = "; ".join(result.errors) if result.errors else ""
        self._store.record_poll(
            source="detection_run",
            models_found=result.total_fetched,
            new_models=result.total_new,
            errors=error_summary,
        )

        logger.info(
            "Detection run complete: %d fetched, %d new, %d errors",
            result.total_fetched,
            result.total_new,
            len(result.errors),
        )

        return result

    # ------------------------------------------------------------------
    # Per-org detection
    # ------------------------------------------------------------------

    def _poll_org(
        self, org: str, *, now: datetime | None = None
    ) -> OrgPollResult:
        """Fetch and detect new models from a single organization.

        Args:
            org: HuggingFace organization/user ID.
            now: Override for current time (for testing).

        Returns:
            OrgPollResult with fetch stats and new model list.
        """
        org_result = OrgPollResult(org=org)

        # Step 1: Fetch models from the HF API
        try:
            model_records = self._client.fetch_org_models(org)
        except HFApiError as exc:
            error_msg = f"{org}: HF API error — {exc} (status={exc.status_code})"
            logger.error(error_msg)
            org_result.error = error_msg
            self._store.record_poll(
                source=org, models_found=0, new_models=0, errors=error_msg
            )
            return org_result
        except Exception as exc:
            error_msg = f"{org}: unexpected error — {exc}"
            logger.exception(error_msg)
            org_result.error = error_msg
            self._store.record_poll(
                source=org, models_found=0, new_models=0, errors=error_msg
            )
            return org_result

        model_dicts = [r.to_dict() for r in model_records]
        org_result.models_fetched = len(model_dicts)

        if not model_dicts:
            logger.info("No models found for org=%s", org)
            self._store.record_poll(
                source=org, models_found=0, new_models=0
            )
            return org_result

        # Step 2: Compare against stored state (upsert returns new ones)
        newly_seen = self._store.upsert_models(model_dicts)

        # Step 3: Filter by polling window
        new_in_window = self._filter_by_polling_window(newly_seen, now=now)
        org_result.new_models = new_in_window

        # Record per-org poll
        self._store.record_poll(
            source=org,
            models_found=len(model_dicts),
            new_models=len(new_in_window),
        )

        if new_in_window:
            logger.info(
                "Detected %d new model(s) from %s within %dh window: %s",
                len(new_in_window),
                org,
                self.polling_interval_hours,
                [m["model_id"] for m in new_in_window],
            )

        return org_result

    # ------------------------------------------------------------------
    # Polling window filter
    # ------------------------------------------------------------------

    def _filter_by_polling_window(
        self,
        models: list[dict],
        *,
        now: datetime | None = None,
    ) -> list[dict]:
        """Filter models to only those created within the polling window.

        Models with unparseable or missing created_at are included
        (benefit of the doubt — we don't want to miss genuinely new models).

        Args:
            models: List of model dicts to filter.
            now: Override for current time (for testing).

        Returns:
            Filtered list of models within the polling window.
        """
        if not models:
            return []

        result = []
        for model in models:
            created_at = model.get("created_at", "N/A")
            if is_within_polling_window(
                created_at,
                self.polling_interval_hours,
                now=now,
            ):
                result.append(model)
            else:
                logger.debug(
                    "Model %s created_at=%s is outside %dh window, skipping",
                    model.get("model_id", "unknown"),
                    created_at,
                    self.polling_interval_hours,
                )
        return result

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def get_last_poll_time(self) -> str | None:
        """Return the timestamp of the most recent poll, or None."""
        return self._store.get_last_poll_time()

    def get_poll_history(self, limit: int = 50) -> list[dict]:
        """Return recent poll history entries."""
        return self._store.get_poll_history(limit)

    def get_all_known_models(self) -> list[dict]:
        """Return all models tracked in the store."""
        return self._store.get_all_models()

    def get_new_unprocessed_models(self) -> list[dict]:
        """Return models with status='new' (not yet fully processed)."""
        return self._store.get_new_models()

    # ------------------------------------------------------------------
    # Post-detection lifecycle
    # ------------------------------------------------------------------

    def mark_models_processed(
        self,
        result: DetectionResult,
        *,
        mark_notified: bool = False,
    ) -> int:
        """Mark all newly detected models in a result as fully processed.

        Transitions their status from 'new' to 'known' in the store.
        This should be called after the detection result has been acted on
        (e.g., Slack notification sent, dashboard updated).

        Args:
            result: The DetectionResult from a ``run()`` call.
            mark_notified: If True, also mark models as notified (Slack sent).

        Returns:
            Number of models marked as processed.
        """
        count = 0
        for model in result.new_models:
            model_id = model.get("model_id", "")
            if not model_id:
                continue
            self._store.mark_as_known(model_id)
            if mark_notified:
                self._store.mark_as_notified(model_id)
            count += 1

        if count:
            logger.info(
                "Marked %d model(s) as processed (notified=%s)",
                count,
                mark_notified,
            )
        return count

    def mark_model_notified(self, model_id: str) -> None:
        """Mark a single model as having its Slack notification sent."""
        self._store.mark_as_notified(model_id)

    def get_detection_stats(self) -> dict:
        """Return a summary of the detection system's current state.

        Useful for dashboard display and health checks.

        Returns:
            Dict with:
            - ``total_known``: Total models tracked in the store
            - ``status_new``: Models with status='new' (pending processing)
            - ``status_known``: Models with status='known' (fully processed)
            - ``unnotified``: Models not yet sent to Slack
            - ``last_poll_time``: ISO timestamp of most recent poll (or None)
            - ``watched_orgs``: Number of watched organizations
            - ``polling_interval_hours``: Current polling window size
        """
        all_models = self._store.get_all_models()
        new_models = [m for m in all_models if m.get("status") == "new"]
        known_models = [m for m in all_models if m.get("status") == "known"]
        unnotified = self._store.get_unnotified_models()

        return {
            "total_known": len(all_models),
            "status_new": len(new_models),
            "status_known": len(known_models),
            "unnotified": len(unnotified),
            "last_poll_time": self._store.get_last_poll_time(),
            "watched_orgs": len(self.watched_orgs),
            "polling_interval_hours": self.polling_interval_hours,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close underlying resources."""
        self._client.close()
        self._store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
