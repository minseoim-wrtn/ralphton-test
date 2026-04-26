"""Tests for hf_model_monitor.detector — new model detection logic.

Covers:
- Time-based polling window filtering
- ISO datetime parsing edge cases
- Per-org polling with mocked HFApiClient
- Full detection runs with state comparison
- Error handling (API failures, empty orgs)
- DetectionResult summary and aggregation
- Integration with ModelStore upsert
- First-run bootstrap with seed models
- Post-detection lifecycle (mark processed, stats)
- State persistence across restarts
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from hf_model_monitor.detector import (
    DetectionResult,
    ModelDetector,
    OrgPollResult,
    _parse_iso_datetime,
    is_within_polling_window,
)
from hf_model_monitor.hf_client import HFApiError, HFRateLimitError, ModelRecord
from hf_model_monitor.model_store import ModelStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_store(tmp_path):
    """Yield a ModelStore backed by a temporary SQLite file."""
    db_path = str(tmp_path / "test_detector.db")
    store = ModelStore(db_path)
    yield store
    store.close()


@pytest.fixture
def mock_client():
    """Return a MagicMock HFApiClient."""
    client = MagicMock(spec=["fetch_org_models", "close"])
    client.fetch_org_models.return_value = []
    return client


@pytest.fixture
def sample_record():
    """Return a factory for creating ModelRecord instances."""
    def _make(
        model_id="org/model",
        author="org",
        created_at="2026-04-25T12:00:00+00:00",
        **kwargs,
    ):
        defaults = {
            "last_modified": created_at,
            "pipeline_tag": "text-generation",
            "tags": ["transformers"],
            "downloads": 1000,
            "likes": 100,
            "library_name": "transformers",
        }
        defaults.update(kwargs)
        return ModelRecord(
            model_id=model_id,
            author=author,
            created_at=created_at,
            **defaults,
        )
    return _make


# ---------------------------------------------------------------------------
# _parse_iso_datetime
# ---------------------------------------------------------------------------
class TestParseIsoDatetime:
    def test_valid_iso_with_timezone(self):
        dt = _parse_iso_datetime("2026-04-25T12:00:00+00:00")
        assert dt is not None
        assert dt.year == 2026
        assert dt.month == 4
        assert dt.day == 25
        assert dt.tzinfo is not None

    def test_valid_iso_with_z_suffix(self):
        dt = _parse_iso_datetime("2026-04-25T12:00:00Z")
        assert dt is not None
        assert dt.tzinfo is not None

    def test_naive_datetime_gets_utc(self):
        dt = _parse_iso_datetime("2026-04-25T12:00:00")
        assert dt is not None
        assert dt.tzinfo == timezone.utc

    def test_returns_none_for_na(self):
        assert _parse_iso_datetime("N/A") is None

    def test_returns_none_for_empty_string(self):
        assert _parse_iso_datetime("") is None

    def test_returns_none_for_garbage(self):
        assert _parse_iso_datetime("not-a-date") is None

    def test_returns_none_for_none_input(self):
        assert _parse_iso_datetime(None) is None


# ---------------------------------------------------------------------------
# is_within_polling_window
# ---------------------------------------------------------------------------
class TestIsWithinPollingWindow:
    def test_model_created_1h_ago_in_24h_window(self):
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
        created = "2026-04-26T11:00:00+00:00"  # 1 hour ago
        assert is_within_polling_window(created, 24, now=now) is True

    def test_model_created_23h_ago_in_24h_window(self):
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
        created = "2026-04-25T13:00:00+00:00"  # 23 hours ago
        assert is_within_polling_window(created, 24, now=now) is True

    def test_model_created_exactly_at_cutoff(self):
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
        created = "2026-04-25T12:00:00+00:00"  # exactly 24 hours ago
        assert is_within_polling_window(created, 24, now=now) is True

    def test_model_created_25h_ago_outside_24h_window(self):
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
        created = "2026-04-25T11:00:00+00:00"  # 25 hours ago
        assert is_within_polling_window(created, 24, now=now) is False

    def test_model_created_48h_ago_outside_24h_window(self):
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
        created = "2026-04-24T12:00:00+00:00"  # 48 hours ago
        assert is_within_polling_window(created, 24, now=now) is False

    def test_na_created_at_returns_true(self):
        """Models with unknown creation time are included (benefit of doubt)."""
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
        assert is_within_polling_window("N/A", 24, now=now) is True

    def test_empty_created_at_returns_true(self):
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
        assert is_within_polling_window("", 24, now=now) is True

    def test_unparseable_created_at_returns_true(self):
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
        assert is_within_polling_window("bad-date", 24, now=now) is True

    def test_48h_polling_window(self):
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
        created = "2026-04-24T13:00:00+00:00"  # 47 hours ago
        assert is_within_polling_window(created, 48, now=now) is True

    def test_1h_polling_window(self):
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
        created = "2026-04-26T11:30:00+00:00"  # 30 min ago
        assert is_within_polling_window(created, 1, now=now) is True

    def test_1h_window_model_2h_ago(self):
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
        created = "2026-04-26T10:00:00+00:00"  # 2 hours ago
        assert is_within_polling_window(created, 1, now=now) is False


# ---------------------------------------------------------------------------
# OrgPollResult
# ---------------------------------------------------------------------------
class TestOrgPollResult:
    def test_success_when_no_error(self):
        r = OrgPollResult(org="meta-llama", models_fetched=10)
        assert r.success is True

    def test_failure_when_error_set(self):
        r = OrgPollResult(org="meta-llama", error="timeout")
        assert r.success is False

    def test_new_count(self):
        r = OrgPollResult(
            org="meta-llama",
            new_models=[{"model_id": "a"}, {"model_id": "b"}],
        )
        assert r.new_count == 2

    def test_new_count_zero(self):
        r = OrgPollResult(org="meta-llama")
        assert r.new_count == 0


# ---------------------------------------------------------------------------
# DetectionResult
# ---------------------------------------------------------------------------
class TestDetectionResult:
    def test_empty_result_defaults(self):
        r = DetectionResult()
        assert r.total_fetched == 0
        assert r.total_new == 0
        assert r.orgs_polled == 0
        assert r.orgs_succeeded == 0
        assert r.orgs_failed == 0
        assert r.errors == []

    def test_aggregation_from_org_results(self):
        r = DetectionResult(
            new_models=[{"model_id": "a"}, {"model_id": "b"}],
            org_results=[
                OrgPollResult(org="org1", models_fetched=10, new_models=[{"model_id": "a"}]),
                OrgPollResult(org="org2", models_fetched=5, new_models=[{"model_id": "b"}]),
                OrgPollResult(org="org3", models_fetched=0, error="timeout"),
            ],
        )
        assert r.total_fetched == 15
        assert r.total_new == 2
        assert r.orgs_polled == 3
        assert r.orgs_succeeded == 2
        assert r.orgs_failed == 1
        assert r.errors == ["timeout"]

    def test_summary_contains_key_info(self):
        r = DetectionResult(
            poll_timestamp="2026-04-26T12:00:00+00:00",
            polling_window_hours=24,
            new_models=[{"model_id": "org/new-model"}],
            org_results=[
                OrgPollResult(org="org", models_fetched=5,
                              new_models=[{"model_id": "org/new-model"}]),
            ],
        )
        summary = r.summary()
        assert "2026-04-26" in summary
        assert "24h" in summary
        assert "org/new-model" in summary
        assert "1" in summary  # new model count

    def test_summary_includes_errors(self):
        r = DetectionResult(
            poll_timestamp="2026-04-26T12:00:00+00:00",
            org_results=[
                OrgPollResult(org="org", error="API timeout"),
            ],
        )
        summary = r.summary()
        assert "API timeout" in summary
        assert "Errors" in summary


# ---------------------------------------------------------------------------
# ModelDetector — from_config
# ---------------------------------------------------------------------------
class TestModelDetectorFromConfig:
    def test_creates_from_config_dict(self, tmp_store, mock_client):
        config = {
            "watched_organizations": ["meta-llama", "google"],
            "polling_interval_hours": 12,
        }
        detector = ModelDetector.from_config(
            config, store=tmp_store, client=mock_client
        )
        assert detector.watched_orgs == ["meta-llama", "google"]
        assert detector.polling_interval_hours == 12

    def test_defaults_when_keys_missing(self, tmp_store, mock_client):
        detector = ModelDetector.from_config(
            {}, store=tmp_store, client=mock_client
        )
        assert detector.watched_orgs == []
        assert detector.polling_interval_hours == 24


# ---------------------------------------------------------------------------
# ModelDetector — run (full detection)
# ---------------------------------------------------------------------------
class TestModelDetectorRun:
    def test_empty_watched_orgs_returns_empty_result(self, tmp_store, mock_client):
        detector = ModelDetector([], store=tmp_store, client=mock_client)
        result = detector.run()
        assert result.total_new == 0
        assert result.orgs_polled == 0
        mock_client.fetch_org_models.assert_not_called()

    def test_detects_new_models_on_first_run(
        self, tmp_store, mock_client, sample_record
    ):
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
        record = sample_record(
            model_id="meta-llama/Llama-5",
            author="meta-llama",
            created_at="2026-04-26T10:00:00+00:00",
        )
        mock_client.fetch_org_models.return_value = [record]

        detector = ModelDetector(
            ["meta-llama"], store=tmp_store, client=mock_client
        )
        result = detector.run(now=now)

        assert result.total_new == 1
        assert result.new_models[0]["model_id"] == "meta-llama/Llama-5"
        assert result.orgs_succeeded == 1
        assert result.orgs_failed == 0

    def test_does_not_flag_already_known_models(
        self, tmp_store, mock_client, sample_record
    ):
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
        record = sample_record(
            model_id="meta-llama/Llama-5",
            created_at="2026-04-26T10:00:00+00:00",
        )
        mock_client.fetch_org_models.return_value = [record]

        detector = ModelDetector(
            ["meta-llama"], store=tmp_store, client=mock_client
        )
        # First run — model is new
        result1 = detector.run(now=now)
        assert result1.total_new == 1

        # Second run — same model, should not be flagged again
        result2 = detector.run(now=now)
        assert result2.total_new == 0

    def test_filters_out_models_outside_polling_window(
        self, tmp_store, mock_client, sample_record
    ):
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)

        # Model created 48 hours ago — outside 24h window
        old_record = sample_record(
            model_id="org/old-model",
            created_at="2026-04-24T12:00:00+00:00",
        )
        # Model created 2 hours ago — inside 24h window
        new_record = sample_record(
            model_id="org/new-model",
            created_at="2026-04-26T10:00:00+00:00",
        )
        mock_client.fetch_org_models.return_value = [old_record, new_record]

        detector = ModelDetector(
            ["org"], polling_interval_hours=24,
            store=tmp_store, client=mock_client,
        )
        result = detector.run(now=now)

        assert result.total_new == 1
        assert result.new_models[0]["model_id"] == "org/new-model"
        # The old model is still upserted into the store (known), just not flagged
        assert tmp_store.is_known("org/old-model")

    def test_includes_models_with_na_created_at(
        self, tmp_store, mock_client, sample_record
    ):
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
        record = sample_record(
            model_id="org/no-date",
            created_at="N/A",
        )
        mock_client.fetch_org_models.return_value = [record]

        detector = ModelDetector(
            ["org"], store=tmp_store, client=mock_client
        )
        result = detector.run(now=now)

        # N/A created_at should be included (benefit of the doubt)
        assert result.total_new == 1
        assert result.new_models[0]["model_id"] == "org/no-date"

    def test_multiple_orgs(self, tmp_store, mock_client, sample_record):
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)

        def side_effect(org, **kwargs):
            if org == "meta-llama":
                return [sample_record(
                    model_id="meta-llama/Llama-5",
                    author="meta-llama",
                    created_at="2026-04-26T10:00:00+00:00",
                )]
            elif org == "google":
                return [sample_record(
                    model_id="google/Gemma-4",
                    author="google",
                    created_at="2026-04-26T11:00:00+00:00",
                )]
            return []

        mock_client.fetch_org_models.side_effect = side_effect

        detector = ModelDetector(
            ["meta-llama", "google"],
            store=tmp_store,
            client=mock_client,
        )
        result = detector.run(now=now)

        assert result.total_new == 2
        assert result.orgs_polled == 2
        assert result.orgs_succeeded == 2
        new_ids = {m["model_id"] for m in result.new_models}
        assert new_ids == {"meta-llama/Llama-5", "google/Gemma-4"}

    def test_mixed_new_and_known_across_orgs(
        self, tmp_store, mock_client, sample_record
    ):
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)

        # Pre-seed a known model
        tmp_store.upsert_model({
            "model_id": "meta-llama/Llama-4",
            "author": "meta-llama",
        })

        def side_effect(org, **kwargs):
            if org == "meta-llama":
                return [
                    sample_record(
                        model_id="meta-llama/Llama-4",
                        author="meta-llama",
                        created_at="2026-04-20T10:00:00+00:00",
                    ),
                    sample_record(
                        model_id="meta-llama/Llama-5",
                        author="meta-llama",
                        created_at="2026-04-26T10:00:00+00:00",
                    ),
                ]
            return []

        mock_client.fetch_org_models.side_effect = side_effect

        detector = ModelDetector(
            ["meta-llama"], store=tmp_store, client=mock_client
        )
        result = detector.run(now=now)

        assert result.total_new == 1
        assert result.new_models[0]["model_id"] == "meta-llama/Llama-5"

    def test_records_poll_history(self, tmp_store, mock_client, sample_record):
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
        record = sample_record(
            model_id="org/model",
            created_at="2026-04-26T10:00:00+00:00",
        )
        mock_client.fetch_org_models.return_value = [record]

        detector = ModelDetector(
            ["org"], store=tmp_store, client=mock_client
        )
        detector.run(now=now)

        history = tmp_store.get_poll_history()
        # Should have 2 entries: one per-org + one overall detection_run
        assert len(history) >= 2
        sources = {h["source"] for h in history}
        assert "org" in sources
        assert "detection_run" in sources


# ---------------------------------------------------------------------------
# ModelDetector — error handling
# ---------------------------------------------------------------------------
class TestModelDetectorErrors:
    def test_api_error_captured_per_org(self, tmp_store, mock_client):
        mock_client.fetch_org_models.side_effect = HFApiError(
            "Server error", status_code=500
        )

        detector = ModelDetector(
            ["meta-llama"], store=tmp_store, client=mock_client
        )
        result = detector.run()

        assert result.orgs_failed == 1
        assert result.orgs_succeeded == 0
        assert len(result.errors) == 1
        assert "meta-llama" in result.errors[0]
        assert "500" in result.errors[0]

    def test_rate_limit_error_captured(self, tmp_store, mock_client):
        mock_client.fetch_org_models.side_effect = HFRateLimitError(
            retry_after=60
        )

        detector = ModelDetector(
            ["meta-llama"], store=tmp_store, client=mock_client
        )
        result = detector.run()

        assert result.orgs_failed == 1
        assert "429" in result.errors[0] or "rate" in result.errors[0].lower()

    def test_unexpected_exception_captured(self, tmp_store, mock_client):
        mock_client.fetch_org_models.side_effect = RuntimeError("network down")

        detector = ModelDetector(
            ["meta-llama"], store=tmp_store, client=mock_client
        )
        result = detector.run()

        assert result.orgs_failed == 1
        assert "network down" in result.errors[0]

    def test_one_org_fails_others_continue(
        self, tmp_store, mock_client, sample_record
    ):
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)

        def side_effect(org, **kwargs):
            if org == "meta-llama":
                raise HFApiError("timeout", status_code=500)
            elif org == "google":
                return [sample_record(
                    model_id="google/Gemma-4",
                    author="google",
                    created_at="2026-04-26T10:00:00+00:00",
                )]
            return []

        mock_client.fetch_org_models.side_effect = side_effect

        detector = ModelDetector(
            ["meta-llama", "google"],
            store=tmp_store,
            client=mock_client,
        )
        result = detector.run(now=now)

        assert result.orgs_polled == 2
        assert result.orgs_failed == 1
        assert result.orgs_succeeded == 1
        assert result.total_new == 1
        assert result.new_models[0]["model_id"] == "google/Gemma-4"

    def test_error_recorded_in_poll_history(self, tmp_store, mock_client):
        mock_client.fetch_org_models.side_effect = HFApiError(
            "Server error", status_code=500
        )

        detector = ModelDetector(
            ["meta-llama"], store=tmp_store, client=mock_client
        )
        detector.run()

        history = tmp_store.get_poll_history()
        error_entries = [h for h in history if h["errors"]]
        assert len(error_entries) >= 1

    def test_empty_org_returns_no_new_models(self, tmp_store, mock_client):
        mock_client.fetch_org_models.return_value = []

        detector = ModelDetector(
            ["empty-org"], store=tmp_store, client=mock_client
        )
        result = detector.run()

        assert result.total_new == 0
        assert result.total_fetched == 0
        assert result.orgs_succeeded == 1  # No error, just empty


# ---------------------------------------------------------------------------
# ModelDetector — polling window edge cases
# ---------------------------------------------------------------------------
class TestPollingWindowEdgeCases:
    def test_custom_polling_interval(
        self, tmp_store, mock_client, sample_record
    ):
        """48h window should include models from the last 2 days."""
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)

        record = sample_record(
            model_id="org/two-day-old",
            created_at="2026-04-24T14:00:00+00:00",  # 46 hours ago
        )
        mock_client.fetch_org_models.return_value = [record]

        detector = ModelDetector(
            ["org"], polling_interval_hours=48,
            store=tmp_store, client=mock_client,
        )
        result = detector.run(now=now)

        assert result.total_new == 1

    def test_short_1h_window_filters_old_models(
        self, tmp_store, mock_client, sample_record
    ):
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)

        recent = sample_record(
            model_id="org/just-now",
            created_at="2026-04-26T11:30:00+00:00",  # 30 min ago
        )
        old = sample_record(
            model_id="org/two-hours-ago",
            created_at="2026-04-26T10:00:00+00:00",  # 2 hours ago
        )
        mock_client.fetch_org_models.return_value = [recent, old]

        detector = ModelDetector(
            ["org"], polling_interval_hours=1,
            store=tmp_store, client=mock_client,
        )
        result = detector.run(now=now)

        assert result.total_new == 1
        assert result.new_models[0]["model_id"] == "org/just-now"

    def test_model_with_unparseable_date_included(
        self, tmp_store, mock_client, sample_record
    ):
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)

        record = sample_record(
            model_id="org/bad-date",
            created_at="invalid-timestamp",
        )
        mock_client.fetch_org_models.return_value = [record]

        detector = ModelDetector(
            ["org"], store=tmp_store, client=mock_client
        )
        result = detector.run(now=now)

        # Unparseable dates → included (benefit of the doubt)
        assert result.total_new == 1

    def test_only_newly_seen_models_are_window_filtered(
        self, tmp_store, mock_client, sample_record
    ):
        """Models already in the store should not appear as new,
        even if their created_at is within the window."""
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)

        # Pre-seed a model that was created recently
        tmp_store.upsert_model({
            "model_id": "org/already-known",
            "author": "org",
            "created_at": "2026-04-26T10:00:00+00:00",
        })

        record = sample_record(
            model_id="org/already-known",
            author="org",
            created_at="2026-04-26T10:00:00+00:00",
        )
        mock_client.fetch_org_models.return_value = [record]

        detector = ModelDetector(
            ["org"], store=tmp_store, client=mock_client
        )
        result = detector.run(now=now)

        assert result.total_new == 0  # Already known — not flagged


# ---------------------------------------------------------------------------
# ModelDetector — convenience methods
# ---------------------------------------------------------------------------
class TestDetectorConvenienceMethods:
    def test_get_last_poll_time(self, tmp_store, mock_client, sample_record):
        detector = ModelDetector(
            ["org"], store=tmp_store, client=mock_client
        )
        assert detector.get_last_poll_time() is None

        mock_client.fetch_org_models.return_value = [
            sample_record(created_at="2026-04-26T10:00:00+00:00")
        ]
        detector.run()
        assert detector.get_last_poll_time() is not None

    def test_get_poll_history(self, tmp_store, mock_client):
        detector = ModelDetector(
            ["org"], store=tmp_store, client=mock_client
        )
        detector.run()
        history = detector.get_poll_history()
        assert len(history) >= 1

    def test_get_all_known_models(
        self, tmp_store, mock_client, sample_record
    ):
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
        mock_client.fetch_org_models.return_value = [
            sample_record(
                model_id="org/model-a",
                created_at="2026-04-26T10:00:00+00:00",
            ),
        ]
        detector = ModelDetector(
            ["org"], store=tmp_store, client=mock_client
        )
        detector.run(now=now)

        known = detector.get_all_known_models()
        assert len(known) == 1
        assert known[0]["model_id"] == "org/model-a"

    def test_get_new_unprocessed_models(
        self, tmp_store, mock_client, sample_record
    ):
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
        mock_client.fetch_org_models.return_value = [
            sample_record(
                model_id="org/fresh",
                created_at="2026-04-26T10:00:00+00:00",
            ),
        ]
        detector = ModelDetector(
            ["org"], store=tmp_store, client=mock_client
        )
        detector.run(now=now)

        unprocessed = detector.get_new_unprocessed_models()
        assert len(unprocessed) == 1
        assert unprocessed[0]["status"] == "new"

        # After marking as known, should no longer appear
        tmp_store.mark_as_known("org/fresh")
        assert detector.get_new_unprocessed_models() == []


# ---------------------------------------------------------------------------
# ModelDetector — context manager
# ---------------------------------------------------------------------------
class TestDetectorContextManager:
    def test_context_manager_closes_resources(self, tmp_store, mock_client):
        with ModelDetector(
            ["org"], store=tmp_store, client=mock_client
        ) as detector:
            detector.run()
        mock_client.close.assert_called_once()


# ---------------------------------------------------------------------------
# Integration: full detection cycle
# ---------------------------------------------------------------------------
class TestDetectionCycleIntegration:
    def test_two_consecutive_runs_detect_only_new(
        self, tmp_store, mock_client, sample_record
    ):
        """Simulate two daily polls: second run should only find genuinely new models."""
        now_day1 = datetime(2026, 4, 25, 12, 0, 0, tzinfo=timezone.utc)
        now_day2 = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)

        day1_models = [
            sample_record(
                model_id="org/model-a",
                created_at="2026-04-25T08:00:00+00:00",
            ),
            sample_record(
                model_id="org/model-b",
                created_at="2026-04-25T09:00:00+00:00",
            ),
        ]
        day2_models = day1_models + [
            sample_record(
                model_id="org/model-c",
                created_at="2026-04-26T07:00:00+00:00",
            ),
        ]

        detector = ModelDetector(
            ["org"], polling_interval_hours=24,
            store=tmp_store, client=mock_client,
        )

        # Day 1 run
        mock_client.fetch_org_models.return_value = day1_models
        result1 = detector.run(now=now_day1)
        assert result1.total_new == 2
        assert {m["model_id"] for m in result1.new_models} == {
            "org/model-a", "org/model-b"
        }

        # Day 2 run — only model-c is new
        mock_client.fetch_org_models.return_value = day2_models
        result2 = detector.run(now=now_day2)
        assert result2.total_new == 1
        assert result2.new_models[0]["model_id"] == "org/model-c"

    def test_model_outside_window_on_first_run_not_flagged(
        self, tmp_store, mock_client, sample_record
    ):
        """On initial run, models older than the polling window are stored
        but NOT flagged as new (they're backfill, not genuinely new)."""
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)

        models = [
            sample_record(
                model_id="org/very-old",
                created_at="2026-01-01T00:00:00+00:00",  # ~4 months old
            ),
            sample_record(
                model_id="org/brand-new",
                created_at="2026-04-26T10:00:00+00:00",  # 2 hours old
            ),
        ]
        mock_client.fetch_org_models.return_value = models

        detector = ModelDetector(
            ["org"], polling_interval_hours=24,
            store=tmp_store, client=mock_client,
        )
        result = detector.run(now=now)

        # Only brand-new should be flagged
        assert result.total_new == 1
        assert result.new_models[0]["model_id"] == "org/brand-new"

        # But both are stored for future comparison
        assert tmp_store.is_known("org/very-old")
        assert tmp_store.is_known("org/brand-new")

    def test_store_state_persists_between_runs(self, tmp_path, mock_client, sample_record):
        """Verify that detection state persists across detector instances
        (simulating process restart between daily polls)."""
        db_path = str(tmp_path / "persist.db")
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)

        record = sample_record(
            model_id="org/model",
            created_at="2026-04-26T10:00:00+00:00",
        )
        mock_client.fetch_org_models.return_value = [record]

        # First run with store1
        store1 = ModelStore(db_path)
        detector1 = ModelDetector(["org"], store=store1, client=mock_client)
        result1 = detector1.run(now=now)
        assert result1.total_new == 1
        store1.close()

        # Second run with store2 (new instance, same DB)
        store2 = ModelStore(db_path)
        detector2 = ModelDetector(["org"], store=store2, client=mock_client)
        result2 = detector2.run(now=now)
        assert result2.total_new == 0  # Already known from first run
        store2.close()


# ---------------------------------------------------------------------------
# ModelDetector — first-run bootstrap (initialize_store)
# ---------------------------------------------------------------------------
class TestInitializeStore:
    def test_is_first_run_on_empty_store(self, tmp_store, mock_client):
        detector = ModelDetector(["org"], store=tmp_store, client=mock_client)
        assert detector.is_first_run() is True

    def test_is_first_run_false_after_insert(self, tmp_store, mock_client):
        tmp_store.upsert_model({"model_id": "org/existing"})
        detector = ModelDetector(["org"], store=tmp_store, client=mock_client)
        assert detector.is_first_run() is False

    def test_initialize_store_with_seed_models(
        self, tmp_store, mock_client, tmp_path
    ):
        """Seed models should be pre-loaded as 'known' + 'notified'."""
        # Create a minimal seed_models.yaml
        seed_yaml = tmp_path / "seed_models.yaml"
        seed_yaml.write_text(
            "categories:\n"
            "  llm:\n"
            "    label: LLMs\n"
            "    models:\n"
            "      - repo_id: meta-llama/Llama-4-Scout-17B-16E\n"
            "        name: Llama 4 Scout\n"
            "      - repo_id: google/gemma-3-27b-it\n"
            "        name: Gemma 3 27B\n"
        )

        detector = ModelDetector(["meta-llama", "google"], store=tmp_store, client=mock_client)
        stats = detector.initialize_store(
            seed_models_path=str(seed_yaml),
            legacy_json_path=str(tmp_path / "nonexistent.json"),
        )

        assert stats["was_first_run"] is True
        assert stats["seed_models_loaded"] == 2
        assert stats["total_known_after"] == 2

        # Verify they are marked as known (not 'new')
        m1 = tmp_store.get_model("meta-llama/Llama-4-Scout-17B-16E")
        assert m1 is not None
        assert m1["status"] == "known"
        assert m1["notified"] == 1

        m2 = tmp_store.get_model("google/gemma-3-27b-it")
        assert m2 is not None
        assert m2["status"] == "known"
        assert m2["notified"] == 1

    def test_initialize_store_idempotent(
        self, tmp_store, mock_client, tmp_path
    ):
        """Calling initialize_store twice should be safe (no-op on second call)."""
        seed_yaml = tmp_path / "seed_models.yaml"
        seed_yaml.write_text(
            "categories:\n"
            "  llm:\n"
            "    label: LLMs\n"
            "    models:\n"
            "      - repo_id: org/model-a\n"
        )

        detector = ModelDetector(["org"], store=tmp_store, client=mock_client)

        stats1 = detector.initialize_store(seed_models_path=str(seed_yaml))
        assert stats1["was_first_run"] is True
        assert stats1["seed_models_loaded"] == 1

        # Second call — store already has data
        stats2 = detector.initialize_store(seed_models_path=str(seed_yaml))
        assert stats2["was_first_run"] is False
        assert stats2["seed_models_loaded"] == 0  # No new loads

    def test_initialize_store_with_legacy_json(
        self, tmp_store, mock_client, tmp_path
    ):
        """Legacy JSON migration should import model IDs as 'known'."""
        import json

        legacy_json = tmp_path / "seen_models.json"
        legacy_json.write_text(json.dumps([
            "org/legacy-a",
            "org/legacy-b",
        ]))

        detector = ModelDetector(["org"], store=tmp_store, client=mock_client)
        stats = detector.initialize_store(
            legacy_json_path=str(legacy_json),
        )

        assert stats["was_first_run"] is True
        assert stats["legacy_models_imported"] == 2
        assert tmp_store.is_known("org/legacy-a")
        assert tmp_store.is_known("org/legacy-b")

    def test_initialize_store_records_poll_history(
        self, tmp_store, mock_client, tmp_path
    ):
        """Bootstrap should record a 'bootstrap' entry in poll history."""
        seed_yaml = tmp_path / "seed_models.yaml"
        seed_yaml.write_text(
            "categories:\n"
            "  llm:\n"
            "    label: LLMs\n"
            "    models:\n"
            "      - repo_id: org/model\n"
        )

        detector = ModelDetector(["org"], store=tmp_store, client=mock_client)
        detector.initialize_store(
            seed_models_path=str(seed_yaml),
            legacy_json_path=str(tmp_path / "nonexistent.json"),
        )

        history = tmp_store.get_poll_history()
        bootstrap_entries = [h for h in history if h["source"] == "bootstrap"]
        assert len(bootstrap_entries) == 1
        assert bootstrap_entries[0]["models_found"] == 1

    def test_seed_models_not_detected_as_new_on_first_run(
        self, tmp_store, mock_client, sample_record, tmp_path
    ):
        """After bootstrap, seed models returned by the API should NOT
        trigger new-model alerts."""
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)

        # Seed catalog knows this model
        seed_yaml = tmp_path / "seed_models.yaml"
        seed_yaml.write_text(
            "categories:\n"
            "  llm:\n"
            "    label: LLMs\n"
            "    models:\n"
            "      - repo_id: meta-llama/Llama-4-Scout-17B-16E\n"
        )

        # API returns the same model
        record = sample_record(
            model_id="meta-llama/Llama-4-Scout-17B-16E",
            author="meta-llama",
            created_at="2026-04-26T08:00:00+00:00",
        )
        mock_client.fetch_org_models.return_value = [record]

        detector = ModelDetector(
            ["meta-llama"], store=tmp_store, client=mock_client,
        )
        detector.initialize_store(seed_models_path=str(seed_yaml))
        result = detector.run(now=now)

        # Already known from seed — not flagged as new
        assert result.total_new == 0

    def test_genuinely_new_model_detected_after_bootstrap(
        self, tmp_store, mock_client, sample_record, tmp_path
    ):
        """After bootstrap, genuinely new models should be detected."""
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)

        seed_yaml = tmp_path / "seed_models.yaml"
        seed_yaml.write_text(
            "categories:\n"
            "  llm:\n"
            "    label: LLMs\n"
            "    models:\n"
            "      - repo_id: meta-llama/Llama-4-Scout-17B-16E\n"
        )

        # API returns known model + a genuinely new one
        known_record = sample_record(
            model_id="meta-llama/Llama-4-Scout-17B-16E",
            author="meta-llama",
            created_at="2026-04-20T08:00:00+00:00",
        )
        new_record = sample_record(
            model_id="meta-llama/Llama-5",
            author="meta-llama",
            created_at="2026-04-26T10:00:00+00:00",
        )
        mock_client.fetch_org_models.return_value = [known_record, new_record]

        detector = ModelDetector(
            ["meta-llama"], store=tmp_store, client=mock_client,
        )
        detector.initialize_store(seed_models_path=str(seed_yaml))
        result = detector.run(now=now)

        # Only the genuinely new model should be flagged
        assert result.total_new == 1
        assert result.new_models[0]["model_id"] == "meta-llama/Llama-5"

    def test_initialize_with_no_seed_file(self, tmp_store, mock_client, tmp_path):
        """Missing seed file should not crash — just load 0 seed models."""
        detector = ModelDetector(["org"], store=tmp_store, client=mock_client)
        stats = detector.initialize_store(
            seed_models_path=str(tmp_path / "nonexistent.yaml"),
            legacy_json_path=str(tmp_path / "nonexistent.json"),
        )

        assert stats["was_first_run"] is True
        assert stats["seed_models_loaded"] == 0
        assert stats["legacy_models_imported"] == 0


# ---------------------------------------------------------------------------
# ModelDetector — mark_models_processed
# ---------------------------------------------------------------------------
class TestMarkModelsProcessed:
    def test_mark_changes_status_to_known(
        self, tmp_store, mock_client, sample_record
    ):
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
        record = sample_record(
            model_id="org/new-model",
            created_at="2026-04-26T10:00:00+00:00",
        )
        mock_client.fetch_org_models.return_value = [record]

        detector = ModelDetector(["org"], store=tmp_store, client=mock_client)
        result = detector.run(now=now)
        assert result.total_new == 1

        # Before marking: status is 'new'
        model = tmp_store.get_model("org/new-model")
        assert model["status"] == "new"

        # Mark as processed
        count = detector.mark_models_processed(result)
        assert count == 1

        # After marking: status is 'known'
        model = tmp_store.get_model("org/new-model")
        assert model["status"] == "known"

    def test_mark_with_notified_flag(
        self, tmp_store, mock_client, sample_record
    ):
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
        record = sample_record(
            model_id="org/model",
            created_at="2026-04-26T10:00:00+00:00",
        )
        mock_client.fetch_org_models.return_value = [record]

        detector = ModelDetector(["org"], store=tmp_store, client=mock_client)
        result = detector.run(now=now)

        # Before: not notified
        assert tmp_store.get_model("org/model")["notified"] == 0

        # Mark as processed AND notified
        detector.mark_models_processed(result, mark_notified=True)

        model = tmp_store.get_model("org/model")
        assert model["status"] == "known"
        assert model["notified"] == 1

    def test_mark_empty_result_returns_zero(self, tmp_store, mock_client):
        detector = ModelDetector(["org"], store=tmp_store, client=mock_client)
        result = DetectionResult()  # empty
        count = detector.mark_models_processed(result)
        assert count == 0

    def test_mark_model_notified_individually(
        self, tmp_store, mock_client, sample_record
    ):
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
        record = sample_record(
            model_id="org/model",
            created_at="2026-04-26T10:00:00+00:00",
        )
        mock_client.fetch_org_models.return_value = [record]

        detector = ModelDetector(["org"], store=tmp_store, client=mock_client)
        detector.run(now=now)

        assert tmp_store.get_model("org/model")["notified"] == 0
        detector.mark_model_notified("org/model")
        assert tmp_store.get_model("org/model")["notified"] == 1


# ---------------------------------------------------------------------------
# ModelDetector — get_detection_stats
# ---------------------------------------------------------------------------
class TestGetDetectionStats:
    def test_stats_on_empty_store(self, tmp_store, mock_client):
        detector = ModelDetector(
            ["org1", "org2"], polling_interval_hours=12,
            store=tmp_store, client=mock_client,
        )
        stats = detector.get_detection_stats()

        assert stats["total_known"] == 0
        assert stats["status_new"] == 0
        assert stats["status_known"] == 0
        assert stats["unnotified"] == 0
        assert stats["last_poll_time"] is None
        assert stats["watched_orgs"] == 2
        assert stats["polling_interval_hours"] == 12

    def test_stats_after_detection(
        self, tmp_store, mock_client, sample_record
    ):
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
        records = [
            sample_record(
                model_id="org/a",
                created_at="2026-04-26T10:00:00+00:00",
            ),
            sample_record(
                model_id="org/b",
                created_at="2026-04-26T11:00:00+00:00",
            ),
        ]
        mock_client.fetch_org_models.return_value = records

        detector = ModelDetector(
            ["org"], polling_interval_hours=24,
            store=tmp_store, client=mock_client,
        )
        result = detector.run(now=now)
        assert result.total_new == 2

        stats = detector.get_detection_stats()
        assert stats["total_known"] == 2
        assert stats["status_new"] == 2  # Not yet processed
        assert stats["status_known"] == 0
        assert stats["unnotified"] == 2
        assert stats["last_poll_time"] is not None

        # After marking processed
        detector.mark_models_processed(result, mark_notified=True)
        stats2 = detector.get_detection_stats()
        assert stats2["status_new"] == 0
        assert stats2["status_known"] == 2
        assert stats2["unnotified"] == 0


# ---------------------------------------------------------------------------
# Full lifecycle integration: bootstrap → detect → mark → verify
# ---------------------------------------------------------------------------
class TestFullDetectionLifecycle:
    def test_complete_lifecycle(
        self, tmp_path, mock_client, sample_record
    ):
        """Simulate the complete lifecycle:
        1. First run: bootstrap with seed models
        2. First detection: find genuinely new model
        3. Mark as processed + notified
        4. Second detection: no new models
        5. Third detection: another new model appears
        """
        db_path = str(tmp_path / "lifecycle.db")
        now_day1 = datetime(2026, 4, 25, 12, 0, 0, tzinfo=timezone.utc)
        now_day2 = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
        now_day3 = datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc)

        # Create seed file with one known model
        seed_yaml = tmp_path / "seed_models.yaml"
        seed_yaml.write_text(
            "categories:\n"
            "  llm:\n"
            "    label: LLMs\n"
            "    models:\n"
            "      - repo_id: org/baseline-model\n"
        )

        # --- Day 1: Bootstrap + first detection ---
        store1 = ModelStore(db_path)
        detector1 = ModelDetector(
            ["org"], polling_interval_hours=24,
            store=store1, client=mock_client,
        )

        # Bootstrap
        bootstrap = detector1.initialize_store(
            seed_models_path=str(seed_yaml),
            legacy_json_path=str(tmp_path / "nonexistent.json"),
        )
        assert bootstrap["was_first_run"] is True
        assert bootstrap["seed_models_loaded"] == 1

        # API returns seed model + new model
        mock_client.fetch_org_models.return_value = [
            sample_record(
                model_id="org/baseline-model",
                created_at="2026-01-01T00:00:00+00:00",
            ),
            sample_record(
                model_id="org/new-day1",
                created_at="2026-04-25T10:00:00+00:00",
            ),
        ]
        result1 = detector1.run(now=now_day1)

        assert result1.total_new == 1
        assert result1.new_models[0]["model_id"] == "org/new-day1"

        # Mark processed + notified
        detector1.mark_models_processed(result1, mark_notified=True)
        stats1 = detector1.get_detection_stats()
        assert stats1["total_known"] == 2  # seed + new
        assert stats1["status_known"] == 2  # both processed
        assert stats1["unnotified"] == 0
        store1.close()

        # --- Day 2: Same models, nothing new ---
        store2 = ModelStore(db_path)
        detector2 = ModelDetector(
            ["org"], polling_interval_hours=24,
            store=store2, client=mock_client,
        )

        # Not first run anymore
        bootstrap2 = detector2.initialize_store(
            seed_models_path=str(seed_yaml),
            legacy_json_path=str(tmp_path / "nonexistent.json"),
        )
        assert bootstrap2["was_first_run"] is False

        mock_client.fetch_org_models.return_value = [
            sample_record(
                model_id="org/baseline-model",
                created_at="2026-01-01T00:00:00+00:00",
            ),
            sample_record(
                model_id="org/new-day1",
                created_at="2026-04-25T10:00:00+00:00",
            ),
        ]
        result2 = detector2.run(now=now_day2)
        assert result2.total_new == 0  # Nothing new
        store2.close()

        # --- Day 3: New model appears ---
        store3 = ModelStore(db_path)
        detector3 = ModelDetector(
            ["org"], polling_interval_hours=24,
            store=store3, client=mock_client,
        )

        mock_client.fetch_org_models.return_value = [
            sample_record(
                model_id="org/baseline-model",
                created_at="2026-01-01T00:00:00+00:00",
            ),
            sample_record(
                model_id="org/new-day1",
                created_at="2026-04-25T10:00:00+00:00",
            ),
            sample_record(
                model_id="org/new-day3",
                created_at="2026-04-27T08:00:00+00:00",
            ),
        ]
        result3 = detector3.run(now=now_day3)

        assert result3.total_new == 1
        assert result3.new_models[0]["model_id"] == "org/new-day3"

        # Verify poll history across all days
        history = store3.get_poll_history()
        assert len(history) >= 3  # at least bootstrap + 3 runs
        store3.close()

    def test_lifecycle_unnotified_tracking(
        self, tmp_store, mock_client, sample_record
    ):
        """Verify that unnotified tracking works correctly:
        detect → (model is unnotified) → mark notified → (model is notified)
        """
        now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)

        mock_client.fetch_org_models.return_value = [
            sample_record(
                model_id="org/model-a",
                created_at="2026-04-26T10:00:00+00:00",
            ),
            sample_record(
                model_id="org/model-b",
                created_at="2026-04-26T11:00:00+00:00",
            ),
        ]

        detector = ModelDetector(
            ["org"], store=tmp_store, client=mock_client,
        )
        result = detector.run(now=now)
        assert result.total_new == 2

        # Both are unnotified
        unnotified = tmp_store.get_unnotified_models()
        assert len(unnotified) == 2

        # Notify just model-a
        detector.mark_model_notified("org/model-a")
        unnotified = tmp_store.get_unnotified_models()
        assert len(unnotified) == 1
        assert unnotified[0]["model_id"] == "org/model-b"

        # Mark all processed with notified flag
        detector.mark_models_processed(result, mark_notified=True)
        unnotified = tmp_store.get_unnotified_models()
        assert len(unnotified) == 0
