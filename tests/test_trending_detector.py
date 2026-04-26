"""Tests for hf_model_monitor.trending_detector — trending/surge detection.

Covers:
- TrendingCandidate dataclass and reason generation
- TrendingDetectionResult aggregation and summary
- Download delta computation against stored snapshots
- Core-org filtering logic
- Full detection runs with mocked HFApiClient
- Threshold evaluation (surge, trending, both, neither)
- Merge/dedup of trending and downloaded sources
- Disabled state handling
- Error handling (API failures)
- Download snapshot persistence
- from_config factory method
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from hf_model_monitor.trending_detector import (
    TrendingCandidate,
    TrendingDetectionResult,
    TrendingDetector,
    compute_download_delta,
    is_core_org_model,
)
from hf_model_monitor.hf_client import HFApiClient, HFApiError, ModelRecord
from hf_model_monitor.model_store import ModelStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_store(tmp_path):
    """Yield a ModelStore backed by a temporary SQLite file."""
    db_path = str(tmp_path / "test_trending.db")
    store = ModelStore(db_path)
    yield store
    store.close()


@pytest.fixture
def mock_client():
    """Return a MagicMock HFApiClient."""
    client = MagicMock(spec=[
        "fetch_trending_models",
        "fetch_most_downloaded_models",
        "close",
    ])
    client.fetch_trending_models.return_value = []
    client.fetch_most_downloaded_models.return_value = []
    return client


@pytest.fixture
def sample_trending_model():
    """Factory for creating trending model dicts."""
    def _make(
        model_id="community-org/cool-model",
        author="community-org",
        downloads=50000,
        likes=200,
        trending_score=80,
        **kwargs,
    ):
        defaults = {
            "model_id": model_id,
            "author": author,
            "downloads": downloads,
            "likes": likes,
            "trending_score": trending_score,
            "pipeline_tag": "text-generation",
            "created_at": "2026-04-25T10:00:00+00:00",
            "last_modified": "2026-04-25T10:00:00+00:00",
            "tags": ["transformers"],
            "library_name": "transformers",
        }
        defaults.update(kwargs)
        return defaults
    return _make


@pytest.fixture
def sample_model_record():
    """Factory for creating ModelRecord instances (for downloaded endpoint)."""
    def _make(
        model_id="community-org/popular-model",
        author="community-org",
        downloads=100000,
        **kwargs,
    ):
        defaults = {
            "last_modified": "2026-04-25T10:00:00+00:00",
            "created_at": "2026-04-25T10:00:00+00:00",
            "pipeline_tag": "text-generation",
            "tags": ["transformers"],
            "likes": 300,
            "library_name": "transformers",
        }
        defaults.update(kwargs)
        return ModelRecord(
            model_id=model_id,
            author=author,
            downloads=downloads,
            **defaults,
        )
    return _make


WATCHED_ORGS = ["meta-llama", "google", "mistralai", "Qwen", "openai"]


# ---------------------------------------------------------------------------
# TrendingCandidate
# ---------------------------------------------------------------------------
class TestTrendingCandidate:
    def test_reason_surge_only(self):
        c = TrendingCandidate(
            model_id="org/m",
            download_delta=15000,
            surge_triggered=True,
            trending_triggered=False,
        )
        assert "download surge" in c.reason
        assert "15,000" in c.reason

    def test_reason_trending_only(self):
        c = TrendingCandidate(
            model_id="org/m",
            trending_score=95,
            surge_triggered=False,
            trending_triggered=True,
        )
        assert "trending score" in c.reason
        assert "95" in c.reason

    def test_reason_both_triggers(self):
        c = TrendingCandidate(
            model_id="org/m",
            download_delta=20000,
            trending_score=95,
            surge_triggered=True,
            trending_triggered=True,
        )
        assert "download surge" in c.reason
        assert "trending score" in c.reason

    def test_reason_no_triggers(self):
        c = TrendingCandidate(model_id="org/m")
        assert c.reason == "unknown"

    def test_to_dict_contains_all_fields(self):
        c = TrendingCandidate(
            model_id="org/m",
            author="org",
            downloads=5000,
            likes=100,
            trending_score=80,
            download_delta=3000,
            surge_triggered=False,
            trending_triggered=True,
            pipeline_tag="text-generation",
            tags=["transformers"],
        )
        d = c.to_dict()
        assert d["model_id"] == "org/m"
        assert d["author"] == "org"
        assert d["downloads"] == 5000
        assert d["trending_score"] == 80
        assert d["download_delta"] == 3000
        assert d["surge_triggered"] is False
        assert d["trending_triggered"] is True
        assert "reason" in d

    def test_to_dict_tags_is_copy(self):
        original_tags = ["a", "b"]
        c = TrendingCandidate(model_id="org/m", tags=original_tags)
        d = c.to_dict()
        d["tags"].append("c")
        assert len(original_tags) == 2  # Not mutated


# ---------------------------------------------------------------------------
# TrendingDetectionResult
# ---------------------------------------------------------------------------
class TestTrendingDetectionResult:
    def test_empty_result_defaults(self):
        r = TrendingDetectionResult()
        assert r.total_candidates == 0
        assert r.surge_candidates == []
        assert r.trending_candidates == []
        assert r.errors == []

    def test_candidate_counts(self):
        r = TrendingDetectionResult(
            candidates=[
                TrendingCandidate(model_id="a", surge_triggered=True),
                TrendingCandidate(model_id="b", trending_triggered=True),
                TrendingCandidate(model_id="c", surge_triggered=True, trending_triggered=True),
            ],
        )
        assert r.total_candidates == 3
        assert len(r.surge_candidates) == 2
        assert len(r.trending_candidates) == 2

    def test_summary_when_disabled(self):
        r = TrendingDetectionResult(enabled=False)
        assert "disabled" in r.summary().lower()

    def test_summary_with_candidates(self):
        r = TrendingDetectionResult(
            poll_timestamp="2026-04-26T12:00:00+00:00",
            trending_fetched=50,
            downloaded_fetched=50,
            filtered_core_org=5,
            below_threshold=40,
            candidates=[
                TrendingCandidate(
                    model_id="org/hot-model",
                    trending_score=90,
                    trending_triggered=True,
                ),
            ],
            enabled=True,
        )
        summary = r.summary()
        assert "2026-04-26" in summary
        assert "50" in summary
        assert "org/hot-model" in summary
        assert "Candidates" in summary

    def test_summary_with_errors(self):
        r = TrendingDetectionResult(
            poll_timestamp="2026-04-26T12:00:00+00:00",
            errors=["API timeout"],
            enabled=True,
        )
        summary = r.summary()
        assert "API timeout" in summary
        assert "Errors" in summary


# ---------------------------------------------------------------------------
# compute_download_delta
# ---------------------------------------------------------------------------
class TestComputeDownloadDelta:
    def test_returns_zero_for_unknown_model(self, tmp_store):
        """First observation has no previous snapshot → delta is 0."""
        delta = compute_download_delta("org/new-model", 50000, tmp_store)
        assert delta == 0

    def test_computes_positive_delta(self, tmp_store):
        """Downloads increased since last snapshot."""
        # Seed a model with 10,000 downloads
        tmp_store.upsert_model({
            "model_id": "org/model",
            "author": "org",
            "downloads": 10000,
        })
        delta = compute_download_delta("org/model", 25000, tmp_store)
        assert delta == 15000

    def test_returns_zero_for_no_change(self, tmp_store):
        """Same download count → delta is 0."""
        tmp_store.upsert_model({
            "model_id": "org/model",
            "author": "org",
            "downloads": 10000,
        })
        delta = compute_download_delta("org/model", 10000, tmp_store)
        assert delta == 0

    def test_returns_zero_for_decrease(self, tmp_store):
        """Downloads decreased (data inconsistency) → delta clamped to 0."""
        tmp_store.upsert_model({
            "model_id": "org/model",
            "author": "org",
            "downloads": 20000,
        })
        delta = compute_download_delta("org/model", 15000, tmp_store)
        assert delta == 0

    def test_large_surge(self, tmp_store):
        """Large download surge correctly computed."""
        tmp_store.upsert_model({
            "model_id": "org/viral",
            "author": "org",
            "downloads": 100000,
        })
        delta = compute_download_delta("org/viral", 500000, tmp_store)
        assert delta == 400000


# ---------------------------------------------------------------------------
# is_core_org_model
# ---------------------------------------------------------------------------
class TestIsCoreOrgModel:
    def test_matches_core_org(self):
        assert is_core_org_model("meta-llama/Llama-5", WATCHED_ORGS) is True

    def test_matches_case_insensitive(self):
        assert is_core_org_model("Meta-Llama/Llama-5", WATCHED_ORGS) is True
        assert is_core_org_model("GOOGLE/gemma-3", WATCHED_ORGS) is True

    def test_non_core_org(self):
        assert is_core_org_model("community-org/model", WATCHED_ORGS) is False

    def test_no_slash_returns_false(self):
        """Model IDs without org prefix can't be matched."""
        assert is_core_org_model("standalone-model", WATCHED_ORGS) is False

    def test_empty_model_id(self):
        assert is_core_org_model("", WATCHED_ORGS) is False

    def test_empty_watched_orgs(self):
        assert is_core_org_model("meta-llama/Llama-5", []) is False

    def test_qwen_case_sensitivity(self):
        """Qwen uses capital Q in the org list."""
        assert is_core_org_model("Qwen/Qwen2.5-72B", WATCHED_ORGS) is True
        assert is_core_org_model("qwen/Qwen2.5-72B", WATCHED_ORGS) is True


# ---------------------------------------------------------------------------
# TrendingDetector — from_config
# ---------------------------------------------------------------------------
class TestTrendingDetectorFromConfig:
    def test_creates_from_full_config(self, tmp_store, mock_client):
        config = {
            "watched_organizations": ["meta-llama", "google"],
            "trending_thresholds": {
                "enabled": True,
                "download_surge_count": 20000,
                "trending_score": 75,
                "time_window_hours": 48,
            },
        }
        detector = TrendingDetector.from_config(
            config, store=tmp_store, client=mock_client
        )
        assert detector.watched_orgs == ["meta-llama", "google"]
        assert detector.download_surge_threshold == 20000
        assert detector.trending_score_threshold == 75
        assert detector.time_window_hours == 48
        assert detector.enabled is True

    def test_defaults_when_thresholds_missing(self, tmp_store, mock_client):
        config = {"watched_organizations": ["org"]}
        detector = TrendingDetector.from_config(
            config, store=tmp_store, client=mock_client
        )
        assert detector.download_surge_threshold == 10000
        assert detector.trending_score_threshold == 50
        assert detector.time_window_hours == 24
        assert detector.enabled is False

    def test_defaults_when_config_empty(self, tmp_store, mock_client):
        detector = TrendingDetector.from_config(
            {}, store=tmp_store, client=mock_client
        )
        assert detector.watched_orgs == []
        assert detector.enabled is False


# ---------------------------------------------------------------------------
# TrendingDetector — disabled state
# ---------------------------------------------------------------------------
class TestTrendingDetectorDisabled:
    def test_disabled_returns_immediately(self, tmp_store, mock_client):
        detector = TrendingDetector(
            WATCHED_ORGS, enabled=False,
            store=tmp_store, client=mock_client,
        )
        result = detector.run()

        assert result.enabled is False
        assert result.total_candidates == 0
        assert "disabled" in result.summary().lower()
        mock_client.fetch_trending_models.assert_not_called()
        mock_client.fetch_most_downloaded_models.assert_not_called()


# ---------------------------------------------------------------------------
# TrendingDetector — trending score detection
# ---------------------------------------------------------------------------
class TestTrendingScoreDetection:
    def test_model_above_trending_threshold(
        self, tmp_store, mock_client, sample_trending_model
    ):
        """Model with high trending score is flagged."""
        mock_client.fetch_trending_models.return_value = [
            sample_trending_model(
                model_id="new-org/hot-model",
                author="new-org",
                trending_score=90,
            ),
        ]

        detector = TrendingDetector(
            WATCHED_ORGS, enabled=True,
            trending_score_threshold=50,
            store=tmp_store, client=mock_client,
        )
        result = detector.run()

        assert result.total_candidates == 1
        assert result.candidates[0].model_id == "new-org/hot-model"
        assert result.candidates[0].trending_triggered is True
        assert result.candidates[0].trending_score == 90

    def test_model_below_trending_threshold(
        self, tmp_store, mock_client, sample_trending_model
    ):
        """Model with low trending score is NOT flagged."""
        mock_client.fetch_trending_models.return_value = [
            sample_trending_model(
                model_id="new-org/meh-model",
                author="new-org",
                trending_score=10,
            ),
        ]

        detector = TrendingDetector(
            WATCHED_ORGS, enabled=True,
            trending_score_threshold=50,
            store=tmp_store, client=mock_client,
        )
        result = detector.run()

        assert result.total_candidates == 0
        assert result.below_threshold == 1

    def test_model_exactly_at_trending_threshold(
        self, tmp_store, mock_client, sample_trending_model
    ):
        """Model at exact threshold boundary IS flagged (>=)."""
        mock_client.fetch_trending_models.return_value = [
            sample_trending_model(
                model_id="new-org/borderline",
                author="new-org",
                trending_score=50,
            ),
        ]

        detector = TrendingDetector(
            WATCHED_ORGS, enabled=True,
            trending_score_threshold=50,
            store=tmp_store, client=mock_client,
        )
        result = detector.run()

        assert result.total_candidates == 1
        assert result.candidates[0].trending_triggered is True


# ---------------------------------------------------------------------------
# TrendingDetector — download surge detection
# ---------------------------------------------------------------------------
class TestDownloadSurgeDetection:
    def test_surge_detected_on_second_run(
        self, tmp_store, mock_client, sample_trending_model
    ):
        """Download surge is detected when delta exceeds threshold."""
        model = sample_trending_model(
            model_id="indie/viral-model",
            author="indie",
            downloads=100000,
            trending_score=10,  # Below trending threshold
        )

        # Seed with lower downloads
        tmp_store.upsert_model({
            "model_id": "indie/viral-model",
            "author": "indie",
            "downloads": 80000,
        })

        mock_client.fetch_trending_models.return_value = [model]

        detector = TrendingDetector(
            WATCHED_ORGS, enabled=True,
            download_surge_threshold=10000,
            trending_score_threshold=50,
            store=tmp_store, client=mock_client,
        )
        result = detector.run()

        assert result.total_candidates == 1
        c = result.candidates[0]
        assert c.model_id == "indie/viral-model"
        assert c.surge_triggered is True
        assert c.download_delta == 20000
        assert c.trending_triggered is False

    def test_no_surge_on_first_observation(
        self, tmp_store, mock_client, sample_trending_model
    ):
        """First time seeing a model — no delta to compute, no surge."""
        model = sample_trending_model(
            model_id="indie/new-model",
            author="indie",
            downloads=50000,
            trending_score=10,  # Below trending threshold too
        )

        mock_client.fetch_trending_models.return_value = [model]

        detector = TrendingDetector(
            WATCHED_ORGS, enabled=True,
            download_surge_threshold=10000,
            trending_score_threshold=50,
            store=tmp_store, client=mock_client,
        )
        result = detector.run()

        # No surge (first observation) and no trending → no candidates
        assert result.total_candidates == 0
        assert result.below_threshold == 1

    def test_surge_below_threshold_not_flagged(
        self, tmp_store, mock_client, sample_trending_model
    ):
        """Small download increase does not trigger surge."""
        tmp_store.upsert_model({
            "model_id": "indie/steady-model",
            "author": "indie",
            "downloads": 45000,
        })

        model = sample_trending_model(
            model_id="indie/steady-model",
            author="indie",
            downloads=50000,  # +5000, below 10000 threshold
            trending_score=10,
        )

        mock_client.fetch_trending_models.return_value = [model]

        detector = TrendingDetector(
            WATCHED_ORGS, enabled=True,
            download_surge_threshold=10000,
            trending_score_threshold=50,
            store=tmp_store, client=mock_client,
        )
        result = detector.run()

        assert result.total_candidates == 0

    def test_surge_exactly_at_threshold(
        self, tmp_store, mock_client, sample_trending_model
    ):
        """Surge at exact threshold boundary IS flagged (>=)."""
        tmp_store.upsert_model({
            "model_id": "indie/edge-model",
            "author": "indie",
            "downloads": 40000,
        })

        model = sample_trending_model(
            model_id="indie/edge-model",
            author="indie",
            downloads=50000,  # +10000, exactly at threshold
            trending_score=10,
        )

        mock_client.fetch_trending_models.return_value = [model]

        detector = TrendingDetector(
            WATCHED_ORGS, enabled=True,
            download_surge_threshold=10000,
            trending_score_threshold=50,
            store=tmp_store, client=mock_client,
        )
        result = detector.run()

        assert result.total_candidates == 1
        assert result.candidates[0].surge_triggered is True


# ---------------------------------------------------------------------------
# TrendingDetector — both triggers
# ---------------------------------------------------------------------------
class TestBothTriggers:
    def test_model_with_surge_and_trending(
        self, tmp_store, mock_client, sample_trending_model
    ):
        """Model can be flagged by both surge AND trending simultaneously."""
        tmp_store.upsert_model({
            "model_id": "indie/mega-hit",
            "author": "indie",
            "downloads": 50000,
        })

        model = sample_trending_model(
            model_id="indie/mega-hit",
            author="indie",
            downloads=100000,  # +50000 surge
            trending_score=95,  # High trending
        )

        mock_client.fetch_trending_models.return_value = [model]

        detector = TrendingDetector(
            WATCHED_ORGS, enabled=True,
            download_surge_threshold=10000,
            trending_score_threshold=50,
            store=tmp_store, client=mock_client,
        )
        result = detector.run()

        assert result.total_candidates == 1
        c = result.candidates[0]
        assert c.surge_triggered is True
        assert c.trending_triggered is True
        assert "download surge" in c.reason
        assert "trending score" in c.reason


# ---------------------------------------------------------------------------
# TrendingDetector — core org filtering
# ---------------------------------------------------------------------------
class TestCoreOrgFiltering:
    def test_core_org_model_filtered_out(
        self, tmp_store, mock_client, sample_trending_model
    ):
        """Models from watched orgs are excluded from trending results."""
        mock_client.fetch_trending_models.return_value = [
            sample_trending_model(
                model_id="meta-llama/Llama-5",
                author="meta-llama",
                trending_score=99,
            ),
            sample_trending_model(
                model_id="indie-lab/cool-model",
                author="indie-lab",
                trending_score=80,
            ),
        ]

        detector = TrendingDetector(
            WATCHED_ORGS, enabled=True,
            trending_score_threshold=50,
            store=tmp_store, client=mock_client,
        )
        result = detector.run()

        assert result.total_candidates == 1
        assert result.candidates[0].model_id == "indie-lab/cool-model"
        assert result.filtered_core_org == 1

    def test_multiple_core_org_models_filtered(
        self, tmp_store, mock_client, sample_trending_model
    ):
        """All core-org models are filtered regardless of how many."""
        mock_client.fetch_trending_models.return_value = [
            sample_trending_model(
                model_id="meta-llama/Model-A", author="meta-llama",
                trending_score=90,
            ),
            sample_trending_model(
                model_id="google/Model-B", author="google",
                trending_score=85,
            ),
            sample_trending_model(
                model_id="Qwen/Model-C", author="Qwen",
                trending_score=80,
            ),
            sample_trending_model(
                model_id="indie/Model-D", author="indie",
                trending_score=75,
            ),
        ]

        detector = TrendingDetector(
            WATCHED_ORGS, enabled=True,
            trending_score_threshold=50,
            store=tmp_store, client=mock_client,
        )
        result = detector.run()

        assert result.filtered_core_org == 3
        assert result.total_candidates == 1
        assert result.candidates[0].model_id == "indie/Model-D"

    def test_empty_watched_orgs_no_filtering(
        self, tmp_store, mock_client, sample_trending_model
    ):
        """With no watched orgs, no models are filtered."""
        mock_client.fetch_trending_models.return_value = [
            sample_trending_model(
                model_id="meta-llama/Llama-5",
                author="meta-llama",
                trending_score=90,
            ),
        ]

        detector = TrendingDetector(
            [], enabled=True,
            trending_score_threshold=50,
            store=tmp_store, client=mock_client,
        )
        result = detector.run()

        assert result.filtered_core_org == 0
        assert result.total_candidates == 1


# ---------------------------------------------------------------------------
# TrendingDetector — merge/dedup
# ---------------------------------------------------------------------------
class TestMergeSources:
    def test_trending_takes_precedence_on_overlap(
        self, tmp_store, mock_client, sample_trending_model, sample_model_record
    ):
        """When a model appears in both sources, trending version wins
        (it has the trending_score)."""
        mock_client.fetch_trending_models.return_value = [
            sample_trending_model(
                model_id="indie/overlap-model",
                author="indie",
                downloads=50000,
                trending_score=80,
            ),
        ]
        mock_client.fetch_most_downloaded_models.return_value = [
            sample_model_record(
                model_id="indie/overlap-model",
                author="indie",
                downloads=50000,
            ),
        ]

        detector = TrendingDetector(
            WATCHED_ORGS, enabled=True,
            trending_score_threshold=50,
            store=tmp_store, client=mock_client,
        )
        result = detector.run()

        # Should have trending_score from the trending source
        assert result.total_candidates == 1
        assert result.candidates[0].trending_score == 80
        assert result.candidates[0].trending_triggered is True

    def test_downloaded_only_models_included(
        self, tmp_store, mock_client, sample_model_record
    ):
        """Models only in the downloaded source are still evaluated."""
        # Seed with lower downloads for surge detection
        tmp_store.upsert_model({
            "model_id": "indie/download-only",
            "author": "indie",
            "downloads": 80000,
        })

        mock_client.fetch_most_downloaded_models.return_value = [
            sample_model_record(
                model_id="indie/download-only",
                author="indie",
                downloads=100000,
            ),
        ]

        detector = TrendingDetector(
            WATCHED_ORGS, enabled=True,
            download_surge_threshold=10000,
            trending_score_threshold=50,
            store=tmp_store, client=mock_client,
        )
        result = detector.run()

        assert result.total_candidates == 1
        assert result.candidates[0].model_id == "indie/download-only"
        assert result.candidates[0].surge_triggered is True

    def test_no_duplicates_in_results(
        self, tmp_store, mock_client, sample_trending_model, sample_model_record
    ):
        """Same model in both sources should appear only once in candidates."""
        model_id = "indie/same-model"
        mock_client.fetch_trending_models.return_value = [
            sample_trending_model(model_id=model_id, author="indie", trending_score=80),
        ]
        mock_client.fetch_most_downloaded_models.return_value = [
            sample_model_record(model_id=model_id, author="indie", downloads=50000),
        ]

        detector = TrendingDetector(
            WATCHED_ORGS, enabled=True,
            trending_score_threshold=50,
            store=tmp_store, client=mock_client,
        )
        result = detector.run()

        candidate_ids = [c.model_id for c in result.candidates]
        assert candidate_ids.count(model_id) == 1


# ---------------------------------------------------------------------------
# TrendingDetector — error handling
# ---------------------------------------------------------------------------
class TestTrendingDetectorErrors:
    def test_trending_api_failure_continues(
        self, tmp_store, mock_client, sample_model_record
    ):
        """If trending fetch fails, downloaded source still works."""
        mock_client.fetch_trending_models.side_effect = HFApiError(
            "Server error", status_code=500
        )

        # Seed for surge detection from downloads
        tmp_store.upsert_model({
            "model_id": "indie/model",
            "author": "indie",
            "downloads": 80000,
        })
        mock_client.fetch_most_downloaded_models.return_value = [
            sample_model_record(
                model_id="indie/model", author="indie", downloads=100000
            ),
        ]

        detector = TrendingDetector(
            WATCHED_ORGS, enabled=True,
            download_surge_threshold=10000,
            store=tmp_store, client=mock_client,
        )
        result = detector.run()

        assert len(result.errors) == 1
        assert "trending" in result.errors[0].lower()
        # Download source still produced a candidate
        assert result.total_candidates == 1

    def test_downloaded_api_failure_continues(
        self, tmp_store, mock_client, sample_trending_model
    ):
        """If downloaded fetch fails, trending source still works."""
        mock_client.fetch_trending_models.return_value = [
            sample_trending_model(
                model_id="indie/hot", author="indie", trending_score=90
            ),
        ]
        mock_client.fetch_most_downloaded_models.side_effect = HFApiError(
            "timeout", status_code=500
        )

        detector = TrendingDetector(
            WATCHED_ORGS, enabled=True,
            trending_score_threshold=50,
            store=tmp_store, client=mock_client,
        )
        result = detector.run()

        assert len(result.errors) == 1
        assert result.total_candidates == 1

    def test_both_sources_fail_returns_empty(self, tmp_store, mock_client):
        """If both sources fail, result is empty with errors."""
        mock_client.fetch_trending_models.side_effect = HFApiError(
            "error1", status_code=500
        )
        mock_client.fetch_most_downloaded_models.side_effect = HFApiError(
            "error2", status_code=500
        )

        detector = TrendingDetector(
            WATCHED_ORGS, enabled=True,
            store=tmp_store, client=mock_client,
        )
        result = detector.run()

        assert result.total_candidates == 0
        assert len(result.errors) == 2


# ---------------------------------------------------------------------------
# TrendingDetector — download snapshot persistence
# ---------------------------------------------------------------------------
class TestDownloadSnapshot:
    def test_snapshots_saved_after_run(
        self, tmp_store, mock_client, sample_trending_model
    ):
        """After a run, download counts are saved for future delta computation."""
        mock_client.fetch_trending_models.return_value = [
            sample_trending_model(
                model_id="indie/snapshot-test",
                author="indie",
                downloads=50000,
                trending_score=10,  # Below threshold
            ),
        ]

        detector = TrendingDetector(
            WATCHED_ORGS, enabled=True,
            trending_score_threshold=50,
            download_surge_threshold=10000,
            store=tmp_store, client=mock_client,
        )
        detector.run()

        # Model should now be in the store with correct downloads
        stored = tmp_store.get_model("indie/snapshot-test")
        assert stored is not None
        assert stored["downloads"] == 50000

    def test_two_runs_detect_surge(
        self, tmp_store, mock_client, sample_trending_model
    ):
        """Two consecutive runs: first snapshots, second detects surge."""
        # Run 1: Establish baseline
        mock_client.fetch_trending_models.return_value = [
            sample_trending_model(
                model_id="indie/growing",
                author="indie",
                downloads=50000,
                trending_score=10,
            ),
        ]

        detector = TrendingDetector(
            WATCHED_ORGS, enabled=True,
            download_surge_threshold=10000,
            trending_score_threshold=50,
            store=tmp_store, client=mock_client,
        )
        result1 = detector.run()
        assert result1.total_candidates == 0  # First run, no delta

        # Run 2: Downloads surged
        mock_client.fetch_trending_models.return_value = [
            sample_trending_model(
                model_id="indie/growing",
                author="indie",
                downloads=75000,  # +25000 since last run
                trending_score=10,
            ),
        ]

        result2 = detector.run()
        assert result2.total_candidates == 1
        assert result2.candidates[0].surge_triggered is True
        assert result2.candidates[0].download_delta == 25000


# ---------------------------------------------------------------------------
# TrendingDetector — poll history recording
# ---------------------------------------------------------------------------
class TestPollHistoryRecording:
    def test_records_poll_history(
        self, tmp_store, mock_client, sample_trending_model
    ):
        mock_client.fetch_trending_models.return_value = [
            sample_trending_model(
                model_id="indie/tracked",
                author="indie",
                trending_score=80,
            ),
        ]

        detector = TrendingDetector(
            WATCHED_ORGS, enabled=True,
            trending_score_threshold=50,
            store=tmp_store, client=mock_client,
        )
        detector.run()

        history = tmp_store.get_poll_history()
        trending_polls = [
            h for h in history if h["source"] == "trending_detection"
        ]
        assert len(trending_polls) == 1
        assert trending_polls[0]["new_models"] == 1

    def test_records_errors_in_poll_history(self, tmp_store, mock_client):
        mock_client.fetch_trending_models.side_effect = HFApiError(
            "fail", status_code=500
        )

        detector = TrendingDetector(
            WATCHED_ORGS, enabled=True,
            store=tmp_store, client=mock_client,
        )
        detector.run()

        history = tmp_store.get_poll_history()
        trending_polls = [
            h for h in history if h["source"] == "trending_detection"
        ]
        assert len(trending_polls) == 1
        assert trending_polls[0]["errors"]  # Non-empty error string


# ---------------------------------------------------------------------------
# TrendingDetector — context manager
# ---------------------------------------------------------------------------
class TestTrendingDetectorContextManager:
    def test_context_manager_closes_resources(self, tmp_store, mock_client):
        with TrendingDetector(
            WATCHED_ORGS, enabled=False,
            store=tmp_store, client=mock_client,
        ) as detector:
            result = detector.run()
            assert not result.enabled
        mock_client.close.assert_called_once()


# ---------------------------------------------------------------------------
# Integration: realistic multi-model scenario
# ---------------------------------------------------------------------------
class TestTrendingIntegration:
    def test_realistic_mixed_scenario(
        self, tmp_store, mock_client, sample_trending_model, sample_model_record
    ):
        """Simulate a realistic run with a mix of:
        - Core org models (filtered)
        - Trending models (some above, some below threshold)
        - Download surges
        """
        # Pre-seed a model for surge detection
        tmp_store.upsert_model({
            "model_id": "indie-lab/sleeper-hit",
            "author": "indie-lab",
            "downloads": 30000,
        })

        mock_client.fetch_trending_models.return_value = [
            # Core org — should be filtered
            sample_trending_model(
                model_id="meta-llama/Llama-5",
                author="meta-llama",
                trending_score=99,
            ),
            # High trending, non-core — should be flagged
            sample_trending_model(
                model_id="new-lab/innovator",
                author="new-lab",
                trending_score=85,
                downloads=20000,
            ),
            # Low trending, non-core — below threshold
            sample_trending_model(
                model_id="hobbyist/tiny-model",
                author="hobbyist",
                trending_score=5,
                downloads=100,
            ),
            # Download surge model — also in trending but low score
            sample_trending_model(
                model_id="indie-lab/sleeper-hit",
                author="indie-lab",
                trending_score=15,
                downloads=60000,  # +30000 surge
            ),
        ]

        mock_client.fetch_most_downloaded_models.return_value = [
            # Core org — will be deduped or filtered
            sample_model_record(
                model_id="google/Gemma-4",
                author="google",
                downloads=500000,
            ),
            # New model only in downloads (not in trending)
            sample_model_record(
                model_id="enterprise/big-model",
                author="enterprise",
                downloads=200000,
            ),
        ]

        detector = TrendingDetector(
            WATCHED_ORGS, enabled=True,
            download_surge_threshold=10000,
            trending_score_threshold=50,
            store=tmp_store, client=mock_client,
        )
        result = detector.run()

        # Verify filtering
        assert result.filtered_core_org >= 1  # At least meta-llama filtered

        # Verify candidates
        candidate_ids = {c.model_id for c in result.candidates}

        # new-lab/innovator: trending_score=85 ≥ 50 → flagged
        assert "new-lab/innovator" in candidate_ids

        # indie-lab/sleeper-hit: +30000 surge ≥ 10000 → flagged
        assert "indie-lab/sleeper-hit" in candidate_ids

        # hobbyist/tiny-model: trending=5, no surge → NOT flagged
        assert "hobbyist/tiny-model" not in candidate_ids

        # meta-llama/Llama-5: core org → filtered
        assert "meta-llama/Llama-5" not in candidate_ids

        # Check the sleeper hit's specifics
        sleeper = next(
            c for c in result.candidates if c.model_id == "indie-lab/sleeper-hit"
        )
        assert sleeper.surge_triggered is True
        assert sleeper.download_delta == 30000

        # Check summary is readable
        summary = result.summary()
        assert "new-lab/innovator" in summary
        assert "indie-lab/sleeper-hit" in summary
