"""Tests for hf_model_monitor.main — CLI entry point and run() function.

Covers:
- run() with the new ModelDetector-based pipeline
- First-run bootstrap integration
- Detection result handling
- Error propagation
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from hf_model_monitor.detector import DetectionResult, OrgPollResult
from hf_model_monitor.main import _get_reference_data, _notify_new_models, run
from hf_model_monitor.model_store import ModelStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_config():
    """Return a minimal config dict for testing."""
    return {
        "watched_organizations": ["meta-llama", "google"],
        "polling_interval_hours": 12,
        "slack_webhook_url": "",
        "run_on_startup": True,
        "trending_thresholds": {
            "enabled": False,
            "download_surge_count": 10000,
            "trending_score": 50,
            "time_window_hours": 24,
        },
    }


@pytest.fixture
def tmp_db_store(tmp_path):
    """Yield a ModelStore backed by a temporary SQLite file."""
    db_path = str(tmp_path / "test_main.db")
    store = ModelStore(db_path)
    yield store
    store.close()


# ---------------------------------------------------------------------------
# run() — new ModelDetector-based pipeline
# ---------------------------------------------------------------------------
class TestRun:
    @patch("hf_model_monitor.main.load_config")
    @patch("hf_model_monitor.main.ModelDetector")
    def test_full_pipeline_with_new_model(self, MockDetector, mock_load_config, mock_config):
        mock_load_config.return_value = mock_config

        # Set up the mock detector
        mock_detector_instance = MagicMock()
        MockDetector.from_config.return_value = mock_detector_instance
        mock_detector_instance.__enter__ = MagicMock(return_value=mock_detector_instance)
        mock_detector_instance.__exit__ = MagicMock(return_value=False)

        # Bootstrap: not first run
        mock_detector_instance.initialize_store.return_value = {
            "was_first_run": False,
            "seed_models_loaded": 0,
            "legacy_models_imported": 0,
            "total_known_after": 50,
        }

        # Detection: found 1 new model
        detection_result = DetectionResult(
            new_models=[{"model_id": "meta-llama/Llama-5"}],
            org_results=[
                OrgPollResult(
                    org="meta-llama",
                    models_fetched=10,
                    new_models=[{"model_id": "meta-llama/Llama-5"}],
                ),
            ],
            poll_timestamp="2026-04-26T12:00:00+00:00",
            polling_window_hours=12,
        )
        mock_detector_instance.run.return_value = detection_result
        mock_detector_instance.mark_models_processed.return_value = 1

        result = run()

        assert result["new_models_found"] == 1
        assert result["models_processed"] == 1
        assert result["errors"] == []
        mock_detector_instance.initialize_store.assert_called_once()
        mock_detector_instance.run.assert_called_once()
        mock_detector_instance.mark_models_processed.assert_called_once_with(detection_result)

    @patch("hf_model_monitor.main.load_config")
    @patch("hf_model_monitor.main.ModelDetector")
    def test_no_new_models(self, MockDetector, mock_load_config, mock_config):
        mock_load_config.return_value = mock_config

        mock_detector_instance = MagicMock()
        MockDetector.from_config.return_value = mock_detector_instance
        mock_detector_instance.__enter__ = MagicMock(return_value=mock_detector_instance)
        mock_detector_instance.__exit__ = MagicMock(return_value=False)

        mock_detector_instance.initialize_store.return_value = {
            "was_first_run": False,
            "seed_models_loaded": 0,
            "legacy_models_imported": 0,
            "total_known_after": 50,
        }

        detection_result = DetectionResult(
            poll_timestamp="2026-04-26T12:00:00+00:00",
        )
        mock_detector_instance.run.return_value = detection_result
        mock_detector_instance.mark_models_processed.return_value = 0

        result = run()

        assert result["new_models_found"] == 0
        assert result["models_processed"] == 0
        assert result["errors"] == []

    @patch("hf_model_monitor.main.load_config")
    @patch("hf_model_monitor.main.ModelDetector")
    def test_first_run_bootstrap(self, MockDetector, mock_load_config, mock_config):
        mock_load_config.return_value = mock_config

        mock_detector_instance = MagicMock()
        MockDetector.from_config.return_value = mock_detector_instance
        mock_detector_instance.__enter__ = MagicMock(return_value=mock_detector_instance)
        mock_detector_instance.__exit__ = MagicMock(return_value=False)

        # First run — bootstrap loads seed models
        mock_detector_instance.initialize_store.return_value = {
            "was_first_run": True,
            "seed_models_loaded": 42,
            "legacy_models_imported": 0,
            "total_known_after": 42,
        }

        detection_result = DetectionResult(
            poll_timestamp="2026-04-26T12:00:00+00:00",
        )
        mock_detector_instance.run.return_value = detection_result
        mock_detector_instance.mark_models_processed.return_value = 0

        result = run()

        assert result["bootstrap_stats"]["was_first_run"] is True
        assert result["bootstrap_stats"]["seed_models_loaded"] == 42

    @patch("hf_model_monitor.main.load_config")
    @patch("hf_model_monitor.main.ModelDetector")
    def test_detection_with_errors(self, MockDetector, mock_load_config, mock_config):
        mock_load_config.return_value = mock_config

        mock_detector_instance = MagicMock()
        MockDetector.from_config.return_value = mock_detector_instance
        mock_detector_instance.__enter__ = MagicMock(return_value=mock_detector_instance)
        mock_detector_instance.__exit__ = MagicMock(return_value=False)

        mock_detector_instance.initialize_store.return_value = {
            "was_first_run": False,
            "seed_models_loaded": 0,
            "legacy_models_imported": 0,
            "total_known_after": 50,
        }

        # Detection with partial failures
        detection_result = DetectionResult(
            new_models=[{"model_id": "google/Gemma-4"}],
            org_results=[
                OrgPollResult(org="meta-llama", error="API timeout"),
                OrgPollResult(
                    org="google",
                    models_fetched=5,
                    new_models=[{"model_id": "google/Gemma-4"}],
                ),
            ],
            poll_timestamp="2026-04-26T12:00:00+00:00",
        )
        mock_detector_instance.run.return_value = detection_result
        mock_detector_instance.mark_models_processed.return_value = 1

        result = run()

        assert result["new_models_found"] == 1
        assert len(result["errors"]) == 1
        assert "API timeout" in result["errors"][0]

    @patch("hf_model_monitor.main.load_config")
    @patch("hf_model_monitor.main.ModelDetector")
    def test_summary_included_in_result(self, MockDetector, mock_load_config, mock_config):
        mock_load_config.return_value = mock_config

        mock_detector_instance = MagicMock()
        MockDetector.from_config.return_value = mock_detector_instance
        mock_detector_instance.__enter__ = MagicMock(return_value=mock_detector_instance)
        mock_detector_instance.__exit__ = MagicMock(return_value=False)

        mock_detector_instance.initialize_store.return_value = {
            "was_first_run": False,
            "seed_models_loaded": 0,
            "legacy_models_imported": 0,
            "total_known_after": 10,
        }

        detection_result = DetectionResult(
            poll_timestamp="2026-04-26T12:00:00+00:00",
            polling_window_hours=12,
        )
        mock_detector_instance.run.return_value = detection_result
        mock_detector_instance.mark_models_processed.return_value = 0

        result = run()

        assert "detection_summary" in result
        assert isinstance(result["detection_summary"], str)
        assert "2026-04-26" in result["detection_summary"]

    @patch("hf_model_monitor.main.send_to_slack")
    @patch("hf_model_monitor.main.format_crawler_errors_summary")
    @patch("hf_model_monitor.main.collect_model_metadata")
    @patch("hf_model_monitor.main.SlackNotifier")
    @patch("hf_model_monitor.main.load_config")
    @patch("hf_model_monitor.main.ModelDetector")
    def test_run_with_slack_notifications(
        self,
        MockDetector,
        mock_load_config,
        MockSlackNotifier,
        mock_collect_metadata,
        mock_format_errors,
        mock_send_to_slack,
        mock_config,
    ):
        """run() sends Slack reports when slack_webhook_url is configured."""
        slack_config = dict(
            mock_config,
            slack_webhook_url="https://hooks.slack.com/services/T/B/x",
        )
        mock_load_config.return_value = slack_config

        # Set up mock detector
        mock_detector_instance = MagicMock()
        MockDetector.from_config.return_value = mock_detector_instance
        mock_detector_instance.__enter__ = MagicMock(return_value=mock_detector_instance)
        mock_detector_instance.__exit__ = MagicMock(return_value=False)

        mock_detector_instance.initialize_store.return_value = {
            "was_first_run": False,
            "seed_models_loaded": 0,
            "legacy_models_imported": 0,
            "total_known_after": 50,
        }

        # Detection found 2 new models
        detection_result = DetectionResult(
            new_models=[
                {"model_id": "meta-llama/Llama-5"},
                {"model_id": "google/Gemma-4"},
            ],
            org_results=[
                OrgPollResult(
                    org="meta-llama",
                    models_fetched=10,
                    new_models=[{"model_id": "meta-llama/Llama-5"}],
                ),
                OrgPollResult(
                    org="google",
                    models_fetched=5,
                    new_models=[{"model_id": "google/Gemma-4"}],
                ),
            ],
            poll_timestamp="2026-04-26T12:00:00+00:00",
            polling_window_hours=12,
        )
        mock_detector_instance.run.return_value = detection_result
        mock_detector_instance.mark_models_processed.return_value = 2

        # Set up mock notifier — configured and sends successfully
        mock_notifier_instance = MagicMock()
        mock_notifier_instance.is_configured = True
        mock_notifier_instance.send_report.return_value = True
        MockSlackNotifier.from_config.return_value = mock_notifier_instance

        # Metadata enrichment returns a fake dict per model
        mock_collect_metadata.return_value = {"basic": {"name": "test-model"}}

        result = run()

        # Notifier was created from config
        MockSlackNotifier.from_config.assert_called_once_with(slack_config)

        # Metadata collected for each new model
        assert mock_collect_metadata.call_count == 2

        # send_report called for each new model
        assert mock_notifier_instance.send_report.call_count == 2

        # Models marked as notified after successful send
        assert mock_detector_instance.mark_model_notified.call_count == 2

        # Result reflects notifications
        assert result["models_notified"] == 2
        assert result["new_models_found"] == 2
        assert result["models_processed"] == 2
        assert result["errors"] == []

    @patch("hf_model_monitor.main.send_to_slack")
    @patch("hf_model_monitor.main.format_crawler_errors_summary")
    @patch("hf_model_monitor.main.collect_model_metadata")
    @patch("hf_model_monitor.main.SlackNotifier")
    @patch("hf_model_monitor.main.load_config")
    @patch("hf_model_monitor.main.ModelDetector")
    def test_run_with_slack_sends_error_summary_on_partial_failure(
        self,
        MockDetector,
        mock_load_config,
        MockSlackNotifier,
        mock_collect_metadata,
        mock_format_errors,
        mock_send_to_slack,
        mock_config,
    ):
        """run() sends crawler error summary via Slack when some orgs fail."""
        slack_config = dict(
            mock_config,
            slack_webhook_url="https://hooks.slack.com/services/T/B/x",
        )
        mock_load_config.return_value = slack_config

        mock_detector_instance = MagicMock()
        MockDetector.from_config.return_value = mock_detector_instance
        mock_detector_instance.__enter__ = MagicMock(return_value=mock_detector_instance)
        mock_detector_instance.__exit__ = MagicMock(return_value=False)

        mock_detector_instance.initialize_store.return_value = {
            "was_first_run": False,
            "seed_models_loaded": 0,
            "legacy_models_imported": 0,
            "total_known_after": 50,
        }

        # Detection: one org succeeded with a new model, one org failed
        detection_result = DetectionResult(
            new_models=[{"model_id": "google/Gemma-4"}],
            org_results=[
                OrgPollResult(org="meta-llama", error="API timeout"),
                OrgPollResult(
                    org="google",
                    models_fetched=5,
                    new_models=[{"model_id": "google/Gemma-4"}],
                ),
            ],
            poll_timestamp="2026-04-26T12:00:00+00:00",
        )
        mock_detector_instance.run.return_value = detection_result
        mock_detector_instance.mark_models_processed.return_value = 1

        mock_notifier_instance = MagicMock()
        mock_notifier_instance.is_configured = True
        mock_notifier_instance.send_report.return_value = True
        MockSlackNotifier.from_config.return_value = mock_notifier_instance

        mock_collect_metadata.return_value = {"basic": {"name": "Gemma-4"}}
        mock_format_errors.return_value = "Error summary text"

        result = run()

        # Error summary was formatted and sent via Slack
        mock_format_errors.assert_called_once()
        call_kwargs = mock_format_errors.call_args
        assert call_kwargs[1]["total_orgs"] == 2
        mock_send_to_slack.assert_called_once_with(
            "Error summary text",
            webhook_url="https://hooks.slack.com/services/T/B/x",
        )

        # Model notification still went through
        assert result["models_notified"] == 1
        assert len(result["errors"]) == 1

    @patch("hf_model_monitor.main._get_reference_data")
    @patch("hf_model_monitor.main.collect_model_metadata")
    @patch("hf_model_monitor.main.TrendingDetector")
    @patch("hf_model_monitor.main.SlackNotifier")
    @patch("hf_model_monitor.main.ModelDetector")
    @patch("hf_model_monitor.main.load_config")
    def test_run_with_trending_enabled(
        self,
        mock_load_config,
        MockDetector,
        MockSlackNotifier,
        MockTrending,
        mock_collect_metadata,
        mock_ref_data,
    ):
        """When trending_thresholds.enabled is True, trending detection runs
        and candidates are counted in the result."""
        from hf_model_monitor.trending_detector import (
            TrendingCandidate,
            TrendingDetectionResult,
        )

        # Config with trending enabled and Slack configured
        mock_load_config.return_value = {
            "watched_organizations": ["meta-llama"],
            "polling_interval_hours": 12,
            "slack_webhook_url": "https://hooks.slack.com/test",
            "run_on_startup": True,
            "trending_thresholds": {
                "enabled": True,
                "download_surge_count": 10000,
                "trending_score": 50,
                "time_window_hours": 24,
            },
        }

        # --- Mock ModelDetector (standard pipeline, no new models) ---
        mock_detector_instance = MagicMock()
        MockDetector.from_config.return_value = mock_detector_instance
        mock_detector_instance.__enter__ = MagicMock(return_value=mock_detector_instance)
        mock_detector_instance.__exit__ = MagicMock(return_value=False)

        mock_detector_instance.initialize_store.return_value = {
            "was_first_run": False,
            "seed_models_loaded": 0,
            "legacy_models_imported": 0,
            "total_known_after": 50,
        }

        detection_result = DetectionResult(
            poll_timestamp="2026-04-26T12:00:00+00:00",
        )
        mock_detector_instance.run.return_value = detection_result
        mock_detector_instance.mark_models_processed.return_value = 0

        # --- Mock SlackNotifier (configured) ---
        mock_notifier = MagicMock()
        mock_notifier.is_configured = True
        mock_notifier.send_report.return_value = True
        MockSlackNotifier.from_config.return_value = mock_notifier

        # --- Mock TrendingDetector with 2 candidates ---
        mock_trending_instance = MagicMock()
        MockTrending.from_config.return_value = mock_trending_instance
        mock_trending_instance.__enter__ = MagicMock(return_value=mock_trending_instance)
        mock_trending_instance.__exit__ = MagicMock(return_value=False)

        candidate_1 = TrendingCandidate(
            model_id="indie-lab/hot-model",
            downloads=500_000,
            trending_score=85,
            trending_triggered=True,
        )
        candidate_2 = TrendingCandidate(
            model_id="startup/surging-model",
            downloads=1_200_000,
            download_delta=50_000,
            surge_triggered=True,
        )
        trending_result = TrendingDetectionResult(
            candidates=[candidate_1, candidate_2],
            trending_fetched=50,
            downloaded_fetched=50,
            enabled=True,
        )
        mock_trending_instance.run.return_value = trending_result

        # --- Mock metadata enrichment for trending candidates ---
        mock_ref_data.return_value = {"gpt-4": {"params": "1.8T"}}
        mock_collect_metadata.side_effect = lambda mid: {"model_id": mid, "params": "7B"}

        result = run()

        # Trending detection was invoked
        MockTrending.from_config.assert_called_once()
        mock_trending_instance.run.assert_called_once()

        # Both candidates counted
        assert result["trending_candidates"] == 2

        # Slack reports sent for each trending candidate
        assert mock_collect_metadata.call_count == 2
        assert mock_notifier.send_report.call_count == 2

        # No errors from trending detection
        assert not any("Trending detection" in e for e in result["errors"])

    @patch("hf_model_monitor.main.TrendingDetector")
    @patch("hf_model_monitor.main.SlackNotifier")
    @patch("hf_model_monitor.main.ModelDetector")
    @patch("hf_model_monitor.main.load_config")
    def test_run_trending_exception_captured(
        self,
        mock_load_config,
        MockDetector,
        MockSlackNotifier,
        MockTrending,
    ):
        """When TrendingDetector.run() raises, the error is captured in result
        and the pipeline does not crash."""
        mock_load_config.return_value = {
            "watched_organizations": ["meta-llama"],
            "polling_interval_hours": 12,
            "slack_webhook_url": "",
            "run_on_startup": True,
            "trending_thresholds": {
                "enabled": True,
                "download_surge_count": 10000,
                "trending_score": 50,
                "time_window_hours": 24,
            },
        }

        # Standard detector mock
        mock_detector_instance = MagicMock()
        MockDetector.from_config.return_value = mock_detector_instance
        mock_detector_instance.__enter__ = MagicMock(return_value=mock_detector_instance)
        mock_detector_instance.__exit__ = MagicMock(return_value=False)
        mock_detector_instance.initialize_store.return_value = {
            "was_first_run": False,
            "seed_models_loaded": 0,
            "legacy_models_imported": 0,
            "total_known_after": 50,
        }
        detection_result = DetectionResult(
            poll_timestamp="2026-04-26T12:00:00+00:00",
        )
        mock_detector_instance.run.return_value = detection_result
        mock_detector_instance.mark_models_processed.return_value = 0

        # Notifier (not configured)
        mock_notifier = MagicMock()
        mock_notifier.is_configured = False
        MockSlackNotifier.from_config.return_value = mock_notifier

        # Trending detector raises on run()
        mock_trending_instance = MagicMock()
        MockTrending.from_config.return_value = mock_trending_instance
        mock_trending_instance.__enter__ = MagicMock(return_value=mock_trending_instance)
        mock_trending_instance.__exit__ = MagicMock(return_value=False)
        mock_trending_instance.run.side_effect = RuntimeError("API unavailable")

        result = run()

        # Trending candidates stay at 0
        assert result["trending_candidates"] == 0
        # Error captured in result
        assert any("Trending detection" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# _notify_new_models — enrichment + Slack notification flow
# ---------------------------------------------------------------------------
class TestNotifyNewModels:
    """Tests for _notify_new_models() covering success, no webhook, and empty list."""

    @patch("hf_model_monitor.main._get_reference_data")
    @patch("hf_model_monitor.main.collect_model_metadata")
    def test_success_notifies_all_models(self, mock_collect, mock_ref_data):
        """When notifier is configured and models are provided, all are notified."""
        mock_ref_data.return_value = {"gpt-4": {"params": "1.8T"}}
        mock_collect.side_effect = lambda mid: {"model_id": mid, "params": "70B"}

        notifier = MagicMock()
        notifier.is_configured = True
        notifier.send_report.return_value = True

        detector = MagicMock()

        new_models = [
            {"model_id": "meta-llama/Llama-5"},
            {"model_id": "google/Gemma-4"},
        ]

        result = _notify_new_models(new_models, notifier, detector)

        assert result == 2
        assert mock_collect.call_count == 2
        assert notifier.send_report.call_count == 2
        detector.mark_model_notified.assert_any_call("meta-llama/Llama-5")
        detector.mark_model_notified.assert_any_call("google/Gemma-4")
        mock_ref_data.assert_called_once()

    def test_no_webhook_returns_zero(self):
        """When notifier is not configured, returns 0 without processing."""
        notifier = MagicMock()
        notifier.is_configured = False

        detector = MagicMock()
        new_models = [{"model_id": "meta-llama/Llama-5"}]

        result = _notify_new_models(new_models, notifier, detector)

        assert result == 0
        notifier.send_report.assert_not_called()
        detector.mark_model_notified.assert_not_called()

    def test_empty_list_returns_zero(self):
        """When new_models is empty, returns 0 without processing."""
        notifier = MagicMock()
        notifier.is_configured = True

        detector = MagicMock()

        result = _notify_new_models([], notifier, detector)

        assert result == 0
        notifier.send_report.assert_not_called()
        detector.mark_model_notified.assert_not_called()


# ---------------------------------------------------------------------------
# _get_reference_data() — seed data path and fallback
# ---------------------------------------------------------------------------
class TestGetReferenceData:
    @patch("hf_model_monitor.main.get_reference_data_from_seed")
    def test_returns_seed_data_when_available(self, mock_get_seed):
        """When seed data returns a non-empty dict, use it directly."""
        seed_refs = {
            "GPT-4o": {
                "params": "~1.8T (estimated)",
                "mmlu": "88.7",
                "humaneval": "90.2",
                "license": "Proprietary",
                "api_price": "$2.50",
                "context_window": "128K",
                "vram": "N/A (API only)",
            },
            "DeepSeek-V3": {
                "params": "671B",
                "mmlu": "87.1",
                "humaneval": "82.6",
                "license": "MIT",
                "api_price": "$0.27",
                "context_window": "128K",
                "vram": "~160GB FP8",
            },
        }
        mock_get_seed.return_value = seed_refs

        result = _get_reference_data()

        mock_get_seed.assert_called_once()
        assert result is seed_refs
        assert "GPT-4o" in result
        assert "DeepSeek-V3" in result

    @patch("hf_model_monitor.main.REFERENCE_MODELS", {"FallbackModel": {"params": "10B"}})
    @patch("hf_model_monitor.main.get_reference_data_from_seed")
    def test_falls_back_to_reference_models_when_seed_empty(self, mock_get_seed):
        """When seed data returns an empty dict, fall back to REFERENCE_MODELS."""
        mock_get_seed.return_value = {}

        result = _get_reference_data()

        mock_get_seed.assert_called_once()
        assert result == {"FallbackModel": {"params": "10B"}}

    @patch("hf_model_monitor.main.REFERENCE_MODELS", {"Fallback": {"params": "7B"}})
    @patch("hf_model_monitor.main.get_reference_data_from_seed")
    def test_falls_back_to_reference_models_when_seed_none(self, mock_get_seed):
        """When seed data returns None, fall back to REFERENCE_MODELS."""
        mock_get_seed.return_value = None

        result = _get_reference_data()

        mock_get_seed.assert_called_once()
        assert result == {"Fallback": {"params": "7B"}}
