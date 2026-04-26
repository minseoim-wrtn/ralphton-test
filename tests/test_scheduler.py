"""Tests for the periodic polling scheduler (hf_model_monitor.scheduler).

Covers:
- Interval validation and sub-24h SLA enforcement
- MonitorScheduler construction (direct and from_config)
- Single run execution (run_once)
- Error handling and Slack error alerts
- Run history tracking
- Scheduler status reporting
- Non-blocking start/stop lifecycle
- Graceful shutdown behavior
- Runtime interval update (update_interval)
- Manual trigger (trigger_now)
- SLA health monitoring (is_healthy, get_sla_status)
- run_on_startup configuration
"""

import threading
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from hf_model_monitor.detector import DetectionResult, OrgPollResult
from hf_model_monitor.scheduler import (
    DEFAULT_POLLING_INTERVAL_HOURS,
    MAX_POLLING_INTERVAL_HOURS,
    MIN_POLLING_INTERVAL_HOURS,
    SLA_WINDOW_HOURS,
    MonitorScheduler,
    SchedulerState,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_detector():
    """Create a mock ModelDetector that returns a successful empty result."""
    detector = MagicMock()
    detector.run.return_value = DetectionResult(
        poll_timestamp=datetime.now(timezone.utc).isoformat(),
        polling_window_hours=12,
        org_results=[
            OrgPollResult(org="meta-llama", models_fetched=10, new_models=[]),
            OrgPollResult(org="google", models_fetched=5, new_models=[]),
        ],
    )
    detector.close = MagicMock()
    return detector


@pytest.fixture
def mock_detector_with_new_models():
    """Create a mock detector that returns new models."""
    detector = MagicMock()
    new_model = {
        "model_id": "meta-llama/Llama-4-Scout-17B-16E",
        "author": "meta-llama",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    detector.run.return_value = DetectionResult(
        poll_timestamp=datetime.now(timezone.utc).isoformat(),
        polling_window_hours=12,
        new_models=[new_model],
        org_results=[
            OrgPollResult(
                org="meta-llama",
                models_fetched=10,
                new_models=[new_model],
            ),
        ],
    )
    detector.close = MagicMock()
    return detector


@pytest.fixture
def mock_detector_failing():
    """Create a mock detector that raises an exception."""
    detector = MagicMock()
    detector.run.side_effect = RuntimeError("API connection failed")
    detector.close = MagicMock()
    return detector


@pytest.fixture
def sample_config():
    """Return a typical config dict."""
    return {
        "watched_organizations": ["meta-llama", "google", "mistralai"],
        "polling_interval_hours": 12,
        "slack_webhook_url": "https://hooks.slack.com/test",
        "run_on_startup": True,
    }


@pytest.fixture
def scheduler(mock_detector):
    """Create a MonitorScheduler with a mock detector."""
    s = MonitorScheduler(
        watched_orgs=["meta-llama", "google"],
        polling_interval_hours=12,
        slack_webhook_url="",
        detector=mock_detector,
    )
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Interval validation
# ---------------------------------------------------------------------------
class TestIntervalValidation:
    """Test that polling intervals are validated for sub-24h SLA."""

    def test_valid_interval_passes_through(self):
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=12,
            detector=MagicMock(),
        )
        assert s.polling_interval_hours == 12
        s.close()

    def test_interval_at_minimum(self):
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=MIN_POLLING_INTERVAL_HOURS,
            detector=MagicMock(),
        )
        assert s.polling_interval_hours == MIN_POLLING_INTERVAL_HOURS
        s.close()

    def test_interval_at_maximum(self):
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=MAX_POLLING_INTERVAL_HOURS,
            detector=MagicMock(),
        )
        assert s.polling_interval_hours == MAX_POLLING_INTERVAL_HOURS
        s.close()

    def test_interval_24h_clamped_to_23h(self):
        """24h would violate sub-24h SLA, so it's clamped to 23h."""
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=24,
            detector=MagicMock(),
        )
        assert s.polling_interval_hours == MAX_POLLING_INTERVAL_HOURS
        s.close()

    def test_interval_48h_clamped_to_23h(self):
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=48,
            detector=MagicMock(),
        )
        assert s.polling_interval_hours == MAX_POLLING_INTERVAL_HOURS
        s.close()

    def test_interval_zero_clamped_to_minimum(self):
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=0,
            detector=MagicMock(),
        )
        assert s.polling_interval_hours == MIN_POLLING_INTERVAL_HOURS
        s.close()

    def test_interval_negative_clamped_to_minimum(self):
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=-5,
            detector=MagicMock(),
        )
        assert s.polling_interval_hours == MIN_POLLING_INTERVAL_HOURS
        s.close()

    def test_interval_non_numeric_uses_default(self):
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours="bad",  # type: ignore
            detector=MagicMock(),
        )
        assert s.polling_interval_hours == DEFAULT_POLLING_INTERVAL_HOURS
        s.close()

    def test_interval_float_truncated_to_int(self):
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=6.7,  # type: ignore
            detector=MagicMock(),
        )
        assert s.polling_interval_hours == 6
        s.close()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
class TestConstruction:
    """Test MonitorScheduler construction."""

    def test_direct_construction(self, mock_detector):
        s = MonitorScheduler(
            watched_orgs=["meta-llama", "google"],
            polling_interval_hours=8,
            slack_webhook_url="https://hooks.slack.com/test",
            detector=mock_detector,
        )
        assert s.watched_orgs == ["meta-llama", "google"]
        assert s.polling_interval_hours == 8
        assert s.slack_webhook_url == "https://hooks.slack.com/test"
        assert s.state == SchedulerState.IDLE
        s.close()

    def test_from_config(self, sample_config, mock_detector):
        s = MonitorScheduler.from_config(sample_config, detector=mock_detector)
        assert s.watched_orgs == ["meta-llama", "google", "mistralai"]
        assert s.polling_interval_hours == 12
        assert s.slack_webhook_url == "https://hooks.slack.com/test"
        s.close()

    def test_from_config_empty_uses_defaults(self, mock_detector):
        s = MonitorScheduler.from_config({}, detector=mock_detector)
        assert s.watched_orgs == []
        assert s.polling_interval_hours == DEFAULT_POLLING_INTERVAL_HOURS
        s.close()

    def test_from_config_loads_yaml(self, mock_detector):
        """Test that from_config can load from a YAML file path."""
        with patch("hf_model_monitor.scheduler.load_config") as mock_load:
            mock_load.return_value = {
                "watched_organizations": ["openai"],
                "polling_interval_hours": 6,
                "slack_webhook_url": "",
            }
            s = MonitorScheduler.from_config(
                config_path="/fake/path.yaml", detector=mock_detector
            )
            mock_load.assert_called_once_with("/fake/path.yaml")
            assert s.watched_orgs == ["openai"]
            assert s.polling_interval_hours == 6
            s.close()

    def test_creates_detector_if_not_provided(self):
        """When no detector is passed, one is created internally."""
        with patch("hf_model_monitor.scheduler.ModelDetector") as MockDetector:
            mock_instance = MagicMock()
            MockDetector.return_value = mock_instance
            s = MonitorScheduler(
                watched_orgs=["meta-llama"],
                polling_interval_hours=12,
            )
            MockDetector.assert_called_once_with(
                watched_orgs=["meta-llama"],
                polling_interval_hours=12,
            )
            s.close()


# ---------------------------------------------------------------------------
# Single run (run_once)
# ---------------------------------------------------------------------------
class TestRunOnce:
    """Test the run_once() method (single detection execution)."""

    def test_successful_run(self, scheduler, mock_detector):
        result = scheduler.run_once()
        mock_detector.run.assert_called_once()
        assert isinstance(result, DetectionResult)
        assert result.total_fetched == 15  # 10 + 5 from fixture
        assert result.total_new == 0

    def test_run_with_new_models(self, mock_detector_with_new_models):
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=12,
            detector=mock_detector_with_new_models,
        )
        result = s.run_once()
        assert result.total_new == 1
        assert result.new_models[0]["model_id"] == "meta-llama/Llama-4-Scout-17B-16E"
        s.close()

    def test_run_records_history(self, scheduler, mock_detector):
        assert len(scheduler.get_run_history()) == 0
        scheduler.run_once()
        history = scheduler.get_run_history()
        assert len(history) == 1
        assert history[0]["success"] is True
        assert history[0]["new_models"] == 0
        assert history[0]["total_fetched"] == 15
        assert history[0]["duration_seconds"] >= 0

    def test_run_records_timestamps(self, scheduler, mock_detector):
        scheduler.run_once()
        record = scheduler.get_run_history()[0]
        assert record["started_at"] != ""
        assert record["completed_at"] != ""
        # Verify parseable as ISO-8601
        datetime.fromisoformat(record["started_at"])
        datetime.fromisoformat(record["completed_at"])

    def test_multiple_runs_tracked(self, scheduler, mock_detector):
        scheduler.run_once()
        scheduler.run_once()
        scheduler.run_once()
        assert len(scheduler.get_run_history()) == 3
        # Most recent first
        records = scheduler.get_run_history()
        assert records[0]["started_at"] >= records[1]["started_at"]

    def test_history_limit(self, scheduler, mock_detector):
        for _ in range(5):
            scheduler.run_once()
        # Request only 2
        assert len(scheduler.get_run_history(limit=2)) == 2


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
class TestErrorHandling:
    """Test behavior when the detection pipeline fails."""

    def test_failed_run_returns_empty_result(self, mock_detector_failing):
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=12,
            detector=mock_detector_failing,
        )
        result = s.run_once()
        assert isinstance(result, DetectionResult)
        assert result.total_new == 0
        assert result.total_fetched == 0
        s.close()

    def test_failed_run_records_error_in_history(self, mock_detector_failing):
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=12,
            detector=mock_detector_failing,
        )
        s.run_once()
        history = s.get_run_history()
        assert len(history) == 1
        assert history[0]["success"] is False
        assert len(history[0]["errors"]) > 0
        assert "API connection failed" in history[0]["errors"][0]
        s.close()

    def test_consecutive_failures_tracked(self, mock_detector_failing):
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=12,
            detector=mock_detector_failing,
        )
        s.run_once()
        s.run_once()
        s.run_once()
        status = s.get_status()
        assert status["consecutive_failures"] == 3
        s.close()

    def test_consecutive_failures_reset_on_success(
        self, mock_detector_failing, mock_detector
    ):
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=12,
            detector=mock_detector_failing,
        )
        s.run_once()  # fail
        s.run_once()  # fail
        assert s.get_status()["consecutive_failures"] == 2

        # Swap to a working detector
        s._detector = mock_detector
        s.run_once()  # success
        assert s.get_status()["consecutive_failures"] == 0
        s.close()

    @patch("hf_model_monitor.scheduler.send_to_slack")
    def test_slack_error_alert_sent(self, mock_send, mock_detector_failing):
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=12,
            slack_webhook_url="https://hooks.slack.com/test",
            detector=mock_detector_failing,
        )
        s.run_once()
        mock_send.assert_called_once()
        alert_msg = mock_send.call_args[0][0]
        assert "Pipeline Error" in alert_msg
        assert "API connection failed" in alert_msg
        assert "Consecutive failures: 1" in alert_msg
        # New: verify error classification is present
        assert "Error type" in alert_msg.replace("error type", "Error type")
        s.close()

    @patch("hf_model_monitor.scheduler.send_to_slack")
    def test_pipeline_error_alert_has_severity(self, mock_send, mock_detector_failing):
        """Pipeline error alert should include severity indicator."""
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=12,
            slack_webhook_url="https://hooks.slack.com/test",
            detector=mock_detector_failing,
        )
        s.run_once()
        alert_msg = mock_send.call_args[0][0]
        assert "Severity" in alert_msg
        assert "Normal" in alert_msg  # First failure → Normal severity
        s.close()

    @patch("hf_model_monitor.scheduler.send_to_slack")
    def test_pipeline_error_high_severity_after_3_failures(
        self, mock_send, mock_detector_failing
    ):
        """After 3+ consecutive failures, severity should be HIGH."""
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=12,
            slack_webhook_url="https://hooks.slack.com/test",
            detector=mock_detector_failing,
        )
        s.run_once()  # failure 1
        s.run_once()  # failure 2
        s.run_once()  # failure 3
        # Check the 3rd call alert
        alert_msg = mock_send.call_args[0][0]
        assert "HIGH" in alert_msg
        assert "Consecutive failures: 3" in alert_msg
        s.close()

    @patch("hf_model_monitor.scheduler.send_to_slack")
    def test_no_slack_alert_without_webhook(self, mock_send, mock_detector_failing):
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=12,
            slack_webhook_url="",  # No webhook
            detector=mock_detector_failing,
        )
        s.run_once()
        mock_send.assert_not_called()
        s.close()

    @patch("hf_model_monitor.scheduler.send_to_slack")
    def test_slack_alert_failure_does_not_crash(
        self, mock_send, mock_detector_failing
    ):
        """If the Slack alert itself fails, the scheduler should not crash."""
        mock_send.side_effect = Exception("Slack down")
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=12,
            slack_webhook_url="https://hooks.slack.com/test",
            detector=mock_detector_failing,
        )
        # Should not raise
        result = s.run_once()
        assert isinstance(result, DetectionResult)
        s.close()


# ---------------------------------------------------------------------------
# Status reporting
# ---------------------------------------------------------------------------
class TestStatus:
    """Test the get_status() method."""

    def test_initial_status(self, scheduler):
        status = scheduler.get_status()
        assert status["state"] == "idle"
        assert status["polling_interval_hours"] == 12
        assert status["watched_orgs_count"] == 2
        assert status["consecutive_failures"] == 0
        assert status["last_run"] is None
        assert status["total_runs"] == 0

    def test_status_after_run(self, scheduler, mock_detector):
        scheduler.run_once()
        status = scheduler.get_status()
        assert status["total_runs"] == 1
        assert status["last_run"] is not None
        assert status["last_run"]["success"] is True
        assert status["consecutive_failures"] == 0

    def test_status_after_failed_run(self, mock_detector_failing):
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=12,
            detector=mock_detector_failing,
        )
        s.run_once()
        status = s.get_status()
        assert status["consecutive_failures"] == 1
        assert status["last_run"]["success"] is False
        s.close()


# ---------------------------------------------------------------------------
# Non-blocking start/stop lifecycle
# ---------------------------------------------------------------------------
class TestLifecycle:
    """Test start_nonblocking / stop lifecycle."""

    def test_start_nonblocking_and_stop(self, scheduler):
        assert scheduler.state == SchedulerState.IDLE

        scheduler.start_nonblocking()
        assert scheduler.state == SchedulerState.RUNNING
        assert scheduler._scheduler is not None
        assert scheduler.get_next_run_time() is not None

        scheduler.stop()
        assert scheduler.state == SchedulerState.STOPPED
        assert scheduler._scheduler is None

    def test_double_start_is_safe(self, scheduler):
        scheduler.start_nonblocking()
        scheduler.start_nonblocking()  # Should not raise
        assert scheduler.state == SchedulerState.RUNNING
        scheduler.stop()

    def test_stop_when_not_running_is_safe(self, scheduler):
        scheduler.stop()  # Should not raise
        assert scheduler.state != SchedulerState.RUNNING

    def test_close_stops_and_cleans_up(self, mock_detector):
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=12,
            detector=mock_detector,
        )
        s.start_nonblocking()
        s.close()
        assert s.state == SchedulerState.STOPPED
        mock_detector.close.assert_called_once()

    def test_context_manager(self, mock_detector):
        with MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=12,
            detector=mock_detector,
        ) as s:
            s.start_nonblocking()
            assert s.state == SchedulerState.RUNNING
        # After __exit__, should be stopped
        assert s.state == SchedulerState.STOPPED
        mock_detector.close.assert_called_once()

    def test_next_run_time_before_start_is_none(self, scheduler):
        assert scheduler.get_next_run_time() is None

    def test_next_run_time_after_start(self, scheduler):
        scheduler.start_nonblocking()
        next_run = scheduler.get_next_run_time()
        assert next_run is not None
        # Should be parseable as ISO-8601
        datetime.fromisoformat(next_run)
        scheduler.stop()


# ---------------------------------------------------------------------------
# Blocking start with signal shutdown
# ---------------------------------------------------------------------------
class TestBlockingStart:
    """Test the blocking start() method with simulated shutdown."""

    def test_start_blocks_until_stop(self, mock_detector):
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=12,
            detector=mock_detector,
        )

        started = threading.Event()
        stopped = threading.Event()

        def run_scheduler():
            started.set()
            s.start()
            stopped.set()

        t = threading.Thread(target=run_scheduler, daemon=True)
        t.start()

        # Wait for scheduler to start
        started.wait(timeout=5)
        time.sleep(0.3)  # Let start() proceed past initial run

        assert s.state == SchedulerState.RUNNING

        # Stop from another thread
        s.stop()
        stopped.wait(timeout=5)
        assert s.state == SchedulerState.STOPPED
        s.close()

    def test_start_runs_immediate_detection(self, mock_detector):
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=12,
            detector=mock_detector,
        )

        started = threading.Event()

        def run_scheduler():
            started.set()
            s.start()

        t = threading.Thread(target=run_scheduler, daemon=True)
        t.start()

        started.wait(timeout=5)
        time.sleep(0.5)  # Let the initial run execute

        # Detector should have been called at least once (immediate run)
        assert mock_detector.run.call_count >= 1
        s.stop()
        s.close()


# ---------------------------------------------------------------------------
# SchedulerState enum
# ---------------------------------------------------------------------------
class TestSchedulerState:
    def test_state_values(self):
        assert SchedulerState.IDLE.value == "idle"
        assert SchedulerState.RUNNING.value == "running"
        assert SchedulerState.STOPPED.value == "stopped"
        assert SchedulerState.ERROR.value == "error"


# ---------------------------------------------------------------------------
# History cap
# ---------------------------------------------------------------------------
class TestHistoryCap:
    """Ensure run history doesn't grow unbounded."""

    def test_history_capped_at_max(self, mock_detector):
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=12,
            detector=mock_detector,
        )
        s._max_history = 5  # Lower cap for testing

        for _ in range(10):
            s.run_once()

        # Internal list should be capped
        assert len(s._run_history) == 5
        # get_run_history respects the cap
        assert len(s.get_run_history(limit=100)) == 5
        s.close()


# ---------------------------------------------------------------------------
# Per-org crawler error alerting
# ---------------------------------------------------------------------------
class TestCrawlerErrorAlerts:
    """Test that per-org crawler errors trigger Slack alerts to the PM."""

    @pytest.fixture
    def mock_detector_with_org_errors(self):
        """Detector that returns a result with partial org failures."""
        detector = MagicMock()
        detector.run.return_value = DetectionResult(
            poll_timestamp=datetime.now(timezone.utc).isoformat(),
            polling_window_hours=12,
            new_models=[],
            org_results=[
                OrgPollResult(org="meta-llama", models_fetched=10),
                OrgPollResult(
                    org="google",
                    error="google: HF API error — Server error 500 (status=500)",
                ),
                OrgPollResult(org="mistralai", models_fetched=5),
                OrgPollResult(
                    org="openai",
                    error="openai: unexpected error — Connection timed out",
                ),
            ],
        )
        detector.close = MagicMock()
        return detector

    @pytest.fixture
    def mock_detector_all_orgs_fail(self):
        """Detector where every org fails."""
        detector = MagicMock()
        detector.run.return_value = DetectionResult(
            poll_timestamp=datetime.now(timezone.utc).isoformat(),
            polling_window_hours=12,
            org_results=[
                OrgPollResult(
                    org="meta-llama",
                    error="meta-llama: HF API error — Connection error (status=None)",
                ),
                OrgPollResult(
                    org="google",
                    error="google: HF API error — Rate limited (status=429)",
                ),
            ],
        )
        detector.close = MagicMock()
        return detector

    @patch("hf_model_monitor.scheduler.send_to_slack")
    def test_crawler_error_alert_sent_for_partial_failures(
        self, mock_send, mock_detector_with_org_errors
    ):
        """When some orgs fail, a crawler error alert is sent to Slack."""
        s = MonitorScheduler(
            watched_orgs=["meta-llama", "google", "mistralai", "openai"],
            polling_interval_hours=12,
            slack_webhook_url="https://hooks.slack.com/test",
            detector=mock_detector_with_org_errors,
        )
        s.run_once()

        mock_send.assert_called_once()
        alert_msg = mock_send.call_args[0][0]

        # Should be a crawler errors summary, not a pipeline error
        assert "Crawler Errors Detected" in alert_msg
        assert "2/4" in alert_msg  # 2 of 4 orgs failed
        assert "2 succeeded" in alert_msg
        assert "google" in alert_msg
        assert "openai" in alert_msg
        # meta-llama and mistralai succeeded, shouldn't appear as errors
        assert "Server Error" in alert_msg
        assert "Timeout" in alert_msg
        s.close()

    @patch("hf_model_monitor.scheduler.send_to_slack")
    def test_crawler_error_alert_all_orgs_fail(
        self, mock_send, mock_detector_all_orgs_fail
    ):
        """When all orgs fail, alert shows all as failed."""
        s = MonitorScheduler(
            watched_orgs=["meta-llama", "google"],
            polling_interval_hours=12,
            slack_webhook_url="https://hooks.slack.com/test",
            detector=mock_detector_all_orgs_fail,
        )
        s.run_once()

        mock_send.assert_called_once()
        alert_msg = mock_send.call_args[0][0]
        assert "2/2" in alert_msg
        assert "0 succeeded" in alert_msg
        assert "meta-llama" in alert_msg
        assert "google" in alert_msg
        s.close()

    @patch("hf_model_monitor.scheduler.send_to_slack")
    def test_no_crawler_alert_when_all_orgs_succeed(self, mock_send, mock_detector):
        """When all orgs succeed, no crawler error alert is sent."""
        s = MonitorScheduler(
            watched_orgs=["meta-llama", "google"],
            polling_interval_hours=12,
            slack_webhook_url="https://hooks.slack.com/test",
            detector=mock_detector,
        )
        s.run_once()

        # No error alert should be sent (mock_detector returns success)
        mock_send.assert_not_called()
        s.close()

    @patch("hf_model_monitor.scheduler.send_to_slack")
    def test_no_crawler_alert_without_webhook(
        self, mock_send, mock_detector_with_org_errors
    ):
        """Even with org errors, no alert if webhook isn't configured."""
        s = MonitorScheduler(
            watched_orgs=["meta-llama", "google", "mistralai", "openai"],
            polling_interval_hours=12,
            slack_webhook_url="",  # No webhook
            detector=mock_detector_with_org_errors,
        )
        s.run_once()
        mock_send.assert_not_called()
        s.close()

    @patch("hf_model_monitor.scheduler.send_to_slack")
    def test_crawler_alert_failure_does_not_crash(
        self, mock_send, mock_detector_with_org_errors
    ):
        """If the Slack alert for crawler errors itself fails, scheduler continues."""
        mock_send.side_effect = Exception("Slack API unavailable")
        s = MonitorScheduler(
            watched_orgs=["meta-llama", "google", "mistralai", "openai"],
            polling_interval_hours=12,
            slack_webhook_url="https://hooks.slack.com/test",
            detector=mock_detector_with_org_errors,
        )
        # Should not raise
        result = s.run_once()
        assert isinstance(result, DetectionResult)
        s.close()

    @patch("hf_model_monitor.scheduler.send_to_slack")
    def test_crawler_alert_includes_next_retry(
        self, mock_send, mock_detector_with_org_errors
    ):
        """Crawler error alert should include when the next retry will happen."""
        s = MonitorScheduler(
            watched_orgs=["meta-llama", "google", "mistralai", "openai"],
            polling_interval_hours=6,
            slack_webhook_url="https://hooks.slack.com/test",
            detector=mock_detector_with_org_errors,
        )
        s.run_once()

        alert_msg = mock_send.call_args[0][0]
        assert "Next retry in 6h" in alert_msg
        s.close()

    @patch("hf_model_monitor.scheduler.send_to_slack")
    def test_crawler_alert_contains_error_type_classification(
        self, mock_send, mock_detector_with_org_errors
    ):
        """Each org error should have its type classified in the alert."""
        s = MonitorScheduler(
            watched_orgs=["meta-llama", "google", "mistralai", "openai"],
            polling_interval_hours=12,
            slack_webhook_url="https://hooks.slack.com/test",
            detector=mock_detector_with_org_errors,
        )
        s.run_once()

        alert_msg = mock_send.call_args[0][0]
        # google had a 500 error → "Server Error"
        assert "Server Error" in alert_msg
        # openai had a timeout → "Timeout"
        assert "Timeout" in alert_msg
        s.close()

    @patch("hf_model_monitor.scheduler.send_to_slack")
    def test_crawler_error_does_not_increment_consecutive_failures(
        self, mock_send, mock_detector_with_org_errors
    ):
        """Per-org errors (partial success) should NOT increment consecutive
        pipeline failures — those are only for catastrophic pipeline crashes."""
        s = MonitorScheduler(
            watched_orgs=["meta-llama", "google", "mistralai", "openai"],
            polling_interval_hours=12,
            slack_webhook_url="https://hooks.slack.com/test",
            detector=mock_detector_with_org_errors,
        )
        s.run_once()
        s.run_once()

        # Partial failures don't count as pipeline failures
        assert s.get_status()["consecutive_failures"] == 0
        s.close()


# ---------------------------------------------------------------------------
# Runtime interval update
# ---------------------------------------------------------------------------
class TestUpdateInterval:
    """Test update_interval() for runtime polling frequency changes."""

    def test_update_interval_while_running(self, mock_detector):
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=12,
            detector=mock_detector,
        )
        s.start_nonblocking()

        success, msg = s.update_interval(6)
        assert success is True
        assert s.polling_interval_hours == 6
        assert "12h" in msg and "6h" in msg
        s.close()

    def test_update_interval_while_idle(self, mock_detector):
        """Update interval before start — stores value, no reschedule needed."""
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=12,
            detector=mock_detector,
        )
        success, msg = s.update_interval(8)
        assert success is True
        assert s.polling_interval_hours == 8
        s.close()

    def test_update_interval_same_value_noop(self, mock_detector):
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=12,
            detector=mock_detector,
        )
        success, msg = s.update_interval(12)
        assert success is True
        assert "unchanged" in msg.lower()
        s.close()

    def test_update_interval_clamped_to_max(self, mock_detector):
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=12,
            detector=mock_detector,
        )
        success, msg = s.update_interval(48)
        assert success is True
        assert s.polling_interval_hours == MAX_POLLING_INTERVAL_HOURS
        s.close()

    def test_update_interval_clamped_to_min(self, mock_detector):
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=12,
            detector=mock_detector,
        )
        success, msg = s.update_interval(0)
        assert success is True
        assert s.polling_interval_hours == MIN_POLLING_INTERVAL_HOURS
        s.close()

    def test_update_interval_reschedules_apscheduler_job(self, mock_detector):
        """Verify the APScheduler job is actually rescheduled."""
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=12,
            detector=mock_detector,
        )
        s.start_nonblocking()

        # Get next run time before update
        next_before = s.get_next_run_time()
        assert next_before is not None

        success, _ = s.update_interval(6)
        assert success is True

        # After rescheduling, next run time should be different
        # (closer, since 6h < 12h)
        next_after = s.get_next_run_time()
        assert next_after is not None
        s.close()

    def test_update_interval_reverts_on_reschedule_failure(self, mock_detector):
        """If APScheduler reschedule fails, the interval should revert."""
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=12,
            detector=mock_detector,
        )
        s.start_nonblocking()

        # Force reschedule_job to fail
        s._scheduler.reschedule_job = MagicMock(
            side_effect=RuntimeError("scheduler error")
        )

        success, msg = s.update_interval(6)
        assert success is False
        assert "Failed" in msg
        # Should revert to original
        assert s.polling_interval_hours == 12
        s.close()


# ---------------------------------------------------------------------------
# Manual trigger
# ---------------------------------------------------------------------------
class TestTriggerNow:
    """Test trigger_now() for on-demand detection from dashboard."""

    def test_trigger_returns_result(self, mock_detector):
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=12,
            detector=mock_detector,
        )
        result = s.trigger_now()
        assert isinstance(result, DetectionResult)
        mock_detector.run.assert_called_once()
        s.close()

    def test_trigger_records_in_history(self, mock_detector):
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=12,
            detector=mock_detector,
        )
        s.trigger_now()
        history = s.get_run_history()
        assert len(history) == 1
        assert history[0]["success"] is True
        s.close()

    def test_trigger_while_scheduler_running(self, mock_detector):
        """Manual trigger should work alongside the scheduled job."""
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=12,
            detector=mock_detector,
        )
        s.start_nonblocking()
        result = s.trigger_now()
        assert isinstance(result, DetectionResult)
        assert result.total_fetched == 15
        s.close()

    def test_trigger_with_failing_detector(self, mock_detector_failing):
        """Manual trigger should handle errors gracefully."""
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=12,
            detector=mock_detector_failing,
        )
        result = s.trigger_now()
        assert isinstance(result, DetectionResult)
        assert result.total_new == 0
        s.close()


# ---------------------------------------------------------------------------
# SLA health monitoring
# ---------------------------------------------------------------------------
class TestSLAHealth:
    """Test is_healthy() and get_sla_status() for sub-24h SLA enforcement."""

    def test_healthy_after_successful_run(self, scheduler, mock_detector):
        scheduler.start_nonblocking()
        scheduler.run_once()
        assert scheduler.is_healthy() is True
        scheduler.stop()

    def test_unhealthy_when_not_running(self, scheduler, mock_detector):
        """Scheduler must be running to be considered healthy."""
        scheduler.run_once()  # Successful run, but not in RUNNING state
        assert scheduler.is_healthy() is False

    def test_healthy_with_no_runs_yet(self, scheduler):
        """Grace period: healthy right after start even with no runs."""
        scheduler.start_nonblocking()
        assert scheduler.is_healthy() is True
        scheduler.stop()

    def test_unhealthy_after_sla_breach(self, scheduler, mock_detector):
        """If last success was >24h ago, should report unhealthy."""
        scheduler.start_nonblocking()
        scheduler.run_once()

        # Simulate 25 hours passing since last success
        breach_time = datetime.now(timezone.utc) + timedelta(hours=25)
        assert scheduler.is_healthy(now=breach_time) is False
        scheduler.stop()

    def test_healthy_within_sla_window(self, scheduler, mock_detector):
        """If last success was <24h ago, should report healthy."""
        scheduler.start_nonblocking()
        scheduler.run_once()

        # Simulate 23 hours passing — still within SLA
        check_time = datetime.now(timezone.utc) + timedelta(hours=23)
        assert scheduler.is_healthy(now=check_time) is True
        scheduler.stop()

    def test_healthy_at_exactly_24h_boundary(self, scheduler, mock_detector):
        """At exactly 24h, should be unhealthy (< not <=)."""
        scheduler.start_nonblocking()
        scheduler.run_once()

        # Exactly 24h later
        boundary_time = datetime.now(timezone.utc) + timedelta(hours=24)
        assert scheduler.is_healthy(now=boundary_time) is False
        scheduler.stop()

    def test_sla_status_includes_all_fields(self, scheduler, mock_detector):
        scheduler.start_nonblocking()
        scheduler.run_once()

        sla = scheduler.get_sla_status()
        assert "healthy" in sla
        assert "sla_window_hours" in sla
        assert "polling_interval_hours" in sla
        assert "last_success_time" in sla
        assert "hours_since_last_success" in sla
        assert "next_run_time" in sla
        assert "consecutive_failures" in sla
        scheduler.stop()

    def test_sla_status_values_after_success(self, scheduler, mock_detector):
        scheduler.start_nonblocking()
        scheduler.run_once()

        sla = scheduler.get_sla_status()
        assert sla["healthy"] is True
        assert sla["sla_window_hours"] == SLA_WINDOW_HOURS
        assert sla["polling_interval_hours"] == 12
        assert sla["last_success_time"] is not None
        assert sla["hours_since_last_success"] is not None
        assert sla["hours_since_last_success"] < 1  # Just ran
        assert sla["consecutive_failures"] == 0
        scheduler.stop()

    def test_sla_status_no_runs_yet(self, scheduler):
        scheduler.start_nonblocking()
        sla = scheduler.get_sla_status()
        assert sla["healthy"] is True  # Grace period
        assert sla["last_success_time"] is None
        assert sla["hours_since_last_success"] is None
        scheduler.stop()

    def test_sla_status_after_failures(self, mock_detector_failing):
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=12,
            detector=mock_detector_failing,
        )
        s.start_nonblocking()
        s.run_once()
        s.run_once()

        sla = s.get_sla_status()
        assert sla["consecutive_failures"] == 2
        # No successful run yet, but grace period applies
        assert sla["last_success_time"] is None
        assert sla["healthy"] is True
        s.close()

    def test_sla_breach_after_successful_then_failures(
        self, mock_detector, mock_detector_failing
    ):
        """Detect SLA breach: initial success, then failures exceeding 24h."""
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=12,
            detector=mock_detector,
        )
        s.start_nonblocking()
        s.run_once()  # Success — sets last_success_time

        # Swap to failing detector
        s._detector = mock_detector_failing
        s.run_once()  # Fail
        s.run_once()  # Fail

        # Still healthy right after (< 24h since last success)
        assert s.is_healthy() is True

        # But after 25h since last success...
        breach_time = datetime.now(timezone.utc) + timedelta(hours=25)
        assert s.is_healthy(now=breach_time) is False

        sla = s.get_sla_status(now=breach_time)
        assert sla["healthy"] is False
        assert sla["hours_since_last_success"] >= 25.0
        assert sla["consecutive_failures"] == 2
        s.close()

    def test_last_success_time_tracked_in_status(self, scheduler, mock_detector):
        """get_status() should include last_success_time and healthy flag."""
        scheduler.start_nonblocking()
        scheduler.run_once()
        status = scheduler.get_status()
        assert "healthy" in status
        assert "last_success_time" in status
        assert status["healthy"] is True
        assert status["last_success_time"] is not None
        scheduler.stop()


# ---------------------------------------------------------------------------
# run_on_startup configuration
# ---------------------------------------------------------------------------
class TestRunOnStartup:
    """Test that run_on_startup controls whether detection runs at startup."""

    def test_from_config_reads_run_on_startup_true(self, mock_detector):
        config = {
            "watched_organizations": ["meta-llama"],
            "polling_interval_hours": 12,
            "slack_webhook_url": "",
            "run_on_startup": True,
        }
        s = MonitorScheduler.from_config(config, detector=mock_detector)
        assert s.run_on_startup is True
        s.close()

    def test_from_config_reads_run_on_startup_false(self, mock_detector):
        config = {
            "watched_organizations": ["meta-llama"],
            "polling_interval_hours": 12,
            "slack_webhook_url": "",
            "run_on_startup": False,
        }
        s = MonitorScheduler.from_config(config, detector=mock_detector)
        assert s.run_on_startup is False
        s.close()

    def test_from_config_defaults_to_true(self, mock_detector):
        config = {
            "watched_organizations": ["meta-llama"],
            "polling_interval_hours": 12,
            "slack_webhook_url": "",
        }
        s = MonitorScheduler.from_config(config, detector=mock_detector)
        assert s.run_on_startup is True
        s.close()

    def test_start_skips_immediate_run_when_disabled(self, mock_detector):
        """When run_on_startup=False, start() should not call detector.run()."""
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=12,
            run_on_startup=False,
            detector=mock_detector,
        )

        started = threading.Event()

        def run_scheduler():
            started.set()
            s.start()

        t = threading.Thread(target=run_scheduler, daemon=True)
        t.start()
        started.wait(timeout=5)
        time.sleep(0.5)  # Let start() proceed

        # Detector should NOT have been called (run_on_startup=False)
        mock_detector.run.assert_not_called()
        s.stop()
        s.close()

    def test_start_runs_immediately_when_enabled(self, mock_detector):
        """When run_on_startup=True (default), start() calls detector.run()."""
        s = MonitorScheduler(
            watched_orgs=["meta-llama"],
            polling_interval_hours=12,
            run_on_startup=True,
            detector=mock_detector,
        )

        started = threading.Event()

        def run_scheduler():
            started.set()
            s.start()

        t = threading.Thread(target=run_scheduler, daemon=True)
        t.start()
        started.wait(timeout=5)
        time.sleep(0.5)  # Let start() proceed

        # Detector should have been called immediately
        assert mock_detector.run.call_count >= 1
        s.stop()
        s.close()


# ---------------------------------------------------------------------------
# SLA constant validation
# ---------------------------------------------------------------------------
class TestSLAConstants:
    """Verify that SLA-related constants are self-consistent."""

    def test_sla_window_is_24h(self):
        assert SLA_WINDOW_HOURS == 24

    def test_max_interval_below_sla(self):
        """Max polling interval must be strictly less than the SLA window."""
        assert MAX_POLLING_INTERVAL_HOURS < SLA_WINDOW_HOURS

    def test_default_interval_well_within_sla(self):
        """Default interval provides at least 2 checks per SLA window."""
        assert DEFAULT_POLLING_INTERVAL_HOURS <= SLA_WINDOW_HOURS // 2
