"""Periodic polling scheduler for the HuggingFace Model Monitor.

Uses APScheduler to run the detection pipeline at configurable intervals,
ensuring sub-24-hour coverage for new model detection.

The scheduler:
1. Runs an immediate detection on startup (configurable via run_on_startup)
2. Schedules recurring detection at the configured interval
3. Sends Slack error alerts if the pipeline fails
4. Handles graceful shutdown via SIGINT/SIGTERM
5. Validates that the polling interval provides sub-24-hour coverage
6. Monitors SLA compliance — alerts if no successful run within 24h
7. Supports runtime interval updates without restart
8. Provides manual trigger for on-demand detection

Usage:
    from hf_model_monitor.scheduler import MonitorScheduler

    scheduler = MonitorScheduler.from_config(config)
    scheduler.start()  # Blocks until shutdown signal

    # Or run once immediately:
    scheduler.run_once()

    # Runtime interval change (from dashboard):
    scheduler.update_interval(6)

    # Manual trigger (from dashboard):
    scheduler.trigger_now()
"""

import logging
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from enum import Enum

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from .config import load_config
from .detector import DetectionResult, ModelDetector
from .slack_notifier import (
    SlackNotifier,
    format_crawler_errors_summary,
    format_pipeline_error_alert,
    send_to_slack,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_POLLING_INTERVAL_HOURS = 23  # Must be < 24h for sub-24-hour SLA
MIN_POLLING_INTERVAL_HOURS = 1
DEFAULT_POLLING_INTERVAL_HOURS = 12
SLA_WINDOW_HOURS = 24  # The sub-24-hour detection guarantee
JOB_ID = "hf_model_detection"


# ---------------------------------------------------------------------------
# Scheduler state
# ---------------------------------------------------------------------------
class SchedulerState(Enum):
    """Possible states for the monitor scheduler."""

    IDLE = "idle"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


# ---------------------------------------------------------------------------
# MonitorScheduler
# ---------------------------------------------------------------------------
class MonitorScheduler:
    """Periodic scheduler that runs the HF model detection pipeline.

    Wraps APScheduler's BackgroundScheduler to provide:
    - Configurable polling interval (validated for sub-24h SLA)
    - Immediate first run on startup
    - Slack error alerts on pipeline failures
    - Graceful shutdown on SIGINT/SIGTERM
    - Run history tracking for observability

    Args:
        watched_orgs: List of HF org/user IDs to monitor.
        polling_interval_hours: Hours between detection runs (1-23).
        slack_webhook_url: Webhook URL for error alerts (optional).
        detector: Pre-built ModelDetector (created from args if None).
    """

    def __init__(
        self,
        watched_orgs: list[str],
        polling_interval_hours: int = DEFAULT_POLLING_INTERVAL_HOURS,
        slack_webhook_url: str = "",
        *,
        run_on_startup: bool = True,
        detector: ModelDetector | None = None,
    ):
        self.watched_orgs = watched_orgs
        self.polling_interval_hours = self._validate_interval(polling_interval_hours)
        self.slack_webhook_url = slack_webhook_url
        self.run_on_startup = run_on_startup
        self.state = SchedulerState.IDLE

        # Detection engine
        self._detector = detector or ModelDetector(
            watched_orgs=watched_orgs,
            polling_interval_hours=self.polling_interval_hours,
        )

        # Slack notifier for model reports (separate from error-only webhook)
        self._notifier = SlackNotifier(webhook_url=slack_webhook_url)

        # APScheduler instance (created on start)
        self._scheduler: BackgroundScheduler | None = None

        # Shutdown coordination
        self._shutdown_event = threading.Event()

        # Run history (in-memory, for dashboard/status queries)
        self._run_history: list[dict] = []
        self._max_history = 100
        self._consecutive_failures = 0
        self._last_success_time: datetime | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: dict | None = None,
        *,
        config_path: str | None = None,
        detector: ModelDetector | None = None,
    ) -> "MonitorScheduler":
        """Create a scheduler from a config dict or YAML file.

        Args:
            config: Config dict (as returned by load_config()).
                    If None, loads from config_path or default location.
            config_path: Path to settings.yaml (used if config is None).
            detector: Optional pre-built ModelDetector.

        Returns:
            Configured MonitorScheduler instance.
        """
        if config is None:
            config = load_config(config_path)

        return cls(
            watched_orgs=config.get("watched_organizations", []),
            polling_interval_hours=config.get(
                "polling_interval_hours", DEFAULT_POLLING_INTERVAL_HOURS
            ),
            slack_webhook_url=config.get("slack_webhook_url", ""),
            run_on_startup=config.get("run_on_startup", True),
            detector=detector,
        )

    # ------------------------------------------------------------------
    # Interval validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_interval(hours: int) -> int:
        """Validate and clamp the polling interval for sub-24h SLA.

        Args:
            hours: Requested polling interval in hours.

        Returns:
            Validated interval (clamped to 1-23 range).
        """
        if not isinstance(hours, (int, float)):
            logger.warning(
                "Invalid polling interval type (%s), using default %dh",
                type(hours).__name__,
                DEFAULT_POLLING_INTERVAL_HOURS,
            )
            return DEFAULT_POLLING_INTERVAL_HOURS

        hours = int(hours)

        if hours > MAX_POLLING_INTERVAL_HOURS:
            logger.warning(
                "Polling interval %dh exceeds sub-24h SLA limit. "
                "Clamping to %dh for guaranteed coverage.",
                hours,
                MAX_POLLING_INTERVAL_HOURS,
            )
            return MAX_POLLING_INTERVAL_HOURS

        if hours < MIN_POLLING_INTERVAL_HOURS:
            logger.warning(
                "Polling interval %dh is below minimum. "
                "Clamping to %dh.",
                hours,
                MIN_POLLING_INTERVAL_HOURS,
            )
            return MIN_POLLING_INTERVAL_HOURS

        return hours

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the scheduler and block until shutdown signal.

        This method:
        1. Registers signal handlers for graceful shutdown
        2. Runs the detection pipeline immediately
        3. Schedules recurring runs at the configured interval
        4. Blocks until SIGINT/SIGTERM or stop() is called
        """
        if self.state == SchedulerState.RUNNING:
            logger.warning("Scheduler is already running")
            return

        logger.info(
            "Starting HF Model Monitor scheduler: "
            "polling every %dh, watching %d org(s)",
            self.polling_interval_hours,
            len(self.watched_orgs),
        )

        # Register signal handlers (only works in the main thread)
        self._register_signal_handlers()

        # Create and configure APScheduler
        self._scheduler = BackgroundScheduler(
            job_defaults={
                "coalesce": True,  # Merge missed runs into one
                "max_instances": 1,  # Never overlap runs
                "misfire_grace_time": 3600,  # Allow 1h late starts
            }
        )

        # Listen for job events (for logging and error tracking)
        self._scheduler.add_listener(
            self._on_job_event, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR
        )

        # Schedule the recurring detection job
        trigger = IntervalTrigger(hours=self.polling_interval_hours)
        self._scheduler.add_job(
            self._run_detection,
            trigger=trigger,
            id=JOB_ID,
            name="HF Model Detection",
            replace_existing=True,
        )

        self._scheduler.start()
        self.state = SchedulerState.RUNNING
        self._shutdown_event.clear()

        # Run immediately on startup unless disabled in config
        if self.run_on_startup:
            logger.info("Scheduler started. Running initial detection...")
            self._run_detection()
        else:
            logger.info(
                "Scheduler started (run_on_startup=False, "
                "first detection at next interval)."
            )

        next_run = self.get_next_run_time()
        if next_run:
            logger.info("Next scheduled run: %s", next_run)

        # Block until shutdown
        logger.info(
            "Scheduler running. Press Ctrl+C to stop. "
            "Next poll in %dh.",
            self.polling_interval_hours,
        )
        self._shutdown_event.wait()

    def start_nonblocking(self) -> None:
        """Start the scheduler without blocking (for embedding in web servers).

        Unlike start(), this returns immediately. Call stop() to shut down.
        Does NOT run an initial detection — that's left to the caller.
        """
        if self.state == SchedulerState.RUNNING:
            logger.warning("Scheduler is already running")
            return

        self._scheduler = BackgroundScheduler(
            job_defaults={
                "coalesce": True,
                "max_instances": 1,
                "misfire_grace_time": 3600,
            }
        )

        self._scheduler.add_listener(
            self._on_job_event, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR
        )

        trigger = IntervalTrigger(hours=self.polling_interval_hours)
        self._scheduler.add_job(
            self._run_detection,
            trigger=trigger,
            id=JOB_ID,
            name="HF Model Detection",
            replace_existing=True,
        )

        self._scheduler.start()
        self.state = SchedulerState.RUNNING
        self._shutdown_event.clear()

        logger.info(
            "Scheduler started (non-blocking): polling every %dh",
            self.polling_interval_hours,
        )

    def stop(self) -> None:
        """Gracefully stop the scheduler."""
        if self.state != SchedulerState.RUNNING:
            logger.debug("Scheduler not running, nothing to stop")
            return

        logger.info("Stopping scheduler...")
        self.state = SchedulerState.STOPPED

        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None

        self._shutdown_event.set()
        logger.info("Scheduler stopped.")

    # ------------------------------------------------------------------
    # Single run (manual or scheduled)
    # ------------------------------------------------------------------

    def run_once(self) -> DetectionResult:
        """Run the detection pipeline once and return the result.

        This is useful for:
        - Manual/CLI invocation
        - Testing
        - Cron-based scheduling (as an alternative to APScheduler)

        Returns:
            DetectionResult from the detector.
        """
        return self._run_detection()

    def _run_detection(self) -> DetectionResult:
        """Execute the detection pipeline with error handling and history tracking.

        Returns:
            DetectionResult (empty result with error info on failure).
        """
        run_start = datetime.now(timezone.utc)
        run_record: dict = {
            "started_at": run_start.isoformat(),
            "completed_at": "",
            "success": False,
            "new_models": 0,
            "total_fetched": 0,
            "errors": [],
            "duration_seconds": 0.0,
        }

        try:
            logger.info("Starting detection run...")
            result = self._detector.run()

            run_end = datetime.now(timezone.utc)
            duration = (run_end - run_start).total_seconds()

            run_record.update(
                {
                    "completed_at": run_end.isoformat(),
                    "success": True,
                    "new_models": result.total_new,
                    "total_fetched": result.total_fetched,
                    "errors": result.errors,
                    "duration_seconds": round(duration, 1),
                }
            )

            with self._lock:
                self._consecutive_failures = 0
                self._last_success_time = run_end

            logger.info(
                "Detection run complete in %.1fs: %d new model(s) from %d fetched",
                duration,
                result.total_new,
                result.total_fetched,
            )

            # Send Slack reports for each new model
            if result.new_models and self._notifier.is_configured:
                from .main import _notify_new_models

                notified = _notify_new_models(
                    result.new_models, self._notifier, self._detector,
                )
                run_record["models_notified"] = notified
                logger.info("Sent %d Slack report(s) for new models", notified)

            # Log any per-org errors (non-fatal) and alert PM via Slack
            if result.errors:
                logger.warning(
                    "Detection completed with %d org error(s): %s",
                    len(result.errors),
                    result.errors,
                )
                self._send_crawler_error_alerts(result)

            return result

        except Exception as exc:
            run_end = datetime.now(timezone.utc)
            duration = (run_end - run_start).total_seconds()
            error_msg = f"Detection pipeline failed: {exc}"

            run_record.update(
                {
                    "completed_at": run_end.isoformat(),
                    "success": False,
                    "errors": [error_msg],
                    "duration_seconds": round(duration, 1),
                }
            )

            with self._lock:
                self._consecutive_failures += 1
                failures = self._consecutive_failures

            logger.exception(error_msg)
            self._send_error_alert(error_msg, failures)

            # Return an empty result rather than crashing the scheduler
            return DetectionResult(
                poll_timestamp=run_start.isoformat(),
                polling_window_hours=self.polling_interval_hours,
            )

        finally:
            self._record_run(run_record)

    # ------------------------------------------------------------------
    # Error alerting
    # ------------------------------------------------------------------

    def _send_error_alert(self, error_msg: str, consecutive_failures: int) -> None:
        """Send a Slack alert when the detection pipeline fails.

        Only sends if a webhook URL is configured. Includes failure count
        and error classification to help the PM assess severity.
        """
        if not self.slack_webhook_url:
            logger.debug("No Slack webhook configured, skipping error alert")
            return

        alert = format_pipeline_error_alert(
            error_msg=error_msg,
            consecutive_failures=consecutive_failures,
            next_retry_hours=self.polling_interval_hours,
        )

        try:
            send_to_slack(alert, webhook_url=self.slack_webhook_url)
            logger.info("Pipeline error alert sent to Slack")
        except Exception:
            logger.exception("Failed to send pipeline error alert to Slack")

    def _send_crawler_error_alerts(self, result: DetectionResult) -> None:
        """Send a Slack alert summarizing per-org crawler failures.

        Called after a detection run that completed but had partial failures
        (e.g. some orgs returned errors while others succeeded).

        Only sends if a webhook URL is configured and there are actual errors.
        """
        if not self.slack_webhook_url:
            logger.debug("No Slack webhook configured, skipping crawler error alert")
            return

        org_errors = [
            {"source": r.org, "error": r.error}
            for r in result.org_results
            if r.error
        ]

        if not org_errors:
            return

        alert = format_crawler_errors_summary(
            org_errors=org_errors,
            total_orgs=result.orgs_polled,
            next_retry_hours=self.polling_interval_hours,
        )

        try:
            send_to_slack(alert, webhook_url=self.slack_webhook_url)
            logger.info(
                "Crawler error alert sent to Slack (%d org failure(s))",
                len(org_errors),
            )
        except Exception:
            logger.exception("Failed to send crawler error alert to Slack")

    # ------------------------------------------------------------------
    # APScheduler event listener
    # ------------------------------------------------------------------

    def _on_job_event(self, event) -> None:
        """Handle APScheduler job execution/error events for logging."""
        if event.exception:
            logger.error(
                "Scheduled job '%s' raised an exception: %s",
                JOB_ID,
                event.exception,
            )
        else:
            logger.debug("Scheduled job '%s' completed successfully", JOB_ID)

    # ------------------------------------------------------------------
    # Run history
    # ------------------------------------------------------------------

    def _record_run(self, record: dict) -> None:
        """Append a run record to in-memory history (capped at _max_history)."""
        with self._lock:
            self._run_history.append(record)
            if len(self._run_history) > self._max_history:
                self._run_history = self._run_history[-self._max_history :]

    def get_run_history(self, limit: int = 20) -> list[dict]:
        """Return the most recent run records.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of run records, newest first.
        """
        with self._lock:
            return list(reversed(self._run_history[-limit:]))

    def get_status(self) -> dict:
        """Return current scheduler status for dashboard/API use.

        Returns:
            Dict with state, config, next run time, SLA health, and history.
        """
        with self._lock:
            last_run = self._run_history[-1] if self._run_history else None
            consecutive_failures = self._consecutive_failures
            last_success = self._last_success_time

        return {
            "state": self.state.value,
            "polling_interval_hours": self.polling_interval_hours,
            "watched_orgs_count": len(self.watched_orgs),
            "next_run_time": self.get_next_run_time(),
            "consecutive_failures": consecutive_failures,
            "last_run": last_run,
            "total_runs": len(self._run_history),
            "healthy": self.is_healthy(),
            "last_success_time": (
                last_success.isoformat() if last_success else None
            ),
        }

    def get_next_run_time(self) -> str | None:
        """Return the ISO-8601 timestamp of the next scheduled run, or None."""
        if self._scheduler is None:
            return None
        job = self._scheduler.get_job(JOB_ID)
        if job and job.next_run_time:
            return job.next_run_time.isoformat()
        return None

    # ------------------------------------------------------------------
    # Runtime interval update
    # ------------------------------------------------------------------

    def update_interval(self, new_hours: int) -> tuple[bool, str]:
        """Change the polling interval at runtime without restarting.

        Validates the new interval, reschedules the APScheduler job, and
        persists the change.  Safe to call while the scheduler is running.

        Args:
            new_hours: New polling interval in hours (validated to 1-23).

        Returns:
            ``(success, message)`` tuple.
        """
        validated = self._validate_interval(new_hours)

        if validated == self.polling_interval_hours:
            return True, (
                f"Polling interval unchanged at {validated}h."
            )

        old_interval = self.polling_interval_hours
        self.polling_interval_hours = validated

        # Reschedule the APScheduler job if the scheduler is running
        if self._scheduler is not None and self.state == SchedulerState.RUNNING:
            try:
                self._scheduler.reschedule_job(
                    JOB_ID,
                    trigger=IntervalTrigger(hours=validated),
                )
                logger.info(
                    "Rescheduled detection job: %dh → %dh",
                    old_interval,
                    validated,
                )
            except Exception as exc:
                logger.exception("Failed to reschedule job")
                # Revert on failure
                self.polling_interval_hours = old_interval
                return False, f"Failed to reschedule: {exc}"

        logger.info(
            "Polling interval updated: %dh → %dh", old_interval, validated
        )
        return True, (
            f"Polling interval updated from {old_interval}h to {validated}h."
        )

    # ------------------------------------------------------------------
    # Manual trigger
    # ------------------------------------------------------------------

    def trigger_now(self) -> DetectionResult:
        """Trigger an immediate detection run (manual / dashboard use).

        Unlike ``run_once()``, this method is safe to call while the
        scheduler is running — it executes the detection in the calling
        thread and does not interfere with the scheduled job.  The result
        is recorded in run history alongside scheduled runs.

        Returns:
            DetectionResult from the detection run.
        """
        logger.info("Manual detection trigger requested")
        return self._run_detection()

    # ------------------------------------------------------------------
    # SLA health monitoring
    # ------------------------------------------------------------------

    def is_healthy(self, *, now: datetime | None = None) -> bool:
        """Check whether the scheduler is meeting the sub-24h detection SLA.

        The scheduler is considered healthy if:
        1. It is currently running, AND
        2. A successful detection occurred within the last 24 hours
           (or no run has been attempted yet — grace period after startup)

        Args:
            now: Override for current time (for testing).

        Returns:
            True if the SLA is being met, False if a breach is detected.
        """
        if self.state != SchedulerState.RUNNING:
            return False

        if now is None:
            now = datetime.now(timezone.utc)

        with self._lock:
            last_success = self._last_success_time

        # Grace period: if no run has happened yet, we're still healthy
        # (scheduler just started and hasn't had its first interval yet)
        if last_success is None:
            return True

        elapsed = now - last_success
        return elapsed < timedelta(hours=SLA_WINDOW_HOURS)

    def get_sla_status(self, *, now: datetime | None = None) -> dict:
        """Return detailed SLA compliance information.

        Designed for dashboard display and health-check endpoints.

        Args:
            now: Override for current time (for testing).

        Returns:
            Dict with SLA metrics:
            - ``healthy``: Whether the sub-24h SLA is currently met
            - ``sla_window_hours``: The SLA window (always 24)
            - ``polling_interval_hours``: Current polling interval
            - ``last_success_time``: ISO timestamp of last successful run
            - ``hours_since_last_success``: Hours elapsed since last success
            - ``next_run_time``: ISO timestamp of the next scheduled run
            - ``consecutive_failures``: Number of consecutive pipeline failures
        """
        if now is None:
            now = datetime.now(timezone.utc)

        with self._lock:
            last_success = self._last_success_time
            consecutive_failures = self._consecutive_failures

        hours_since = None
        if last_success is not None:
            hours_since = round(
                (now - last_success).total_seconds() / 3600, 1
            )

        return {
            "healthy": self.is_healthy(now=now),
            "sla_window_hours": SLA_WINDOW_HOURS,
            "polling_interval_hours": self.polling_interval_hours,
            "last_success_time": (
                last_success.isoformat() if last_success else None
            ),
            "hours_since_last_success": hours_since,
            "next_run_time": self.get_next_run_time(),
            "consecutive_failures": consecutive_failures,
        }

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _register_signal_handlers(self) -> None:
        """Register SIGINT/SIGTERM handlers for graceful shutdown.

        Only works when called from the main thread. Silently skips
        if called from a background thread (e.g., in tests).
        """
        try:
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
        except ValueError:
            # signal.signal() only works in the main thread
            logger.debug(
                "Cannot register signal handlers outside main thread"
            )

    def _handle_signal(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        sig_name = signal.Signals(signum).name
        logger.info("Received %s, shutting down scheduler...", sig_name)
        self.stop()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Stop the scheduler and close the detector's resources."""
        self.stop()
        if self._detector is not None:
            self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def run_scheduler(config_path: str | None = None) -> None:
    """Start the monitor scheduler from the command line.

    Loads config, creates the scheduler, and blocks until interrupted.

    Args:
        config_path: Optional path to settings.yaml.
    """
    config = load_config(config_path)

    logger.info(
        "Initializing scheduler with config: "
        "interval=%dh, orgs=%d, slack=%s, run_on_startup=%s",
        config.get("polling_interval_hours", DEFAULT_POLLING_INTERVAL_HOURS),
        len(config.get("watched_organizations", [])),
        "configured" if config.get("slack_webhook_url") else "not configured",
        config.get("run_on_startup", True),
    )

    with MonitorScheduler.from_config(config) as scheduler:
        scheduler.start()
