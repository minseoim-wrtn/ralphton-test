import argparse
import logging
import sys

from .config import load_config, REFERENCE_MODELS
from .detector import ModelDetector
from .metadata_collector import collect_model_metadata
from .seed_data import get_reference_data_from_seed
from .slack_notifier import SlackNotifier, format_crawler_errors_summary, send_to_slack
from .trending_detector import TrendingDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _get_reference_data() -> dict:
    """Load reference data for Slack comparison tables.

    Tries seed data first (richer), falls back to config.REFERENCE_MODELS.
    """
    refs = get_reference_data_from_seed()
    return refs if refs else REFERENCE_MODELS


def _notify_new_models(
    new_models: list[dict],
    notifier: SlackNotifier,
    detector: ModelDetector,
) -> int:
    """Enrich new models with metadata and send Slack reports.

    Returns the number of models successfully notified.
    """
    if not new_models or not notifier.is_configured:
        return 0

    reference_data = _get_reference_data()
    notified = 0

    for model in new_models:
        model_id = model.get("model_id", "")
        if not model_id:
            continue

        logger.info("Collecting metadata for %s", model_id)
        metadata = collect_model_metadata(model_id)

        success = notifier.send_report(metadata, reference_data)
        if success:
            detector.mark_model_notified(model_id)
            notified += 1
            logger.info("Slack report sent for %s", model_id)
        else:
            logger.warning("Failed to send Slack report for %s", model_id)

    return notified


def run(config_path: str | None = None) -> dict:
    """Run the full detection + notification pipeline once.

    This is the primary one-shot entry point. It:
    1. Loads config from settings.yaml
    2. Creates a ModelDetector with SQLite-backed state
    3. Bootstraps the store with seed models (on first run)
    4. Runs detection across all watched organizations
    5. Enriches new models with metadata and sends Slack reports
    6. Runs trending/surge detection if enabled
    7. Marks detected models as processed
    8. Returns a summary dict

    Args:
        config_path: Optional path to settings.yaml override.

    Returns:
        Dict with new_models_found, models_processed, detection_summary, errors.
    """
    logger.info("Starting HF Model Monitor run")
    config = load_config(config_path)

    result = {
        "new_models_found": 0,
        "models_processed": 0,
        "models_notified": 0,
        "trending_candidates": 0,
        "detection_summary": "",
        "errors": [],
        "bootstrap_stats": {},
    }

    notifier = SlackNotifier.from_config(config)

    with ModelDetector.from_config(config) as detector:
        # Step 1: Bootstrap store on first run (prevents false positives)
        bootstrap_stats = detector.initialize_store()
        result["bootstrap_stats"] = bootstrap_stats
        if bootstrap_stats["was_first_run"]:
            logger.info(
                "First run bootstrap: %d seed models loaded",
                bootstrap_stats["seed_models_loaded"],
            )

        # Step 2: Run detection across all watched orgs
        detection = detector.run()
        result["new_models_found"] = detection.total_new
        result["detection_summary"] = detection.summary()
        result["errors"] = detection.errors

        if detection.total_new == 0:
            logger.info("No new models detected")
        else:
            logger.info(
                "Detected %d new model(s): %s",
                detection.total_new,
                [m.get("model_id") for m in detection.new_models],
            )

        # Step 3: Send Slack notifications for new models
        notified = _notify_new_models(
            detection.new_models, notifier, detector,
        )
        result["models_notified"] = notified

        # Step 4: Send crawler error summary if there were partial failures
        if detection.errors and notifier.is_configured:
            org_errors = [
                {"source": r.org, "error": r.error}
                for r in detection.org_results
                if r.error
            ]
            if org_errors:
                alert = format_crawler_errors_summary(
                    org_errors=org_errors,
                    total_orgs=detection.orgs_polled,
                    next_retry_hours=config.get("polling_interval_hours", 12),
                )
                send_to_slack(alert, webhook_url=config.get("slack_webhook_url", ""))

        # Step 5: Mark detected models as processed
        processed = detector.mark_models_processed(detection)
        result["models_processed"] = processed

    # Step 6: Run trending/surge detection if enabled
    trending_thresholds = config.get("trending_thresholds", {})
    if trending_thresholds.get("enabled", False):
        try:
            with TrendingDetector.from_config(config) as trending:
                trending_result = trending.run()
                result["trending_candidates"] = trending_result.total_candidates

                if trending_result.candidates and notifier.is_configured:
                    reference_data = _get_reference_data()
                    for candidate in trending_result.candidates:
                        metadata = collect_model_metadata(candidate.model_id)
                        notifier.send_report(metadata, reference_data)
                        logger.info(
                            "Trending alert sent for %s (%s)",
                            candidate.model_id,
                            candidate.reason,
                        )
        except Exception as exc:
            logger.exception("Trending detection failed: %s", exc)
            result["errors"].append(f"Trending detection: {exc}")

    logger.info(
        "Run complete: %d new, %d notified, %d processed, %d trending, %d errors",
        result["new_models_found"],
        result["models_notified"],
        result["models_processed"],
        result["trending_candidates"],
        len(result["errors"]),
    )
    return result


def run_scheduled(config_path: str | None = None) -> None:
    """Start the periodic scheduler for continuous monitoring.

    This is the primary entry point for production use. It:
    1. Loads config from settings.yaml (or provided path)
    2. Runs an immediate detection on startup
    3. Schedules recurring detections at the configured interval
    4. Blocks until interrupted (Ctrl+C or SIGTERM)

    Args:
        config_path: Optional path to settings.yaml override.
    """
    from .scheduler import run_scheduler

    run_scheduler(config_path)


def main() -> None:
    """CLI entry point with subcommand support.

    Usage:
        python -m hf_model_monitor              # one-shot run (legacy)
        python -m hf_model_monitor --once       # one-shot run (explicit)
        python -m hf_model_monitor --schedule   # start periodic scheduler
        python -m hf_model_monitor --status     # show scheduler status
    """
    parser = argparse.ArgumentParser(
        description="HuggingFace Model Monitor — detect new AI model releases",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--once",
        action="store_true",
        default=True,
        help="Run detection once and exit (default)",
    )
    mode.add_argument(
        "--schedule",
        action="store_true",
        help="Start periodic polling scheduler (runs continuously)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to settings.yaml config file",
    )
    args = parser.parse_args()

    if args.schedule:
        run_scheduled(args.config)
    else:
        summary = run(args.config)
        print(f"Result: {summary}")
        sys.exit(1 if summary["errors"] else 0)


if __name__ == "__main__":
    main()
