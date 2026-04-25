import logging
import sys

from .hf_fetcher import fetch_trending_models
from .metadata_collector import collect_model_metadata
from .reference_models import get_reference_data
from .state import detect_new_models, load_previous_models, save_current_models
from .analyzer import analyze_model
from .slack_notifier import format_report, send_to_slack

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run() -> dict:
    logger.info("Starting HF Model Monitor run")
    result = {"new_models_found": 0, "reports_sent": 0, "errors": []}

    trending = fetch_trending_models()
    if not trending:
        logger.info("No trending models fetched")
        return result

    previous = load_previous_models()
    new_models = detect_new_models(trending, previous)

    if not new_models:
        logger.info("No new trending models detected")
        all_ids = [m["model_id"] for m in trending]
        save_current_models(all_ids)
        return result

    result["new_models_found"] = len(new_models)
    logger.info("Found %d new trending models", len(new_models))

    reference_data = get_reference_data()

    for model in new_models:
        model_id = model["model_id"]
        try:
            logger.info("Processing model: %s", model_id)
            metadata = collect_model_metadata(model_id, trending_data=model)
            analysis = analyze_model(metadata, reference_data)
            report = format_report(metadata, analysis, reference_data)

            if send_to_slack(report):
                result["reports_sent"] += 1
                logger.info("Report sent for %s", model_id)
            else:
                logger.warning("Failed to send report for %s", model_id)
        except Exception as e:
            logger.exception("Error processing model %s", model_id)
            result["errors"].append(f"{model_id}: {e}")

    all_ids = [m["model_id"] for m in trending]
    save_current_models(all_ids)

    logger.info(
        "Run complete: %d new, %d sent, %d errors",
        result["new_models_found"],
        result["reports_sent"],
        len(result["errors"]),
    )
    return result


if __name__ == "__main__":
    summary = run()
    print(f"Result: {summary}")
    sys.exit(1 if summary["errors"] else 0)
