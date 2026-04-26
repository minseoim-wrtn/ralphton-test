import logging

import requests

from .config import SLACK_WEBHOOK_URL

logger = logging.getLogger(__name__)


def format_report(model_metadata: dict, reference_data: dict) -> str:
    basic = model_metadata.get("basic", {})
    community = model_metadata.get("community", {})

    header = f":bar_chart: *New Trending: {basic.get('name', 'Unknown')}* ({basic.get('org', 'N/A')})"
    info = f":mag: License: {basic.get('license', 'N/A')} | Downloads: {community.get('downloads', 0):,} | Likes: {community.get('likes', 0):,}"

    table = _build_comparison_table(model_metadata, reference_data)

    parts = [header, "", info, "", table]
    return "\n".join(parts)


def _build_comparison_table(model_metadata: dict, reference_data: dict) -> str:
    basic = model_metadata.get("basic", {})
    perf = model_metadata.get("performance", {})
    deploy = model_metadata.get("deployment", {})
    cost = model_metadata.get("cost", {})
    practical = model_metadata.get("practical", {})

    col_names = ["", basic.get("name", "New")] + list(reference_data.keys())

    metrics = [
        ("Params", basic.get("params", "N/A"), "params"),
        ("MMLU", perf.get("mmlu", "N/A"), "mmlu"),
        ("HumanEval", perf.get("humaneval", "N/A"), "humaneval"),
        ("License", basic.get("license", "N/A"), "license"),
        ("API$/1M", cost.get("api_price_per_million_tokens", "N/A"), "api_price"),
        ("Context", practical.get("context_window", "N/A"), "context_window"),
        ("VRAM", deploy.get("vram_estimate", "N/A"), "vram"),
    ]

    rows = []
    for label, new_val, ref_key in metrics:
        ref_vals = [str(ref.get(ref_key, "N/A")) for ref in reference_data.values()]
        rows.append([label, str(new_val)] + ref_vals)

    all_rows = [col_names] + rows
    col_widths = [
        max(len(row[i]) for row in all_rows) for i in range(len(col_names))
    ]

    def fmt(row: list[str]) -> str:
        return "  ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))

    sep = "  ".join("-" * w for w in col_widths)
    lines = [fmt(col_names), sep] + [fmt(r) for r in rows]
    return "```\n" + "\n".join(lines) + "\n```"


def send_to_slack(message: str, webhook_url: str | None = None) -> bool:
    url = webhook_url or SLACK_WEBHOOK_URL
    if not url:
        logger.warning("No SLACK_WEBHOOK_URL configured, skipping send")
        return False
    if not message:
        return False

    try:
        resp = requests.post(
            url,
            json={"text": message},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception:
        logger.exception("Failed to send Slack message")
        return False
