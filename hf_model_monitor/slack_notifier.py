import logging

import requests

from .config import SLACK_WEBHOOK_URL

logger = logging.getLogger(__name__)


def format_report(
    model_metadata: dict, analysis: dict, reference_data: dict
) -> str:
    basic = model_metadata.get("basic", {})

    header = f":bar_chart: *New Trending: {basic.get('name', 'Unknown')}* ({basic.get('org', 'N/A')})"
    summary = f"*Summary:* {analysis.get('summary', 'N/A')}"

    table_header = "| | " + basic.get("name", "New") + " | " + " | ".join(reference_data.keys()) + " |"
    table_sep = "|------|" + "------|" * (1 + len(reference_data))

    rows = _build_comparison_rows(model_metadata, reference_data)
    table = "\n".join([table_header, table_sep] + rows)

    b2b = analysis.get("b2b_assessment", {})
    pros = b2b.get("pros", [])
    warnings = b2b.get("warnings", [])
    b2b_lines = ["*B2B Assessment:*"]
    for p in pros:
        b2b_lines.append(f"  :white_check_mark: {p}")
    for w in warnings:
        b2b_lines.append(f"  :warning: {w}")
    b2b_section = "\n".join(b2b_lines)

    takeaway = f":star: *Takeaway:* {analysis.get('takeaway', 'N/A')}"

    parts = [header, "", summary, "", table, "", b2b_section, "", takeaway]
    return "\n".join(parts)


def _build_comparison_rows(model_metadata: dict, reference_data: dict) -> list[str]:
    basic = model_metadata.get("basic", {})
    perf = model_metadata.get("performance", {})
    deploy = model_metadata.get("deployment", {})
    cost = model_metadata.get("cost", {})
    practical = model_metadata.get("practical", {})

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
        ref_vals = [ref.get(ref_key, "N/A") for ref in reference_data.values()]
        row = f"| {label} | {new_val} | " + " | ".join(str(v) for v in ref_vals) + " |"
        rows.append(row)
    return rows


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
