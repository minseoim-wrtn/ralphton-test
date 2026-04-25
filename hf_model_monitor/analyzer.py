import json
import logging

import anthropic

from .config import ANTHROPIC_API_KEY

logger = logging.getLogger(__name__)

ANALYSIS_PROMPT = """You are an AI model analyst for a B2B AI PM team. Analyze the following new trending model and compare it against reference models.

## New Model Data
{model_data}

## Reference Models
{reference_data}

Respond in JSON with exactly these keys:
- "summary": 1-2 sentence summary of the model's key characteristics and positioning
- "differentiation_points": list of 2-4 strings describing how this model differs from existing options
- "b2b_assessment": object with "pros" (list of strings) and "warnings" (list of strings) for B2B adoption
- "takeaway": single sentence with the key decision-relevant insight for a PM choosing models

Focus on practical B2B implications: licensing, API availability, cost, performance trade-offs, and deployment requirements.
Respond ONLY with valid JSON, no markdown."""


def analyze_model(model_metadata: dict, reference_data: dict) -> dict:
    if not ANTHROPIC_API_KEY:
        logger.warning("No ANTHROPIC_API_KEY set, returning fallback analysis")
        return _fallback_analysis(model_metadata)

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        prompt = ANALYSIS_PROMPT.format(
            model_data=json.dumps(model_metadata, indent=2, default=str),
            reference_data=json.dumps(reference_data, indent=2, default=str),
        )
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text
        return _parse_analysis(text)
    except Exception:
        logger.exception("Claude API analysis failed")
        return _fallback_analysis(model_metadata)


def _parse_analysis(text: str) -> dict:
    try:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
        result = json.loads(cleaned)
        return {
            "summary": result.get("summary", "Analysis unavailable"),
            "differentiation_points": result.get("differentiation_points", []),
            "b2b_assessment": result.get("b2b_assessment", {"pros": [], "warnings": []}),
            "takeaway": result.get("takeaway", "N/A"),
        }
    except (json.JSONDecodeError, IndexError):
        return _fallback_analysis({})


def _fallback_analysis(model_metadata: dict) -> dict:
    basic = model_metadata.get("basic", {})
    name = basic.get("name", "Unknown")
    return {
        "summary": f"{name} is a newly trending model on HuggingFace. Manual review recommended.",
        "differentiation_points": ["Newly trending on HuggingFace"],
        "b2b_assessment": {
            "pros": ["Community interest (trending)"],
            "warnings": ["Requires manual evaluation for production use"],
        },
        "takeaway": f"Evaluate {name} against your specific use case requirements.",
    }
