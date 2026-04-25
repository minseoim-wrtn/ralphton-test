import logging

import requests

logger = logging.getLogger(__name__)

HF_TRENDING_URL = "https://huggingface.co/api/trending"
HF_MODELS_URL = "https://huggingface.co/api/models"


def fetch_trending_models(limit: int = 20) -> list[dict]:
    try:
        resp = requests.get(
            HF_TRENDING_URL,
            params={"type": "model", "limit": limit},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        recently_trending = data if isinstance(data, list) else data.get("recentlyTrending", [])

        models = []
        for item in recently_trending:
            repo_data = item.get("repoData", item)
            models.append({
                "model_id": repo_data.get("id", item.get("id", "")),
                "author": repo_data.get("author", item.get("author", "unknown")),
                "downloads": repo_data.get("downloads", 0),
                "likes": repo_data.get("likes", 0),
            })
        return models
    except Exception:
        logger.exception("Failed to fetch trending models")
        return []


def fetch_model_info(model_id: str) -> dict:
    try:
        resp = requests.get(f"{HF_MODELS_URL}/{model_id}", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        logger.exception("Failed to fetch model info for %s", model_id)
        return {}
