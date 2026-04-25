import logging

from .hf_fetcher import fetch_model_info

logger = logging.getLogger(__name__)


def _safe_get(data: dict, *keys, default="N/A"):
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    return current if current is not None else default


def _estimate_vram(params_str: str) -> str:
    try:
        text = str(params_str).upper().replace(",", "")
        if "B" in text:
            num = float(text.replace("B", "").strip())
            multiplier = 1
        elif "M" in text:
            num = float(text.replace("M", "").strip())
            multiplier = 0.001
        else:
            num = float(text)
            multiplier = 1e-9
        params_b = num * multiplier
        vram_fp16_gb = params_b * 2
        return f"~{vram_fp16_gb:.0f}GB FP16"
    except (ValueError, TypeError):
        return "N/A"


def collect_model_metadata(model_id: str, trending_data: dict | None = None) -> dict:
    info = fetch_model_info(model_id)
    if not info and not trending_data:
        return _empty_metadata(model_id)

    safetensors = _safe_get(info, "safetensors", default={})
    params_info = _safe_get(safetensors, "total", default=None)
    if params_info and params_info != "N/A":
        params_str = _format_params(params_info)
    else:
        params_str = _safe_get(info, "config", "num_parameters", default="N/A")
        if params_str != "N/A":
            params_str = _format_params(params_str)

    card_data = _safe_get(info, "cardData", default={})
    if not isinstance(card_data, dict):
        card_data = {}

    model_card_eval = _extract_eval_results(card_data)

    tags = info.get("tags", []) if info else []
    license_tag = _safe_get(card_data, "license", default="N/A")
    if license_tag == "N/A":
        for tag in tags:
            if tag.startswith("license:"):
                license_tag = tag.split(":", 1)[1]
                break

    arch = _safe_get(info, "config", "model_type", default="N/A")

    trending = trending_data or {}

    return {
        "basic": {
            "name": model_id.split("/")[-1] if "/" in model_id else model_id,
            "org": model_id.split("/")[0] if "/" in model_id else "N/A",
            "params": params_str,
            "architecture": arch,
            "license": license_tag,
        },
        "performance": {
            "mmlu": model_card_eval.get("mmlu", "N/A"),
            "humaneval": model_card_eval.get("humaneval", "N/A"),
            "arena_rank": "N/A",
        },
        "practical": {
            "context_window": _extract_context_window(info),
            "multilingual": _check_multilingual(tags),
            "fine_tuning_support": _check_fine_tuning(info),
        },
        "deployment": {
            "vram_estimate": _estimate_vram(params_str),
            "quantization_options": _extract_quantizations(tags),
            "api_available": _check_api_available(info),
        },
        "community": {
            "downloads": trending.get("downloads", info.get("downloads", 0)),
            "likes": trending.get("likes", info.get("likes", 0)),
            "trending_rank": trending.get("trending_rank", "N/A"),
        },
        "cost": {
            "api_price_per_million_tokens": "N/A",
            "hosting_cost_estimate": "N/A",
        },
    }


def _empty_metadata(model_id: str) -> dict:
    name = model_id.split("/")[-1] if "/" in model_id else model_id
    org = model_id.split("/")[0] if "/" in model_id else "N/A"
    return {
        "basic": {"name": name, "org": org, "params": "N/A", "architecture": "N/A", "license": "N/A"},
        "performance": {"mmlu": "N/A", "humaneval": "N/A", "arena_rank": "N/A"},
        "practical": {"context_window": "N/A", "multilingual": "N/A", "fine_tuning_support": "N/A"},
        "deployment": {"vram_estimate": "N/A", "quantization_options": "N/A", "api_available": "N/A"},
        "community": {"downloads": 0, "likes": 0, "trending_rank": "N/A"},
        "cost": {"api_price_per_million_tokens": "N/A", "hosting_cost_estimate": "N/A"},
    }


def _format_params(value) -> str:
    try:
        num = float(value)
        if num >= 1e12:
            return f"{num / 1e12:.1f}T"
        if num >= 1e9:
            return f"{num / 1e9:.1f}B"
        if num >= 1e6:
            return f"{num / 1e6:.0f}M"
        return str(int(num))
    except (ValueError, TypeError):
        return str(value) if value else "N/A"


def _extract_eval_results(card_data: dict) -> dict:
    results = {}
    eval_results = card_data.get("eval_results", card_data.get("model-index", []))
    if isinstance(eval_results, list):
        for entry in eval_results:
            if isinstance(entry, dict):
                results_list = entry.get("results", [])
                if isinstance(results_list, list):
                    for r in results_list:
                        dataset = _safe_get(r, "dataset", "name", default="")
                        metrics = r.get("metrics", [])
                        if isinstance(metrics, list):
                            for m in metrics:
                                name = m.get("name", "").lower()
                                val = m.get("value", "N/A")
                                if "mmlu" in dataset.lower() or "mmlu" in name:
                                    results["mmlu"] = str(val)
                                if "humaneval" in dataset.lower() or "humaneval" in name:
                                    results["humaneval"] = str(val)
    return results


def _extract_context_window(info: dict) -> str:
    config = info.get("config", {}) if info else {}
    if not isinstance(config, dict):
        return "N/A"
    for key in ("max_position_embeddings", "max_seq_len", "seq_length", "n_positions"):
        val = config.get(key)
        if val:
            try:
                num = int(val)
                if num >= 1000:
                    return f"{num // 1024}K" if num >= 1024 else str(num)
                return str(num)
            except (ValueError, TypeError):
                pass
    return "N/A"


def _check_multilingual(tags: list) -> str:
    ml_tags = {"multilingual", "zh", "ja", "ko", "de", "fr", "es", "ar"}
    return "Yes" if any(t.lower() in ml_tags for t in tags) else "N/A"


def _check_fine_tuning(info: dict) -> str:
    tags = info.get("tags", []) if info else []
    if any("gguf" in t.lower() or "adapter" in t.lower() or "lora" in t.lower() for t in tags):
        return "Yes (LoRA/GGUF)"
    library = info.get("library_name", "") if info else ""
    if library in ("transformers", "peft"):
        return "Yes"
    return "N/A"


def _extract_quantizations(tags: list) -> str:
    quant_tags = [t for t in tags if any(q in t.lower() for q in ("gguf", "gptq", "awq", "bnb", "quant"))]
    return ", ".join(quant_tags) if quant_tags else "N/A"


def _check_api_available(info: dict) -> str:
    if not info:
        return "N/A"
    pipeline_tag = info.get("pipeline_tag", "")
    if info.get("inference") or pipeline_tag:
        return "HF Inference API"
    return "N/A"
