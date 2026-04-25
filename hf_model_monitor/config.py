import os

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")

SEEN_MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "seen_models.json")

REFERENCE_MODELS = {
    "GPT-4o": {
        "params": "~1.8T (estimated)",
        "mmlu": "88.7",
        "humaneval": "90.2",
        "license": "Proprietary",
        "api_price": "$2.50",
        "context_window": "128K",
        "vram": "N/A (API only)",
    },
    "Claude Sonnet 4": {
        "params": "N/A",
        "mmlu": "88.8",
        "humaneval": "93.0",
        "license": "Proprietary",
        "api_price": "$3.00",
        "context_window": "200K",
        "vram": "N/A (API only)",
    },
    "Gemini 2.5 Pro": {
        "params": "N/A",
        "mmlu": "89.0",
        "humaneval": "84.0",
        "license": "Proprietary",
        "api_price": "$1.25",
        "context_window": "1M",
        "vram": "N/A (API only)",
    },
    "Llama-3.1-405B": {
        "params": "405B",
        "mmlu": "87.3",
        "humaneval": "61.0",
        "license": "Llama 3.1 Community",
        "api_price": "$0.90",
        "context_window": "128K",
        "vram": "~800GB FP16",
    },
    "Qwen-2.5-72B": {
        "params": "72B",
        "mmlu": "85.3",
        "humaneval": "86.4",
        "license": "Apache 2.0",
        "api_price": "$0.30",
        "context_window": "128K",
        "vram": "~144GB FP16",
    },
    "Mistral-Large": {
        "params": "123B",
        "mmlu": "84.0",
        "humaneval": "72.0",
        "license": "Mistral Research",
        "api_price": "$2.00",
        "context_window": "128K",
        "vram": "~246GB FP16",
    },
}
