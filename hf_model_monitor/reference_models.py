from .config import REFERENCE_MODELS


def get_reference_data() -> dict[str, dict]:
    return dict(REFERENCE_MODELS)
