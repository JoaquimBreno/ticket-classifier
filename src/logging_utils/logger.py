import json
import logging
from pathlib import Path

import config

LLM_USAGE_LOGGER = "ticket_classifier.llm_usage"


def get_usage_logger() -> logging.Logger:
    logger = logging.getLogger(LLM_USAGE_LOGGER)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_llm_usage(
    source: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int | None = None,
    requests: int = 1,
    **extra: str | int | float,
) -> None:
    total = total_tokens if total_tokens is not None else input_tokens + output_tokens
    payload = {
        "event": "llm_usage",
        "source": source,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total,
        "requests": requests,
        **{k: v for k, v in extra.items() if v is not None},
    }
    get_usage_logger().info("%s", json.dumps(payload, ensure_ascii=False))


def log_knn_classification(confidence: float) -> None:
    payload = {
        "event": "llm_usage",
        "source": "knn",
        "model": "n/a",
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "requests": 0,
        "confidence": round(confidence, 4),
    }
    get_usage_logger().info("%s", json.dumps(payload, ensure_ascii=False))


def log_result(payload: dict, path: Path | None = None, append: bool = True):
    path = path or config.OUTPUTS / "results_sample.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
