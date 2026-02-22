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


def log_classification(
    classifier: str,
    classe: str,
    confidence: float | None = None,
    model: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    total_tokens: int | None = None,
    instance_id: str | None = None,
) -> None:
    payload: dict = {
        "event": "classification",
        "classifier": classifier,
        "classe": classe,
    }
    if instance_id is not None:
        payload["id"] = instance_id
    if confidence is not None:
        payload["confidence"] = round(confidence, 4)
    if model is not None:
        payload["model"] = model
    if input_tokens is not None:
        payload["input_tokens"] = input_tokens
    if output_tokens is not None:
        payload["output_tokens"] = output_tokens
    if total_tokens is not None:
        payload["total_tokens"] = total_tokens
    get_usage_logger().info("%s", json.dumps(payload, ensure_ascii=False))


def log_justification(
    model: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int | None = None,
    instance_id: str | None = None,
) -> None:
    total = total_tokens if total_tokens is not None else input_tokens + output_tokens
    payload = {
        "event": "justification",
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total,
    }
    if instance_id is not None:
        payload["id"] = instance_id
    get_usage_logger().info("%s", json.dumps(payload, ensure_ascii=False))


def log_inference(
    classification_source: str,
    classe: str,
    inference_time_sec: float,
    classification_tokens: int | None = None,
    justification_tokens: int | None = None,
    instance_id: str | None = None,
) -> None:
    payload: dict = {
        "event": "inference",
        "classification_source": classification_source,
        "classe": classe,
        "inference_time_sec": round(inference_time_sec, 4),
    }
    if instance_id is not None:
        payload["id"] = instance_id
    if classification_tokens is not None:
        payload["classification_tokens"] = classification_tokens
    if justification_tokens is not None:
        payload["justification_tokens"] = justification_tokens
    get_usage_logger().info("%s", json.dumps(payload, ensure_ascii=False))


def log_result(payload: dict, path: Path | None = None, append: bool = True):
    path = path or config.OUTPUTS / "results_sample.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
