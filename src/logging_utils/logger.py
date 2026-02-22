import json
import logging
from pathlib import Path

import config

LLM_USAGE_LOGGER = "ticket_classifier.llm_usage"


def get_usage_logger() -> logging.Logger:
    logger = logging.getLogger(LLM_USAGE_LOGGER)
    if not logger.handlers:
        stream = logging.StreamHandler()
        stream.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


_STEP_EVENT: dict[str, str] = {
    "knn_classify": "classification",
    "agent_justify": "justification",
    "agent_classify_and_justify": "classify_and_justify",
    "inference": "inference",
}


def log_usage(step: str, **kwargs) -> None:
    payload: dict = {"step": step, "event": _STEP_EVENT.get(step, step)}
    for k, v in kwargs.items():
        if v is None:
            continue
        if k == "confidence" or k == "inference_time_sec":
            payload[k] = round(float(v), 4)
        else:
            payload[k] = v
    if "input_tokens" in payload and "output_tokens" in payload and "total_tokens" not in payload:
        payload["total_tokens"] = payload["input_tokens"] + payload["output_tokens"]
    get_usage_logger().info("%s", json.dumps(payload, ensure_ascii=False))


def log_inference(
    classification_source: str,
    classe: str,
    inference_time_sec: float,
    classification_tokens: int | None = None,
    justification_tokens: int | None = None,
    instance_id: str | None = None,
) -> None:
    log_usage(
        "inference",
        classification_source=classification_source,
        classe=classe,
        inference_time_sec=inference_time_sec,
        classification_tokens=classification_tokens,
        justification_tokens=justification_tokens,
        instance_id=instance_id,
    )


def log_result(payload: dict, path: Path | None = None, append: bool = True):
    path = path or config.OUTPUTS / "results_sample.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
