from typing import TypedDict


class PipelineState(TypedDict, total=False):
    instance_id: str
    ticket_text: str
    cleaned_text: str
    embedding: list[float]
    classe: str
    confidence: float
    justificativa: str
    used_llm_for_class: bool
    classes: list[str]
    classification_tokens: int
    justification_tokens: int
    confidence_threshold: float


class PipelineResult(TypedDict, total=False):
    classe: str
    justificativa: str
    classification_source: str
    confidence: float
    inference_time_sec: float
    classification_tokens: int | None
    justification_tokens: int | None
