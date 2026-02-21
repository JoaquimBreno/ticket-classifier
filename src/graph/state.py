from typing import TypedDict


class PipelineState(TypedDict, total=False):
    ticket_text: str
    cleaned_text: str
    embedding: list[float]
    classe: str
    confidence: float
    justificativa: str
    used_llm_for_class: bool
    classes: list[str]
