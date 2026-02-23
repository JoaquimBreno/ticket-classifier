import time

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

import config
from .state import PipelineState, PipelineResult
from . import nodes
from src.logging_utils import log_inference
from src.rag import Embedder, VectorStore
from src.classification import KNNClassifier


def build_pipeline(
    vector_store: VectorStore,
    embedder: Embedder | None = None,
    *,
    use_checkpointer: bool = False,
):
    embedder = embedder or Embedder()
    knn = KNNClassifier(vector_store)
    builder = StateGraph(PipelineState)
    builder.add_node("preprocess", nodes.preprocess)
    builder.add_node("embed", lambda s: nodes.embed(s, embedder))
    builder.add_node("knn_classify", lambda s: nodes.knn_classify(s, knn))
    builder.add_node("llm_classify_and_justify", nodes.llm_classify_and_justify)
    builder.add_node(
        "generate_justification",
        lambda s: nodes.justify(s, vector_store, k=config.KNN_K),
    )
    builder.add_node("log_and_return", nodes.log_and_return)

    builder.set_entry_point("preprocess")
    builder.add_edge("preprocess", "embed")
    builder.add_edge("embed", "knn_classify")
    builder.add_conditional_edges(
        "knn_classify",
        nodes.route_after_knn,
        {
            "generate_justification": "generate_justification",
            "llm_classify_and_justify": "llm_classify_and_justify",
        },
    )
    builder.add_edge("llm_classify_and_justify", "log_and_return")
    builder.add_edge("generate_justification", "log_and_return")
    builder.add_edge("log_and_return", END)

    if use_checkpointer:
        compiled = builder.compile(checkpointer=MemorySaver())
    else:
        compiled = builder.compile()
    return compiled, embedder, knn


def run_pipeline(
    compiled,
    ticket_text: str,
    classes: list[str],
    *,
    thread_id: str = "default",
    instance_id: str | None = None,
    confidence_threshold: float | None = None,
) -> PipelineResult:
    run_config = {"configurable": {"thread_id": thread_id}}
    initial: PipelineState = {"ticket_text": ticket_text, "classes": classes}
    if instance_id is not None:
        initial["instance_id"] = instance_id
    if confidence_threshold is not None:
        initial["confidence_threshold"] = confidence_threshold
    t0 = time.perf_counter()
    final = compiled.invoke(initial, run_config)
    inference_time_sec = time.perf_counter() - t0
    values = final if isinstance(final, dict) else getattr(final, "values", dict(final))
    classification_source = "llm" if values.get("used_llm_for_class") else "knn"
    log_inference(
        classification_source=classification_source,
        classe=values.get("classe", ""),
        inference_time_sec=inference_time_sec,
        classification_tokens=values.get("classification_tokens"),
        justification_tokens=values.get("justification_tokens"),
        instance_id=values.get("instance_id"),
    )
    return {
        "classe": values.get("classe", ""),
        "justificativa": values.get("justificativa", ""),
        "classification_source": classification_source,
        "confidence": values.get("confidence"),
        "inference_time_sec": inference_time_sec,
        "classification_tokens": values.get("classification_tokens"),
        "justification_tokens": values.get("justification_tokens"),
    }
