import re
import time
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

import config
from .state import PipelineState
from src.logging_utils import log_classification, log_inference, log_justification
from src.rag import Embedder, VectorStore
from src.classification import KNNClassifier, LLMClassifier
from src.justification.generator import generate_justification_text


def _clean_text(t: str) -> str:
    if not isinstance(t, str) or not t.strip():
        return ""
    t = re.sub(r"\s+", " ", t).strip()
    return t[:10000]


def _preprocess(state: PipelineState) -> PipelineState:
    text = state.get("ticket_text") or ""
    cleaned = _clean_text(text)
    return {**state, "cleaned_text": cleaned}


def _embed(state: PipelineState, embedder: Embedder) -> PipelineState:
    text = state.get("cleaned_text") or state.get("ticket_text") or ""
    vec = embedder.embed(text)
    if hasattr(vec, "ndim") and vec.ndim > 1:
        vec = vec[0]
    emb_list = vec.tolist() if hasattr(vec, "tolist") else list(vec)
    return {**state, "embedding": emb_list}


def _knn_classify(state: PipelineState, knn: KNNClassifier) -> PipelineState:
    text = state.get("cleaned_text") or state.get("ticket_text") or ""
    classe, confidence = knn.predict(text)
    log_classification(classifier="knn", classe=classe, confidence=confidence)
    return {
        **state,
        "classe": classe,
        "confidence": confidence,
        "used_llm_for_class": False,
    }


def _llm_classify(state: PipelineState, llm: LLMClassifier) -> PipelineState:
    text = state.get("cleaned_text") or state.get("ticket_text") or ""
    classes = state.get("classes") or []
    if not classes:
        return {**state, "used_llm_for_class": True}
    classe, usage_info = llm.predict(text, classes)
    if usage_info:
        log_classification(
            classifier="llm",
            classe=classe,
            model=llm.model,
            input_tokens=usage_info["input_tokens"],
            output_tokens=usage_info["output_tokens"],
            total_tokens=usage_info.get("total_tokens"),
        )
        total = usage_info.get("total_tokens")
        state_extra = {"classification_tokens": total} if total is not None else {}
    else:
        log_classification(classifier="llm", classe=classe, model=llm.model)
        state_extra = {}
    return {
        **state,
        **state_extra,
        "classe": classe,
        "confidence": 0.0,
        "used_llm_for_class": True,
    }


def _justify(state: PipelineState, vector_store: VectorStore, k: int) -> PipelineState:
    text = state.get("cleaned_text") or state.get("ticket_text") or ""
    classe = state.get("classe") or ""
    confidence = state.get("confidence")
    neighbors = vector_store.search(text, k=k) if text else []
    justificativa, usage_info = generate_justification_text(
        ticket_text=text,
        classe=classe,
        confidence=confidence,
        neighbors=neighbors,
    )
    if usage_info:
        log_justification(
            model=config.OPENROUTER_MODEL,
            input_tokens=usage_info["input_tokens"],
            output_tokens=usage_info["output_tokens"],
            total_tokens=usage_info.get("total_tokens"),
        )
        total = usage_info.get("total_tokens")
        state_extra = {"justification_tokens": total} if total is not None else {}
    else:
        state_extra = {}
    return {**state, "justificativa": justificativa, **state_extra}


def _route_after_knn(state: PipelineState) -> Literal["generate_justification", "llm_classify"]:
    if state.get("used_llm_for_class") is True:
        return "generate_justification"
    conf = state.get("confidence", 0.0)
    if conf >= config.KNN_CONFIDENCE_THRESHOLD:
        return "generate_justification"
    return "llm_classify"


def _log_and_return(state: PipelineState) -> PipelineState:
    return state


def build_pipeline(
    vector_store: VectorStore,
    classes: list[str],
    embedder: Embedder | None = None,
):
    embedder = embedder or Embedder()
    knn = KNNClassifier(vector_store)
    llm = LLMClassifier(vector_store)

    builder = StateGraph(PipelineState)
    builder.add_node("preprocess", _preprocess)
    builder.add_node("embed", lambda s: _embed(s, embedder))
    builder.add_node("knn_classify", lambda s: _knn_classify(s, knn))
    builder.add_node("llm_classify", lambda s: _llm_classify(s, llm))
    builder.add_node(
        "generate_justification",
        lambda s: _justify(s, vector_store, k=config.KNN_K),
    )
    builder.add_node("log_and_return", _log_and_return)

    builder.set_entry_point("preprocess")
    builder.add_edge("preprocess", "embed")
    builder.add_edge("embed", "knn_classify")
    builder.add_conditional_edges(
        "knn_classify",
        _route_after_knn,
        {"generate_justification": "generate_justification", "llm_classify": "llm_classify"},
    )
    builder.add_edge("llm_classify", "generate_justification")
    builder.add_edge("generate_justification", "log_and_return")
    builder.add_edge("log_and_return", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory), embedder, knn, llm


def run_pipeline(compiled, ticket_text: str, classes: list[str], *, thread_id: str = "default"):
    run_config = {"configurable": {"thread_id": thread_id}}
    initial: PipelineState = {
        "ticket_text": ticket_text,
        "classes": classes,
    }
    t0 = time.perf_counter()
    final = compiled.invoke(initial, run_config)
    inference_time_sec = time.perf_counter() - t0
    if isinstance(final, dict):
        values = final
    else:
        values = getattr(final, "values", dict(final))
    classification_source = "llm" if values.get("used_llm_for_class") else "knn"
    log_inference(
        classification_source=classification_source,
        classe=values.get("classe", ""),
        inference_time_sec=inference_time_sec,
        classification_tokens=values.get("classification_tokens"),
        justification_tokens=values.get("justification_tokens"),
    )
    return {
        "classe": values.get("classe", ""),
        "justificativa": values.get("justificativa", ""),
        "classification_source": classification_source,
        "inference_time_sec": inference_time_sec,
    }
