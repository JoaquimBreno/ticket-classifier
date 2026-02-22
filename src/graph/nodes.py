import re
from typing import Literal

import config
from src.logging_utils import log_usage
from src.rag import Embedder, VectorStore
from src.classification import KNNClassifier
from src.llm_local import agent_classify_and_justify, agent_justify

from .state import PipelineState

MAX_TICKET_CHARS = 10_000
WINNING_VOTER_SNIPPET_CHARS = 200


def clean_text(t: str) -> str:
    if not isinstance(t, str) or not t.strip():
        return ""
    t = re.sub(r"\s+", " ", t).strip()
    return t[:MAX_TICKET_CHARS]


def preprocess(state: PipelineState) -> PipelineState:
    text = state.get("ticket_text") or ""
    return {**state, "cleaned_text": clean_text(text)}


def embed(state: PipelineState, embedder: Embedder) -> PipelineState:
    text = state.get("cleaned_text") or state.get("ticket_text") or ""
    vec = embedder.embed(text)
    if hasattr(vec, "ndim") and vec.ndim > 1:
        vec = vec[0]
    emb_list = vec.tolist() if hasattr(vec, "tolist") else list(vec)
    return {**state, "embedding": emb_list}


def knn_classify(state: PipelineState, knn: KNNClassifier) -> PipelineState:
    embedding = state.get("embedding")
    text = state.get("cleaned_text") or state.get("ticket_text") or ""
    classe, confidence = knn.predict(text, embedding=embedding)
    log_usage(
        "knn_classify", classe=classe, confidence=confidence,
        instance_id=state.get("instance_id"),
    )
    return {
        **state,
        "classe": classe,
        "confidence": confidence,
        "used_llm_for_class": False,
    }


def llm_classify_and_justify(state: PipelineState) -> PipelineState:
    text = state.get("cleaned_text") or state.get("ticket_text") or ""
    classes = state.get("classes") or []
    knn_classe = state.get("classe")
    knn_conf = state.get("confidence")
    if not classes or knn_classe is None or knn_conf is None:
        return {**state, "used_llm_for_class": True}
    classe, justificativa, inp_tok, out_tok = agent_classify_and_justify(
        text, classes, (knn_classe, knn_conf)
    )
    log_usage(
        "agent_classify_and_justify",
        classe=classe,
        input_tokens=inp_tok,
        output_tokens=out_tok,
        instance_id=state.get("instance_id"),
    )
    total = inp_tok + out_tok
    return {
        **state,
        "classe": classe,
        "justificativa": justificativa,
        "confidence": 0.0,
        "used_llm_for_class": True,
        "classification_tokens": total,
        "justification_tokens": total,
    }


def justify(state: PipelineState, vector_store: VectorStore, k: int) -> PipelineState:
    text = state.get("cleaned_text") or state.get("ticket_text") or ""
    classe = state.get("classe") or ""
    confidence = state.get("confidence") or 0.0
    embedding = state.get("embedding")
    if embedding is not None:
        neighbors = vector_store.search_by_vector(embedding, k=k)
    else:
        neighbors = vector_store.search(text, k=k)
    winning_voters = [
        (i, dist, (txt or "").strip()[:WINNING_VOTER_SNIPPET_CHARS])
        for i, (label, txt, dist) in enumerate(neighbors)
        if label == classe
    ]
    justificativa, inp_tok, out_tok = agent_justify(
        text, classe, confidence, winning_voters
    )
    log_usage(
        "agent_justify",
        input_tokens=inp_tok,
        output_tokens=out_tok,
        instance_id=state.get("instance_id"),
    )
    return {
        **state,
        "justificativa": justificativa,
        "justification_tokens": inp_tok + out_tok,
    }


def route_after_knn(
    state: PipelineState,
) -> Literal["generate_justification", "llm_classify_and_justify"]:
    if state.get("used_llm_for_class") is True:
        return "generate_justification"
    if (state.get("confidence") or 0.0) >= config.KNN_CONFIDENCE_THRESHOLD:
        return "generate_justification"
    return "llm_classify_and_justify"


def log_and_return(state: PipelineState) -> PipelineState:
    return state
