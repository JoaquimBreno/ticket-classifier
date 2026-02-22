import json
from pathlib import Path

from pydantic import BaseModel, Field

import config

_llm = None


class TicketClassification(BaseModel):
    classe: str = Field(
        description="A categoria exata do ticket (deve pertencer à lista fornecida)."
    )
    justificativa: str = Field(
        description="Explicação de 1 a 3 frases do motivo da classificação, citando termos do ticket, em português."
    )


class ClasseOnly(BaseModel):
    classe: str = Field(description="A categoria exata do ticket (uma das classes fornecidas).")


class JustificativaOnly(BaseModel):
    justificativa: str = Field(
        description="Explicação de 1 a 3 frases em português que justifica a classificação, citando termos do ticket."
    )


def _project_root() -> Path:
    root = Path(config.ROOT)
    s = str(root).rstrip(":")
    return Path(s) if s else root


def _download_model_to(local_dir: Path) -> Path:
    import os
    from huggingface_hub import hf_hub_download
    local_dir.mkdir(parents=True, exist_ok=True)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    try:
        path = hf_hub_download(
            repo_id=config.LLAMA_HF_REPO,
            filename=config.LLAMA_HF_FILENAME,
            local_dir=str(local_dir),
            token=token,
        )
        return Path(path)
    except Exception as e:
        if "401" in str(e) or "Unauthorized" in str(e) or "RepositoryNotFound" in str(type(e).__name__):
            raise RuntimeError(
                "Download do modelo no Hugging Face falhou (401/Unauthorized). "
                "Defina HF_TOKEN no .env (token em https://huggingface.co/settings/tokens)."
            ) from e
        raise


def get_model_path() -> Path:
    if config.LLAMA_MODEL_PATH:
        p = Path(config.LLAMA_MODEL_PATH)
        if not p.is_absolute():
            p = _project_root() / p
        if p.exists():
            return p
        alt = Path(config.LLAMA_MODEL_PATH).resolve()
        if alt.exists():
            return alt
        return _download_model_to(p.parent)
    import os
    models_dir = _project_root() / os.getenv("MODELS_DIR", "models")
    return _download_model_to(models_dir)


def get_llm():
    global _llm
    if _llm is not None:
        return _llm
    from llama_cpp import Llama
    model_path = get_model_path()
    _llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=config.LLAMA_N_GPU_LAYERS,
        n_ctx=config.LLAMA_N_CTX,
        verbose=False,
    )
    return _llm

def _format_neighbors(neighbors: list[tuple[str, str, float]], max_len: int = 120) -> str:
    lines = []
    for label, text, dist in neighbors:
        s = (text or "").strip()
        snippet = s[:max_len]
        if len(s) > max_len:
            snippet += "..."
        lines.append(f"- [{label}] (dist {dist:.2f}): {snippet}")
    return "\n".join(lines) if lines else ""


def _format_winning_voters(winning_voters: list[tuple[int, float, str]], max_len: int = 120) -> str:
    lines = []
    for idx, dist, text in winning_voters:
        s = (text or "").strip()[:max_len]
        if len((text or "").strip()) > max_len:
            s += "..."
        lines.append(f"- #{idx} (dist {dist:.2f}): {s}")
    return "\n".join(lines) if lines else ""

def agent_classifier(
    texto_ticket: str,
    classes_validas: list[str],
    neighbors: list[tuple[str, str, float]] | None = None,
) -> tuple[str, int, int]:
    llm = get_llm()
    user_parts = [f"Classifique o ticket estritamente em uma destas categorias: {classes_validas}.\n\nTicket:\n{texto_ticket[:2000]}"]
    if neighbors:
        user_parts.append("\n\nTickets similares do treino (use como referência):\n" + _format_neighbors(neighbors))
    user_content = "".join(user_parts)
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are an expert IT analyst. Output only the exact category name, nothing else. Be objective.",
            },
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
        max_tokens=64,
        response_format={
            "type": "json_object",
            "schema": ClasseOnly.model_json_schema(),
        },
    )
    usage = response.get("usage") or {}
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    raw = response["choices"][0]["message"]["content"]
    parsed = json.loads(raw)
    out = ClasseOnly(**parsed)
    return out.classe.strip(), input_tokens, output_tokens


def _justify_knn(
    llm,
    ticket_text: str,
    classe: str,
    confidence: float,
    winning_voters: list[tuple[int, float, str]],
) -> tuple[str, int, int]:
    system = "Analista de tickets. Explique em 1-3 frases (PT-BR) por que os vizinhos KNN sustentam a classe. Só a justificativa."
    user = (
        f"Ticket:\n{ticket_text[:1500]}\n\n"
        f"Classe: {classe}. Confiança KNN: {confidence:.2f}.\n"
        f"Vizinhos que votaram nessa classe:\n{_format_winning_voters(winning_voters)}\n\n"
        "Justifique em português."
    )
    return _call_justify(llm, user, system)


def _justify_llm(llm, ticket_text: str, classe: str, confidence: float | None = None) -> tuple[str, int, int]:
    system = "Analista de tickets. Explique em 1-3 frases (PT-BR) que a confiança do KNN foi baixa, a classificação foi feita pelo LLM, e cite trechos do ticket que indicaram a classe. Só a justificativa."
    conf_str = f" Confiança KNN (baixa): {confidence:.2f}." if confidence is not None else ""
    user = f"Ticket:\n{ticket_text[:1500]}\n\nClasse: {classe}.{conf_str}\n\nJustifique: confiança baixa → LLM classificou; cite trechos do texto que indicaram a classe."
    return _call_justify(llm, user, system)


def _call_justify(llm, user: str, system: str) -> tuple[str, int, int]:
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        max_tokens=config.JUSTIFICATION_MAX_TOKENS,
        response_format={
            "type": "json_object",
            "schema": JustificativaOnly.model_json_schema(),
        },
    )
    usage = response.get("usage") or {}
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    raw = response["choices"][0]["message"]["content"]
    parsed = json.loads(raw)
    out = JustificativaOnly(**parsed)
    return out.justificativa or "", input_tokens, output_tokens


def agent_justify(
    ticket_text: str,
    classe: str,
    confidence: float | None = None,
    winning_voters: list[tuple[int, float, str]] | None = None,
    used_llm_for_class: bool = False,
) -> tuple[str, int, int]:
    llm = get_llm()
    if not used_llm_for_class and winning_voters and confidence is not None:
        return _justify_knn(llm, ticket_text, classe, confidence, winning_voters)
    return _justify_llm(llm, ticket_text, classe, confidence=confidence)
