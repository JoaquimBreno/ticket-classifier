import json

from pydantic import BaseModel, Field

import config
from .backend import get_llm_backend


class TicketClassification(BaseModel):
    classe: str = Field(
        min_length=1,
        description="Categoria exata do ticket (deve ser uma das classes fornecidas).",
    )
    justificativa: str = Field(
        min_length=1,
        description="Explicação em 1 a 3 frases do motivo da classificação, citando termos do ticket, em português.",
    )


class ClassificationResponse(BaseModel):
    classe: str = Field(
        min_length=1,
        description="Categoria exata do ticket. Deve ser exatamente uma das classes fornecidas na lista.",
    )


class JustificationResponse(BaseModel):
    justificativa: str = Field(
        min_length=1,
        max_length=500,
        description="Explicação em 1 a 3 frases em português que justifica a classificação, citando termos do ticket.",
    )


JustificativaOnly = JustificationResponse


def _extract_json(raw: str) -> str | None:
    if not raw or not raw.strip():
        return None
    s = raw.strip()
    for prefix in ("```json", "```"):
        if s.startswith(prefix):
            s = s[len(prefix):].lstrip()
    if s.endswith("```"):
        s = s[:-3].rstrip()
    start = s.find("{")
    if start == -1:
        return None
    end = s.rfind("}") + 1
    if end <= start:
        return None
    return s[start:end]


def _format_winning_voters(winning_voters: list[tuple[int, float, str]], max_len: int = 120) -> str:
    lines = []
    for idx, dist, text in winning_voters:
        s = (text or "").strip()[:max_len]
        if len((text or "").strip()) > max_len:
            s += "..."
        lines.append(f"- #{idx} (dist {dist:.2f}): {s}")
    return "\n".join(lines) if lines else ""


def _normalize_class(c: str, classes_validas: list[str]) -> str:
    if not classes_validas:
        return c.strip() if c else ""
    cnorm = (c or "").strip().lower()
    for valid in classes_validas:
        if (valid or "").strip().lower() == cnorm:
            return valid
    return classes_validas[0]


def _parse_classification(raw: str, classes_validas: list[str]) -> str:
    blob = _extract_json(raw)
    if blob:
        try:
            data = json.loads(blob)
            if isinstance(data, dict) and "classe" in data:
                out = ClassificationResponse.model_validate(data)
                return _normalize_class(out.classe, classes_validas)
        except (json.JSONDecodeError, Exception):
            pass
    return classes_validas[0] if classes_validas else ""


def _parse_justification(raw: str) -> str:
    blob = _extract_json(raw)
    if blob:
        try:
            data = json.loads(blob)
            if isinstance(data, dict) and "justificativa" in data:
                out = JustificationResponse.model_validate(data)
                return out.justificativa.strip()
        except (json.JSONDecodeError, Exception):
            pass
    return (raw or "").strip()


def agent_classifier(
    texto_ticket: str,
    classes_validas: list[str],
    taxonomy: str | None = None,
    knn_hint: tuple[str, float] | None = None,
) -> tuple[str, int, int]:
    backend = get_llm_backend()
    system = "You are an expert IT analyst. Output only valid JSON with a single key 'classe' whose value is the exact category name. Be objective."
    if taxonomy:
        system = system + "\n\n" + taxonomy
    hint_line = ""
    if knn_hint:
        knn_classe, knn_conf = knn_hint
        hint_line = f"KNN sugeriu a classe {knn_classe!r} com confiança {knn_conf:.2f} (abaixo do limiar). Use como hint.\n\n"
    user_content = (
        f"Classifique o ticket estritamente em uma destas categorias: {classes_validas}.\n\n"
        f"{hint_line}"
        f"Ticket:\n{texto_ticket[:config.CLASSIFICATION_MAX_CHARS]}"
    )
    content, input_tokens, output_tokens = backend.chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
        max_tokens=64,
        response_format={
            "type": "json_object",
            "schema": ClassificationResponse.model_json_schema(),
        },
    )
    classe = _parse_classification(content, classes_validas)
    return classe, input_tokens, output_tokens


def _justify_knn(
    backend,
    ticket_text: str,
    classe: str,
    confidence: float,
    winning_voters: list[tuple[int, float, str]],
) -> tuple[str, int, int]:
    system = "Analista de tickets. Responda em JSON com uma única chave 'justificativa'. Explique em 1-3 frases (PT-BR) por que os vizinhos KNN sustentam a classe. Só a justificativa."
    user = (
        f"Ticket:\n{ticket_text[:config.JUSTIFICATION_MAX_CHARS]}\n\n"
        f"Classe: {classe}. Confiança KNN: {confidence:.2f}.\n"
        f"Vizinhos que votaram nessa classe:\n{_format_winning_voters(winning_voters)}\n\n"
        "Justifique em português."
    )
    return _call_justify(backend, user, system)


def _justify_llm(
    backend, ticket_text: str, classe: str, confidence: float | None = None
) -> tuple[str, int, int]:
    system = "Analista de tickets. Responda em JSON com uma única chave 'justificativa'. Explique em 1-3 frases (PT-BR) que a confiança do KNN foi baixa, a classificação foi feita pelo LLM, e cite trechos do ticket que indicaram a classe. Só a justificativa."
    conf_str = f" Confiança KNN (baixa): {confidence:.2f}." if confidence is not None else ""
    user = (
        f"Ticket:\n{ticket_text[:config.JUSTIFICATION_MAX_CHARS]}\n\nClasse: {classe}.{conf_str}\n\n"
        "Justifique: confiança baixa → LLM classificou; cite trechos do texto que indicaram a classe."
    )
    return _call_justify(backend, user, system)


def _call_justify(backend, user: str, system: str) -> tuple[str, int, int]:
    content, input_tokens, output_tokens = backend.chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        max_tokens=config.JUSTIFICATION_MAX_TOKENS,
        response_format={
            "type": "json_object",
            "schema": JustificationResponse.model_json_schema(),
        },
    )
    justificativa = _parse_justification(content)
    return justificativa, input_tokens, output_tokens


def agent_justify(
    ticket_text: str,
    classe: str,
    confidence: float | None = None,
    winning_voters: list[tuple[int, float, str]] | None = None,
    used_llm_for_class: bool = False,
) -> tuple[str, int, int]:
    backend = get_llm_backend()
    if not used_llm_for_class and winning_voters and confidence is not None:
        return _justify_knn(backend, ticket_text, classe, confidence, winning_voters)
    return _justify_llm(backend, ticket_text, classe, confidence=confidence)
