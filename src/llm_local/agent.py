import json

from pydantic import BaseModel, Field

import config
from .backend import get_llm_backend


def _load_class_schema() -> dict | None:
    if not config.CLASS_SCHEMA_PATH.exists():
        return None
    try:
        with open(config.CLASS_SCHEMA_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _build_schema_and_few_shot(classes_validas: list[str]) -> tuple[str, str]:
    data = _load_class_schema()
    if not data or "classes" not in data:
        return "", ""
    definitions = []
    examples_by_class = []
    for c in classes_validas:
        info = data["classes"].get(c)
        if not info:
            continue
        definitions.append(f"- **{c}**: {info.get('description', '').strip()}")
        ex_list = info.get("examples") or []
        for ex in ex_list[:2]:
            snippet = (ex or "").strip()[:config.MAX_EXAMPLE_CHARS]
            if snippet:
                examples_by_class.append(f"  Ticket: {snippet}\n  Classe: {c}")
    def_block = "\n".join(definitions) if definitions else ""
    few_shot_block = "\n\n".join(examples_by_class) if examples_by_class else ""
    return def_block, few_shot_block


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


def _parse_classify_and_justify(raw: str, classes_validas: list[str]) -> tuple[str, str]:
    blob = _extract_json(raw)
    if blob and classes_validas:
        try:
            data = json.loads(blob)
            if isinstance(data, dict) and "classe" in data and "justificativa" in data:
                out = TicketClassification.model_validate(data)
                return _normalize_class(out.classe, classes_validas), out.justificativa.strip()
        except (json.JSONDecodeError, Exception):
            pass
    return classes_validas[0], (raw or "").strip()


def agent_classify_and_justify(
    ticket_text: str,
    classes_validas: list[str],
    knn_hint: tuple[str, float],
) -> tuple[str, str, int, int]:
    backend = get_llm_backend()
    knn_classe, knn_conf = knn_hint
    def_block, few_shot_block = _build_schema_and_few_shot(classes_validas)
    schema_section = (
        f"\n\nDefinição das categorias:\n{def_block}"
        if def_block else ""
    )
    system = (
        "You are an expert IT analyst. Output only valid JSON with keys 'classe' and 'justificativa'. "
        "classe: exact category name from the list. justificativa: 1-3 sentences in PT-BR explaining the classification. "
        f"{schema_section}"
    )
    few_shot_section = (
        f"Exemplos por categoria:\n{few_shot_block}\n\n"
        if few_shot_block else ""
    )
    user = (
        f"KNN sugeriu a classe {knn_classe!r} com confiança {knn_conf:.2f} (abaixo do limiar). Use como hint.\n\n"
        f"Categorias válidas: {classes_validas}.\n\n"
        f"{few_shot_section}"
        f"Classifique o ticket abaixo em uma das categorias e justifique em 1-3 frases (PT-BR).\n\n"
        f"Ticket:\n{ticket_text[:config.CLASSIFICATION_MAX_CHARS]}"
    )
    print(user)
    print(system)
    
    content, input_tokens, output_tokens = backend.chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        max_tokens=config.JUSTIFICATION_MAX_TOKENS,
        response_format={
            "type": "json_object",
            "schema": TicketClassification.model_json_schema(),
        },
    )
    print(content)
    classe, justificativa = _parse_classify_and_justify(content, classes_validas)
    
    return classe, justificativa, input_tokens, output_tokens


def agent_classifier(
    texto_ticket: str,
    classes_validas: list[str],
    knn_hint: tuple[str, float] | None = None,
) -> tuple[str, int, int]:
    backend = get_llm_backend()
    system = "You are an expert IT analyst. Output only valid JSON with a single key 'classe' whose value is the exact category name. Be objective."
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
