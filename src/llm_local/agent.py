import json

import config
from .backend import get_llm_backend
from .schemas import JustificationResponse, TicketClassification

MAX_J = config.JUSTIFICATION_MAX_LENGTH


def _json_from_raw(raw: str) -> dict | None:
    if not (raw and raw.strip()):
        return None
    s = raw.strip()
    for prefix in ("```json", "```"):
        if s.startswith(prefix):
            s = s[len(prefix) :].lstrip()
    if s.endswith("```"):
        s = s[:-3].rstrip()
    start, end = s.find("{"), s.rfind("}") + 1
    if start == -1 or end <= start:
        return None
    try:
        out = json.loads(s[start:end])
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        return None


def _truncate(text: str, max_len: int) -> str:
    if not text or len(text) <= max_len:
        return text
    cut = text[: max_len + 1]
    last = cut.rfind(" ")
    return text[: last].strip() if last > 0 else text[:max_len]


def _format_winning_voters(winning_voters: list[tuple[int, float, str]], max_len: int = 120) -> str:
    lines = []
    for idx, dist, text in winning_voters:
        t = (text or "").strip()[:max_len]
        if len((text or "").strip()) > max_len:
            t += "..."
        lines.append(f"- #{idx} (dist {dist:.2f}): {t}")
    return "\n".join(lines) if lines else ""


def _normalize_class(c: str, classes_validas: list[str]) -> str:
    if not classes_validas:
        return (c or "").strip()
    cnorm = (c or "").strip().lower()
    for valid in classes_validas:
        if (valid or "").strip().lower() == cnorm:
            return valid
    return classes_validas[0]


def _parse_justification(raw: str) -> str:
    data = _json_from_raw(raw)
    s = None
    if data and "justificativa" in data:
        try:
            s = JustificationResponse.model_validate(data).justificativa.strip()
        except Exception:
            pass
    return _truncate(s or (raw or "").strip(), MAX_J)


def _parse_classify_and_justify(raw: str, classes_validas: list[str]) -> tuple[str, str]:
    data = _json_from_raw(raw)
    classe = classes_validas[0] if classes_validas else ""
    j = (raw or "").strip()
    if data and "classe" in data and "justificativa" in data and classes_validas:
        try:
            out = TicketClassification.model_validate(data)
            classe = _normalize_class(out.classe, classes_validas)
            j = out.justificativa.strip()
        except Exception:
            pass
    return classe, _truncate(j, MAX_J)


def agent_classify_and_justify(
    ticket_text: str,
    classes_validas: list[str],
    knn_hint: tuple[str, float],
) -> tuple[str, str, int, int]:
    backend = get_llm_backend()
    knn_classe, knn_conf = knn_hint
    system = (
        "Analista de tickets. Saída apenas JSON com chaves 'classe' e 'justificativa'. "
        "classe: nome exato de uma das categorias fornecidas. "
        "justificativa: em 1-3 frases (PT-BR) você DEVE (1) citar o que o KNN sugeriu (classe e confiança), "
        "(2) em seguida dar sua contrapartida: concordar (mesma classe) ou discordar (outra classe) e o porquê, citando termos do ticket."
    )
    user = (
        f"KNN sugeriu: classe {knn_classe!r}, confiança {knn_conf:.2f} (abaixo do limiar). Use como guia.\n\n"
        f"Categorias válidas: {classes_validas}\n\n"
        f"Classifique o ticket. Na justificativa: diga o que o KNN sugeriu e depois sua contrapartida (mesma classe ou não, e por quê), citando o ticket.\n\n"
        f"Ticket:\n{ticket_text[:config.TICKET_MAX_CHARS]}"
    )
    
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
    classe, justificativa = _parse_classify_and_justify(content, classes_validas)
    
    return classe, justificativa, input_tokens, output_tokens


def _justify_knn(
    backend,
    ticket_text: str,
    classe: str,
    confidence: float,
    winning_voters: list[tuple[int, float, str]],
) -> tuple[str, int, int]:
    system = (
        "Analista de tickets. Responda em JSON com uma única chave 'justificativa'. "
        "Em no máximo 3 frases curtas (PT-BR), indique quais termos do ticket correlacionam à classe e aos vizinhos KNN. "
        "Seja conciso. Só a justificativa."
    )
    user = (
        f"Ticket:\n{ticket_text[:config.TICKET_MAX_CHARS]}\n\n"
        f"Classe atribuída: {classe}. Confiança KNN: {confidence:.2f}.\n\n"
        f"Vizinhos KNN que votaram nessa classe:\n{_format_winning_voters(winning_voters)}\n\n"
        "Quais termos do ticket acima correlacionam com a classe e com os vizinhos? Justifique em português citando o texto."
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
    confidence: float,
    winning_voters: list[tuple[int, float, str]],
) -> tuple[str, int, int]:
    backend = get_llm_backend()
    return _justify_knn(backend, ticket_text, classe, confidence, winning_voters)
