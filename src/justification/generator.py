import json
import os
import time
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.exceptions import ModelHTTPError, ToolRetryError, UnexpectedModelBehavior

import nest_asyncio

import config

nest_asyncio.apply()


class ClassificationOutput(BaseModel):
    classe: str
    justificativa: str


_agent = None


def _get_agent():
    global _agent
    if _agent is not None:
        return _agent
    api_key = os.getenv("OPENROUTER_API_KEY") or config.OPENROUTER_API_KEY
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set.")
    _profile = OpenAIModelProfile(
        default_structured_output_mode="prompted",
        supports_json_object_output=True,
        openai_supports_tool_choice_required=False,
    )
    model = OpenRouterModel(
        config.OPENROUTER_MODEL,
        provider=OpenRouterProvider(api_key=api_key),
        profile=_profile,
    )
    _agent = Agent(model, output_type=ClassificationOutput, output_retries=3)

    @_agent.system_prompt
    def _sys() -> str:
        return (
            "You are a support ticket analyst. Given the ticket text and the assigned class, "
            "output the same class and a short justification (1 to 3 sentences) that highlights "
            "specific words or patterns in the ticket that support this classification. Be concrete. "
            "Always write the justification in Portuguese (Brazilian)."
        )

    return _agent


def _wait_for_rate_limit(body: object | None) -> float:
    if body is None:
        return 65.0
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            return 65.0
    if not isinstance(body, dict):
        return 65.0
    err = body.get("error", body) if isinstance(body.get("error"), dict) else body
    meta = err.get("metadata", {}) if isinstance(err, dict) else {}
    headers = meta.get("headers", {}) if isinstance(meta, dict) else {}
    reset_ms = headers.get("X-RateLimit-Reset")
    if reset_ms is not None:
        try:
            reset_ts = int(reset_ms) / 1000.0
            wait = max(1.0, min(120.0, reset_ts - time.time()))
            return wait
        except (TypeError, ValueError):
            pass
    return 65.0


def _format_neighbors(neighbors: list[tuple[str, str, float]], max_snippet_len: int = 120) -> str:
    lines = []
    for label, text, dist in neighbors:
        snippet = (text or "").strip()[:max_snippet_len]
        if len((text or "").strip()) > max_snippet_len:
            snippet += "..."
        lines.append(f"- [{label}] (dist {dist:.2f}): {snippet}")
    return "\n".join(lines) if lines else ""


def generate_justification_text(
    ticket_text: str,
    classe: str,
    confidence: float | None = None,
    neighbors: list[tuple[str, str, float]] | None = None,
    max_retries: int = 2,
) -> tuple[str, dict | None]:
    agent = _get_agent()
    prompt_parts = [
        f"Texto do ticket:\n{ticket_text[:2000]}\n\nClasse atribuída: {classe}.",
    ]
    if confidence is not None:
        prompt_parts.append(
            f" Confiança da classificação (acordo entre vizinhos KNN): {confidence:.2f}."
        )
    if neighbors:
        prompt_parts.append(
            "\nTickets similares do treino (use para embasar a justificativa):\n"
            + _format_neighbors(neighbors)
        )
    prompt_parts.append(
        "\nRetorne a mesma classe e uma justificativa curta (1 a 3 frases) em português, com evidências concretas do ticket."
    )
    prompt = "".join(prompt_parts)
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            result = agent.run_sync(
                prompt,
                model_settings={"max_tokens": config.JUSTIFICATION_MAX_TOKENS},
            )
            u = result.usage()
            usage_info = {
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "total_tokens": getattr(u, "total_tokens", None)
                or u.input_tokens + u.output_tokens,
            }
            out = result.output
            if isinstance(out, ClassificationOutput):
                return (out.justificativa or "", usage_info)
            if isinstance(out, dict):
                return (out.get("justificativa", "") or "", usage_info)
            return (str(out) if out else "", usage_info)
        except (UnexpectedModelBehavior, ToolRetryError):
            return ("", None)
        except ModelHTTPError as e:
            last_exc = e
            if getattr(e, "status_code", None) == 402:
                return ("", None)
            if getattr(e, "status_code", None) == 429 and attempt < max_retries:
                wait = _wait_for_rate_limit(getattr(e, "body", None))
                time.sleep(wait)
                continue
            raise
        except Exception as e:
            last_exc = e
            if "402" in str(e) or "insufficient credits" in str(e).lower():
                return ("", None)
            if "429" in str(e) or "rate limit" in str(e).lower():
                if attempt < max_retries:
                    time.sleep(_wait_for_rate_limit(None))
                    continue
            raise
    if last_exc:
        raise last_exc
    return ("", None)
