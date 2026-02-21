import os
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

import config


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
    model = OpenRouterModel(
        config.OPENROUTER_MODEL,
        provider=OpenRouterProvider(api_key=api_key),
    )
    _agent = Agent(model, result_type=ClassificationOutput)

    @_agent.system_prompt
    def _sys() -> str:
        return (
            "You are a support ticket analyst. Given the ticket text and the assigned class, "
            "output the same class and a short justification (1 to 3 sentences) that highlights "
            "specific words or patterns in the ticket that support this classification. Be concrete."
        )

    return _agent


def generate_justification_text(ticket_text: str, classe: str) -> str:
    agent = _get_agent()
    result = agent.run_sync(
        f"Ticket text:\n{ticket_text[:2000]}\n\nAssigned class: {classe}. "
        "Return the same class and a short justification (1-3 sentences) with concrete evidence from the ticket."
    )
    out = result.data
    if isinstance(out, ClassificationOutput):
        return out.justificativa or ""
    if isinstance(out, dict):
        return out.get("justificativa", "")
    return str(out)
