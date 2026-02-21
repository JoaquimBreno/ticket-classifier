import os
from openai import OpenAI

import config
from src.logging_utils import log_llm_usage


def get_openrouter_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY") or config.OPENROUTER_API_KEY
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set. Set it in .env or environment.")
    return OpenAI(
        base_url=config.OPENROUTER_BASE_URL,
        api_key=api_key,
    )


class LLMClassifier:
    def __init__(self, vector_store, model: str | None = None, client: OpenAI | None = None):
        self.store = vector_store
        self.model = model or config.OPENROUTER_MODEL
        self._client = client

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = get_openrouter_client()
        return self._client

    def predict(self, text: str, classes: list[str]) -> str:
        examples = self.store.search(text, k=3)
        examples_str = "\n".join(
            f"- Ticket: {t[:200]}... -> {label}" for label, t, _ in examples
        )
        prompt = f"""Classify the following support ticket into exactly one of these classes: {", ".join(classes)}.

Example classifications (ticket -> class):
{examples_str}

Ticket to classify:
{text[:1500]}

Reply with only the class name, nothing else."""

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
        )
        if resp.usage:
            log_llm_usage(
                source="classification",
                model=self.model,
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
            )
        raw = (resp.choices[0].message.content or "").strip()
        for c in classes:
            if c.lower() in raw.lower() or raw.lower() == c.lower():
                return c
        return classes[0] if classes else raw or "Unknown"
