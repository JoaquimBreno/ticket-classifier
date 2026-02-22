from typing import Any, Protocol


class LLMBackend(Protocol):
    def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.1,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str, int, int]:
        """
        messages: [{"role": "system"|"user"|"assistant", "content": "..."}, ...]
        response_format: e.g. {"type": "json_object", "schema": {...}}
        Returns: (assistant_content_str, input_tokens, output_tokens).
        """
        ...
