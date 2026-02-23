import os
import threading
import time

REQUESTS_PER_MINUTE = 30
MIN_INTERVAL = 60.0 / REQUESTS_PER_MINUTE

_lock = threading.Lock()
_last_request_time: float = 0.0


def _rate_limit() -> None:
    global _last_request_time
    with _lock:
        now = time.monotonic()
        elapsed = now - _last_request_time
        if elapsed < MIN_INTERVAL and _last_request_time > 0:
            time.sleep(MIN_INTERVAL - elapsed)
        _last_request_time = time.monotonic()


class GroqBackend:
    def __init__(self) -> None:
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY não definido. Defina no .env (chave em https://console.groq.com/keys)."
            )
        from groq import Groq
        self._client = Groq(api_key=api_key)
        self._model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant").strip()

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.1,
        response_format: dict | None = None,
    ) -> tuple[str, int, int]:
        _rate_limit()
        kwargs = dict(
            messages=messages,
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if response_format and response_format.get("type") == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        choice = (resp.choices or [None])[0]
        content = (choice.message.content or "") if choice and choice.message else ""
        usage = getattr(resp, "usage", None)
        inp = getattr(usage, "prompt_tokens", 0) or 0
        out = getattr(usage, "completion_tokens", 0) or 0
        return (content or "", inp, out)
