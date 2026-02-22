import os

_backend = None


def get_llm_backend():
    global _backend
    if _backend is not None:
        return _backend
    name = os.getenv("LLM_BACKEND", "llama_cpp").strip().lower()
    if name == "llama_cpp":
        from .backends.llama_cpp import LlamaCppBackend
        _backend = LlamaCppBackend()
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {name}")
    return _backend
