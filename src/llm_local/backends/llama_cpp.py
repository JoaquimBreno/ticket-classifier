import os
from pathlib import Path

import config


def _project_root() -> Path:
    root = Path(config.ROOT)
    s = str(root).rstrip(":")
    return Path(s) if s else root


def _download_model_to(local_dir: Path) -> Path:
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


def _resolve_model_path() -> Path:
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
    return _download_model_to(config.MODELS_DIR)


class LlamaCppBackend:
    def __init__(self) -> None:
        self._model = None

    def _get_model(self):
        if self._model is None:
            from llama_cpp import Llama
            path = _resolve_model_path()
            self._model = Llama(
                model_path=str(path),
                n_gpu_layers=config.LLAMA_N_GPU_LAYERS,
                n_ctx=config.LLAMA_N_CTX,
                n_batch=config.LLAMA_N_BATCH,
                n_threads=config.LLAMA_N_THREADS,
                verbose=False,
            )
        return self._model

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.1,
        response_format: dict | None = None,
    ) -> tuple[str, int, int]:
        llm = self._get_model()
        resp = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format or {},
        )
        usage = resp.get("usage") or {}
        content = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "")
        return (
            content or "",
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )
