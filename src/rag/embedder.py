from sentence_transformers import SentenceTransformer

import config


class Embedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, text: str | list[str]):
        if isinstance(text, str):
            text = [text]
        return self.model.encode(text, convert_to_numpy=True)
