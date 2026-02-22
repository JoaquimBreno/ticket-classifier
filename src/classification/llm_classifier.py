from src.llm_local import agent_classifier
from src.rag import VectorStore

import config


class LLMClassifier:
    def __init__(self, vector_store: VectorStore, k: int | None = None):
        self.store = vector_store
        self.k = k or config.KNN_K

    def predict(self, text: str, classes: list[str], embedding: list[float] | None = None) -> tuple[str, int, int]:
        if embedding is not None:
            neighbors = self.store.search_by_vector(embedding, k=self.k)
        elif text:
            neighbors = self.store.search(text, k=self.k)
        else:
            neighbors = []
        classe, input_tokens, output_tokens = agent_classifier(
            text,
            classes,
            neighbors=neighbors,
        )
        cnorm = (classe or "").strip().lower()
        for c in classes:
            if (c or "").strip().lower() == cnorm:
                return c, input_tokens, output_tokens
        return classes[0] if classes else (classe or ""), input_tokens, output_tokens
