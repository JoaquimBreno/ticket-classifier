from collections import Counter

from src.rag import VectorStore

import config


class KNNClassifier:
    def __init__(self, vector_store: VectorStore, k: int | None = None):
        self.store = vector_store
        self.k = k or config.KNN_K

    def predict(self, text: str, embedding: list[float] | None = None) -> tuple[str, float]:
        if embedding is not None:
            neighbors = self.store.search_by_vector(embedding, k=self.k)
        else:
            neighbors = self.store.search(text, k=self.k)
        if not neighbors:
            return "", 0.0
        labels = [n[0] for n in neighbors]
        counts = Counter(labels)
        majority_label, count = counts.most_common(1)[0]
        confidence = count / len(labels)
        return majority_label, float(confidence)
