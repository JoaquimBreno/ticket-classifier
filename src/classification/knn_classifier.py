from collections import Counter

from src.rag import VectorStore

import config


class KNNClassifier:
    def __init__(self, vector_store: VectorStore, k: int | None = None, confidence_threshold: float | None = None):
        self.store = vector_store
        self.k = k or config.KNN_K
        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else config.KNN_CONFIDENCE_THRESHOLD

    def predict(self, text: str) -> tuple[str, float]:
        neighbors = self.store.search(text, k=self.k)
        if not neighbors:
            return "", 0.0
        labels = [n[0] for n in neighbors]
        counts = Counter(labels)
        majority_label, count = counts.most_common(1)[0]
        confidence = count / len(labels)
        return majority_label, float(confidence)

    def use_knn(self, confidence: float) -> bool:
        return confidence >= self.confidence_threshold
