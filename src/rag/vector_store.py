import numpy as np
import faiss

from .embedder import Embedder


class VectorStore:
    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder()
        self.index: faiss.IndexFlatL2 | None = None
        self.labels: list[str] = []
        self.texts: list[str] = []

    def build(self, texts: list[str], labels: list[str]):
        self.texts = list(texts)
        self.labels = list(labels)
        vectors = self.embedder.embed(self.texts)
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        self.index.add(vectors)
        return self

    def search(self, query_text: str, k: int = 5):
        if self.index is None:
            raise RuntimeError("VectorStore not built. Call build() first.")
        q = self.embedder.embed([query_text]).astype(np.float32)
        distances, indices = self.index.search(q, min(k, self.index.ntotal))
        return [
            (self.labels[i], self.texts[i], float(distances[0][j]))
            for j, i in enumerate(indices[0])
        ]
