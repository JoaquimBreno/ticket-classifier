import json
from pathlib import Path

import faiss
import numpy as np

from .embedder import Embedder

try:
    import config
except ImportError:
    config = None

MANIFEST_VERSION = 1
MANIFEST_FILENAME = "manifest.json"


class VectorStore:
    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder()
        self.index: faiss.IndexFlatL2 | None = None
        self.labels: list[str] = []
        self.texts: list[str] = []

    def build(self, texts: list[str], labels: list[str]):
        if len(texts) != len(labels):
            raise ValueError(
                f"texts and labels length mismatch: {len(texts)} texts vs {len(labels)} labels"
            )
        if not texts:
            raise ValueError("Cannot build VectorStore with empty texts.")
        self.texts = list(texts)
        self.labels = list(labels)
        vectors = self.embedder.embed(self.texts)
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        self.index.add(vectors)
        return self

    def save(self, path: Path | None = None):
        if self.index is None:
            raise RuntimeError("VectorStore not built. Call build() first.")
        path = path or (config.ARTIFACTS_DIR if config else Path("outputs/artifacts"))
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "labels.json", "w", encoding="utf-8") as f:
            json.dump(self.labels, f, ensure_ascii=False)
        with open(path / "texts.json", "w", encoding="utf-8") as f:
            json.dump(self.texts, f, ensure_ascii=False)
        classes = sorted(set(self.labels))
        with open(path / "classes.json", "w", encoding="utf-8") as f:
            json.dump(classes, f, ensure_ascii=False)
        manifest = {
            "version": MANIFEST_VERSION,
            "embedding_model": self.embedder.model_name,
            "dim": int(self.index.d),
            "n_vectors": int(self.index.ntotal),
        }
        with open(path / MANIFEST_FILENAME, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return path

    @classmethod
    def load(cls, path: Path | None = None, embedder: Embedder | None = None):
        path = path or (config.ARTIFACTS_DIR if config else Path("outputs/artifacts"))
        path = Path(path)
        store = cls(embedder=embedder)
        store.index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "labels.json", encoding="utf-8") as f:
            store.labels = json.load(f)
        with open(path / "texts.json", encoding="utf-8") as f:
            store.texts = json.load(f)
        manifest_path = path / MANIFEST_FILENAME
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            expected_dim = manifest.get("dim")
            if expected_dim is not None:
                actual_dim = store.embedder.embed(["x"]).shape[1]
                if actual_dim != expected_dim:
                    raise ValueError(
                        f"Embedder dimension mismatch: artifact was built with dim={expected_dim}, "
                        f"current embedder produces dim={actual_dim}. Use embedding_model={manifest.get('embedding_model')!r}."
                    )
        return store

    def search(self, query_text: str, k: int = 5):
        if self.index is None:
            raise RuntimeError("VectorStore not built. Call build() first.")
        q = self.embedder.embed([query_text]).astype(np.float32)
        distances, indices = self.index.search(q, min(k, self.index.ntotal))
        return [
            (self.labels[i], self.texts[i], float(distances[0][j]))
            for j, i in enumerate(indices[0])
        ]
