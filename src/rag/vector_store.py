import json
from pathlib import Path

import faiss
import numpy as np

from src.prep.loader import stable_id as _stable_id
from .embedder import Embedder

try:
    import config
except ImportError:
    config = None

MANIFEST_VERSION = 2
MANIFEST_FILENAME = "manifest.json"
IDS_FILENAME = "ids.json"


class VectorStore:
    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder()
        self.index: faiss.IndexFlatL2 | None = None
        self.ids: list[str] = []
        self.labels: list[str] = []
        self.texts: list[str] = []

    def build(
        self,
        texts: list[str],
        labels: list[str],
        ids: list[str] | None = None,
    ) -> "VectorStore":
        if len(texts) != len(labels):
            raise ValueError(
                f"texts and labels length mismatch: {len(texts)} texts vs {len(labels)} labels"
            )
        if not texts:
            raise ValueError("Cannot build VectorStore with empty texts.")
        self.texts = list(texts)
        self.labels = list(labels)
        self.ids = list(ids) if ids is not None else [_stable_id(t) for t in self.texts]
        if len(self.ids) != len(self.texts):
            raise ValueError("ids length must match texts length.")
        vectors = self.embedder.embed(self.texts)
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.ascontiguousarray(vectors.astype(np.float32)))
        return self

    def save(self, path: Path | None = None) -> Path:
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
        with open(path / IDS_FILENAME, "w", encoding="utf-8") as f:
            json.dump(self.ids, f, ensure_ascii=False)
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
    def load(cls, path: Path | None = None, embedder: Embedder | None = None) -> "VectorStore":
        path = path or (config.ARTIFACTS_DIR if config else Path("outputs/artifacts"))
        path = Path(path)
        store = cls(embedder=embedder)
        store.index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "labels.json", encoding="utf-8") as f:
            store.labels = json.load(f)
        with open(path / "texts.json", encoding="utf-8") as f:
            store.texts = json.load(f)
        ids_path = path / IDS_FILENAME
        if ids_path.exists():
            with open(ids_path, encoding="utf-8") as f:
                store.ids = json.load(f)
        else:
            store.ids = [_stable_id(t) for t in store.texts]
        n = len(store.texts)
        if len(store.labels) != n or len(store.ids) != n:
            raise ValueError(
                f"Artifact length mismatch: texts={n}, labels={len(store.labels)}, ids={len(store.ids)}"
            )
        manifest_path = path / MANIFEST_FILENAME
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            expected_dim = manifest.get("dim")
            if expected_dim is not None:
                actual_dim = store.embedder.embed(["x"]).shape[1]
                if actual_dim != expected_dim:
                    raise ValueError(
                        f"Embedder dimension mismatch: artifact dim={expected_dim}, "
                        f"embedder dim={actual_dim}. Use embedding_model={manifest.get('embedding_model')!r}."
                    )
        return store

    def search(self, query_text: str, k: int = 5) -> list[tuple[str, str, float]]:
        if self.index is None:
            raise RuntimeError("VectorStore not built. Call build() first.")
        q = self.embedder.embed([query_text]).astype(np.float32)
        return self._search_q(q, k)

    def search_by_vector(self, vector: list[float] | np.ndarray, k: int = 5) -> list[tuple[str, str, float]]:
        if self.index is None:
            raise RuntimeError("VectorStore not built. Call build() first.")
        q = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        return self._search_q(q, k)

    def _search_q(self, q: np.ndarray, k: int) -> list[tuple[str, str, float]]:
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(q, k)
        return [
            (self.labels[i], self.texts[i], float(distances[0][j]))
            for j, i in enumerate(indices[0])
        ]

    def get_by_index(self, i: int) -> tuple[str, str, str]:
        if i < 0 or i >= len(self.texts):
            raise IndexError(f"Index {i} out of range [0, {len(self.texts)}).")
        return (self.texts[i], self.labels[i], self.ids[i])

    def get_by_id(self, id: str) -> tuple[str, str] | None:
        if id not in self.ids:
            return None
        i = self.ids.index(id)
        return (self.texts[i], self.labels[i])
