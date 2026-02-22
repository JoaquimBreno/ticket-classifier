from pathlib import Path

import pandas as pd

import config
from src.llm_local import agent_classifier


def build_taxonomy(
    csv_path: Path | str,
    label_col: str | None = None,
    text_col: str | None = None,
) -> tuple[list[str], str]:
    csv_path = Path(csv_path)
    label_col = label_col or config.LABEL_COLUMN

    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(f"CSV must have column {label_col!r}")

    classes_ordered = df[label_col].dropna().unique().astype(str).tolist()
    classes_ordered = [c.strip() for c in classes_ordered if c.strip()]
    taxonomy_prompt = f"Categorias (use exatamente um destes nomes): {classes_ordered}."
    return classes_ordered, taxonomy_prompt


class LLMClassifier:
    def __init__(self, taxonomy_prompt: str, classes: list[str] | None = None):
        self.taxonomy_prompt = taxonomy_prompt.strip()
        self.classes = classes or []

    @classmethod
    def from_dataset(
        cls,
        csv_path: Path | str | None = None,
        label_col: str | None = None,
    ) -> "LLMClassifier":
        csv_path = Path(csv_path) if csv_path else config.DATA_PROCESSED / config.PROCESSED_CSV_FILENAME
        classes_ordered, taxonomy_prompt = build_taxonomy(csv_path, label_col=label_col)
        return cls(taxonomy_prompt=taxonomy_prompt, classes=classes_ordered)

    def predict(
        self,
        text: str,
        classes: list[str],
        embedding: list[float] | None = None,
        knn_hint: tuple[str, float] | None = None,
    ) -> tuple[str, int, int]:
        classe, input_tokens, output_tokens = agent_classifier(
            text,
            classes,
            taxonomy=self.taxonomy_prompt,
            knn_hint=knn_hint,
        )
        cnorm = (classe or "").strip().lower()
        for c in classes:
            if (c or "").strip().lower() == cnorm:
                return c, input_tokens, output_tokens
        return (classes[0] if classes else (classe or "")), input_tokens, output_tokens
