import json
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, classification_report

import config


def compute_metrics(y_true: list[str], y_pred: list[str], labels: list[str] | None = None):
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "classification_report": report,
    }


def save_metrics_report(metrics: dict, path: Path | None = None):
    path = path or config.OUTPUTS / "metrics_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return path
