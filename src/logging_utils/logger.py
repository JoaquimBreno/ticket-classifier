import json
from pathlib import Path

import config


def log_result(payload: dict, path: Path | None = None, append: bool = True):
    path = path or config.OUTPUTS / "results_sample.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
