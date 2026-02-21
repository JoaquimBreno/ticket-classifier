import pandas as pd
from pathlib import Path

import config

KAGGLE_DATASET = "aniketg11/supportticketsclassification"


def _find_csv(path: Path) -> Path | None:
    for f in path.rglob("*.csv"):
        return f
    return None


def download_from_kaggle() -> Path:
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    existing = _find_csv(config.DATA_RAW)
    if existing:
        return existing
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(KAGGLE_DATASET, path=config.DATA_RAW, unzip=True)
    except Exception as e:
        raise RuntimeError(
            "Falha ao baixar o dataset do Kaggle. Configure ~/.kaggle/kaggle.json com suas credenciais "
            "e aceite as regras do dataset em https://www.kaggle.com/datasets/aniketg11/supportticketsclassification"
        ) from e
    out = _find_csv(config.DATA_RAW)
    if out is None:
        raise FileNotFoundError(
            f"Dataset baixado em {config.DATA_RAW} mas nenhum CSV foi encontrado."
        )
    return out


def load_dataset(csv_path: Path | None = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = _find_csv(config.DATA_RAW)
    if csv_path is None:
        raise FileNotFoundError(
            f"Nenhum CSV em {config.DATA_RAW}. É obrigatório baixar o dataset do Kaggle. "
            "Execute: python -c \"from src.prep.loader import download_from_kaggle; download_from_kaggle()\""
        )
    df = pd.read_csv(csv_path)
    return df


def get_text_and_label_columns(df: pd.DataFrame):
    label_col = None
    for cand in config.LABEL_COLUMN_CANDIDATES:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        label_col = config.LABEL_COLUMN
    text_cols = [c for c in config.TEXT_COLUMNS if c in df.columns]
    if not text_cols:
        candidates = ["Body", "Description", "body", "description", "text", "Ticket"]
        text_cols = [c for c in candidates if c in df.columns]
    if not text_cols:
        text_cols = [df.columns[0]]
    return text_cols, label_col
