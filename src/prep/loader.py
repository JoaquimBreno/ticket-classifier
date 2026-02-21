import pandas as pd
from pathlib import Path

import config

KAGGLE_DATASET = "adisongoh/it-service-ticket-classification-dataset"
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset"


def _find_csv(path: Path) -> Path | None:
    preferred = path / config.RAW_CSV_FILENAME
    if preferred.exists():
        return preferred
    for f in path.rglob("*.csv"):
        return f
    return None


def download_from_kaggle() -> Path:
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    existing = _find_csv(config.DATA_RAW)
    if existing:
        return existing
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter

        raw = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            KAGGLE_DATASET,
            "",
        )
        if isinstance(raw, dict):
            df = next(iter(raw.values()))
        else:
            df = raw
        if df is None or (hasattr(df, "empty") and df.empty):
            raise ValueError("Dataset retornou vazio.")
        out_path = config.DATA_RAW / config.RAW_CSV_FILENAME
        df.to_csv(out_path, index=False)
        return out_path
    except Exception as e:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(KAGGLE_DATASET, path=config.DATA_RAW, unzip=True)
            out = _find_csv(config.DATA_RAW)
            if out is None:
                raise FileNotFoundError(
                    f"Dataset baixado em {config.DATA_RAW} mas nenhum CSV foi encontrado."
                ) from e
            return out
        except Exception as e2:
            raise RuntimeError(
                f"Falha ao baixar o dataset do Kaggle. Configure ~/.kaggle/kaggle.json e aceite as regras em {KAGGLE_DATASET_URL}"
            ) from (e2 or e)


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
    label_col = config.LABEL_COLUMN
    if label_col not in df.columns:
        raise ValueError(f"Coluna de rótulo '{label_col}' não existe no dataset. Colunas: {list(df.columns)}")
    text_cols = [c for c in config.TEXT_COLUMNS if c in df.columns]
    if not text_cols:
        raise ValueError(f"Nenhuma coluna de texto em {config.TEXT_COLUMNS} existe no dataset. Colunas: {list(df.columns)}")
    return text_cols, label_col
