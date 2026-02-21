import pandas as pd
import numpy as np

import config


def stratified_sample(
    df: pd.DataFrame,
    label_col: str,
    n: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    n = n or config.SAMPLE_SIZE
    seed = seed if seed is not None else config.SEED
    if n >= len(df):
        return df.reset_index(drop=True)
    if label_col not in df.columns:
        return df.sample(n=n, random_state=seed).reset_index(drop=True)
    y = df[label_col].astype(str)
    classes = y.unique()
    n_per_class = max(1, n // len(classes))
    indices = []
    for c in classes:
        idx = df.index[y == c].tolist()
        k = min(n_per_class, len(idx))
        chosen = np.random.RandomState(seed).choice(idx, size=k, replace=False)
        indices.extend(chosen.tolist())
    if len(indices) > n:
        indices = np.random.RandomState(seed).choice(indices, size=n, replace=False).tolist()
    elif len(indices) < n:
        remaining = [i for i in df.index if i not in indices]
        extra = np.random.RandomState(seed + 1).choice(remaining, size=min(n - len(indices), len(remaining)), replace=False)
        indices = indices + extra.tolist()
    return df.loc[indices].reset_index(drop=True)
