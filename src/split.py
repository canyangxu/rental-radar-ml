"""Pipeline step 3: train / validation / test split by time.

Rows are snapshots of every building at a cutoff date, so the split is by cutoff, not random:
  train = cutoff 2025-05-01, validation = 2025-11-01, test = 2026-05-01 (config.yaml: split).
A random split would put the same building's earlier and later snapshots on both sides and let the model peek at
the future. Splitting by time mirrors real use: learn from the past, predict the next six months.
"""
from __future__ import annotations

import pandas as pd


def time_split(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s = cfg["split"]
    cut = pd.to_datetime(df["cutoff"]).dt.strftime("%Y-%m-%d")
    parts = tuple(df[cut == s[k]].reset_index(drop=True) for k in ("train_cutoff", "val_cutoff", "test_cutoff"))
    for name, part in zip(("train", "val", "test"), parts):
        if part.empty:
            raise ValueError(f"no rows for the {name} cutoff; check split in config.yaml and the dataset")
    return parts
