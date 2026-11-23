"""Load config.yaml so every step of the pipeline shares one set of parameters."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = ("project", "data", "features", "target", "split", "models", "evaluation", "mlflow", "serving")


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path or os.getenv("CONFIG_PATH") or ROOT / "config.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    _apply_env(cfg)
    _validate(cfg)
    return cfg


def _apply_env(cfg: dict) -> None:
    """A few values differ per machine (bucket name, tracking server); let the environment override them."""
    if os.getenv("GCS_BUCKET"):
        cfg["data"]["gcs"]["bucket"] = os.environ["GCS_BUCKET"]
    if os.getenv("MLFLOW_TRACKING_URI"):
        cfg["mlflow"]["tracking_uri"] = os.environ["MLFLOW_TRACKING_URI"]
    if os.getenv("MODEL_DIR"):
        cfg["serving"]["model_dir"] = os.environ["MODEL_DIR"]


def _validate(cfg: dict) -> None:
    for key in REQUIRED:
        if key not in cfg:
            raise ValueError(f"config.yaml is missing the '{key}' section")
    s = cfg["split"]
    if not s["train_cutoff"] < s["val_cutoff"] < s["test_cutoff"] < s["live_cutoff"]:
        raise ValueError("split cutoffs must be in time order: train < val < test < live")
    if cfg["target"]["horizon_months"] <= 0:
        raise ValueError("target.horizon_months must be positive")
    if not set(cfg["target"]["violation_classes"]) <= {"A", "B", "C"}:
        raise ValueError("target.violation_classes must be a subset of A, B, C")
    if cfg["evaluation"]["primary_metric"] not in {"pr_auc", "roc_auc", "f1"}:
        raise ValueError("evaluation.primary_metric must be pr_auc, roc_auc or f1")


def path(cfg_value: str) -> Path:
    """Resolve a path from config.yaml relative to the project root."""
    p = Path(cfg_value)
    return p if p.is_absolute() else ROOT / p
