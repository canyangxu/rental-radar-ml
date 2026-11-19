"""Load config.yaml so every step of the pipeline shares one set of parameters."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    for key in ("data", "features", "target", "split", "models", "mlflow", "serving"):
        if key not in cfg:
            raise ValueError(f"config.yaml is missing the '{key}' section")
    s = cfg["split"]
    if not s["train_cutoff"] < s["val_cutoff"] < s["test_cutoff"]:
        raise ValueError("split cutoffs must be in time order: train < val < test")
