"""Load the trained model and score buildings. Used by the API and by `python -m src.predict` for a quick check.

The model folder (serving.model_dir) holds:
  model.joblib   the full scikit-learn pipeline: FeatureBuilder (preprocessing) + classifier
  metadata.json  model name and version, decision threshold, test metrics, baselines, score percentiles for
                 the live snapshot and the observed violation rate per predicted-probability band
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import LABELS

# shown to people as one driver: where the building is
GROUPS = {"lat": "location", "lng": "location", "boro_bronx": "location", "boro_brooklyn": "location",
          "boro_manhattan": "location", "boro_queens": "location", "boro_staten_island": "location"}


class Predictor:
    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        self.pipeline = joblib.load(model_dir / "model.joblib")
        self.meta = json.loads((model_dir / "metadata.json").read_text())
        self.builder = self.pipeline.named_steps["features"]
        self.model = self.pipeline.steps[-1][1]
        self.threshold = float(self.meta["threshold"])
        self._quantiles = np.asarray(self.meta["live_score_quantiles"])
        # same fitted preprocessing without the log transform, to show people the real values of derived features
        self._display = copy.copy(self.builder)
        self._display.log_counts = False

    def predict(self, rows: list[dict]) -> list[dict]:
        df = pd.DataFrame(rows)
        p = self.pipeline.predict_proba(df)[:, 1]
        contribs = self._contributions(df)
        shown = self._display.transform(df).to_dict("records")
        return [self._explain(float(pi), c, row, s) for pi, c, row, s in zip(p, contribs, rows, shown)]

    def _contributions(self, df: pd.DataFrame) -> np.ndarray:
        """Per-feature contribution to the log-odds of each prediction."""
        X = self.builder.transform(df)
        if self.meta["model_type"] == "xgboost":
            import xgboost as xgb

            return self.model.get_booster().predict(xgb.DMatrix(X, feature_names=list(X.columns)), pred_contribs=True)[:, :-1]
        Z = self.pipeline[:-1].transform(df)  # features + scaler: contribution = coefficient x standardized value
        return np.asarray(Z) * self.model.coef_[0]

    def _explain(self, p: float, contrib: np.ndarray, row: dict, shown: dict) -> dict:
        grouped: dict[str, float] = {}
        for name, v in zip(self.builder.feature_names_, contrib):
            key = GROUPS.get(name, name)
            grouped[key] = grouped.get(key, 0.0) + float(v)
        top = sorted(grouped.items(), key=lambda kv: -abs(kv[1]))[:6]
        drivers = [{"feature": f, "label": LABELS.get(f, f), "value": _value(f, row, shown),
                    "effect": "raises" if v > 0 else "lowers", "contribution": round(v, 4)} for f, v in top]
        band = next((b for b in self.meta["calibration_bands"] if b["low"] <= p < b["high"] or (p >= 1 and b["high"] >= 1)), None)
        return {
            "probability": round(p, 4),
            "label": int(p >= self.threshold),
            "risk_level": "high" if p >= self.threshold else ("medium" if p >= self.meta["base_rate"] else "low"),
            "percentile": round(float(np.searchsorted(self._quantiles, p) / (len(self._quantiles) - 1)), 4),
            "threshold": self.threshold,
            "base_rate": self.meta["base_rate"],
            "backtest_band": band,
            "drivers": drivers,
        }


def _value(feature: str, row: dict, shown: dict):
    """The value a person can read next to a driver: the request value, or the derived value (age, rates)."""
    if feature == "location":
        return None
    v = row.get(feature, shown.get(feature))
    if v is None or (isinstance(v, float) and np.isnan(v)):
        v = shown.get(feature)
    return None if v is None else round(float(v), 2)


if __name__ == "__main__":
    from src.config import load_config, path

    cfg = load_config()
    pred = Predictor(path(cfg["serving"]["model_dir"]))
    example = {"borough": "MANHATTAN", "units": 837, "stories": 42, "year_built": 2015, "c12_all": 34, "c12_mold": 6, "v12_b": 1}
    print(json.dumps(pred.predict([example])[0], indent=2))
