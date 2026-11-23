"""Pipeline step 2: preprocessing and feature engineering.

Input: one row per building per cutoff with raw history counts and building attributes (see data_prep/make_dataset.py).
Output: the numeric feature matrix the models train on.

What happens here:
  * missing building attributes (units, stories, year built, lot size, coordinates) are filled with the
    TRAINING-set median; the medians are learned in fit() and stored with the model, so validation, test and
    live rows are filled with the same numbers and nothing leaks from them
  * building age = cutoff year - year built
  * borough becomes five 0/1 columns
  * per-unit rates (complaints and B/C violations per 10 units), so large buildings are not risky by size alone
  * optional log1p of counts for the linear model (counts are heavily skewed)

The same FeatureBuilder object is saved inside the model pipeline, so the API transforms requests exactly like training did.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

BOROUGHS = ["BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"]
ATTRIBUTES = ["units", "stories", "year_built", "bldgs_lot", "lat", "lng"]
COUNT_PREFIXES = ("c3_", "c12_", "v3_", "v12_", "bb24_")

# Plain-English names, used by the API and the front end to explain predictions
LABELS = {
    "c12_all": "311 complaints, past year", "c3_all": "311 complaints, past 3 months",
    "c12_noise": "noise complaints, past year", "c12_heat": "heat complaints, past year",
    "c12_mold": "mold or leak complaints, past year", "c12_pests": "pest complaints, past year",
    "c12_upkeep": "repair complaints, past year", "c12_sanitation": "sanitation complaints, past year",
    "c12_street": "street noise complaints, past year",
    "v12_a": "class A violations, past year", "v12_b": "class B violations, past year",
    "v12_c": "class C violations, past year", "v3_a": "class A violations, past 3 months",
    "v3_b": "class B violations, past 3 months", "v3_c": "class C violations, past 3 months",
    "v12_ri": "rent-impairing violations, past year", "v12_mold": "mold violations, past year",
    "v12_pests": "pest violations, past year", "v12_heat": "heat violations, past year",
    "bb24_infested": "units with bedbugs, past 2 years", "bb24_reinfested": "re-infested units, past 2 years",
    "units": "number of units", "stories": "number of stories", "age": "building age",
    "bldgs_lot": "buildings on the lot", "lat": "location (north-south)", "lng": "location (east-west)",
    "has_mgmt": "registered management company",
    "c12_per10u": "complaints per 10 units, past year", "c3_per10u": "complaints per 10 units, past 3 months",
    "vbc12_per10u": "B/C violations per 10 units, past year",
}
_TOPIC = {"noise": "noise", "heat": "heat", "street": "street noise", "upkeep": "repair", "mold": "mold or leak",
          "pests": "pest", "sanitation": "sanitation"}
for _k, _t in _TOPIC.items():  # every 3-month complaint column gets a readable name too
    LABELS.setdefault(f"c3_{_k}", f"{_t} complaints, past 3 months")
LABELS.setdefault("v3_ri", "rent-impairing violations, past 3 months")
for _b in BOROUGHS:
    LABELS.setdefault("boro_" + _b.lower().replace(" ", "_"), f"being in {_b.title()}")


def count_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(COUNT_PREFIXES)]


class FeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, log_counts: bool = False, per_unit_rates: bool = True):
        self.log_counts = log_counts
        self.per_unit_rates = per_unit_rates

    def fit(self, X: pd.DataFrame, y=None):
        self.counts_ = count_columns(X)
        self.medians_ = {}
        for c in ATTRIBUTES:
            m = pd.to_numeric(X[c], errors="coerce").median() if c in X else np.nan
            self.medians_[c] = 0.0 if pd.isna(m) else float(m)
        self.feature_names_ = self._build(X.head(1)).columns.tolist()
        return self

    def _build(self, X: pd.DataFrame) -> pd.DataFrame:
        """Missing columns are allowed (an API request may send only some fields): a missing count means 0,
        a missing attribute gets the training median."""
        def col(name, default):
            return X[name] if name in X else pd.Series(default, index=X.index)

        out = pd.DataFrame(index=X.index)
        for c in self.counts_:
            out[c] = pd.to_numeric(col(c, 0), errors="coerce").fillna(0).clip(lower=0)
        for c in ATTRIBUTES:
            out[c] = pd.to_numeric(col(c, np.nan), errors="coerce").fillna(self.medians_[c])
        cutoff = pd.to_datetime(col("cutoff", pd.NaT), errors="coerce")
        year = cutoff.dt.year.fillna(pd.Timestamp.now().year)
        out["age"] = (year - out.pop("year_built")).clip(lower=0)
        out["units"] = out["units"].clip(lower=1)
        out["has_mgmt"] = col("has_mgmt", False).fillna(False).astype(bool).astype(int)
        borough = col("borough", "").fillna("").astype(str).str.upper().str.strip()
        for b in BOROUGHS:
            out["boro_" + b.lower().replace(" ", "_")] = (borough == b).astype(int)
        if self.per_unit_rates:
            out["c12_per10u"] = out["c12_all"] / out["units"] * 10
            out["c3_per10u"] = out["c3_all"] / out["units"] * 10
            out["vbc12_per10u"] = (out["v12_b"] + out["v12_c"]) / out["units"] * 10
        if self.log_counts:
            skewed = self.counts_ + ["units", "stories", "bldgs_lot"] + [c for c in out if c.endswith("per10u")]
            out[skewed] = np.log1p(out[skewed])
        return out.astype(float)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = self._build(X)
        return out[self.feature_names_] if hasattr(self, "feature_names_") else out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)
