"""Feature, split, evaluation and end-to-end training tests (synthetic data, runs in seconds)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src import evaluate as ev
from src.config import ROOT, load_config
from src.features import FeatureBuilder
from src.split import time_split


def synthetic(n=600, seed=0):
    """Snapshots at the three configured cutoffs; the label depends on past B/C violations and complaints."""
    rng = np.random.default_rng(seed)
    cfg = load_config()
    parts = []
    for cut in (cfg["split"][k] for k in ("train_cutoff", "val_cutoff", "test_cutoff")):
        d = pd.DataFrame({
            "bid": np.arange(n), "cutoff": pd.Timestamp(cut),
            "borough": rng.choice(["BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"], n),
            "units": rng.integers(3, 300, n).astype(float), "stories": rng.integers(2, 30, n).astype(float),
            "year_built": rng.integers(1900, 2020, n).astype(float), "bldgs_lot": 1.0,
            "lat": 40.7 + rng.normal(0, .05, n), "lng": -73.9 + rng.normal(0, .05, n), "has_mgmt": rng.random(n) > .5,
        })
        for c in ["c3_all", "c12_all", "c12_mold", "v3_b", "v12_a", "v12_b", "v12_c", "bb24_infested"]:
            d[c] = rng.poisson(2, n)
        logit = -3 + 0.6 * d.v12_b + 0.4 * d.v12_c + 0.1 * d.c12_all
        d["y"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
        d["y_count"] = d.y * 2
        d.loc[rng.random(n) < .1, "units"] = np.nan  # missing attributes get imputed
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def test_feature_builder_imputes_with_training_medians_only():
    train = pd.DataFrame({"cutoff": ["2025-05-01"] * 3, "units": [10, 20, np.nan], "c12_all": [1, 2, 3],
                          "c3_all": [0, 0, 0], "v12_b": [0, 1, 0], "v12_c": [0, 0, 0], "borough": ["BRONX"] * 3})
    fb = FeatureBuilder().fit(train)
    other = pd.DataFrame({"cutoff": ["2026-05-01"], "units": [np.nan], "borough": ["queens"]})
    X = fb.transform(other)
    assert X.loc[0, "units"] == 15  # median of the training rows, not of the new row
    assert X.loc[0, "boro_queens"] == 1 and X.loc[0, "boro_bronx"] == 0
    assert X.loc[0, "c12_all"] == 0  # missing count means zero events
    assert not X.isna().any().any()
    assert list(X.columns) == fb.feature_names_


def test_per_unit_rates():
    df = pd.DataFrame({"cutoff": ["2025-05-01"], "units": [50], "c12_all": [10], "c3_all": [5], "v12_b": [2], "v12_c": [3]})
    X = FeatureBuilder().fit(df).transform(df)
    assert X.loc[0, "c12_per10u"] == 2 and X.loc[0, "vbc12_per10u"] == 1


def test_time_split_is_by_cutoff():
    df = synthetic(50)
    train, val, test = time_split(df, load_config())
    assert len(train) == len(val) == len(test) == 50
    assert train.cutoff.max() < val.cutoff.min() < test.cutoff.min()


def test_metrics_confusion_and_baseline_topk():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 1])
    m = ev.metrics(y, np.zeros(10), 0.5, 0.1)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (0, 0, 3, 7)
    assert m["accuracy"] == 0.7 and m["precision_top_10pct"] == 0.3  # constant scores rank nothing
    p = np.array([.9, .1, .2, .8, .1, .1, .3, .1, .1, .7])
    m = ev.metrics(y, p, 0.5, 0.1)
    assert m["tp"] == 3 and m["fp"] == 0 and m["f1"] == 1.0


def test_threshold_is_picked_from_validation_scores():
    y = np.array([0, 0, 1, 1])
    assert ev.pick_threshold(y, np.array([.1, .2, .6, .7])) <= 0.6


def test_training_pipeline_end_to_end(tmp_path, monkeypatch):
    """Runs src.train.main on synthetic data and checks the model, metadata, metrics and MLflow run."""
    data = tmp_path / "processed"
    data.mkdir()
    snaps = synthetic()
    snaps.to_parquet(data / "snapshots.parquet")
    snaps[snaps.cutoff == snaps.cutoff.max()].drop(columns=["y", "y_count"]).to_parquet(data / "live.parquet")
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    cfg["data"]["processed_dir"] = str(data)
    cfg["serving"]["model_dir"] = str(tmp_path / "model")
    cfg["mlflow"]["tracking_uri"] = f"sqlite:///{tmp_path / 'mlflow.db'}"
    cfg["models"]["xgboost"]["n_estimators"] = 50
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("CONFIG_PATH", str(cfg_path))
    monkeypatch.setenv("DATA_SOURCE", "local")
    monkeypatch.chdir(tmp_path)  # MLflow artifacts and reports/ land in the temp folder
    import src.train as train
    monkeypatch.setattr(train, "path", lambda p: Path(p) if Path(p).is_absolute() else tmp_path / p)

    summary = train.main()

    assert summary["selected_model"] in {"logistic_regression", "xgboost"}
    meta = json.loads((tmp_path / "model" / "metadata.json").read_text())
    assert 0 < meta["threshold"] < 1 and len(meta["live_score_quantiles"]) == 1001
    best = summary["results"][summary["selected_model"]]["test"]
    assert best["pr_auc"] > summary["results"]["baseline_always_no"]["test"]["pr_auc"]

    from src.predict import Predictor
    pred = Predictor(tmp_path / "model")
    low, high = pred.predict([{"v12_b": 0, "c12_all": 0}, {"v12_b": 8, "v12_c": 4, "c12_all": 30}])
    assert 0 <= low["probability"] < high["probability"] <= 1
    assert high["drivers"] and {"feature", "label", "effect", "contribution"} <= set(high["drivers"][0])


def test_every_feature_has_a_readable_label():
    from src.features import LABELS
    df = synthetic(20)
    names = FeatureBuilder().fit(df).feature_names_
    assert [n for n in names if n not in LABELS] == []
