"""API tests against a small model trained on synthetic data (no cloud access, no real data needed)."""
import importlib
from pathlib import Path

import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from src.config import ROOT
from tests.test_model import synthetic


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api")
    data = tmp / "processed"
    data.mkdir()
    snaps = synthetic(300)
    snaps.to_parquet(data / "snapshots.parquet")
    live = snaps[snaps.cutoff == snaps.cutoff.max()].drop(columns=["y", "y_count"])
    live.to_parquet(data / "live.parquet")
    pd.DataFrame({"bid": [1, 2, 3], "name": ["The Max", None, "Bay Park Two"],
                  "address": ["606 WEST 57 STREET", "3405 NEPTUNE AVENUE", "3395 NEPTUNE AVENUE"],
                  "borough": ["MANHATTAN", "BROOKLYN", "BROOKLYN"], "zip": ["10019", "11224", "11224"],
                  "units": [837.0, 134.0, 200.0], "stories": [42.0, 20.0, 20.0], "year_built": [2015.0, 1990.0, 1990.0],
                  "lat": [40.77, 40.58, 40.58], "lng": [-73.99, -73.99, -73.99]}).to_parquet(data / "buildings.parquet")
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    cfg["data"]["processed_dir"] = str(data)
    cfg["serving"]["model_dir"] = str(tmp / "model")
    cfg["mlflow"]["tracking_uri"] = f"sqlite:///{tmp / 'mlflow.db'}"
    cfg["models"]["xgboost"]["n_estimators"] = 30
    (tmp / "config.yaml").write_text(yaml.safe_dump(cfg))

    mp = pytest.MonkeyPatch()
    mp.setenv("CONFIG_PATH", str(tmp / "config.yaml"))
    mp.setenv("DATA_SOURCE", "local")
    mp.chdir(tmp)
    import src.train as train
    mp.setattr(train, "path", lambda p: Path(p) if Path(p).is_absolute() else tmp / p)
    train.main()
    import api.main as main
    main = importlib.reload(main)  # picks up CONFIG_PATH
    with TestClient(main.app) as c:
        yield c
    mp.undo()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_predict_returns_probability_and_drivers(client):
    r = client.post("/predict", json={"borough": "MANHATTAN", "units": 837, "c12_all": 34, "v12_b": 3})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability"] <= 1 and body["label"] in (0, 1)
    assert body["risk_level"] in ("low", "medium", "high") and 0 <= body["percentile"] <= 1
    assert body["drivers"] and body["model_version"]


def test_predict_accepts_empty_object(client):
    assert client.post("/predict", json={}).status_code == 200  # every field has a default


@pytest.mark.parametrize("bad", [{"units": -5}, {"c12_all": -1}, {"borough": "NEW JERSEY"}, {"typo_field": 1}, {"lat": 10}])
def test_predict_rejects_invalid_input(client, bad):
    assert client.post("/predict", json=bad).status_code == 422


def test_more_violations_means_higher_risk(client):
    lo = client.post("/predict", json={"v12_b": 0, "v12_c": 0, "c12_all": 0}).json()["probability"]
    hi = client.post("/predict", json={"v12_b": 10, "v12_c": 6, "c12_all": 40}).json()["probability"]
    assert hi > lo


def test_batch(client):
    r = client.post("/predict/batch", json={"instances": [{}, {"v12_b": 4}]})
    assert r.status_code == 200 and len(r.json()["predictions"]) == 2
    assert client.post("/predict/batch", json={"instances": []}).status_code == 422


def test_search_by_name_and_abbreviated_address(client):
    assert client.get("/buildings/search", params={"q": "the max"}).json()["results"][0]["bid"] == 1
    res = client.get("/buildings/search", params={"q": "3405 Neptune Ave"}).json()["results"]
    assert [r["bid"] for r in res] == [2]
    assert client.get("/buildings/search", params={"q": "x"}).status_code == 422


def test_building_features_round_trip_to_predict(client):
    r = client.get("/buildings/1")
    assert r.status_code == 200
    feats = r.json()["features"]
    assert client.post("/predict", json=feats).status_code == 200  # what the front end does
    assert client.get("/buildings/999999").status_code == 404


def test_model_info_and_cors(client):
    m = client.get("/model").json()
    assert {"test_metrics", "baselines", "threshold", "version"} <= set(m)
    r = client.options("/predict", headers={"Origin": "https://example.hf.space", "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") in ("*", "https://example.hf.space")


@pytest.mark.skipif(not (ROOT / "artifacts/model/live.parquet").exists(), reason="needs a trained model bundle")
def test_every_live_building_passes_request_validation():
    """What GET /buildings/{id} returns must be accepted by POST /predict, for every building we serve."""
    from api.main import _py
    from api.schemas import BuildingFeatures
    live = pd.read_parquet(ROOT / "artifacts/model/live.parquet")
    fields = set(BuildingFeatures.model_fields)
    bad = []
    for row in live.to_dict("records"):
        feats = {k: _py(v) for k, v in row.items() if k in fields}
        feats["cutoff"] = str(row["cutoff"])[:10]
        try:
            BuildingFeatures(**feats)
        except Exception:
            bad.append(row["bid"])
    assert bad == [], f"{len(bad)} buildings would get 422, e.g. {bad[:5]}"
