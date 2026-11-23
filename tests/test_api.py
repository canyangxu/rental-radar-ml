"""API tests against a small model trained on synthetic data (no cloud access, no real data needed)."""
import importlib
from pathlib import Path

import pandas as pd
import pytest
import requests
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
    # bids 4 and 5 are real buildings next to 5Pointz (real coordinates), for the search backup tests
    pd.DataFrame({"bid": [1, 2, 3, 4, 5], "name": ["The Max", None, "Bay Park Two", None, None],
                  "address": ["606 WEST 57 STREET", "3405 NEPTUNE AVENUE", "3395 NEPTUNE AVENUE",
                              "22-44 JACKSON AVENUE", "45-19 DAVIS STREET"],
                  "borough": ["MANHATTAN", "BROOKLYN", "BROOKLYN", "QUEENS", "QUEENS"],
                  "zip": ["10019", "11224", "11224", "11101", "11101"],
                  "units": [837.0, 134.0, 200.0, 1122.0, 128.0], "stories": [42.0, 20.0, 20.0, 41.0, 6.0],
                  "year_built": [2015.0, 1990.0, 1990.0, 2019.0, 1931.0],
                  "lat": [40.77, 40.58, 40.58, 40.745640, 40.745198],
                  "lng": [-73.99, -73.99, -73.99, -73.946090, -73.945231]}).to_parquet(data / "buildings.parquet")
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


# ---------- search backup: hyphen-free addresses and OpenStreetMap name lookup ----------
FIVE_POINTZ = [{"osm_type": "way", "osm_id": 1146353417, "lat": "40.7449071", "lon": "-73.9456593",
                "name": "5 Pointz", "display_name": "5 Pointz, Jackson Avenue, Long Island City, Queens, New York",
                "address": {"road": "Jackson Avenue", "suburb": "Queens", "city": "New York"}}]


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Tests never call the real Nominatim: any un-mocked call fails loudly. Also resets the cache and rate limit."""
    from api import lookup
    def refuse(*a, **k):
        raise AssertionError("test tried to call the real network")
    monkeypatch.setattr(lookup.requests, "get", refuse)
    monkeypatch.setattr(lookup, "_CACHE", {})
    monkeypatch.setattr(lookup, "_last_call", 0.0)
    return lookup


@pytest.mark.parametrize("q", ["22 44 Jackson", "22-44 Jackson Ave", "2244 Jackson"])
def test_search_queens_address_with_or_without_hyphen(client, q):
    res = client.get("/buildings/search", params={"q": q}).json()["results"]
    assert res and res[0]["bid"] == 4


def test_search_street_number_is_not_joined(client):
    # '3405 Neptune' must not turn into something else, and '1 5 Avenue' style queries keep their meaning
    assert [r["bid"] for r in client.get("/buildings/search", params={"q": "3405 Neptune"}).json()["results"]] == [2]
    assert client.get("/buildings/search", params={"q": "5Pointz"}).json()["results"] == []  # what lookup is for


def test_lookup_5pointz_picks_the_building_on_the_osm_street(client, no_network, monkeypatch):
    calls = []
    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(params)
        assert "nominatim.openstreetmap.org" in url and headers["User-Agent"] and timeout == 6
        assert params["q"] == "5Pointz, New York" and params["bounded"] == 1
        return FakeResponse(FIVE_POINTZ)
    monkeypatch.setattr(no_network.requests, "get", fake_get)

    body = client.get("/buildings/lookup", params={"q": "5Pointz"}).json()
    first = body["results"][0]
    # 45-19 Davis St is closer (48 m) but on another street; 22-44 Jackson Ave (89 m) is on OSM's road
    assert first["bid"] == 4 and first["address"] == "22-44 JACKSON AVENUE" and first["units"] == 1122
    assert 80 <= first["distance_m"] <= 100
    assert first["source"] == "OpenStreetMap (Nominatim)" and first["source_url"].startswith("https://www.openstreetmap.org/way/")
    assert first["matched_name"] == "5 Pointz" and body["reason"]
    assert 5 not in [r["bid"] for r in body["results"]]

    client.get("/buildings/lookup", params={"q": "5pointz "})  # same name: served from the cache
    assert len(calls) == 1


def test_lookup_without_street_match_keeps_only_very_close_buildings(client, no_network, monkeypatch):
    place = {**FIVE_POINTZ[0], "address": {"road": "Some Other Road"}}
    monkeypatch.setattr(no_network.requests, "get", lambda *a, **k: FakeResponse([place]))
    res = client.get("/buildings/lookup", params={"q": "Somewhere"}).json()["results"]
    assert [r["bid"] for r in res] == [5]  # 48 m: within 60 m; Jackson at 89 m is not


@pytest.mark.parametrize("error", [requests.Timeout("slow"), requests.ConnectionError("down"), ValueError("bad json")])
def test_lookup_network_failure_returns_empty_with_reason(client, no_network, monkeypatch, error):
    def fail(*a, **k):
        raise error
    monkeypatch.setattr(no_network.requests, "get", fail)
    r = client.get("/buildings/lookup", params={"q": "5Pointz"})
    assert r.status_code == 200
    assert r.json()["results"] == [] and r.json()["reason"]


def test_lookup_waits_between_nominatim_calls(client, no_network, monkeypatch):
    slept = []
    monkeypatch.setattr(no_network.time, "sleep", slept.append)
    monkeypatch.setattr(no_network.requests, "get", lambda *a, **k: FakeResponse([]))
    client.get("/buildings/lookup", params={"q": "first name"})
    body = client.get("/buildings/lookup", params={"q": "second name"}).json()  # a different name: not cached
    assert body["results"] == [] and body["reason"]
    assert len(slept) == 1 and 0 < slept[0] <= 1.0  # the second call waited so we stay under 1 request/second


def test_lookup_does_not_shadow_building_route(client):
    assert client.get("/buildings/1").status_code == 200
    assert client.get("/buildings/lookup").status_code == 422  # q is required


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
