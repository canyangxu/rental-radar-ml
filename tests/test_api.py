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
    # bids 4 and 5 are real buildings next to 5Pointz (real coordinates), for the search backup tests;
    # bids 6-8 test the house-number check: 6 is across the street from The Dakota (1 West 72nd St, not in HPD),
    # 7 is The Eugene's address and 8 a closer neighbour on the same street (approximate coordinates)
    pd.DataFrame({"bid": [1, 2, 3, 4, 5, 6, 7, 8], "name": ["The Max", None, "Bay Park Two", None, None, None, None, None],
                  "address": ["606 WEST 57 STREET", "3405 NEPTUNE AVENUE", "3395 NEPTUNE AVENUE",
                              "22-44 JACKSON AVENUE", "45-19 DAVIS STREET", "12 WEST 72 STREET",
                              "435 WEST 31 STREET", "445 WEST 31 STREET"],
                  "borough": ["MANHATTAN", "BROOKLYN", "BROOKLYN", "QUEENS", "QUEENS", "MANHATTAN", "MANHATTAN",
                              "MANHATTAN"],
                  "zip": ["10019", "11224", "11224", "11101", "11101", "10023", "10001", "10001"],
                  "units": [837.0, 134.0, 200.0, 1122.0, 128.0, 60.0, 844.0, 40.0],
                  "stories": [42.0, 20.0, 20.0, 41.0, 6.0, 9.0, 62.0, 6.0],
                  "year_built": [2015.0, 1990.0, 1990.0, 2019.0, 1931.0, 1920.0, 2017.0, 1910.0],
                  "lat": [40.77, 40.58, 40.58, 40.745640, 40.745198, 40.77680, 40.75290, 40.75300],
                  "lng": [-73.99, -73.99, -73.99, -73.946090, -73.945231, -73.97700, -73.99940, -73.99990],
                  }).to_parquet(data / "buildings.parquet")
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
    assert first["matched_name"] == "5 Pointz" and first["match"] == "same street" and body["reason"]
    assert 5 not in [r["bid"] for r in body["results"]]

    client.get("/buildings/lookup", params={"q": "5pointz "})  # same name: served from the cache
    assert len(calls) == 1


def test_lookup_without_street_match_returns_nearby_candidates(client, no_network, monkeypatch):
    place = {**FIVE_POINTZ[0], "address": {"road": "Some Other Road"}}
    monkeypatch.setattr(no_network.requests, "get", lambda *a, **k: FakeResponse([place]))
    body = client.get("/buildings/lookup", params={"q": "Somewhere"}).json()
    res = body["results"]
    # no street to check: both buildings within 100 m come back, nearest first, labelled as unconfirmed
    assert [r["bid"] for r in res] == [5, 4] and {r["match"] for r in res} == {"nearby"}
    assert "cannot tell which one" in body["reason"]


DAKOTA = {"osm_type": "way", "osm_id": 1, "lat": "40.7764", "lon": "-73.9762", "name": "The Dakota",
          "address": {"house_number": "1", "road": "West 72nd Street"}}
EUGENE = {"osm_type": "way", "osm_id": 2, "lat": "40.75300", "lon": "-73.99990", "name": "The Eugene",
          "address": {"house_number": "435", "road": "West 31st Street"}}


def test_lookup_house_number_not_in_hpd_returns_nothing(client, no_network, monkeypatch):
    # OSM knows the exact address and HPD has no building there: say so, never offer the building across the street
    area = {**DAKOTA, "osm_type": "relation", "address": {"road": "West 72nd Street"}}  # a second, address-less hit
    monkeypatch.setattr(no_network.requests, "get", lambda *a, **k: FakeResponse([DAKOTA, area]))
    body = client.get("/buildings/lookup", params={"q": "The Dakota"}).json()
    assert body["results"] == []
    assert "1 West 72 Street" in body["reason"] and "matches no address" in body["reason"]


def test_lookup_house_number_picks_that_address_over_a_closer_neighbour(client, no_network, monkeypatch):
    # the OSM point sits on 445 W 31st St, but OSM's address is 435: only 435 counts
    monkeypatch.setattr(no_network.requests, "get", lambda *a, **k: FakeResponse([EUGENE]))
    body = client.get("/buildings/lookup", params={"q": "The Eugene"}).json()
    assert [r["bid"] for r in body["results"]] == [7]
    assert body["results"][0]["address"] == "435 WEST 31 STREET" and body["results"][0]["match"] == "same address"
    assert "same address" in body["reason"]


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


def test_lookup_address_range_outside_queens_tries_both_ends():
    from api.lookup import address_tries
    manhattan = {"address": {"house_number": "145-146", "road": "Central Park West", "suburb": "Manhattan"}}
    queens = {"address": {"house_number": "22-44", "road": "Jackson Avenue", "suburb": "Queens"}}
    assert address_tries(["145-146"], manhattan) == [["145-146"], ["145", "146"]]
    assert address_tries(["22-44"], queens) == [["22-44"]]  # a Queens house number, not a range


def test_lookup_unknown_street_spelling_falls_back_to_nearby(client, no_network, monkeypatch):
    # OSM's street name matches none of ours (different spelling): the house number cannot be checked,
    # so the usual nearby rule applies instead of a wrong "not found"
    place = {**DAKOTA, "lat": "40.77670", "lon": "-73.97690", "address": {"house_number": "12", "road": "W Seventy-Second St"}}
    monkeypatch.setattr(no_network.requests, "get", lambda *a, **k: FakeResponse([place]))
    res = client.get("/buildings/lookup", params={"q": "Somewhere on 72nd"}).json()["results"]
    assert [r["bid"] for r in res] == [6] and res[0]["match"] == "nearby"  # 12 West 72nd, next to the point


def test_predict_accepts_borough_in_any_case(client):
    for b in ["bronx", "Bronx", " staten  island "]:
        assert client.post("/predict", json={"borough": b, "units": 20}).status_code == 200
    assert client.post("/predict", json={"borough": "Paris"}).status_code == 422
