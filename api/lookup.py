"""Name lookup fallback: find a building by a name we do not have (e.g. "5Pointz") through OpenStreetMap.

Only 3,867 of our 162,997 buildings have a name. When /buildings/search finds nothing, the front end can ask
/buildings/lookup, which:
  1. asks OpenStreetMap Nominatim (free, no API key) where that name is;
  2. checks the answer against our own HPD building table:
     - when OSM gives a house number, only our building at that exact address counts; if none of ours has it
       (HPD may list the building under another address, or not at all, e.g. a co-op), the answer is
       "not found", never the building next door;
     - without a house number, a building must be close to the OSM point AND on the same street (the nearest
       building alone is often the wrong one: next door, around the corner);
  3. returns at most 3 of our buildings, each with where the match came from.

Nominatim usage policy: an identifying User-Agent and at most 1 request per second. Answers are cached in memory,
so the same name never calls Nominatim twice while the server runs.
"""
from __future__ import annotations

import re
import threading
import time
from typing import Callable

import numpy as np
import pandas as pd
import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "rental-radar-ml/1.0 (course project)"
NYC_VIEWBOX = "-74.26,40.92,-73.68,40.49"  # left, top, right, bottom
TIMEOUT_S = 6
SOURCE = "OpenStreetMap (Nominatim)"
STREET_RADIUS_M = 200  # same street: allow a big lot (5Pointz's building entrance is 89 m from its OSM point)
NEAR_RADIUS_M = 60     # no street match: only something right at the point
ADDRESS_RADIUS_M = 1000  # exact address: same number and street, and within 1 km (the street may exist in two boroughs)
MAX_RESULTS = 3

_CACHE: dict[str, dict] = {}
_LOCK = threading.Lock()  # FastAPI runs sync endpoints in a thread pool, so two requests could race
_last_call = 0.0


def _rate_limited_get(params: dict) -> requests.Response:
    """GET Nominatim, waiting first if the previous call was less than 1 second ago."""
    global _last_call
    with _LOCK:
        wait = 1.0 - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            return requests.get(NOMINATIM_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_S)
        finally:
            _last_call = time.monotonic()


def search_osm(q: str) -> list[dict]:
    """Nominatim places for `q`, limited to New York City. Raises on network or format errors."""
    params = {"q": f"{q}, New York", "format": "jsonv2", "addressdetails": 1, "limit": 5, "countrycodes": "us",
              "viewbox": NYC_VIEWBOX, "bounded": 1}
    r = _rate_limited_get(params)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise ValueError("unexpected Nominatim response")
    return data


def distance_m(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Haversine distance in meters from one point to many."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat, lon, lats, lons))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6_371_000 * 2 * np.arcsin(np.sqrt(a))


def street_of(addr: str) -> str:
    """'22-44 JACKSON AVENUE' -> 'JACKSON AVENUE' (drop the house number, which may have a hyphen or a letter)."""
    return re.sub(r"^\d+(?:-\d+)?[A-Z]?\s+", "", addr)


def house_numbers(place: dict) -> list[str]:
    """OSM house number(s) of a place: '435' -> ['435']; '1;3' or '1, 3' -> ['1', '3']; none -> []."""
    hn = str((place.get("address") or {}).get("house_number") or "").upper()
    return [h.strip() for h in re.split(r"[;,]", hn) if h.strip()]


def address_tries(numbers: list[str], place: dict) -> list[list[str]]:
    """House numbers to try, in order. A range outside Queens ('145-146' Central Park West) is also tried as its two
    ends, since HPD lists one of them. In Queens '22-44' is a single house number, so it is never split."""
    tries = [numbers]
    borough = " ".join(str(v) for v in (place.get("address") or {}).values()).upper()
    if "QUEENS" not in borough:
        ends = [p for h in numbers if (m := re.fullmatch(r"(\d+)-(\d+)", h)) for p in m.groups()]
        if ends:
            tries.append(ends)
    return tries


def match_buildings(place: dict, b: pd.DataFrame, norm: Callable[[str], str]) -> tuple[pd.DataFrame, str | None]:
    """Our buildings that one OSM place points to, with distance_m and street_match columns.

    Also returns the OSM address when OSM gives a house number and none of our buildings has it (else None).
    """
    try:
        lat, lon = float(place["lat"]), float(place["lon"])
    except (KeyError, TypeError, ValueError):
        return b.iloc[0:0], None
    b = b.dropna(subset=["lat", "lng"])
    d = distance_m(lat, lon, b["lat"].to_numpy(float), b["lng"].to_numpy(float))
    road = norm((place.get("address") or {}).get("road", ""))

    numbers = house_numbers(place)
    close = d <= ADDRESS_RADIUS_M
    # Trust OSM's address only when we know its street: if OSM spells the street differently from HPD
    # ('Adam Clayton Powell Jr. Boulevard' vs 'ADAM C POWELL BOULEVARD'), no address can match, so fall back below
    street_known = bool(road) and bool((b.loc[close, "_addr"].map(street_of) == road).any())
    if numbers and street_known:  # OSM knows the exact address: accept that address only
        hit = np.zeros(len(b), bool)
        for tries in address_tries(numbers, place):  # as written first, then the two ends of a range
            hit = close & b["_addr"].isin({norm(f"{h} {road}") for h in tries}).to_numpy()
            if hit.any():
                break
        exact = b[hit].assign(distance_m=d[hit], street_match=True, exact_address=True)
        missing = None if len(exact) else f"{numbers[0]} {road}"
        return exact, missing

    near = b[d <= STREET_RADIUS_M].assign(distance_m=d[d <= STREET_RADIUS_M])
    near = near.assign(street_match=bool(road) & (near["_addr"].map(street_of) == road), exact_address=False)
    if near["street_match"].any():
        return near[near["street_match"]], None
    return near[near["distance_m"] <= NEAR_RADIUS_M], None


def lookup(q: str, buildings: pd.DataFrame, norm: Callable[[str], str]) -> dict:
    """Find up to 3 of our buildings for a place name. Never raises: on any failure returns no results and a reason."""
    key = re.sub(r"\s+", " ", q.strip().lower())
    if key in _CACHE:
        return {**_CACHE[key], "query": q}
    if buildings.empty:
        return {"query": q, "results": [], "reason": "The building table is not loaded."}
    try:
        places = search_osm(q)
    except requests.Timeout:
        return {"query": q, "results": [], "reason": "OpenStreetMap did not answer in time. Try again in a moment."}
    except (requests.RequestException, ValueError):  # connection error, HTTP error, bad JSON
        return {"query": q, "results": [], "reason": "OpenStreetMap could not be reached. Try the street address."}

    found, missing = [], []
    for place in places:
        m, miss = match_buildings(place, buildings, norm)
        if miss:
            name = place.get("name") or str(place.get("display_name", "")).split(",")[0]
            missing.append((name, miss))
        if len(m):
            osm_url = f"https://www.openstreetmap.org/{place.get('osm_type', 'node')}/{place.get('osm_id', '')}"
            name = place.get("name") or str(place.get("display_name", "")).split(",")[0]
            found.append(m.assign(source_url=osm_url, matched_name=name))

    if missing or any(f["exact_address"].any() for f in found):
        # OSM gave an exact address for this name: approximate matches from its other results (a land-use area,
        # a neighbourhood) would only bring back buildings next door, so keep exact-address matches only
        found = [f[f["exact_address"]] for f in found if f["exact_address"].any()]

    if not places:
        answer = {"results": [], "reason": "OpenStreetMap does not know this name in New York City."}
    elif not found and missing:
        name, addr = missing[0]
        answer = {"results": [], "reason": f"OpenStreetMap places “{name}” at {addr.title()}, which matches no address "
                                           "in HPD's list of buildings with 3+ units. Try the building's street "
                                           "address."}
    elif not found:
        answer = {"results": [], "reason": "OpenStreetMap found the place, but no HPD building with 3+ units is there."}
    else:
        hits = (pd.concat(found)
                .sort_values(["exact_address", "street_match", "distance_m", "units"], ascending=[False, False, True, False])
                .drop_duplicates("bid")  # the same building can sit near two OSM results: keep its best match
                .head(MAX_RESULTS))
        results = [{"bid": int(r.bid), "name": r.name if isinstance(r.name, str) else None, "address": r.address,
                    "borough": r.borough, "zip": None if pd.isna(r.zip) else str(r.zip),
                    "units": None if pd.isna(r.units) else float(r.units), "distance_m": round(float(r.distance_m)),
                    "source": SOURCE, "source_url": r.source_url, "matched_name": r.matched_name}
                   for r in hits.itertuples(index=False)]
        top = hits.iloc[0]
        where = (f"at {top['address'].title()}, the same address as this HPD building" if top["exact_address"] else
                 f"on {street_of(top['_addr']).title()}, next to these HPD buildings" if top["street_match"] else
                 f"within {NEAR_RADIUS_M} m of these HPD buildings")
        answer = {"results": results, "reason": f"OpenStreetMap places “{top['matched_name']}” {where}."}
    _CACHE[key] = answer  # cache real answers (including "not found"), never network failures
    return {**answer, "query": q}
