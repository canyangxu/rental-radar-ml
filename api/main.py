"""Rental Radar ML API: serves the violation-risk model.

Endpoints
  GET  /                     service info and links
  GET  /health               liveness check (Cloud Run uses it)
  GET  /model                model version, threshold, test metrics and baselines
  POST /predict              JSON features of one building -> probability, risk level and the drivers behind it
  POST /predict/batch        up to 500 buildings at once
  GET  /buildings/search     find a building by name or address (q=...)
  GET  /buildings/lookup     fallback: find a building by a name we do not have, via OpenStreetMap (q=...)
  GET  /buildings/{bid}      a building's latest history (the JSON the front end sends to /predict)

Run locally:  uvicorn api.main:app --port 8080     docs at http://localhost:8080/docs
"""
from __future__ import annotations

import re
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from api import lookup
from api.schemas import BatchRequest, BatchResponse, BuildingFeatures, Prediction
from src.config import load_config, path
from src.predict import Predictor

CFG = load_config()
STATE: dict = {}


def norm_address(s: str) -> str:
    """Same rules the data step uses for street names, so '625 W 57th St' finds '625 WEST 57 STREET'."""
    s = re.sub(r"[^\w\s-]", " ", (s or "").upper())
    s = re.sub(r"\b(\d+)(ST|ND|RD|TH)\b", r"\1", s)
    for a, b in ((r"\bAVE?\b", "AVENUE"), (r"\bST\b", "STREET"), (r"\bW\b", "WEST"), (r"\bE\b", "EAST"),
                 (r"\bN\b", "NORTH"), (r"\bS\b", "SOUTH"), (r"\bPL\b", "PLACE"), (r"\bRD\b", "ROAD"),
                 (r"\bBLVD\b", "BOULEVARD"), (r"\bPKWY\b", "PARKWAY"), (r"\bDR\b", "DRIVE")):
        s = re.sub(a, b, s)
    return re.sub(r"\s+", " ", s).strip()


STREET_WORDS = r"(?:AVENUE|STREET|PLACE|ROAD|BOULEVARD|PARKWAY|DRIVE|LANE|TERRACE|COURT)"


def addr_key(s: str) -> str:
    """Stored address with a two-part Queens house number written as one number: '22-44 JACKSON AVENUE' -> '2244 JACKSON AVENUE'."""
    return re.sub(r"^(\d+)-(\d+)\b", r"\1\2", s)


def query_key(s: str) -> str:
    """Normalized query with its first two numbers joined, so '22 44 JACKSON', '22-44 JACKSON' and '2244 JACKSON' agree.

    Not joined when the second number is a street number ('1 5 AVENUE' is One Fifth Avenue, not 15 Avenue).
    """
    return re.sub(rf"^(\d+)[\s-]+(\d+)(?=$|\s(?!{STREET_WORDS}\b))", r"\1\2", s)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = path(CFG["serving"]["model_dir"])
    STATE["predictor"] = Predictor(model_dir)
    live = pd.read_parquet(model_dir / "live.parquet") if (model_dir / "live.parquet").exists() else pd.DataFrame()
    STATE["live"] = live.set_index("bid") if len(live) else live
    b = pd.read_parquet(model_dir / "buildings.parquet") if (model_dir / "buildings.parquet").exists() else pd.DataFrame()
    if len(b):
        b["_addr"] = b["address"].fillna("").map(norm_address)
        b["_key"] = b["_addr"].map(addr_key)
        b["_name"] = b["name"].fillna("").str.upper()
    STATE["buildings"] = b
    yield
    STATE.clear()


app = FastAPI(title="Rental Radar ML API", version="1.0.0", lifespan=lifespan,
              description="Predicts whether a NYC apartment building will get a class B or C HPD housing violation "
                          "in the next 6 months. Model trained on NYC Open Data; see /model for its test metrics.")
app.add_middleware(CORSMiddleware, allow_origins=CFG["serving"]["cors_origins"], allow_methods=["GET", "POST"],
                   allow_headers=["*"])


@app.middleware("http")
async def timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
    return response


def version() -> str:
    return STATE["predictor"].meta["version"]


def score(rows: list[BuildingFeatures]) -> list[dict]:
    records = [r.model_dump() for r in rows]
    for r in records:
        r["cutoff"] = r["cutoff"] or CFG["split"]["live_cutoff"]
    return [{**p, "model_version": version()} for p in STATE["predictor"].predict(records)]


@app.get("/")
def root():
    return {"service": "rental-radar-ml", "model_version": version(),
            "predict": "POST /predict with building features as JSON", "docs": "/docs", "model": "/model"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "predictor" in STATE, "model_version": version()}


@app.get("/model")
def model_info():
    m = STATE["predictor"].meta
    keys = ("model_name", "model_type", "version", "trained_at", "target", "cutoffs", "threshold", "threshold_rule",
            "base_rate", "test_metrics", "val_metrics", "baselines", "calibration_bands", "mlflow_run_id")
    return {k: m[k] for k in keys} | {"n_features": len(m["features"])}


@app.post("/predict", response_model=Prediction)
def predict(features: BuildingFeatures):
    return score([features])[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    return {"predictions": score(req.instances), "model_version": version()}


@app.get("/buildings/search")
def search(q: str = Query(..., min_length=2, max_length=80), limit: int = Query(8, ge=1, le=25)):
    b: pd.DataFrame = STATE["buildings"]
    if b.empty:
        return {"query": q, "results": []}
    qa, qn = norm_address(q), q.upper().strip()
    qk = query_key(qa)  # '22 44 JACKSON' -> '2244 JACKSON'
    if re.match(r"^\d", qa):  # starts with a house number: match the start, as written or as a key
        by_addr = b["_addr"].str.startswith(qa) | b["_key"].str.startswith(qk)
    else:
        by_addr = b["_addr"].str.contains(qa, regex=False)
    by_name = b["_name"].str.contains(qn, regex=False) if len(qn) >= 3 else pd.Series(False, index=b.index)
    hits = b[by_addr | by_name].copy()
    addr_eq = (hits["_addr"] == qa) | (hits["_key"] == qk)
    addr_start = hits["_addr"].str.startswith(qa) | hits["_key"].str.startswith(qk)
    hits["_rank"] = np.select([hits["_name"] == qn, hits["_name"].str.startswith(qn), addr_eq, addr_start],
                              [0, 1, 2, 3], 4)
    hits = hits.sort_values(["_rank", "units"], ascending=[True, False]).head(limit)
    cols = ["bid", "name", "address", "borough", "zip", "units"]
    return {"query": q, "results": _clean(hits[cols]).to_dict("records")}


@app.get("/buildings/lookup")
def building_lookup(q: str = Query(..., min_length=2, max_length=80)):
    """Fallback for names we do not have: ask OpenStreetMap where the place is, then keep only our buildings on that
    street nearby. Always 200; when nothing is found, `results` is empty and `reason` says why."""
    return lookup.lookup(q, STATE["buildings"], norm_address)


@app.get("/buildings/{bid}")
def building(bid: int):
    b, live = STATE["buildings"], STATE["live"]
    if live.empty or bid not in live.index:
        raise HTTPException(404, f"building {bid} not found")
    row = live.loc[bid]
    fields = set(BuildingFeatures.model_fields)
    features = {k: _py(v) for k, v in row.items() if k in fields}
    features["cutoff"] = str(row["cutoff"])[:10]
    info = b[b.bid == bid].drop(columns=["_addr", "_key", "_name"], errors="ignore") if "bid" in b else pd.DataFrame()
    info = _clean(info).to_dict("records")[0] if len(info) else {"bid": bid}
    return {"building": info, "features": features}


def _py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(object).where(df.notna(), None)
