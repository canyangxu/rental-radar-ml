"""Step 1 (local, run once): download the raw NYC Open Data snapshot.

Each dataset is fetched through the Socrata SODA API as pages of CSV and written to
data/raw/<name>/part-NNNNN.csv.gz. Before downloading, a count(*) query records how many
rows to expect, and the count is saved next to the data in _info.json. The run is resumable:
finished pages are skipped.

Dataset ids and the date window come from config.yaml (data.sources, data.start_date, data.end_date).

Usage:
    python -m data_prep.download              # every dataset
    python -m data_prep.download violations   # one dataset

An optional SOCRATA_APP_TOKEN environment variable raises the API rate limit; it is not required.
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests

from src.config import load_config, path

PAGE = 50_000

# 311 complaint types kept besides every HPD complaint: noise, pests, sanitation and building systems.
EXTRA_311_TYPES = [
    "Noise - Residential", "Noise - Street/Sidewalk", "Noise - Commercial", "Noise - Vehicle",
    "Rodent", "Dirty Condition", "Residential Disposal Complaint", "Indoor Air Quality",
    "Lead", "Elevator", "Boilers", "Indoor Sewage",
]


def dataset_specs(cfg: dict) -> dict[str, dict]:
    """Columns and filters per dataset. Only the columns the pipeline uses are downloaded."""
    d = cfg["data"]
    start, end = f"{d['start_date']}T00:00:00", f"{d['end_date']}T00:00:00"
    types = ",".join(f"'{t}'" for t in EXTRA_311_TYPES)
    specs = {
        "buildings": {
            "select": "buildingid,boro,housenumber,lowhousenumber,highhousenumber,streetname,zip,block,lot,bin,"
                      "legalstories,legalclassa,legalclassb,registrationid,lifecycle,recordstatus,managementprogram,dobbuildingclass",
            "where": None},
        "pluto": {
            "select": "bbl,borough,address,zipcode,unitsres,unitstotal,numbldgs,numfloors,yearbuilt,bldgclass,ownername,latitude,longitude",
            "where": f"unitsres >= {d['min_units']}"},
        "registrations": {
            "select": "registrationid,buildingid,bin,boro,housenumber,streetname,zip,block,lot,lastregistrationdate,registrationenddate",
            "where": None},
        "contacts": {  # company names only; names of individual people are never downloaded
            "select": "registrationcontactid,registrationid,type,contactdescription,corporationname",
            "where": None},
        "names_hny": {
            "select": "project_id,project_name,project_start_date,project_completion_date,building_id,house_number,street_name,"
                      "borough,postcode,bbl,bin,latitude,longitude,total_units,counted_rental_units",
            "where": None},
        "bedbugs": {
            "select": "building_id,registration_id,borough,house_number,street_name,postcode,of_dwelling_units,"
                      "infested_dwelling_unit_count,eradicated_unit_count,re_infested_dwelling_unit,filing_date,"
                      "filing_period_start_date,filling_period_end_date,latitude,longitude,bin,bbl",
            "where": f"filling_period_end_date >= '{d['bedbug_start_date']}T00:00:00'"},
        "violations": {
            "select": "violationid,buildingid,registrationid,bin,bbl,boro,housenumber,streetname,zip,apartment,story,class,"
                      "inspectiondate,novdescription,currentstatus,currentstatusdate,violationstatus,rentimpairing,latitude,longitude",
            "where": f"(inspectiondate >= '{start}' AND inspectiondate < '{end}') OR violationstatus = 'Open'"},
        "complaints": {
            "select": "unique_key,created_date,closed_date,agency,complaint_type,descriptor,descriptor_2,location_type,status,"
                      "resolution_description,incident_address,borough,incident_zip,bbl,latitude,longitude",
            "where": f"created_date >= '{start}' AND created_date < '{end}' AND (agency = 'HPD' OR complaint_type in ({types}))"},
    }
    for name, spec in specs.items():
        spec["id"] = d["sources"][name]
    return specs


session = requests.Session()
if os.getenv("SOCRATA_APP_TOKEN"):
    session.headers["X-App-Token"] = os.environ["SOCRATA_APP_TOKEN"]


def get(url: str, params: dict, tries: int = 6) -> requests.Response:
    for attempt in range(tries):
        try:
            r = session.get(url, params=params, timeout=300)
            if r.status_code == 200:
                return r
            msg = f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.RequestException as e:  # network hiccup: back off and retry
            msg = str(e)
        wait = 5 * 2 ** attempt
        print(f"  retry in {wait}s ({msg})", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"failed after {tries} tries: {url} {params}")


def download(name: str, spec: dict, base: str, raw_dir) -> dict:
    out = raw_dir / name
    out.mkdir(parents=True, exist_ok=True)
    params = {"$select": "count(*) as n"}
    if spec["where"]:
        params["$where"] = spec["where"]
    expected = int(get(f"{base}/{spec['id']}.json", params).json()[0]["n"])
    pages = (expected + PAGE - 1) // PAGE
    print(f"[{name}] {expected:,} rows in {pages} pages", flush=True)
    for p in range(pages):
        part = out / f"part-{p:05d}.csv.gz"
        if part.exists() and part.stat().st_size > 0:
            continue
        q = {"$select": spec["select"], "$order": ":id", "$limit": PAGE, "$offset": p * PAGE}
        if spec["where"]:
            q["$where"] = spec["where"]
        text = get(f"{base}/{spec['id']}.csv", q).text
        tmp = part.with_suffix(".tmp")
        with gzip.open(tmp, "wt") as f:
            f.write(text)
        tmp.rename(part)  # atomic: a page is either complete or absent
    info = {"dataset_id": spec["id"], "where": spec["where"], "select": spec["select"],
            "expected_rows": expected, "pages": pages,
            "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": f"https://data.cityofnewyork.us/d/{spec['id']}"}
    (out / "_info.json").write_text(json.dumps(info, indent=2))
    print(f"[{name}] done", flush=True)
    return info


def main(names: list[str]) -> None:
    cfg = load_config()
    specs = dataset_specs(cfg)
    raw_dir = path(cfg["data"]["raw_dir"])
    names = names or list(specs)
    unknown = set(names) - set(specs)
    if unknown:
        raise SystemExit(f"unknown dataset(s): {', '.join(sorted(unknown))}")
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda n: download(n, specs[n], cfg["data"]["socrata_base"], raw_dir), names))
    print("all downloads finished", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
