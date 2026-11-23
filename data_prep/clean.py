"""Step 2 (local): clean the raw CSV snapshot into tidy Parquet tables keyed by HPD building id.

Tables written to data/clean/:
  buildings.parquet   one row per active HPD building with 3+ units: address, units, stories, year built,
                      coordinates, whether a management company is registered
  violations.parquet  HPD violations: building id, class (A/B/C), inspection date, category, rent impairing
  complaints.parquet  311 complaints matched to a building by normalized address, with a category
  bedbugs.parquet     latest bedbug filing per building and reporting period
  names.parquet       official building names (HPD affordable housing data + a small curated list), for search

Cleaning rules (all in SQL below):
  * text dates are parsed with TRY_CAST; rows whose date cannot be parsed are dropped from the event tables
  * street names are normalized ("W 57TH ST" -> "WEST 57 STREET") so 311 addresses can be joined to HPD buildings
  * only active buildings with at least data.min_units legal units are kept
  * duplicate bedbug filings for the same period keep the latest filing
  * 311 complaints that do not match a building keep a NULL building id and are excluded from features

Usage: python -m data_prep.clean
"""
from __future__ import annotations

import json

import duckdb

from src.config import load_config, path

MACROS = r"""
CREATE OR REPLACE MACRO norm_street(s) AS
  trim(regexp_replace(regexp_replace(regexp_replace(regexp_replace(regexp_replace(regexp_replace(
    upper(coalesce(s, '')),
    '(\d+)(ST|ND|RD|TH)\b', '\1', 'g'),
    '\bAVE\b', 'AVENUE', 'g'),
    '\bST\b', 'STREET', 'g'),
    '\b(W)\b', 'WEST', 'g'),
    '\b(E)\b', 'EAST', 'g'),
    '\s+', ' ', 'g'));
CREATE OR REPLACE MACRO clean_name(s) AS
  trim(regexp_replace(regexp_replace(regexp_replace(regexp_replace(
    trim(s),
    '\s*\(\d+\)\s*$', ''),
    '(\.[A-Za-z0-9]+)+$', ''),
    '\s+(HDFC|H\.D\.F\.C\.|LLC|L\.P\.|LP|INC\.?|CORP\.?|HOUSING DEVELOPMENT FUND CORP(ORATION)?)(\s+(INC\.?|LLC))?\b', '', 'gi'),
    '\s+', ' ', 'g'));
CREATE OR REPLACE MACRO boro_code(b) AS
  CASE upper(b) WHEN 'MANHATTAN' THEN '1' WHEN 'BRONX' THEN '2' WHEN 'BROOKLYN' THEN '3'
                WHEN 'QUEENS' THEN '4' WHEN 'STATEN ISLAND' THEN '5' ELSE NULL END;
CREATE OR REPLACE MACRO cat_hpd(t) AS
  CASE WHEN regexp_matches(upper(coalesce(t, '')), 'MICE|ROACH|INFESTATION|BEDBUG|BED BUG|RATS|RODENT|PEST') THEN 'pests'
       WHEN regexp_matches(upper(coalesce(t, '')), 'HEAT|HOT WATER|RADIATOR|BOILER|HEATER') THEN 'heat'
       WHEN regexp_matches(upper(coalesce(t, '')), 'MOLD|LEAK|WATER') THEN 'mold'
       ELSE 'upkeep' END;
CREATE OR REPLACE MACRO cat_311(ct, d) AS
  CASE WHEN ct = 'Noise - Residential' THEN 'noise'
       WHEN ct LIKE 'Noise%' THEN 'street'
       WHEN regexp_matches(upper(ct), 'HEAT|BOILER') THEN 'heat'
       WHEN regexp_matches(upper(coalesce(d, '')), 'PEST|MICE|ROACH|RAT|BED BUG') OR ct = 'Rodent' THEN 'pests'
       WHEN ct IN ('PLUMBING', 'WATER LEAK', 'PAINT/PLASTER', 'Indoor Air Quality', 'Indoor Sewage')
            OR regexp_matches(upper(coalesce(d, '')), 'MOLD') THEN 'mold'
       WHEN ct IN ('Dirty Condition', 'Residential Disposal Complaint') THEN 'sanitation'
       WHEN ct IN ('DOOR/WINDOW', 'FLOORING/STAIRS', 'GENERAL', 'ELECTRIC', 'Elevator', 'ELEVATOR', 'APPLIANCE',
                   'UNSANITARY CONDITION', 'SAFETY', 'Lead', 'OUTSIDE BUILDING', 'NONCONST') THEN 'upkeep'
       ELSE 'other' END;
"""


def main() -> dict:
    cfg = load_config()
    raw_dir, out = path(cfg["data"]["raw_dir"]), path(cfg["data"]["clean_dir"])
    out.mkdir(parents=True, exist_ok=True)
    min_units = int(cfg["data"]["min_units"])
    curated = path("data_prep/names_curated.csv")

    def raw(name: str) -> str:
        return f"read_csv('{raw_dir / name}/part-*.csv.gz', header=true, all_varchar=true, union_by_name=true)"

    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    con.execute(MACROS)

    print("buildings ...", flush=True)
    con.execute(f"""
    CREATE TABLE pluto AS
    SELECT CAST(CAST(TRY_CAST(bbl AS DOUBLE) AS BIGINT) AS VARCHAR) AS bbl,
           TRY_CAST(numbldgs AS INT) AS bldgs_lot, TRY_CAST(yearbuilt AS INT) AS year_built,
           TRY_CAST(latitude AS DOUBLE) AS lat, TRY_CAST(longitude AS DOUBLE) AS lng
    FROM {raw('pluto')}""")
    # Coordinates per building from the records that carry them (more precise than the tax lot centroid)
    con.execute(f"""
    CREATE TABLE bcoords AS
    SELECT bid, median(lat) AS lat, median(lng) AS lng FROM (
      SELECT buildingid AS bid, TRY_CAST(latitude AS DOUBLE) lat, TRY_CAST(longitude AS DOUBLE) lng FROM {raw('violations')}
      UNION ALL
      SELECT building_id, TRY_CAST(latitude AS DOUBLE), TRY_CAST(longitude AS DOUBLE) FROM {raw('bedbugs')}
    ) WHERE lat IS NOT NULL AND lat <> 0 GROUP BY bid""")
    con.execute(f"""
    CREATE TABLE mgmt AS
    WITH reg AS (
      SELECT buildingid, registrationid,
             row_number() OVER (PARTITION BY buildingid ORDER BY TRY_CAST(lastregistrationdate AS TIMESTAMP) DESC NULLS LAST) rn
      FROM {raw('registrations')}),
    c AS (SELECT registrationid, type, nullif(trim(corporationname), '') corp FROM {raw('contacts')})
    SELECT reg.buildingid, max(CASE WHEN c.type = 'Agent' THEN c.corp END) AS management
    FROM reg JOIN c USING (registrationid) WHERE reg.rn = 1 GROUP BY reg.buildingid""")
    con.execute(f"""
    CREATE TABLE buildings AS
    SELECT CAST(b.buildingid AS BIGINT) AS bid,
           upper(b.boro) AS borough, b.housenumber AS house, norm_street(b.streetname) AS street, b.zip,
           TRY_CAST(b.legalclassa AS INT) + coalesce(TRY_CAST(b.legalclassb AS INT), 0) AS units,
           TRY_CAST(b.legalstories AS INT) AS stories, b.managementprogram AS program,
           CASE WHEN p.year_built > 1700 THEN p.year_built END AS year_built, p.bldgs_lot,
           coalesce(c.lat, p.lat) AS lat, coalesce(c.lng, p.lng) AS lng,
           m.management IS NOT NULL AS has_mgmt
    FROM {raw('buildings')} b
    LEFT JOIN pluto p ON p.bbl = boro_code(b.boro) || lpad(b.block, 5, '0') || lpad(b.lot, 4, '0')
    LEFT JOIN bcoords c ON c.bid = b.buildingid
    LEFT JOIN mgmt m ON m.buildingid = b.buildingid
    WHERE b.recordstatus = 'Active' AND TRY_CAST(b.buildingid AS BIGINT) IS NOT NULL
      AND coalesce(TRY_CAST(b.legalclassa AS INT), 0) + coalesce(TRY_CAST(b.legalclassb AS INT), 0) >= {min_units}""")

    print("violations ...", flush=True)
    con.execute(f"""
    CREATE TABLE violations AS
    SELECT DISTINCT ON (violationid)
           CAST(buildingid AS BIGINT) AS bid, upper(class) AS class,
           CAST(TRY_CAST(inspectiondate AS TIMESTAMP) AS DATE) AS d,
           cat_hpd(novdescription) AS cat, rentimpairing = 'Y' AS rent_impairing
    FROM {raw('violations')}
    WHERE TRY_CAST(buildingid AS BIGINT) IN (SELECT bid FROM buildings)
      AND TRY_CAST(inspectiondate AS TIMESTAMP) IS NOT NULL AND upper(class) IN ('A', 'B', 'C')""")

    print("bedbugs ...", flush=True)
    con.execute(f"""
    CREATE TABLE bedbugs AS
    SELECT bid, infested, reinfested, filed FROM (
      SELECT CAST(building_id AS BIGINT) AS bid,
             coalesce(TRY_CAST(infested_dwelling_unit_count AS INT), 0) infested,
             coalesce(TRY_CAST(re_infested_dwelling_unit AS INT), 0) reinfested,
             CAST(TRY_CAST(filing_date AS TIMESTAMP) AS DATE) filed,
             row_number() OVER (PARTITION BY building_id, filling_period_end_date ORDER BY TRY_CAST(filing_date AS TIMESTAMP) DESC) rn
      FROM {raw('bedbugs')} WHERE TRY_CAST(building_id AS BIGINT) IS NOT NULL)
    WHERE rn = 1 AND filed IS NOT NULL AND bid IN (SELECT bid FROM buildings)""")

    print("complaints ...", flush=True)
    con.execute(f"""
    CREATE TABLE complaints AS
    WITH c AS (
      SELECT DISTINCT ON (unique_key)
             CAST(TRY_CAST(created_date AS TIMESTAMP) AS DATE) AS d, upper(borough) AS borough,
             cat_311(complaint_type, descriptor) AS cat,
             regexp_extract(upper(trim(incident_address)), '^([0-9]+(-[0-9]+)?[A-Z]?)\\s', 1) AS house,
             norm_street(regexp_replace(upper(trim(incident_address)), '^[0-9]+(-[0-9]+)?[A-Z]?\\s+', '')) AS street
      FROM {raw('complaints')} WHERE TRY_CAST(created_date AS TIMESTAMP) IS NOT NULL)
    SELECT a.bid, c.d, c.cat
    FROM c LEFT JOIN (SELECT house, street, borough, min(bid) bid FROM buildings GROUP BY ALL) a
      ON a.house = c.house AND a.street = c.street AND a.borough = c.borough""")

    print("names ...", flush=True)
    con.execute(f"""
    CREATE TABLE names AS
    SELECT DISTINCT clean_name(project_name) AS name, CAST(building_id AS BIGINT) AS bid
    FROM {raw('names_hny')}
    WHERE project_name IS NOT NULL AND TRY_CAST(building_id AS BIGINT) IN (SELECT bid FROM buildings)
      AND NOT regexp_matches(upper(project_name), '^(CONFIDENTIAL|[0-9-]+ .*)$')""")
    if curated.exists():
        con.execute(f"""
        INSERT INTO names
        SELECT n.name, b.bid FROM read_csv('{curated}', header=true, all_varchar=true) n
        JOIN buildings b ON b.house = n.house AND b.street = norm_street(n.street) AND b.borough = upper(n.borough)""")

    stats = {}
    for t in ["buildings", "violations", "complaints", "bedbugs", "names"]:
        con.execute(f"COPY {t} TO '{out / (t + '.parquet')}' (FORMAT parquet)")
        stats[t] = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
    stats["complaints_matched_to_building"] = con.execute("SELECT count(*) FROM complaints WHERE bid IS NOT NULL").fetchone()[0]
    (out / "clean_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    main()
