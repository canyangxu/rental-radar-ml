"""Step 3 (local): turn the clean event tables into the modeling dataset, then check it.

For every cutoff date T in config.yaml (train, val, test) each building gets one row:
  * history counts that use only events strictly before T (311 complaints by category, HPD violations by class,
    bedbug filings), over the look-back windows in features.lookback_months
  * the label: 1 if the building receives at least one violation of target.violation_classes in [T, T + horizon)

The live cutoff gets the same history counts but no label: these are the rows the API scores today.

Outputs in data/processed/ (these are the files uploaded to GCS):
  snapshots.parquet  train/val/test rows: bid, cutoff, raw counts, building attributes, y_count, y
  live.parquet       live rows: bid, cutoff, raw counts, building attributes
  buildings.parquet  search index: bid, name, address, borough, zip, units, lat, lng
  manifest.json      row counts, positive rates, date windows and data checks

Usage: python -m data_prep.make_dataset
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone

import duckdb
from dateutil.relativedelta import relativedelta

from src.config import load_config, path

ATTRS = ["borough", "zip", "units", "stories", "year_built", "bldgs_lot", "lat", "lng", "has_mgmt"]


def snapshot_sql(cfg: dict, cutoff: str, with_label: bool) -> str:
    f, t = cfg["features"], cfg["target"]
    looks = f["lookback_months"]
    comp = []
    for m in looks:
        comp.append(f"count(*) FILTER (WHERE d >= T - INTERVAL {m} MONTH) AS c{m}_all")
        comp += [f"count(*) FILTER (WHERE cat = '{k}' AND d >= T - INTERVAL {m} MONTH) AS c{m}_{k}"
                 for k in f["complaint_categories"]]
    vio = []
    for m in looks:
        vio += [f"count(*) FILTER (WHERE class = '{c}' AND d >= T - INTERVAL {m} MONTH) AS v{m}_{c.lower()}" for c in "ABC"]
        vio.append(f"count(*) FILTER (WHERE rent_impairing AND d >= T - INTERVAL {m} MONTH) AS v{m}_ri")
    for k in ["mold", "pests", "heat"]:
        vio.append(f"count(*) FILTER (WHERE cat = '{k}' AND d >= T - INTERVAL {max(looks)} MONTH) AS v{max(looks)}_{k}")
    bb = f["bedbug_lookback_months"]
    classes = ",".join(f"'{c}'" for c in t["violation_classes"])
    label = f"""
    , lab AS (SELECT bid, count(*) AS y_count FROM violations, p
              WHERE class IN ({classes}) AND d >= T AND d < T + INTERVAL {t['horizon_months']} MONTH GROUP BY bid)""" if with_label else ""
    label_cols = ", coalesce(lab.y_count, 0) AS y_count, (coalesce(lab.y_count, 0) > 0)::INT AS y" if with_label else ""
    label_join = "LEFT JOIN lab USING (bid)" if with_label else ""
    return f"""
    WITH p AS (SELECT DATE '{cutoff}' AS T),
    comp AS (SELECT bid, {', '.join(comp)} FROM complaints, p
             WHERE bid IS NOT NULL AND d < T AND d >= T - INTERVAL {max(looks)} MONTH GROUP BY bid),
    vio AS (SELECT bid, {', '.join(vio)} FROM violations, p
            WHERE d < T AND d >= T - INTERVAL {max(looks)} MONTH GROUP BY bid),
    bb AS (SELECT bid, max(infested) AS bb{bb}_infested, max(reinfested) AS bb{bb}_reinfested FROM bedbugs, p
           WHERE filed < T AND filed >= T - INTERVAL {bb} MONTH GROUP BY bid)
    {label}
    SELECT b.bid, DATE '{cutoff}' AS cutoff, {', '.join('b.' + a for a in ATTRS)},
           comp.* EXCLUDE (bid), vio.* EXCLUDE (bid), bb.* EXCLUDE (bid) {label_cols}
    FROM buildings b LEFT JOIN comp USING (bid) LEFT JOIN vio USING (bid) LEFT JOIN bb USING (bid) {label_join}
    """


def check(con: duckdb.DuckDBPyConnection, cfg: dict) -> dict:
    """Data checks. Any failure stops the script, so a broken dataset never reaches GCS."""
    s, d = cfg["split"], cfg["data"]
    end = date.fromisoformat(d["end_date"])
    for name in ("train", "val", "test"):
        label_end = date.fromisoformat(s[f"{name}_cutoff"]) + relativedelta(months=cfg["target"]["horizon_months"])
        assert label_end <= end, f"{name} label window ends {label_end}, after the data ends {end}"
        first = date.fromisoformat(s[f"{name}_cutoff"]) - relativedelta(months=max(cfg["features"]["lookback_months"]))
        assert first >= date.fromisoformat(d["start_date"]), f"{name} look-back starts {first}, before the data starts"
    n_b = con.execute("SELECT count(*) FROM buildings").fetchone()[0]
    per_cut = con.execute("SELECT cutoff, count(*), count(DISTINCT bid), avg(y) FROM snaps GROUP BY 1 ORDER BY 1").fetchall()
    for cutoff, n, n_distinct, _ in per_cut:
        assert n == n_distinct == n_b, f"{cutoff}: {n} rows for {n_b} buildings (duplicates or missing rows)"
    nulls = con.execute("SELECT count(*) FROM snaps WHERE y IS NULL OR y_count IS NULL").fetchone()[0]
    assert nulls == 0, "labels contain NULLs"
    neg = con.execute("SELECT count(*) FROM snaps WHERE c12_all < 0 OR v12_b < 0").fetchone()[0]
    assert neg == 0, "negative counts"
    return {
        "buildings": n_b,
        "rows_per_cutoff": {str(c): n for c, n, _, _ in per_cut},
        "positive_rate_per_cutoff": {str(c): round(r, 4) for c, _, _, r in per_cut},
        "checks": ["label windows end before the data does", "look-back windows start after the data does",
                   "one row per building per cutoff", "no NULL labels", "no negative counts"],
    }


def main() -> dict:
    cfg = load_config()
    clean, out = path(cfg["data"]["clean_dir"]), path(cfg["data"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    for t in ["buildings", "violations", "complaints", "bedbugs", "names"]:
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM '{clean / (t + '.parquet')}'")

    s = cfg["split"]
    labeled = [s["train_cutoff"], s["val_cutoff"], s["test_cutoff"]]
    con.execute("CREATE TABLE snaps AS " + " UNION ALL BY NAME ".join(f"({snapshot_sql(cfg, c, True)})" for c in labeled))
    con.execute(f"CREATE TABLE live AS {snapshot_sql(cfg, s['live_cutoff'], False)}")
    # Counts are NULL where a building had no events in the window; that means zero events, not missing data.
    count_cols = [c for c, in con.execute(
        "SELECT column_name FROM (DESCRIBE snaps) WHERE regexp_matches(column_name, '^(c|v|bb)[0-9]+_')").fetchall()]
    for t in ["snaps", "live"]:
        con.execute(f"UPDATE {t} SET " + ", ".join(f"{c} = coalesce({c}, 0)" for c in count_cols))

    con.execute("""
    CREATE TABLE search AS
    SELECT b.bid, n.name, b.house || ' ' || b.street AS address, b.borough, b.zip, b.units, b.stories,
           b.year_built, b.lat, b.lng
    FROM buildings b LEFT JOIN (SELECT bid, string_agg(DISTINCT name, ' / ') AS name FROM names GROUP BY bid) n USING (bid)""")

    report = check(con, cfg)
    con.execute(f"COPY (SELECT * FROM snaps ORDER BY cutoff, bid) TO '{out / 'snapshots.parquet'}' (FORMAT parquet)")
    con.execute(f"COPY (SELECT * FROM live ORDER BY bid) TO '{out / 'live.parquet'}' (FORMAT parquet)")
    con.execute(f"COPY (SELECT * FROM search ORDER BY bid) TO '{out / 'buildings.parquet'}' (FORMAT parquet)")
    report.update({
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_window": [cfg["data"]["start_date"], cfg["data"]["end_date"]],
        "cutoffs": {k: s[k] for k in ("train_cutoff", "val_cutoff", "test_cutoff", "live_cutoff")},
        "label": f"at least one class {'/'.join(cfg['target']['violation_classes'])} HPD violation "
                 f"in the {cfg['target']['horizon_months']} months after the cutoff",
        "count_columns": count_cols,
        "sources": {k: f"https://data.cityofnewyork.us/d/{v}" for k, v in cfg["data"]["sources"].items()},
    })
    (out / "manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("buildings", "rows_per_cutoff", "positive_rate_per_cutoff")}, indent=2))
    return report


if __name__ == "__main__":
    main()
