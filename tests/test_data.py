"""Data step tests: no time leakage in the snapshot SQL, and GCS download integrity."""
import json
import shutil
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from data_prep.make_dataset import snapshot_sql
from src import ingest
from src.config import load_config


@pytest.fixture
def con():
    """Two buildings with events placed just before, on and after the cutoff 2025-05-01."""
    c = duckdb.connect()
    c.execute("""CREATE TABLE buildings AS SELECT * FROM (VALUES
        (1, 'BROOKLYN', '11201', 10, 4, 1950, 1, 40.7, -73.9, true),
        (2, 'QUEENS',   '11101', 20, 6, 1990, 1, 40.7, -73.8, false))
        t(bid, borough, zip, units, stories, year_built, bldgs_lot, lat, lng, has_mgmt)""")
    c.execute("""CREATE TABLE violations AS SELECT * FROM (VALUES
        (1, 'B', DATE '2025-04-30', 'mold', false),   -- history: the day before the cutoff
        (1, 'C', DATE '2025-05-01', 'heat', true),    -- label: on the cutoff day
        (1, 'B', DATE '2025-10-31', 'mold', false),   -- label: last day of the 6-month horizon
        (1, 'C', DATE '2025-11-01', 'heat', false),   -- outside both windows
        (2, 'A', DATE '2025-06-01', 'upkeep', false)) -- class A: not part of the label
        t(bid, class, d, cat, rent_impairing)""")
    c.execute("""CREATE TABLE complaints AS SELECT * FROM (VALUES
        (1, DATE '2025-04-30', 'noise'), (1, DATE '2025-05-01', 'noise'), (2, DATE '2024-04-30', 'mold'),
        (NULL, DATE '2025-04-01', 'noise')) t(bid, d, cat)""")
    c.execute("CREATE TABLE bedbugs AS SELECT * FROM (VALUES (1, 2, 0, DATE '2024-01-01')) t(bid, infested, reinfested, filed)")
    return c


def rows(con, with_label=True):
    df = con.execute(snapshot_sql(load_config(), "2025-05-01", with_label)).df().set_index("bid")
    return df.fillna(0)


def test_features_use_only_events_before_cutoff(con):
    r = rows(con)
    assert r.loc[1, "c12_all"] == 1 and r.loc[1, "c12_noise"] == 1  # the complaint on the cutoff day is excluded
    assert r.loc[1, "v12_b"] == 1 and r.loc[1, "v12_c"] == 0         # the C on the cutoff day is excluded
    assert r.loc[2, "c12_all"] == 0                                    # 2024-04-30 is older than 12 months


def test_label_uses_only_horizon_after_cutoff(con):
    r = rows(con)
    assert r.loc[1, "y_count"] == 2 and r.loc[1, "y"] == 1  # 2025-05-01 and 2025-10-31, not 2025-11-01
    assert r.loc[2, "y"] == 0                                # class A does not count


def test_live_rows_have_no_label(con):
    assert "y" not in rows(con, with_label=False).columns


def test_bedbug_window(con):
    assert rows(con).loc[1, "bb24_infested"] == 2  # 2024-01-01 is within 24 months of 2025-05-01


class FakeBlob:
    def __init__(self, root: Path, name: str):
        self.p = root / name

    def download_to_filename(self, dest):
        shutil.copy(self.p, dest)


class FakeClient:
    """Stands in for google.cloud.storage.Client: the bucket is a local folder."""
    def __init__(self, root: Path):
        self.root = root

    def bucket(self, name):
        root = self.root
        return type("B", (), {"blob": lambda self, n: FakeBlob(root, n)})()


def fake_bucket(tmp_path, corrupt=False):
    cfg = load_config()
    remote = tmp_path / "remote" / cfg["data"]["gcs"]["prefix"]
    remote.mkdir(parents=True)
    pd.DataFrame({"a": [1, 2]}).to_parquet(remote / "snapshots.parquet")
    hashes = {"snapshots.parquet": ingest.sha256(remote / "snapshots.parquet")}
    if corrupt:
        hashes["snapshots.parquet"] = "0" * 64
    (remote / "manifest.json").write_text(json.dumps({"sha256": hashes}))
    cfg["data"]["gcs"]["files"] = ["snapshots.parquet", "manifest.json"]
    cfg["data"]["cache_dir"] = str(tmp_path / "cache")
    return cfg, FakeClient(tmp_path / "remote")


def test_ingest_downloads_and_verifies(tmp_path):
    cfg, client = fake_bucket(tmp_path)
    out = ingest.download_dataset(cfg, client)
    assert pd.read_parquet(out / "snapshots.parquet").a.tolist() == [1, 2]


def test_ingest_rejects_corrupt_download(tmp_path):
    cfg, client = fake_bucket(tmp_path, corrupt=True)
    with pytest.raises(RuntimeError, match="SHA-256"):
        ingest.download_dataset(cfg, client)
