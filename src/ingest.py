"""Pipeline step 1: download the processed dataset from Google Cloud Storage.

Files listed in data.gcs.files are downloaded from gs://<bucket>/<prefix>/ into data.cache_dir. Each data file
is checked against the SHA-256 recorded in manifest.json at upload time; a mismatch stops the pipeline.
Files already in the cache with the right hash are not downloaded again.

Usage: python -m src.ingest
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from src.config import load_config, path


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _client():
    from google.cloud import storage  # imported here so the API container does not need GCS credentials

    # GOOGLE_CLOUD_PROJECT is needed when the credentials do not carry a project (e.g. inside Docker)
    return storage.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT"))


def download_dataset(cfg: dict | None = None, client=None) -> Path:
    """Download (or reuse) the dataset and return the folder that holds it."""
    cfg = cfg or load_config()
    g = cfg["data"]["gcs"]
    cache = path(cfg["data"]["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    bucket = (client or _client()).bucket(g["bucket"])

    def fetch(name: str) -> Path:
        dest = cache / name
        bucket.blob(f"{g['prefix']}/{name}").download_to_filename(str(dest))
        print(f"downloaded gs://{g['bucket']}/{g['prefix']}/{name}")
        return dest

    manifest = json.loads(fetch("manifest.json").read_text())
    hashes = manifest.get("sha256", {})
    for name in g["files"]:
        if name == "manifest.json":
            continue
        dest = cache / name
        if dest.exists() and hashes.get(name) == sha256(dest):
            print(f"cached   {name} (hash matches)")
            continue
        fetch(name)
        if name in hashes and sha256(dest) != hashes[name]:
            dest.unlink()
            raise RuntimeError(f"{name}: SHA-256 does not match the manifest; the download is corrupt")
    return cache


def dataset_dir(cfg: dict) -> Path:
    """Where the training step reads data from.

    The default is the GCS download. DATA_SOURCE=local reads data/processed/ directly; it exists only so the
    unit tests and a first offline run work without cloud credentials.
    """
    if os.getenv("DATA_SOURCE", "gcs") == "local":
        return path(cfg["data"]["processed_dir"])
    return download_dataset(cfg)


if __name__ == "__main__":
    print(f"dataset ready in {download_dataset()}")
