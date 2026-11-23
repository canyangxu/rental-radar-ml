"""Step 4 (local): upload the cleaned, processed dataset to Google Cloud Storage.

Uploads every file in data.gcs.files from data/processed/ to gs://<bucket>/<prefix>/ and writes the
SHA-256 of each file into manifest.json first, so the training pipeline can verify what it downloads.

Authentication uses Application Default Credentials:
    gcloud auth application-default login      # once, on your machine
Bucket name: data.gcs.bucket in config.yaml, or the GCS_BUCKET environment variable.

Usage: python -m data_prep.upload
"""
from __future__ import annotations

import hashlib
import json

from google.cloud import storage

from src.config import load_config, path


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    cfg = load_config()
    g = cfg["data"]["gcs"]
    src = path(cfg["data"]["processed_dir"])
    manifest_path = src / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    data_files = [f for f in g["files"] if f != "manifest.json"]
    missing = [f for f in data_files if not (src / f).exists()]
    if missing:
        raise SystemExit(f"run python -m data_prep.make_dataset first; missing {missing}")
    manifest["sha256"] = {f: sha256(src / f) for f in data_files}
    manifest_path.write_text(json.dumps(manifest, indent=2))

    bucket = storage.Client().bucket(g["bucket"])
    for f in data_files + ["manifest.json"]:  # manifest last: it only appears once the data is in place
        blob = bucket.blob(f"{g['prefix']}/{f}")
        blob.upload_from_filename(str(src / f))
        print(f"uploaded gs://{g['bucket']}/{g['prefix']}/{f} ({(src / f).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
