#!/usr/bin/env bash
# One-time: create the GCS bucket and upload the processed dataset.
# Needs: gcloud CLI, `gcloud auth login` and `gcloud auth application-default login`.
# Usage: GCP_PROJECT=my-project GCS_BUCKET=my-unique-bucket scripts/setup_gcs.sh
set -euo pipefail
: "${GCP_PROJECT:?set GCP_PROJECT}"; : "${GCS_BUCKET:?set GCS_BUCKET (globally unique)}"
REGION="${REGION:-us-east1}"
gcloud config set project "$GCP_PROJECT"
gcloud services enable storage.googleapis.com
gcloud storage buckets describe "gs://$GCS_BUCKET" >/dev/null 2>&1 || \
  gcloud storage buckets create "gs://$GCS_BUCKET" --location="$REGION" --uniform-bucket-level-access
python -m data_prep.upload
gcloud storage ls -l "gs://$GCS_BUCKET/$(python -c 'from src.config import load_config;print(load_config()["data"]["gcs"]["prefix"])')/"
