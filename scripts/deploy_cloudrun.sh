#!/usr/bin/env bash
# Build the image with Cloud Build and deploy the API to Cloud Run, public (no auth), then curl it.
# The trained model bundle in artifacts/model/ is part of the upload (.gcloudignore keeps it).
# Needs: gcloud CLI and `gcloud auth login`. Usage: GCP_PROJECT=my-project scripts/deploy_cloudrun.sh
set -euo pipefail
: "${GCP_PROJECT:?set GCP_PROJECT}"
REGION="${REGION:-us-east1}"
SERVICE="${SERVICE:-rental-radar-ml-api}"
[ -f artifacts/model/model.joblib ] || { echo "train first: python -m src.train"; exit 1; }
gcloud config set project "$GCP_PROJECT"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud run deploy "$SERVICE" --source . --region "$REGION" \
  --allow-unauthenticated --port 8080 --memory 2Gi --cpu 1 --min-instances 0 --max-instances 3 --quiet
URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)')
echo "deployed: $URL"
scripts/smoke_test.sh "$URL"
