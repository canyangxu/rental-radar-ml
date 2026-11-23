#!/usr/bin/env bash
# Publish frontend/ as a public static Hugging Face Space that calls the deployed API.
# Needs: the hf CLI and `hf auth login`.
# Usage: API_URL=https://rental-radar-ml-api-xxxx.a.run.app HF_SPACE=user/rental-radar-forecast scripts/deploy_frontend_hf.sh
set -euo pipefail
: "${API_URL:?set API_URL to the Cloud Run URL}"
: "${HF_SPACE:?set HF_SPACE, e.g. your-hf-user/rental-radar-forecast}"
curl -sf "$API_URL/health" >/dev/null || { echo "API at $API_URL is not answering /health"; exit 1; }
tmp=$(mktemp -d)
cp -R frontend/. "$tmp/"
sed -i.bak "s#API_URL: \"[^\"]*\"#API_URL: \"$API_URL\"#" "$tmp/config.js" && rm "$tmp/config.js.bak"
cat > "$tmp/README.md" <<MD
---
title: Rental Radar Forecast
emoji: 🔮
colorFrom: yellow
colorTo: gray
sdk: static
app_file: index.html
pinned: false
short_description: Will a NYC building get a serious violation in 6 months?
---
Front end for [rental-radar-ml](https://github.com/canyangxu/rental-radar-ml). It calls the model API at $API_URL.
MD
hf repos create "$HF_SPACE" --repo-type space --space-sdk static --public --exist-ok
hf upload "$HF_SPACE" "$tmp" . --repo-type space --commit-message "Deploy front end (API: $API_URL)"
echo "front end: https://huggingface.co/spaces/$HF_SPACE"
