#!/usr/bin/env bash
# Calls every API endpoint with curl and fails on any non-200 answer.
# Usage: scripts/smoke_test.sh [BASE_URL]      (default http://localhost:8080)
set -euo pipefail
BASE="${1:-http://localhost:8080}"
call() { echo; echo "\$ curl $*"; curl -sS --fail-with-body "$@"; echo; }

call "$BASE/health"
call "$BASE/model" -o /dev/null -w "HTTP %{http_code}, /model returned %{size_download} bytes\n"
call "$BASE/buildings/search?q=the%20max&limit=3"
call -X POST "$BASE/predict" -H "Content-Type: application/json" \
  -d '{"borough":"MANHATTAN","units":837,"stories":42,"year_built":2015,"has_mgmt":true,"c12_all":34,"c12_noise":13,"c12_mold":6,"c12_pests":5,"c3_all":5,"v12_a":1,"v12_b":1,"bb24_infested":1}'
call -X POST "$BASE/predict/batch" -H "Content-Type: application/json" \
  -d '{"instances":[{"borough":"BRONX","units":40,"v12_b":6,"v12_c":3,"c12_all":25},{"borough":"QUEENS","units":12}]}'
echo; echo "\$ curl -X POST $BASE/predict -d '{\"units\": -3}'   (expect 422)"
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/predict" -H "Content-Type: application/json" -d '{"units": -3}')
echo "HTTP $code"; [ "$code" = "422" ]
echo; echo "all endpoints OK"
