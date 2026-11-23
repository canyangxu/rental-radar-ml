#!/bin/sh
# Container entry point: serve (default), train or test.
set -e
case "${1:-serve}" in
  serve) exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8080}" --workers "${WORKERS:-1}" ;;
  train) shift; exec python -W ignore -m src.train "$@" ;;
  ingest) exec python -m src.ingest ;;
  test)  shift; exec python -m pytest -q -W ignore "$@" ;;
  *)     exec "$@" ;;
esac
