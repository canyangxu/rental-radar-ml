# Contributing

1. Branch from `main`: `feature/<short-name>` or `fix/<short-name>`.
2. Change parameters in `config.yaml`, not in code.
3. Run `pytest -q` before pushing; for API changes also run `scripts/smoke_test.sh`.
4. Open a pull request with what changed, why, and how you tested it. Results that change metrics should include the new `reports/metrics.json` and the MLflow run id.
5. Never commit secrets (`.env`, service account keys) or data (`data/`, `mlruns/`, `artifacts/` are ignored).
