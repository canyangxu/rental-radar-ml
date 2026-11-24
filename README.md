# Rental Radar Forecast

**Will this New York City apartment building get a serious housing violation in the next six months?**

An end-to-end ML system on public NYC Open Data. It cleans the raw data locally and uploads it to Google Cloud Storage. A training pipeline downloads it, builds features, splits the data by time, trains and evaluates two models, and tracks every run in MLflow. The selected model is served by a FastAPI app in Docker on Cloud Run. A web front end on Hugging Face Spaces calls that API.

Duke AIPI 510 · Module Project 3: Data/ML Engineering

| | Link |
|---|---|
| **Front end** (Hugging Face Spaces) | **https://huggingface.co/spaces/canyangxu077/rental-radar-forecast** |
| **API** (Cloud Run) | **https://rental-radar-ml-api-976733016944.us-east1.run.app** · interactive docs at [https://rental-radar-ml-api-976733016944.us-east1.run.app/docs](https://rental-radar-ml-api-976733016944.us-east1.run.app/docs) |
| **Predict endpoint** | `POST https://rental-radar-ml-api-976733016944.us-east1.run.app/predict` |
| **Dataset in the cloud** (GCS) | `gs://rental-radar-ml-7631fd34/v1/` (Google Cloud Storage, project `project-7631fd34-7bbc-4858-b2d`) |

![Front end: forecast for The Max](docs/img/frontend-forecast.png)

## Assignment checklist

Every requirement of the assignment, with where to find it. All links point at the live services or at files in this repository.

| Requirement | Where it is | Evidence |
|---|---|---|
| Public dataset, supervised task, target and metric | [§1](#1-overview-and-goals), [§2](#2-dataset) | 8 public datasets from NYC Open Data, the city's official portal (open under the NYC Open Data Law, downloadable without an account): official titles, publishers, dataset IDs, the API used, filters, rows downloaded and the download date are in §2, with a one-line `curl` anyone can run; binary target "class B/C HPD violation in the next 6 months"; primary metric PR-AUC |
| Cleaned locally, stored in the cloud | [`data_prep/`](data_prep), [§4](#4-cloud-services) | cleaned with DuckDB, uploaded to **Google Cloud Storage** `gs://rental-radar-ml-7631fd34/v1/` with SHA-256 hashes |
| Script that downloads the data in the pipeline | [`src/ingest.py`](src/ingest.py) | `python -m src.train` downloads from GCS and verifies every file before training |
| Simple model | [`src/train.py`](src/train.py), [§3](#3-model-and-evaluation) | two baselines, logistic regression, XGBoost (selected on validation PR-AUC) |
| Pipeline: ingest, preprocessing and features, split, train, evaluate | [`src/`](src) | `ingest.py` → `features.py` → `split.py` (time-based train/val/test) → `train.py` → `evaluate.py` |
| `config.yaml` | [`config.yaml`](config.yaml) | data sources, dates, features, target, split, hyperparameters, evaluation, MLflow, serving |
| Experiment tracking (params, metrics, artifacts) | [§3 MLflow](#experiment-tracking-mlflow) | MLflow parent run + 4 nested runs; params, metrics, datasets, figures, model; registered as `violation-risk-classifier` |
| Dockerfile with train and serve entry points, built and run locally | [`Dockerfile`](Dockerfile), [`scripts/entrypoint.sh`](scripts/entrypoint.sh), [§6.3](#63-build-and-run-the-docker-image-locally) | `docker run rental-radar-ml` (serve), `... train`, `... test`; local run output (health, smoke test, 46 tests passing in the container) in §6.3 |
| FastAPI `/predict` with JSON input, tested with curl | [`api/main.py`](api/main.py), [`scripts/smoke_test.sh`](scripts/smoke_test.sh), [§6.4](#64-test-the-api-locally-with-curl) | `POST /predict` returns probability, label and drivers; invalid input returns 422 |
| Deployed to the cloud, `/predict` public and verified over HTTP | [§6.5](#65-deploy) | **Cloud Run**: `POST https://rental-radar-ml-api-976733016944.us-east1.run.app/predict` (live response shown in §6.5) |
| Front end that calls the deployed API, interactive, hosted publicly | [`frontend/`](frontend), [§7](#7-front-end) | **https://huggingface.co/spaces/canyangxu077/rental-radar-forecast** — renter page (search → forecast card) and a project page with a form that sends your own input to `/predict` |
| README sections | this file | overview and goals, dataset, model and evaluation, cloud services, setup, usage (train, build, test the API, deploy), links |
| `requirements.txt` | [`requirements.txt`](requirements.txt) | pinned versions, Python 3.12 |
| Tests | [`tests/`](tests) | 46 tests: config, leakage at the cutoff, GCS integrity, features, training end to end, API |
| Branches and pull requests | [pull requests](https://github.com/canyangxu/rental-radar-ml/pulls?q=is%3Apr) | one branch and PR per part: skeleton → data → model → API → Docker/Cloud Run → front end → docs → fixes → deployment → search backup → two-page front end; both team members opened and merged PRs from their own accounts (#1–#10 Canyang Xu, #11–#12 Xianyu Wang) |
| Ethics and limitations | [§9](#9-ethics-and-limitations) | label bias from complaint-driven inspections, neighborhood signal, size bias, error costs, drift |

---

## Contents
1. [Overview and goals](#1-overview-and-goals)
2. [Dataset](#2-dataset)
3. [Model and evaluation](#3-model-and-evaluation)
4. [Cloud services](#4-cloud-services)
5. [Setup](#5-setup)
6. [Usage: train, build, test the API, deploy](#6-usage-train-build-test-the-api-deploy)
7. [Front end](#7-front-end)
8. [Repository structure](#8-repository-structure)
9. [Ethics and limitations](#9-ethics-and-limitations)

---

## 1. Overview and goals

Renters in NYC sign leases with little idea how a building is maintained. The city publishes every 311 complaint and every Housing Preservation and Development (HPD) violation, but raw records do not say what is likely to happen next. This project turns those records into one forecast a renter can read in seconds.

**Supervised task.** Binary classification with one row per building per cutoff date *T*:

| | Definition |
|---|---|
| **Target** `y` | 1 if the building receives **at least one class B (hazardous) or class C (immediately hazardous) HPD violation in the 6 months after *T***, otherwise 0 |
| **Features** | only events **before *T***: 311 complaints by topic, HPD violations by class, bedbug filings, and building attributes (44 features) |
| **Primary metric** | **PR-AUC** (average precision). Only 14% of buildings are positive, so accuracy is misleading: always answering "no" scores 86% accuracy. |
| **Also reported** | ROC-AUC; precision, recall, F1 and the full confusion matrix at a threshold chosen on validation; precision among the top 10% riskiest; calibration |

**Goals**
1. A reproducible pipeline from raw public data to a deployed model, with every parameter in `config.yaml`.
2. An honest evaluation: time-based split, a held-out test period, and comparison against simple baselines.
3. A forecast a renter can act on: the probability, the reasons behind it, and one question to ask at the viewing.

## 2. Dataset

All data is public. It comes from [NYC Open Data](https://opendata.cityofnewyork.us/), the City of New York's official open data portal. Under the city's Open Data Law (Local Law 11 of 2012, NYC Administrative Code Title 23, Chapter 5), agency data is "open by default" and free to use ([what NYC Open Data is](https://www.nyc.gov/opendata/get-started/what-is-open-data-)). No account, API key or payment is needed to download it.

**How it was downloaded.** [`data_prep/download.py`](data_prep/download.py) calls the portal's public Socrata API. For each dataset it first runs a `count(*)` query, then downloads pages of 50,000 rows as CSV from
`https://data.cityofnewyork.us/resource/<dataset id>.csv`. Each request passes `$select` (only the columns we use), `$where` (the date window below) and `$limit`/`$offset`. The dataset IDs and dates are in [`config.yaml`](config.yaml) (`data.sources`, `data.start_date`, `data.end_date`). The script saves the expected row count, the source URL and the download time in `data/raw/<name>/_info.json`, and `manifest.json` (uploaded to GCS with the data) lists every source URL. Anyone can check that the data is public with one anonymous request:

```bash
curl "https://data.cityofnewyork.us/resource/wvxf-dwi5.csv?\$limit=5"        # first 5 HPD violations, no login
curl "https://data.cityofnewyork.us/api/views/wvxf-dwi5.json"                 # the dataset's official metadata
```

**Snapshot.** Downloaded on **2026-11-14** between 18:01 and 18:42 UTC (about 514 MB of gzipped CSV). It covers events from 2024-01-01 through 2026-11-14 (`end_date` 2026-11-15 is exclusive). Bedbug filings start at 2021-01-01 because the 24-month bedbug feature needs a longer history. NYC Open Data updates these datasets daily to quarterly, so a download on another day will differ slightly. The cleaned, frozen copy used for training is the one in Google Cloud Storage ([§4](#4-cloud-services)).

| Dataset (official title, publisher) | ID | Filter | Rows downloaded | Used for |
|---|---|---|---|---|
| [Housing Maintenance Code Violations](https://data.cityofnewyork.us/d/wvxf-dwi5), HPD | `wvxf-dwi5` | inspected 2024-01-01 to 2026-11-14, or still open | 4,495,365 | label (B/C violations after *T*) and violation features before *T*; 3,687,956 rows after cleaning |
| [311 Service Requests from 2020 to Present](https://data.cityofnewyork.us/d/erm2-nwe9), Office of Technology and Innovation | `erm2-nwe9` | created 2024-01-01 to 2026-11-14; HPD complaints plus 12 noise, pest, sanitation and building-system types | 4,739,363 | complaint counts by topic: noise, heat, mold or leaks, pests, repairs, sanitation; 3,314,046 matched to a building |
| [Bedbug Reporting](https://data.cityofnewyork.us/d/wz6d-d3jb), HPD | `wz6d-d3jb` | filing period ending 2021-01-01 or later | 556,876 | infested and re-infested units in the past 24 months; 490,201 filings after cleaning |
| [Buildings Subject to HPD Jurisdiction](https://data.cityofnewyork.us/d/kj4p-ruqc), HPD | `kj4p-ruqc` | none | 381,934 | the building list (active, 3+ legal units): **162,997 buildings** |
| [Primary Land Use Tax Lot Output (PLUTO)](https://data.cityofnewyork.us/d/64uk-42ks), Department of City Planning | `64uk-42ks` | 3+ residential units | 172,485 | year built, buildings on the lot, coordinates |
| [Multiple Dwelling Registrations](https://data.cityofnewyork.us/d/tesw-yqqr) and [Registration Contacts](https://data.cityofnewyork.us/d/feu5-w2e2), HPD | `tesw-yqqr`, `feu5-w2e2` | none; contacts: company names only | 187,127 and 744,753 | whether a management company is registered |
| [Affordable Housing Production by Building](https://data.cityofnewyork.us/d/hg8x-zxpr), HPD | `hg8x-zxpr` | none | 9,551 | official building names, used for search only |

Raw and cleaned files are not committed to git (about 514 MB); `python -m data_prep.download` rebuilds them from the public source.

**Modeling table.** Each building gets a snapshot at three cutoffs, giving 488,991 labeled rows:

| Split | Cutoff *T* | Features from | Label window | Rows | Positive |
|---|---|---|---|---|---|
| train | 2025-05-01 | 2024-05-01 – 2025-04-30 | 2025-05-01 – 2025-10-31 | 162,997 | 13.3% |
| validation | 2025-11-01 | 2024-11-01 – 2025-10-31 | 2025-11-01 – 2026-04-30 | 162,997 | 14.3% |
| test | 2026-05-01 | 2025-05-01 – 2026-04-30 | 2026-05-01 – 2026-10-31 | 162,997 | 14.0% |
| live (served) | 2026-11-14 | 2025-11-14 – 2026-11-13 | the future | 162,997 | – |

**Cleaning** (`data_prep/clean.py`)
- Dates are parsed with `TRY_CAST`; rows with dates that cannot be parsed are dropped.
- Street names are normalized (`W 57TH ST` → `WEST 57 STREET`) so 311 addresses join to HPD buildings.
- Only active buildings with at least 3 legal units are kept.
- Violations are de-duplicated by ID, and repeated bedbug filings keep the latest one.
- Impossible values are treated as data-entry errors:
  - bedbug counts are capped at the building's unit count (one 4-unit building reported 414,754 infested units);
  - stories outside 1–200 and buildings-per-lot outside 1–500 become missing and are imputed later.

**Data checks** (`data_prep/make_dataset.py`) stop the build if any of these fail:
- a label window extends past the end of the data;
- a look-back window starts before the data does;
- a building has anything other than one row per cutoff;
- any label is NULL, or any count is negative.

`tests/test_data.py` places events the day before, on, and after a cutoff and checks that each lands on the right side.

## 3. Model and evaluation

**Preprocessing** (`src/features.py`, a scikit-learn transformer saved inside the model)
- Missing units, stories, year built, lot size and coordinates are filled with **training-set medians**. The medians are learned in `fit()` on the training rows only, so validation, test and API requests are filled with the same numbers.
- Derived features: building age, and borough as five 0/1 columns.
- Complaints and B/C violations per 10 units, so large buildings are not flagged for size alone.
- `log1p` on counts for the linear model.

**Models** (`src/train.py`, hyperparameters in `config.yaml`)

| Model | Setup |
|---|---|
| Baseline: always no | predicts 0 for every building |
| Baseline: B/C last year | predicts 1 if the building had any B/C violation in the past 12 months |
| Logistic regression | StandardScaler + L2 logistic regression (C = 1.0) |
| **XGBoost (selected)** | 600 trees max, depth 6, learning rate 0.05, subsample 0.8, early stopping on validation PR-AUC |

The decision threshold maximizes F1 on **validation**. The served model is the one with the best **validation** PR-AUC. The test set is used only to report results.

**Test results** (cutoff 2026-05-01, n = 162,997, 22,861 positives; from [`reports/metrics.json`](reports/metrics.json))

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Top-10% precision | Accuracy |
|---|---|---|---|---|---|---|---|
| Always no | 0.140 | 0.500 | 0.000 | 0.000 | 0.000 | 0.140 | 0.860 |
| B/C last year (rule) | 0.622 | 0.836 | 0.536 | 0.737 | 0.620 | 0.716 | 0.874 |
| Logistic regression | 0.728 | 0.913 | 0.640 | 0.693 | 0.665 | 0.763 | 0.902 |
| **XGBoost** | **0.745** | **0.922** | **0.644** | 0.708 | **0.674** | **0.774** | **0.904** |

Against the "last year" rule, XGBoost finds almost the same number of real cases: 16,178 against 16,850. It raises **5,673 fewer false alarms**: 8,935 against 14,608.

| Confusion matrix, XGBoost (threshold 0.32) | Precision-recall curves |
|---|---|
| ![](reports/figures/confusion_xgboost.png) | ![](reports/figures/pr_curves.png) |

**Calibration.** Within each predicted-risk band on the test set, the share of buildings that really got a violation matches the prediction:

| Predicted band | Buildings | Mean predicted | Observed rate |
|---|---|---|---|
| 0–10% | 118,230 | 2.9% | 2.5% |
| 20–30% | 6,316 | 24.7% | 24.6% |
| 50–60% | 3,445 | 54.9% | 53.4% |
| 90–100% | 4,610 | 95.8% | 95.5% |

The front end can therefore say: "Of the 6,316 buildings scored 20–30%, 25% really got a B/C violation."

**Explanations.** For XGBoost the API returns per-building SHAP contributions (`pred_contribs`) and lists the strongest drivers with their values. Coordinates and borough are reported together as "location". By total gain, the most important features are B/C violations per 10 units, class B violations in the past year, 311 complaints in the past year, and class C violations (see [`reports/figures/importance_xgboost.png`](reports/figures/importance_xgboost.png)).

### Experiment tracking (MLflow)

Every `python -m src.train` creates one parent run with four nested runs: two baselines and two models. The selected model is logged and registered from the parent run. MLflow uses a SQLite backend in `mlruns/mlflow.db`, with artifacts in `mlruns/`. It records:

| What | Where |
|---|---|
| Parameters | all `features`, `target`, `split` and `evaluation` settings, and each model's hyperparameters |
| Metrics | validation and test PR-AUC, ROC-AUC, precision, recall, F1, accuracy, confusion counts, best XGBoost iteration |
| Datasets | train, validation and test logged as MLflow inputs, with the GCS source URI and the SHA-256 of the data |
| Artifacts | confusion matrices, PR curves, feature importance, `metrics.json`, the serving bundle, and each model logged with a signature and input example |
| Model registry | the selected model registered as `violation-risk-classifier` |

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db   # http://localhost:5000
```
![MLflow parent run: four child runs (two baselines, two models), the three datasets, tags with the git SHA and data hash, and the registered model](docs/img/mlflow-run.png)

## 4. Cloud services

| Service | Role |
|---|---|
| **Google Cloud Storage** | Holds the cleaned, processed dataset under `gs://rental-radar-ml-7631fd34/v1/` (region us-east1): `snapshots.parquet`, `live.parquet`, `buildings.parquet` and `manifest.json` with SHA-256 hashes. The training pipeline downloads from here and verifies every file. |
| **Cloud Build + Artifact Registry** | Build the Docker image from this repo (`gcloud run deploy --source .`). |
| **Cloud Run** | Runs the API container at https://rental-radar-ml-api-976733016944.us-east1.run.app (service `rental-radar-ml-api`, us-east1): public (no auth), 2 GiB memory, scales to zero, at most 3 instances. |
| **Hugging Face Spaces** (static) | Hosts the front end, which calls the Cloud Run API from the browser (CORS enabled). |

```
 NYC Open Data ──download──▶ data/raw ──clean──▶ data/clean ──snapshots + checks──▶ data/processed
   (Socrata API)            data_prep/download.py  data_prep/clean.py  data_prep/make_dataset.py
                                                                                 │ data_prep/upload.py
                                                                                 ▼
                                                                  Google Cloud Storage  gs://rental-radar-ml-7631fd34/v1/
                                                                                 │ src/ingest.py (SHA-256 checked)
                                                                                 ▼
  config.yaml ──▶  src/features.py → src/split.py → src/train.py → src/evaluate.py ──▶ MLflow (runs, metrics, registry)
                                                                                 │ artifacts/model/
                                                                                 ▼
                                          Docker image ──Cloud Build──▶ Cloud Run: FastAPI /predict
                                                                                 ▲ HTTPS + JSON
                                                         Hugging Face Space: front end (search → forecast)
```

## 5. Setup

Requirements: Python 3.12 and Docker. Cloud steps also need the [gcloud CLI](https://cloud.google.com/sdk/docs/install) and a GCP project with billing enabled.

```bash
git clone https://github.com/canyangxu/rental-radar-ml.git && cd rental-radar-ml
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # pinned versions
pytest -q                                 # 46 tests; all but one run offline on synthetic data
```

Every parameter is in [`config.yaml`](config.yaml): data sources and dates, feature windows, target, split cutoffs, model hyperparameters, evaluation, MLflow and serving. Three environment variables override per-machine values:

| Variable | Overrides | Example |
|---|---|---|
| `GCS_BUCKET` | `data.gcs.bucket` | `rental-radar-ml-7631fd34` |
| `MLFLOW_TRACKING_URI` | `mlflow.tracking_uri` | `sqlite:///mlruns/mlflow.db` |
| `MODEL_DIR` | `serving.model_dir` | `artifacts/model` |

## 6. Usage: train, build, test the API, deploy

### 6.1 Prepare the data locally and upload it to GCS (once)
```bash
python -m data_prep.download        # NYC Open Data → data/raw/   (about 500 MB, resumable)
python -m data_prep.clean           # → data/clean/*.parquet
python -m data_prep.make_dataset    # → data/processed/ (snapshots, live, search index, manifest + checks)

gcloud auth login && gcloud auth application-default login
GCP_PROJECT=<project> GCS_BUCKET=<bucket> scripts/setup_gcs.sh   # creates the bucket, runs data_prep/upload.py
```

### 6.2 Train (downloads from GCS)
```bash
GCS_BUCKET=<bucket> python -m src.train
```
This step:
1. downloads and verifies the dataset from GCS;
2. trains both baselines and both models;
3. logs everything to MLflow;
4. writes the serving bundle to `artifacts/model/` (`model.joblib`, `metadata.json`, `live.parquet`, `buildings.parquet`);
5. writes `reports/metrics.json` and the figures.

`DATA_SOURCE=local` reads `data/processed/` instead of GCS, for offline runs only.

### 6.3 Build and run the Docker image locally
One image, two entry points: `serve` (the default) and `train`, plus `test`.
```bash
docker build -t rental-radar-ml .
docker run --rm -p 8080:8080 rental-radar-ml                 # serve: FastAPI on :8080
docker run --rm rental-radar-ml test                         # run the test suite in the container
docker run --rm -e GCS_BUCKET=<bucket> -e GOOGLE_CLOUD_PROJECT=<project> \
  -v ~/.config/gcloud/application_default_credentials.json:/creds.json:ro -e GOOGLE_APPLICATION_CREDENTIALS=/creds.json \
  -v "$PWD/artifacts:/app/artifacts" -v "$PWD/mlruns:/app/mlruns" -v "$PWD/reports:/app/reports" \
  rental-radar-ml train                                      # train: full pipeline inside the container
```
Output on 2026-11-23, with the image built from `main` (serving on port 8090 on the host):
```text
$ docker run -d -p 8090:8080 rental-radar-ml && curl localhost:8090/health
{"status":"ok","model_loaded":true,"model_version":"20261123-170838-d806870"}
$ scripts/smoke_test.sh http://localhost:8090
...
$ curl -X POST http://localhost:8090/predict -d '{"units": -3}'   (expect 422)
HTTP 422
all endpoints OK
$ docker run --rm rental-radar-ml test
..............................................                           [100%]
46 passed in 28.22s
```
The same Dockerfile is what Cloud Build uses for the Cloud Run deployment (§6.5).

### 6.4 Test the API locally with curl
```bash
uvicorn api.main:app --port 8080          # or the docker run above
scripts/smoke_test.sh http://localhost:8080
```
`smoke_test.sh` calls every endpoint and fails on any unexpected status. A single prediction:
```bash
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" \
  -d '{"borough":"MANHATTAN","units":837,"stories":42,"year_built":2015,"has_mgmt":true,
       "c12_all":34,"c12_noise":13,"c12_mold":6,"c12_pests":5,"c3_all":5,"v12_a":1,"v12_b":1,"bb24_infested":1}'
```
```json
{"probability":0.288,"label":0,"risk_level":"medium","percentile":0.832,"threshold":0.32,"base_rate":0.1461,
 "backtest_band":{"low":0.2,"high":0.3,"n":6316,"mean_predicted":0.2465,"observed_rate":0.2459},
 "drivers":[{"feature":"c12_all","label":"311 complaints, past year","value":34.0,"effect":"raises","contribution":0.62}, "..."],
 "model_version":"20261123-170838-d806870"}
```

| Endpoint | Description |
|---|---|
| `POST /predict` | JSON features of one building → probability, 0/1 label, risk level, NYC percentile, backtest band, drivers |
| `POST /predict/batch` | `{"instances": [...]}`, up to 500 buildings |
| `GET /buildings/search?q=` | find a building by name or address (`625 W 57th St`, `22 44 Jackson`, `The Max`) |
| `GET /buildings/lookup?q=` | fallback for names we do not have (`5Pointz`): OpenStreetMap Nominatim, verified against our HPD buildings. When OSM gives a house number, only that exact address counts (otherwise "not found", never the building next door); without one, the building must be on the same street nearby; with neither, buildings within 100 m come back marked `"match": "nearby"` for the user to confirm |
| `GET /buildings/{bid}` | the building's latest features: the exact JSON the front end sends to `/predict` |
| `GET /model` | version, threshold, test metrics, baselines, calibration |
| `GET /health`, `GET /docs` | health check; OpenAPI docs |

Every field is optional: a missing count means 0, and a missing attribute gets the training median. Invalid input returns **422**: negative counts, unknown boroughs, coordinates outside NYC, or unknown field names.

### 6.5 Deploy
```bash
GCP_PROJECT=<project> scripts/deploy_cloudrun.sh
```
This builds with Cloud Build, deploys a public Cloud Run service, then runs `smoke_test.sh` against the live URL.
```bash
API_URL=<cloud run url> HF_SPACE=<hf-user>/rental-radar-forecast scripts/deploy_frontend_hf.sh
```
This publishes the front end with `API_URL` set.

Verify the deployed endpoint with a plain HTTP request:
```bash
curl -X POST https://rental-radar-ml-api-976733016944.us-east1.run.app/predict -H "Content-Type: application/json" -d '{"borough":"BRONX","units":40,"v12_b":6,"v12_c":3,"c12_all":25}'
```
Response from the live service (drivers trimmed):
```json
{"probability":0.3032,"label":0,"risk_level":"medium","percentile":0.837,"threshold":0.32,"base_rate":0.1461,
 "backtest_band":{"low":0.3,"high":0.4,"n":4887,"mean_predicted":0.3477,"observed_rate":0.3427},
 "drivers":[{"feature":"vbc12_per10u","label":"B/C violations per 10 units, past year","value":2.25,"effect":"raises","contribution":0.8858}, ...],
 "model_version":"20261123-170838-d806870"}
```
`scripts/smoke_test.sh https://rental-radar-ml-api-976733016944.us-east1.run.app` checks every endpoint of the live service, including a 422 for invalid input.

## 7. Front end

A custom web app (HTML, CSS and JavaScript `fetch`, no build step) in [`frontend/`](frontend), in the visual style of Rental Radar NYC, our team's renter app from another course. It has two pages, and both send every request to the deployed API.

**`index.html`: for renters**
1. **Search** a building by name or address, or try an example. `GET /buildings/search`. When nothing matches, a button looks the name up on OpenStreetMap: `GET /buildings/lookup`. Each result says how it was matched (same address, same street, or nearby and still to be confirmed).
2. **Forecast card.** `GET /buildings/{id}` returns the features, then `POST /predict`:
   - the probability, with a count-up animation;
   - a gauge with the NYC average marked;
   - a "Top N% riskiest in NYC" sticker from the percentile (The Max: top 17%);
   - the reasons behind the score;
   - one question to ask at the viewing.

**`project.html`: for the course and graders**
1. **Links** to the front end, the API docs, `/predict` and the repository.
2. **Pipeline and results.** `GET /model` supplies PR-AUC against the baselines, the confusion matrix and calibration.
3. **Try the API.** A form collects the building's attributes and history, sends them to `POST /predict` and shows the probability, risk level, backtest band and drivers, with the exact request and response JSON, the latency and a copy-as-curl button. It can start from a preset or from a real building's record (`GET /buildings/{id}`).

The API base URL is set in `frontend/config.js` (`?api=` overrides it). Run it locally:
```bash
python -m http.server 5173 --directory frontend     # with the API on :8080
```

| Renter page: search | Project page: model results | Project page: try the API |
|---|---|---|
| ![](docs/img/frontend-hero.png) | ![](docs/img/frontend-model.png) | ![](docs/img/frontend-api.png) |

## 8. Repository structure

```
config.yaml              every parameter of the pipeline
requirements.txt         pinned dependencies (Python 3.12)
Dockerfile               one image: serve | train | test
data_prep/               LOCAL, run once: download → clean → make_dataset (+ checks) → upload to GCS
src/
  config.py              load and validate config.yaml (+ env overrides)
  ingest.py              download the dataset from GCS, verify SHA-256
  features.py            preprocessing and feature engineering (fitted on train only)
  split.py               time-based train / validation / test split
  train.py               baselines + models, threshold and model selection on validation, MLflow
  evaluate.py            metrics, calibration, figures
  predict.py             load the model bundle, predict, explain
api/                     FastAPI app (main.py) and request/response schemas (schemas.py)
frontend/                web front end (Hugging Face Space): index.html for renters, project.html for graders
scripts/                 setup_gcs.sh, deploy_cloudrun.sh, deploy_frontend_hf.sh, smoke_test.sh, entrypoint.sh
tests/                   46 tests: config, leakage at the cutoff, GCS integrity, features, training end to end, API
reports/                 test metrics (metrics.json) and figures from the latest run
docs/img/                screenshots
```

Development history: each part landed through its own branch and [pull request](https://github.com/canyangxu/rental-radar-ml/pulls?q=is%3Apr): skeleton (#1) → data and GCS (#2) → model and MLflow (#3) → API (#4) → Docker and Cloud Run (#5) → front end (#6) → README (#7) → code-review fixes (#8) → live deployment links (#9) → front-end polish (#10) → search backup: hyphen-free addresses and OpenStreetMap name lookup (#11) → renter page and project page (#12). PRs #1–#10 by Canyang Xu ([@canyangxu](https://github.com/canyangxu)); #11 and #12 by Xianyu Wang ([@sienna-wangxy](https://github.com/sienna-wangxy)).

## 9. Ethics and limitations

**What the label really measures.** A violation exists only if an inspector visited, and inspections mostly follow tenant complaints. The model therefore predicts *recorded* violations, not true building condition.
- Buildings whose tenants complain less, for example because of language barriers, fear of retaliation or immigration status, can look safer than they are.
- Complaint-heavy buildings can look worse.

The page says plainly that this is "a forecast, not an inspection".

**Fairness and harm.**
- **Neighborhood signal.** Location features carry neighborhood patterns, which in NYC correlate with income and race. Used against tenants, for example to steer applicants or set rents, the score could reinforce existing inequity.
- **Intended use.** The model is meant to help renters ask better questions. It is not meant for landlords, insurers or lenders to screen buildings or people, and should not be used that way.
- **Ownership data.** The model uses no personal data about tenants. Ownership data is limited to whether a management company is registered, from company names only; names of individual people are never downloaded.

**Size bias.** Large buildings get more complaints simply because more people live there. Per-unit rates reduce this, but the number of units is still a strong feature. A small building with one serious violation can score lower than it should.

**Errors have costs.**
- At the chosen threshold the model misses 29% of the buildings that did get a B/C violation: 6,683 false negatives.
- 36% of its alerts are false alarms: 8,935.

The page shows the probability and the calibration band instead of a yes/no verdict, and pairs every forecast with a concrete question to verify in person.

**Time and drift.**
- The data is frozen on 2026-11-14.
- The test label window ends on 2026-10-31, two weeks before that. Violations from late October that HPD had not yet posted count as negatives, so test results may be slightly pessimistic.
- Test results come from one six-month window.
- HPD enforcement priorities, inspection staffing and seasons (heat violations peak in winter) shift the base rate.

The model should be retrained on new snapshots and re-checked on a later period before anyone relies on it.

**Other limitations.**
- 311 complaints are matched to buildings by normalized address, and about 30% do not match and are excluded.
- Building names come from HPD data plus a short curated list, so some marketed names are not searchable; the street address always is.
- The model is correlational: it does not explain *why* a building gets violations.
