"""Training pipeline: ingest from GCS -> features -> time split -> train -> evaluate -> track -> save.

Every run is tracked in MLflow (experiment mlflow.experiment):
  parent run     the config, dataset row counts and hash, the selected model, its test metrics and all figures
  nested runs    one per baseline and per candidate model, with its parameters, validation and test metrics,
                 confusion matrix and the logged scikit-learn model

The served model is selected by validation PR-AUC (evaluation.select_model_by); the test set is only used to report.
Outputs:
  artifacts/model/model.joblib + metadata.json   the model the API loads (also baked into the Docker image)
  reports/metrics.json + reports/figures/*.png   evaluation results shown in the README

Usage:
    python -m src.train                 # data from GCS (default)
    DATA_SOURCE=local python -m src.train   # data/processed/, offline only
"""
from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src import evaluate as ev
from src.config import load_config, path
from src.features import LABELS, FeatureBuilder
from src.ingest import dataset_dir, sha256
from src.split import time_split


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = json.dumps(v) if isinstance(v, list) else v
    return out


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "nogit"


def build_candidates(cfg: dict, seed: int) -> dict:
    lr, xg = cfg["models"]["logistic_regression"], dict(cfg["models"]["xgboost"])
    per_unit = cfg["features"]["per_unit_rates"]
    early = xg.pop("early_stopping_rounds")
    return {
        "logistic_regression": Pipeline([
            ("features", FeatureBuilder(log_counts=True, per_unit_rates=per_unit)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=lr["C"], max_iter=lr["max_iter"], random_state=seed)),
        ]),
        "xgboost": Pipeline([
            ("features", FeatureBuilder(log_counts=False, per_unit_rates=per_unit)),
            ("model", XGBClassifier(**xg, early_stopping_rounds=early, eval_metric="aucpr", random_state=seed, n_jobs=-1)),
        ]),
    }


def fit(name: str, pipe: Pipeline, train: pd.DataFrame, val: pd.DataFrame) -> Pipeline:
    if name == "xgboost":  # early stopping watches validation PR-AUC; the features step is fitted on train only
        feats = pipe.named_steps["features"].fit(train)
        pipe.named_steps["model"].fit(feats.transform(train), train["y"], eval_set=[(feats.transform(val), val["y"])], verbose=False)
        return pipe
    return pipe.fit(train, train["y"])


def importance(name: str, pipe: Pipeline) -> tuple[list[str], np.ndarray]:
    names = pipe.named_steps["features"].feature_names_
    model = pipe.named_steps["model"]
    if name == "xgboost":
        gain = model.get_booster().get_score(importance_type="gain")
        return names, np.array([gain.get(n, 0.0) for n in names])
    return names, np.abs(model.coef_[0])


def example_rows(df: pd.DataFrame) -> pd.DataFrame:
    """A few request rows for the MLflow signature; integers as floats because real requests may omit values."""
    ex = df.drop(columns=["y", "y_count"]).head(3).copy()
    ints = ex.select_dtypes("integer").columns
    ex[ints] = ex[ints].astype("float64")
    return ex


def main() -> dict:
    cfg = load_config()
    seed = cfg["project"]["seed"]
    np.random.seed(seed)
    ecfg = cfg["evaluation"]
    k = ecfg["top_k_fraction"]

    # 1. ingest
    src_dir = dataset_dir(cfg)
    snaps = pd.read_parquet(src_dir / "snapshots.parquet")
    live = pd.read_parquet(src_dir / "live.parquet")
    data_hash = sha256(src_dir / "snapshots.parquet")
    # 3. split (features are built inside each model pipeline, fitted on train only)
    train, val, test = time_split(snaps, cfg)
    print(f"rows: train {len(train):,}  val {len(val):,}  test {len(test):,}  "
          f"positive rate {train.y.mean():.3f} / {val.y.mean():.3f} / {test.y.mean():.3f}")

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    reports = path("reports")
    figs = reports / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    gcs = cfg["data"]["gcs"]
    data_uri = f"gs://{gcs['bucket']}/{gcs['prefix']}/snapshots.parquet"

    with mlflow.start_run(run_name=f"train-{stamp}") as parent:
        mlflow.log_params(flatten({k2: cfg[k2] for k2 in ("features", "target", "split", "evaluation")}))
        mlflow.set_tags({"git_sha": git_sha(), "data_sha256": data_hash, "data_uri": data_uri})
        mlflow.log_metrics({"rows_train": len(train), "rows_val": len(val), "rows_test": len(test),
                            "positive_rate_train": train.y.mean(), "positive_rate_test": test.y.mean()})
        for split_name, part in (("train", train), ("validation", val), ("test", test)):
            mlflow.log_input(mlflow.data.from_pandas(part, source=data_uri, name=split_name, targets="y"), context=split_name)

        results, curves = {}, {}
        # baselines, on the same test set
        hist = (test.v12_b + test.v12_c).to_numpy()
        baselines = {
            "baseline_always_no": (np.zeros(len(test)), 0.5),
            "baseline_bc_last_year": (hist, 0.5),  # rule: predict yes if the building had a B/C violation last year
        }
        for name, (score, th) in baselines.items():
            with mlflow.start_run(run_name=name, nested=True):
                m = ev.metrics(test.y, score, th, k)
                mlflow.log_metrics({f"test_{key}": v for key, v in m.items()})
                results[name] = {"test": m}
        curves["B/C last year (rule)"] = hist

        # candidate models
        fitted = {}
        for name, pipe in build_candidates(cfg, seed).items():
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_params({f"model.{k2}": v for k2, v in cfg["models"][name].items()})
                pipe = fit(name, pipe, train, val)
                p_val = pipe.predict_proba(val)[:, 1]
                p_test = pipe.predict_proba(test)[:, 1]
                th = ev.pick_threshold(val.y, p_val, ecfg["threshold_metric"])
                mv, mt = ev.metrics(val.y, p_val, th, k), ev.metrics(test.y, p_test, th, k)
                mlflow.log_metrics({**{f"val_{a}": b for a, b in mv.items()}, **{f"test_{a}": b for a, b in mt.items()}})
                if name == "xgboost":
                    mlflow.log_metric("best_iteration", pipe.named_steps["model"].best_iteration)
                cm_path = figs / f"confusion_{name}.png"
                ev.plot_confusion(mt, f"{name}, test (threshold {th:.2f})", cm_path)
                imp_path = figs / f"importance_{name}.png"
                ev.plot_importance(*importance(name, pipe), imp_path,
                                   "XGBoost feature importance (gain)" if name == "xgboost" else "Logistic regression |coefficient|")
                mlflow.log_artifact(str(cm_path))
                mlflow.log_artifact(str(imp_path))
                example = example_rows(test)
                mlflow.sklearn.log_model(pipe, name="model", input_example=example, serialization_format="cloudpickle",
                                         signature=infer_signature(example, pipe.predict_proba(example)[:, 1]))
                results[name] = {"val": mv, "test": mt}
                fitted[name] = (pipe, th, p_test)
                curves[name] = p_test
                print(f"{name:20s} val PR-AUC {mv['pr_auc']:.3f} | test PR-AUC {mt['pr_auc']:.3f} "
                      f"ROC-AUC {mt['roc_auc']:.3f} F1 {mt['f1']:.3f} @ {th:.2f}")

        # 5. select on validation, then save
        best = max(fitted, key=lambda n: results[n]["val"]["pr_auc"])
        pipe, th, p_test = fitted[best]
        ev.plot_pr_curves(test.y.to_numpy(), curves, figs / "pr_curves.png", f"Test set, cutoff {cfg['split']['test_cutoff']}")
        ev.plot_confusion(results["baseline_bc_last_year"]["test"], "rule: B/C last year, test", figs / "confusion_baseline_rule.png")
        live_scores = pipe.predict_proba(live)[:, 1]
        version = f"{stamp}-{git_sha()}"
        meta = {
            "model_name": cfg["mlflow"]["registered_model"],
            "model_type": best,
            "version": version,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mlflow_run_id": parent.info.run_id,
            "target": f"at least one class {'/'.join(cfg['target']['violation_classes'])} HPD violation "
                      f"in the next {cfg['target']['horizon_months']} months",
            "cutoffs": {k2: cfg["split"][k2] for k2 in ("train_cutoff", "val_cutoff", "test_cutoff", "live_cutoff")},
            "threshold": th,
            "threshold_rule": f"maximizes {ecfg['threshold_metric']} on validation",
            "base_rate": round(float(live_scores.mean()), 4),
            "test_metrics": results[best]["test"],
            "val_metrics": results[best]["val"],
            "baselines": {n: results[n]["test"] for n in baselines},
            "calibration_bands": ev.calibration_bands(test.y, p_test),
            "live_score_quantiles": np.round(np.quantile(live_scores, np.linspace(0, 1, 1001)), 6).tolist(),
            "features": pipe.named_steps["features"].feature_names_,
            "feature_labels": {f: LABELS.get(f, f) for f in pipe.named_steps["features"].feature_names_},
            "data_sha256": data_hash,
        }
        model_dir = path(cfg["serving"]["model_dir"])
        if model_dir.exists():
            shutil.rmtree(model_dir)
        model_dir.mkdir(parents=True)
        joblib.dump(pipe, model_dir / "model.joblib")
        (model_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        summary = {"selected_model": best, "selected_by": ecfg["select_model_by"], "version": version,
                   "mlflow_run_id": parent.info.run_id, "rows": {"train": len(train), "val": len(val), "test": len(test)},
                   "results": results, "calibration_bands": meta["calibration_bands"]}
        (reports / "metrics.json").write_text(json.dumps(summary, indent=2))

        mlflow.set_tags({"selected_model": best, "model_version": version})
        mlflow.log_metrics({f"selected_test_{a}": b for a, b in results[best]["test"].items()})
        mlflow.log_artifacts(str(figs), artifact_path="figures")
        mlflow.log_artifact(str(reports / "metrics.json"))
        mlflow.log_artifacts(str(model_dir), artifact_path="serving_model")
        example = example_rows(test)
        info = mlflow.sklearn.log_model(pipe, name="selected_model", input_example=example, serialization_format="cloudpickle",
                                        signature=infer_signature(example, pipe.predict_proba(example)[:, 1]))
        mv = mlflow.register_model(info.model_uri, cfg["mlflow"]["registered_model"])
        mlflow.set_tag("registered_version", mv.version)
        print(f"selected {best} (val PR-AUC {results[best]['val']['pr_auc']:.3f}); saved to {model_dir}")
    return summary


if __name__ == "__main__":
    main()
