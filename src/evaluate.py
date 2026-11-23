"""Pipeline step 4: evaluation.

About 14% of buildings get a B/C violation in the next 6 months, so a model that always says "no" is 86% accurate
and useless. Metrics are therefore chosen for an imbalanced target:
  * PR-AUC (average precision): primary metric, compares against the 14% base rate
  * ROC-AUC: ranking quality
  * precision, recall, F1 and the full confusion matrix at a threshold chosen on VALIDATION, never on test
  * precision among the top 10% riskiest buildings: what an inspector or renter shortlist would see
Every model is compared with two baselines on the same test set: "always no" and "had a B/C violation last year".
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)


def pick_threshold(y_val, p_val, metric: str = "f1") -> float:
    """Decision threshold that maximizes the metric on the validation set."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scorer = {"f1": f1_score, "precision": precision_score, "recall": recall_score}[metric]
    scores = [scorer(y_val, p_val >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def metrics(y, p, threshold: float, top_k_fraction: float = 0.10) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    k = max(1, int(len(p) * top_k_fraction))
    top = np.argsort(-p, kind="stable")[:k]
    out = {
        "pr_auc": average_precision_score(y, p) if len(set(y)) > 1 else float("nan"),
        "roc_auc": roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "accuracy": float((pred == y).mean()),
        # constant scores cannot rank buildings: a top-k list is then a random pick at the base rate
        f"precision_top_{int(top_k_fraction * 100)}pct": float(y.mean() if np.ptp(p) == 0 else y[top].mean()),
        "base_rate": float(y.mean()),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "threshold": float(threshold),
    }
    return {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in out.items()}


def calibration_bands(y, p, edges=(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001)) -> list[dict]:
    """Observed violation rate per predicted-probability band on the test set (used to say how often the model is right)."""
    y, p = np.asarray(y), np.asarray(p)
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            bands.append({"low": round(lo, 2), "high": round(min(hi, 1.0), 2), "n": int(m.sum()),
                          "mean_predicted": round(float(p[m].mean()), 4), "observed_rate": round(float(y[m].mean()), 4)})
    return bands


def plot_confusion(m: dict, title: str, path) -> None:
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Oranges")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=13, color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["predicted no", "predicted yes"])
    ax.set_yticks([0, 1], ["actual no", "actual yes"])
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pr_curves(y, curves: dict, path, title: str = "Test set") -> None:
    """Precision-recall curve per model, with the base rate as the 'no skill' line."""
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for name, p in curves.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, label=f"{name} (PR-AUC {average_precision_score(y, p):.3f})")
    ax.axhline(np.mean(y), ls="--", c="gray", label=f"always no / base rate {np.mean(y):.3f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(names, values, path, title="Feature importance (gain)") -> None:
    order = np.argsort(values)[-15:]
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.barh(np.array(names)[order], np.array(values)[order], color="#F2B632")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
