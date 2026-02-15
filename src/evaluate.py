# src/evaluate.py

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
)

from src.utils import ensure_dir, write_text


def _save_confusion_matrix(cm: np.ndarray, out_path: str, title: str) -> None:
    """
    Saves a confusion matrix plot.
    """
    ensure_dir(os.path.dirname(out_path))

    fig = plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Annotate cells
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_roc_curve(y_true, y_prob, out_path: str, title: str) -> None:
    """
    Saves ROC curve plot using sklearn's RocCurveDisplay.
    """
    ensure_dir(os.path.dirname(out_path))

    fig = plt.figure()
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_model(
    model_name: str,
    y_true,
    y_pred,
    y_prob,
    run_dir: str,
) -> dict:
    """
    Computes required metrics and saves evaluation artifacts for a single model:
    - classification report (txt)
    - confusion matrix (png)
    - ROC curve (png)
    Returns a dict of metrics for comparison table.
    """
    # Metrics required by assignment
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    metrics = {
        "model": model_name,
        "accuracy": round(acc, 6),
        "auc": round(auc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "mcc": round(mcc, 6),
    }

    # Artifacts
    report = classification_report(y_true, y_pred, digits=4)
    report_path = os.path.join(run_dir, f"classification_report_{model_name}.txt")
    write_text(report_path, report)

    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(run_dir, f"confusion_matrix_{model_name}.png")
    _save_confusion_matrix(cm, cm_path, title=f"Confusion Matrix - {model_name}")

    roc_path = os.path.join(run_dir, f"roc_curve_{model_name}.png")
    _save_roc_curve(y_true, y_prob, roc_path, title=f"ROC Curve - {model_name}")

    return metrics