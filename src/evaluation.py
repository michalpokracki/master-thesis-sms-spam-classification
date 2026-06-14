"""
Unified evaluation for all models.

Computing every model's metrics through the same function guarantees that
"accuracy" or "F1" means exactly the same thing for Naive Bayes, SVM,
Logistic Regression and (later) DistilBERT.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


@dataclass
class EvaluationResult:
    """Container for one model's predictive and basic operational metrics."""

    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    train_time_s: float
    inference_ms_per_msg: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    def as_row(self) -> dict:
        """Flat dict suitable for a pandas DataFrame row."""
        return asdict(self)


def evaluate_model(model_name, model, X_test, y_test, train_time_s, y_score=None):
    """
    Evaluate a fitted model on the test set.

    Parameters
    ----------
    model_name : str
    model : fitted classifier with ``predict`` (and ideally ``predict_proba``)
    X_test, y_test : test features and labels
    train_time_s : float
        Training time measured by the caller.
    y_score : optional array of positive-class scores. If None, the function
        tries ``predict_proba`` and falls back to ``decision_function``.
    """
    # Inference-time benchmark (averaged per message, in milliseconds).
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_ms = (time.perf_counter() - start) / len(y_test) * 1000.0

    # Positive-class score for ROC-AUC.
    if y_score is None:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:  # pragma: no cover - all our models support one of the above
            y_score = y_pred

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return EvaluationResult(
        model_name=model_name,
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall=recall_score(y_test, y_pred, zero_division=0),
        f1=f1_score(y_test, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_test, y_score),
        train_time_s=train_time_s,
        inference_ms_per_msg=inference_ms,
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
    )
