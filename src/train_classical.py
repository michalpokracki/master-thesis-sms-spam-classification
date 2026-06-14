"""
End-to-end training and comparison pipeline for the classical models.

This ties the pieces together:
  load data -> shared split -> shared TF-IDF -> train each model -> evaluate.

Run as a script to print the comparison table and (optionally) save it:

    python -m src.train_classical
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from .data_loader import load_sms_spam
from .preprocessing import build_vectorizer, split_data
from .models import get_classical_models
from .evaluation import evaluate_model

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_comparison(save: bool = False) -> pd.DataFrame:
    """Train and evaluate all classical models, returning a results table."""
    df = load_sms_spam()

    X_train_raw, X_test_raw, y_train, y_test = split_data(
        df["message"], df["label_binary"]
    )

    # Shared TF-IDF fitted on the training data only (no leakage).
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    results = []
    for name, model in get_classical_models().items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start

        result = evaluate_model(name, model, X_test, y_test, train_time)
        results.append(result.as_row())

    results_df = pd.DataFrame(results)

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(RESULTS_DIR / "classical_models_comparison.csv", index=False)

    return results_df


def _format_for_display(results_df: pd.DataFrame) -> pd.DataFrame:
    """Round and rename columns for a readable printout."""
    display = results_df.copy()
    for col in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        display[col] = (display[col] * 100).round(2)
    display["train_time_s"] = display["train_time_s"].round(4)
    display["inference_ms_per_msg"] = display["inference_ms_per_msg"].round(4)
    return display


if __name__ == "__main__":
    table = run_comparison(save=True)
    display = _format_for_display(table)
    cols = [
        "model_name", "accuracy", "precision", "recall", "f1", "roc_auc",
        "train_time_s", "inference_ms_per_msg",
        "false_positives", "false_negatives",
    ]
    print(display[cols].to_string(index=False))
