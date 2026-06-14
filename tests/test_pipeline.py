"""
Initial test suite for the classical-models pipeline.

These tests verify the correctness and stability of the core components rather
than chasing a specific accuracy number. They are the "initial system testing"
for the engineering project: confirm the data loads, the shapes are right, the
split preserves the class balance, every model trains and produces metrics in a
valid range, and results are reproducible across runs.

Run with:  pytest -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_sms_spam
from src.preprocessing import build_vectorizer, split_data, RANDOM_SEED
from src.models import get_classical_models
from src.evaluation import evaluate_model


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    """Load the dataset once for all tests (uses local cache if present)."""
    return load_sms_spam()


@pytest.fixture(scope="module")
def split(dataset):
    """A single shared split reused across tests."""
    return split_data(dataset["message"], dataset["label_binary"])


# --------------------------------------------------------------------------- #
# Data integrity
# --------------------------------------------------------------------------- #

def test_dataset_loads_and_has_expected_columns(dataset):
    assert {"label", "message", "label_binary"}.issubset(dataset.columns)
    assert len(dataset) > 5000  # the collection has ~5.5k messages


def test_labels_are_only_ham_or_spam(dataset):
    assert set(dataset["label"].unique()) == {"ham", "spam"}


def test_binary_label_matches_text_label(dataset):
    # label_binary must be 1 exactly when label == "spam"
    assert (dataset["label_binary"] == (dataset["label"] == "spam").astype(int)).all()


def test_no_missing_messages(dataset):
    assert dataset["message"].notna().all()


def test_dataset_is_imbalanced_ham_majority(dataset):
    spam_ratio = dataset["label_binary"].mean()
    # ham should clearly dominate (~13% spam in the real dataset)
    assert 0.05 < spam_ratio < 0.30


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #

def test_split_sizes_add_up(split, dataset):
    X_train, X_test, y_train, y_test = split
    assert len(X_train) + len(X_test) == len(dataset)
    # default test size is 20%
    assert abs(len(X_test) / len(dataset) - 0.20) < 0.01


def test_split_is_stratified(split):
    _, _, y_train, y_test = split
    # spam ratio should be (almost) identical in train and test
    assert abs(y_train.mean() - y_test.mean()) < 0.02


# --------------------------------------------------------------------------- #
# Vectorizer
# --------------------------------------------------------------------------- #

def test_vectorizer_no_leakage_and_consistent_features(split):
    X_train_raw, X_test_raw, _, _ = split
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    # same feature dimensionality on train and test
    assert X_train.shape[1] == X_test.shape[1]
    assert X_train.shape[0] == len(X_train_raw)
    assert X_test.shape[0] == len(X_test_raw)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

def test_all_models_train_and_metrics_in_valid_range(split):
    X_train_raw, X_test_raw, y_train, y_test = split
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    for name, model in get_classical_models().items():
        model.fit(X_train, y_train)
        result = evaluate_model(name, model, X_test, y_test, train_time_s=0.0)
        for metric_name, value in [
            ("accuracy", result.accuracy),
            ("precision", result.precision),
            ("recall", result.recall),
            ("f1", result.f1),
            ("roc_auc", result.roc_auc),
        ]:
            assert 0.0 <= value <= 1.0, f"{name} {metric_name} out of range: {value}"


def test_confusion_matrix_counts_sum_to_test_size(split):
    X_train_raw, X_test_raw, y_train, y_test = split
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    model = get_classical_models()["Multinomial Naive Bayes"]
    model.fit(X_train, y_train)
    result = evaluate_model("NB", model, X_test, y_test, train_time_s=0.0)
    total = (
        result.true_negatives
        + result.false_positives
        + result.false_negatives
        + result.true_positives
    )
    assert total == len(y_test)


def test_reproducibility_same_seed_same_predictions(split):
    """Training twice with the fixed seed must give identical predictions."""
    X_train_raw, X_test_raw, y_train, y_test = split
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    model_a = get_classical_models()["Logistic Regression"]
    model_b = get_classical_models()["Logistic Regression"]
    model_a.fit(X_train, y_train)
    model_b.fit(X_train, y_train)

    np.testing.assert_array_equal(
        model_a.predict(X_test), model_b.predict(X_test)
    )
