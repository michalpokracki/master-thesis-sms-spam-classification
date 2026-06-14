"""
Classical model definitions.

Keeping the three classical models in one place makes the comparison explicit
and ensures consistent settings (e.g. the shared random seed).
"""

from __future__ import annotations

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from .preprocessing import RANDOM_SEED


def get_classical_models() -> dict:
    """
    Return the dictionary of classical models to benchmark.

    - Multinomial Naive Bayes: the historical text-classification baseline.
    - Linear SVM: strong linear classifier for high-dimensional sparse TF-IDF.
    - Logistic Regression: probabilistic linear baseline, naturally calibrated.

    ``class_weight="balanced"`` is used for the two discriminative models to
    counteract the ~87/13 class imbalance. Naive Bayes is left at its default,
    as it is the historical reference point.
    """
    return {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
    }
