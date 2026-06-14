"""
Shared preprocessing for the classical models.

All classical models (Naive Bayes, SVM, Logistic Regression) use the *same*
TF-IDF configuration so that the comparison is fair: any difference in results
comes from the classifier, not from a different feature representation.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Single source of truth for the split and the seed, reused everywhere.
RANDOM_SEED = 42
TEST_SIZE = 0.2


def build_vectorizer() -> TfidfVectorizer:
    """Return the shared TF-IDF vectorizer configuration."""
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=10000,
    )


def split_data(messages, labels):
    """Stratified train/test split that preserves the class imbalance."""
    return train_test_split(
        messages,
        labels,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=RANDOM_SEED,
    )
