"""Traditional classifier definitions."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_models():
    return {
        "naive_bayes": MultinomialNB(alpha=0.5),
        "linear_svm": LinearSVC(C=1.0, class_weight="balanced", random_state=42),
    }


def make_pipeline(model, max_features: int, min_df: int) -> Pipeline:
    vectorizer = TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.95,
        max_features=max_features,
        sublinear_tf=True,
    )
    return Pipeline(
        [
            ("tfidf", vectorizer),
            ("classifier", model),
        ]
    )
