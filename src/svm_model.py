"""
SVM Sentiment Classifier with TF-IDF Features
--------------------------------------------
Reference implementation of the classical machine learning baseline
as described in Chapter 3 (Methodology – Baseline Models) of the thesis:
"Fortified Financial Sentiment Analysis Using AI for APT Decisioning".

This model uses:
- TF-IDF features (unigrams + bigrams)
- Linear Support Vector Machine (SVM)

The implementation is intended for methodological reproducibility,
not for exact numerical replication of reported results.
"""

from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


def train_svm(
    train_texts: List[str],
    train_labels: List[str],
    max_features: int = 5000
) -> Tuple[LinearSVC, TfidfVectorizer]:
    """
    Train a linear SVM using TF-IDF features.

    Parameters:
    - train_texts: list of preprocessed text strings
    - train_labels: list of sentiment labels
    - max_features: vocabulary size cap
    """

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        lowercase=False
    )

    X_train = vectorizer.fit_transform(train_texts)

    clf = LinearSVC(class_weight="balanced")
    clf.fit(X_train, train_labels)

    return clf, vectorizer


def evaluate_svm(
    clf: LinearSVC,
    vectorizer: TfidfVectorizer,
    test_texts: List[str],
    test_labels: List[str]
) -> None:
    """
    Evaluate the SVM classifier on a test set.
    """
    X_test = vectorizer.transform(test_texts)
    preds = clf.predict(X_test)

    print("Accuracy:", accuracy_score(test_labels, preds))
    print("\nClassification Report:\n")
    print(classification_report(test_labels, preds))


if __name__ == "__main__":
    # Example usage with dummy data
    from preprocess import preprocess_text

    texts = [
        "company reports strong profit growth",
        "firm posts heavy loss after weak demand",
        "shares unchanged after earnings announcement",
        "revenue beats estimates and stock jumps",
        "company cuts guidance amid market uncertainty"
    ]

    labels = [
        "Positive",
        "Negative",
        "Neutral",
        "Positive",
        "Negative"
    ]

    texts = [preprocess_text(t) for t in texts]

    model, vec = train_svm(texts, labels)
    evaluate_svm(model, vec, texts, labels)
