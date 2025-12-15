"""
Evaluation Utilities for Financial Sentiment Analysis
----------------------------------------------------
Reference implementation of the evaluation methodology described in
Chapter 4 (Results & Discussion) of the thesis:
"Fortified Financial Sentiment Analysis Using AI for APT Decisioning".

This module provides:
- Classification metrics (Accuracy, Precision, Recall, F1)
- Confusion matrix
- Daily sentiment index computation

This is a lightweight, reproducible implementation intended to
support methodological validation rather than exact result replication.
"""

from typing import List, Dict
from collections import Counter, defaultdict

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)


LABELS = ["Positive", "Neutral", "Negative"]


def classification_metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    """
    Compute accuracy, precision, recall, and macro-F1.
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1
    }


def confusion(y_true: List[str], y_pred: List[str]):
    """
    Compute confusion matrix.
    """
    return confusion_matrix(y_true, y_pred, labels=LABELS)


def daily_sentiment_index(predictions_by_day: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Compute daily sentiment index as:

    ( #Positive - #Negative ) / ( #Positive + #Negative + #Neutral )

    Parameters:
    - predictions_by_day: dict mapping date -> list of sentiment labels
    """
    index = {}

    for day, labels in predictions_by_day.items():
        counts = Counter(labels)
        pos = counts.get("Positive", 0)
        neg = counts.get("Negative", 0)
        neu = counts.get("Neutral", 0)

        total = pos + neg + neu
        if total == 0:
            index[day] = 0.0
        else:
            index[day] = (pos - neg) / total

    return index


if __name__ == "__main__":
    # Example usage
    y_true = ["Positive", "Negative", "Neutral", "Positive", "Negative"]
    y_pred = ["Positive", "Negative", "Neutral", "Neutral", "Negative"]

    metrics = classification_metrics(y_true, y_pred)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    print("\nConfusion Matrix:")
    print(confusion(y_true, y_pred))

    predictions_by_day = {
        "2025-01-01": ["Positive", "Positive", "Neutral"],
        "2025-01-02": ["Negative", "Negative", "Neutral"],
    }

    print("\nDaily Sentiment Index:")
    for day, value in daily_sentiment_index(predictions_by_day).items():
        print(day, value)
