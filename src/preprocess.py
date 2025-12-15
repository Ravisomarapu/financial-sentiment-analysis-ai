"""
Preprocessing module for Financial Sentiment Analysis
--------------------------------------------------
This script implements the text cleaning and preprocessing steps
as described in Chapter 3 (Methodology – Data Preprocessing)
of the thesis:
"Fortified Financial Sentiment Analysis Using AI for APT Decisioning".

This is a reference implementation intended to demonstrate
methodological reproducibility.
"""

import re
import random
from typing import List

# Finance-relevant terms that should NOT be removed as stopwords
FINANCE_TERMS = {
    "gain", "gains", "loss", "losses", "up", "down",
    "rise", "rises", "fall", "falls", "profit", "profits",
    "revenue", "earnings", "beat", "miss"
}

# Basic English stopwords (minimal, illustrative list)
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while",
    "of", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after",
    "to", "from", "in", "out", "on", "off", "over", "under"
}


def clean_text(text: str) -> str:
    """
    Clean and normalize raw financial text.

    Steps:
    - Remove HTML tags
    - Remove URLs
    - Lowercase text
    - Preserve numbers and finance-relevant terms
    - Remove extraneous punctuation
    """
    if not text:
        return ""

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Lowercase
    text = text.lower()

    # Remove punctuation except % and $ (financially relevant)
    text = re.sub(r"[^a-z0-9%$\s-]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into words while preserving finance-relevant tokens.
    """
    tokens = text.split()
    return tokens


def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Remove stopwords while keeping finance-relevant terms.
    """
    filtered = []
    for token in tokens:
        if token in FINANCE_TERMS:
            filtered.append(token)
        elif token not in STOPWORDS:
            filtered.append(token)
    return filtered


def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline for a single text input.
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    return " ".join(tokens)


def train_val_test_split(data: List, labels: List, seed: int = 42):
    """
    Split data into train (80%), validation (10%), and test (10%).
    Stratification is assumed to be handled upstream if required.
    """
    random.seed(seed)
    indices = list(range(len(data)))
    random.shuffle(indices)

    n = len(indices)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X_train = [data[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]

    X_val = [data[i] for i in val_idx]
    y_val = [labels[i] for i in val_idx]

    X_test = [data[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    # Example usage
    sample_text = "Company reports 10% revenue growth but cuts guidance"
    processed = preprocess_text(sample_text)
    print("Original:", sample_text)
    print("Processed:", processed)
