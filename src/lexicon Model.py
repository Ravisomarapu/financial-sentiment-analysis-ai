"""
Lexicon-Based Sentiment Classifier (Loughran–McDonald)
----------------------------------------------------
Reference implementation of the lexicon-based baseline described in
Chapter 3 (Methodology – Baseline Models) of the thesis:
"Fortified Financial Sentiment Analysis Using AI for APT Decisioning".

This model uses the Loughran–McDonald financial sentiment lexicon to
classify text into Positive / Negative / Neutral based on word counts.

NOTE:
- This is a reference implementation for methodological reproducibility.
- The actual Loughran–McDonald word lists must be obtained from the
  original authors and are not redistributed here.
"""

from typing import List, Dict

# Placeholder lexicon lists (illustrative only)
# Replace with full Loughran–McDonald lists loaded from files
POSITIVE_WORDS = {
    "gain", "gains", "profit", "profits", "growth", "improve",
    "improves", "beat", "beats", "strong", "increase"
}

NEGATIVE_WORDS = {
    "loss", "losses", "decline", "declines", "fall", "falls",
    "drop", "drops", "miss", "misses", "weak", "decrease"
}


def score_text(tokens: List[str]) -> int:
    """
    Compute sentiment score as:
        (#positive words) - (#negative words)
    """
    pos_count = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg_count = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    return pos_count - neg_count


def classify_score(score: int) -> str:
    """
    Convert sentiment score to class label.
    """
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"


def predict(texts: List[str], tokenizer) -> List[str]:
    """
    Predict sentiment labels for a list of texts.

    Parameters:
    - texts: list of preprocessed text strings
    - tokenizer: function that converts text -> list of tokens
    """
    predictions = []
    for text in texts:
        tokens = tokenizer(text)
        score = score_text(tokens)
        label = classify_score(score)
        predictions.append(label)
    return predictions


if __name__ == "__main__":
    # Example usage
    from preprocess import preprocess_text, tokenize

    examples = [
        "company reports strong profit growth",
        "firm posts loss but forecasts improvement",
        "shares unchanged after earnings"
    ]

    processed = [preprocess_text(t) for t in examples]
    labels = predict(processed, tokenize)

    for text, label in zip(examples, labels):
        print(f"Text: {text} -> Sentiment: {label}")
