"""
GPT-based Sentiment Classifier (FinSentGPT-style) – Reference Stub
----------------------------------------------------------------
This module provides a **reference implementation** of a GPT-style
generative sentiment classifier, inspired by FinSentGPT, as described in
Chapter 3 (Advanced AI Models) of the thesis:
"Fortified Financial Sentiment Analysis Using AI for APT Decisioning".

Purpose:
- Demonstrate how a generative language model can be adapted for
  financial sentiment classification
- Show prompt/format design used for fine-tuning or inference
- Avoid dependence on proprietary APIs or releasing trained weights

IMPORTANT:
- This is a conceptual and runnable stub
- It does NOT require access to OpenAI or other proprietary services
- It focuses on methodological clarity and reproducibility
"""

from typing import List

# Canonical label set
LABELS = ["Positive", "Neutral", "Negative"]


def format_training_example(text: str, label: str) -> str:
    """
    Format a training example in a FinSentGPT-style manner.

    Example format:
    <TEXT> ### <LABEL>
    """
    return f"{text} ### {label}"


def build_prompt(text: str) -> str:
    """
    Build an inference prompt for a GPT-style sentiment classifier.
    """
    prompt = (
        "Classify the financial sentiment of the following text as "
        "Positive, Neutral, or Negative.\n\n"
        f"Text: {text}\nSentiment:"
    )
    return prompt


class GPTSentimentStub:
    """
    Reference GPT-style sentiment classifier.
    """

    def __init__(self):
        pass

    def predict(self, texts: List[str]) -> List[str]:
        """
        Predict sentiment labels using heuristic logic.

        NOTE:
        - In a real system, this prompt would be passed to a fine-tuned GPT model
        - Here we use simple rules to keep the implementation self-contained
        """
        predictions = []
        for text in texts:
            text_lower = text.lower()
            if any(w in text_lower for w in ["loss", "decline", "drop", "miss"]):
                predictions.append("Negative")
            elif any(w in text_lower for w in ["profit", "gain", "beat", "growth"]):
                predictions.append("Positive")
            else:
                predictions.append("Neutral")
        return predictions


if __name__ == "__main__":
    # Example usage
    examples = [
        "company reports strong profit growth",
        "firm posts loss after earnings miss",
        "shares trade flat ahead of results"
    ]

    gpt_stub = GPTSentimentStub()
    labels = gpt_stub.predict(examples)

    for t, l in zip(examples, labels):
        print(f"Text: {t} -> Sentiment: {l}")
