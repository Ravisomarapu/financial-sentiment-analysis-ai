"""
FinBERT Sentiment Classifier (Reference Implementation)
-----------------------------------------------------
This module provides a reference implementation of a FinBERT-based
sentiment classifier as described in Chapter 3 (Methodology – Advanced
AI Models) of the thesis:
"Fortified Financial Sentiment Analysis Using AI for APT Decisioning".

Purpose:
- Demonstrate how a finance-domain transformer (FinBERT) can be used
  for 3-class financial sentiment classification (Positive/Negative/Neutral).
- Enable methodological reproducibility without assuming access to
  proprietary datasets or heavy compute.

Notes:
- This code uses HuggingFace Transformers.
- Fine-tuning is optional; inference-only usage is supported.
- Exact numerical results may differ from the thesis due to randomness,
  dataset reconstruction, and hardware differences.
"""

from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Default FinBERT model checkpoint (finance-domain BERT)
FINBERT_CHECKPOINT = "ProsusAI/finbert"

# Label mapping commonly used with FinBERT
LABEL_MAP = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


class FinBERTSentiment:
    """
    Wrapper class for FinBERT sentiment inference.
    """

    def __init__(self, model_name: str = FINBERT_CHECKPOINT, device: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(self.device)
        self.model.eval()

    def predict(self, texts: List[str], max_length: int = 128) -> List[str]:
        """
        Predict sentiment labels for a list of texts.

        Parameters:
        - texts: list of preprocessed text strings
        - max_length: maximum token length
        """
        predictions = []

        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=max_length
                )

                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                logits = outputs.logits

                pred_id = torch.argmax(logits, dim=1).item()
                predictions.append(LABEL_MAP.get(pred_id, "Neutral"))

        return predictions


if __name__ == "__main__":
    # Example usage
    from preprocess import preprocess_text

    texts = [
        "company reports strong profit growth",
        "firm posts loss after missing earnings estimates",
        "shares unchanged ahead of earnings announcement"
    ]

    texts = [preprocess_text(t) for t in texts]

    finbert = FinBERTSentiment()
    labels = finbert.predict(texts)

    for t, l in zip(texts, labels):
        print(f"Text: {t} -> Sentiment: {l}")
