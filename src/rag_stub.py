"""
Retrieval-Augmented Generation (RAG) Sentiment Pipeline – Reference Stub
---------------------------------------------------------------------
This module provides a **reference (non-proprietary) implementation** of
a Retrieval-Augmented LLM (RAG) approach for financial sentiment analysis,
as described in Chapter 3 (Advanced AI Models) of the thesis:
"Fortified Financial Sentiment Analysis Using AI for APT Decisioning".

Purpose:
- Demonstrate the *architecture* and *logic* of a RAG-based sentiment system
- Show how external context can improve sentiment classification
- Avoid dependence on proprietary APIs or confidential data

IMPORTANT:
- This is an architectural stub, not a production RAG system
- No external APIs are called by default
- The design mirrors the thesis methodology for reproducibility
"""

from typing import List


class SimpleRetriever:
    """
    Simple keyword-based retriever (illustrative).
    In practice, this could be replaced with TF-IDF, BM25, or vector search.
    """

    def __init__(self, corpus: List[str]):
        self.corpus = corpus

    def retrieve(self, query: str, top_k: int = 2) -> List[str]:
        """
        Retrieve top-k context snippets relevant to the query.
        """
        query_terms = set(query.split())
        scored = []

        for doc in self.corpus:
            score = len(query_terms.intersection(doc.split()))
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored[:top_k] if score > 0]


class RAGSentimentStub:
    """
    Reference RAG pipeline for sentiment classification.
    """

    def __init__(self, retriever: SimpleRetriever):
        self.retriever = retriever

    def build_prompt(self, text: str, context: List[str]) -> str:
        """
        Construct a few-shot style prompt for sentiment classification.
        """
        prompt = """
You are a financial analyst.
Classify the sentiment of the following financial text as
Positive, Neutral, or Negative.

Context:
"""
        for c in context:
            prompt += f"- {c}\n"

        prompt += f"\nText: {text}\nSentiment:"
        return prompt

    def predict(self, texts: List[str]) -> List[str]:
        """
        Predict sentiment labels using retrieved context.

        NOTE:
        - This stub returns heuristic outputs for demonstration
        - In a real system, the prompt would be passed to an LLM
        """
        predictions = []

        for text in texts:
            context = self.retriever.retrieve(text)
            prompt = self.build_prompt(text, context)

            # Heuristic placeholder decision logic
            if "loss" in text or "decline" in text:
                predictions.append("Negative")
            elif "profit" in text or "gain" in text:
                predictions.append("Positive")
            else:
                predictions.append("Neutral")

        return predictions


if __name__ == "__main__":
    # Example usage
    corpus = [
        "company reports strong profit growth",
        "market reacts negatively to earnings miss",
        "shares trade flat ahead of results"
    ]

    retriever = SimpleRetriever(corpus)
    rag = RAGSentimentStub(retriever)

    texts = [
        "company posts profit despite weak demand",
        "firm reports loss after earnings miss"
    ]

    labels = rag.predict(texts)

    for t, l in zip(texts, labels):
        print(f"Text: {t} -> Sentiment: {l}")
