# Dataset Description and Ethics Compliance

## Overview

This project uses a **composite public dataset** constructed exclusively from publicly available sources, as described in Section 3.2 (Dataset Collection) of the thesis *“Fortified Financial Sentiment Analysis Using AI for APT Decisioning.”*

The dataset supports three-class financial sentiment analysis:
**Positive, Neutral, Negative**.

---

## Data Sources

### Financial News Headlines (Public)
- Publicly accessible financial news outlets (e.g., Reuters, Bloomberg, CNBC)
- Headlines used instead of full articles to capture market-relevant sentiment

**Redistribution note:**  
Raw headlines are not redistributed due to publisher copyright and terms of service.

---

### Finance-Related Social Media Posts (Public)
- Public posts related to financial markets (e.g., cashtag-based tweets)
- No private, protected, or restricted accounts accessed
- No user identifiers retained

**Redistribution note:**  
Raw social media text is not redistributed to comply with platform terms of service.

---

### Financial PhraseBank (Academic Dataset)
- Malo et al. (2014), *Financial PhraseBank*
- Expert-annotated financial sentences labeled as Positive, Neutral, or Negative

**Access:**  
The dataset must be obtained directly from the original authors or repository.

---

## Dataset Composition

- **Total size:** ~5,000 instances
- **Language:** English
- **Classes:** Positive / Neutral / Negative
- **Class distribution:** Approximately balanced

---

## Annotation Process

- Financial PhraseBank labels are used as provided
- Additional texts manually annotated using finance-specific guidelines
- Annotation focuses on **financial impact**, not emotional tone

---

## Preprocessing Summary

- Removal of HTML tags and URLs
- Lowercasing
- Preservation of finance-relevant terms and numerical information
- Optional anonymization of monetary values

---

## Reproducibility

Raw datasets are not included in this repository due to redistribution constraints.

Reproducibility is ensured through:
- Data collection scripts
- Preprocessing pipeline
- Annotation schema
- Model training and evaluation code

---

## Ethical Compliance

- All data sources are public
- No personally identifiable information (PII) is used
- No proprietary or confidential data is involved
- **No NDA is required**

This dataset complies with academic research ethics and platform usage policies.
