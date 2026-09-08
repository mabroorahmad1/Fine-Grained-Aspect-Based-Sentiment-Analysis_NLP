# Fine-Grained Aspect-Based Sentiment Analysis (ABSA)

End-to-end Aspect-Based Sentiment Analysis pipeline using pre-trained BERT architectures on customer reviews. The project decomposes reviews into granular target aspects (ATE) and determines the exact sentiment polarity toward each aspect (ASC).

---

## Architecture Overview

1. **Aspect Term Extraction (ATE):** Sequence labeling using BERT with an IOB token tagging scheme (`B-ASP`, `I-ASP`, `O`).
2. **Aspect Sentiment Classification (ASC):** Aspect-sentence pair classification (`[CLS] sentence [SEP] aspect [SEP]`) targeting 3 classes: Positive, Neutral, Negative.

---

## Datasets

- **Primary Exploration:** [Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) (Kaggle)
- **Supervised Benchmarks:** SemEval-2014 Task 4 & SemEval-2016 Task 5

---

## Evaluation Metrics

- **ATE:** Strict & Relaxed F1-Score
- **ASC:** Accuracy & Macro F1-Score

---

## Quickstart

### Prerequisites
- Python 3.10+
- PyTorch / Hugging Face Transformers

### Installation

```bash
git clone [https://github.com/](https://github.com/)<your-username>/aspect-based-sentiment-analysis.git
cd aspect-based-sentiment-analysis
pip install -r requirements.txt
