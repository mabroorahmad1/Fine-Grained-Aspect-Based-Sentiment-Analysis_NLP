# Fine-Grained Aspect-Based Sentiment Analysis (ABSA) on E-Commerce Reviews

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Natural Language Processing pipeline that decomposes raw consumer reviews into fine-grained opinion targets and evaluates sentiment polarity at the aspect level.

Unlike traditional document-level sentiment classification that flattens mixed opinions into vague aggregate scores, this project decouples customer feedback into specific entities (e.g., *fabric*, *fit*, *sizing*, *zipper*, *shipping*) and predicts sentiment individually for each extracted aspect.

---

## Repository Architecture

```text
Fine-Grained-Aspect-Based-Sentiment-Analysis_NLP/
├── Aspect based-Women Clothing Ecomerce-  sentiment-analysis-nlp.ipynb  # Primary notebook (EDA, preprocessing, ATE & ASC)
├── Aspect Based Sentiment Analysis                                      # Model pipeline scripts & experiment modules
├── NLP_PROJECT_SEM                                                      # SemEval benchmark adaptation & evaluation runs
├── README.md                                                            # Project documentation
└── requirements.txt                                                     # Dependency configuration
```

---

## Pipeline Overview

The project operates as a unified two-stage deep learning pipeline:

```
                  +-------------------------------+
                  |      Raw Customer Review      |
                  +---------------+---------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Stage 1: Aspect Term Extraction (ATE)                             |
| Sequence Tagging with BERT (IOB: B-ASP, I-ASP, O)                 |
+---------------------------------+---------------------------------+
                                  |
               [ "dress material", "zipper" ]
                                  |
                                  v
+-------------------------------------------------------------------+
| Stage 2: Aspect Sentiment Classification (ASC)                    |
| Sentence-Aspect Pair: [CLS] Sentence [SEP] Aspect [SEP]           |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Output: Structured Granular Sentiments                            |
| {"aspect": "dress material", "polarity": "Positive"}              |
| {"aspect": "zipper",         "polarity": "Negative"}              |
+-------------------------------------------------------------------+
```

### 1. Aspect Term Extraction (ATE)
- **Framing:** Token-level sequence labeling (NER-style).
- **Tagging Scheme:** `IOB` format (`B-ASP`, `I-ASP`, `O`).
- **Model:** Contextual transformer encoder (BERT / DistilBERT) with a token classification head.
- **Evaluation Metric:** Aspect-level Strict and Relaxed **F1-Score**.

### 2. Aspect Sentiment Classification (ASC)
- **Framing:** Sentence-pair classification.
- **Input Representation:** `[CLS] <Review Sentence> [SEP] <Target Aspect> [SEP]`
- **Classes:** `Positive`, `Neutral`, `Negative`.
- **Model:** Pre-trained BERT encoder with cross-attention between review context and extracted aspect target, pooled via classification head.
- **Evaluation Metric:** Classification **Accuracy** and **Macro F1-Score**.

---

## Datasets

| Dataset | Type | Size | Primary Role | Source |
| :--- | :--- | :--- | :--- | :--- |
| **Women's E-Commerce Clothing Reviews** | CSV | 23,486 reviews | Unsupervised/Domain ATE & Real-world Validation | [Kaggle](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) |
| **SemEval-2014 Task 4** | XML / JSON | 3,000+ labeled sentences | Supervised ATE & ASC Benchmarking | SemEval Workshop |
| **SemEval-2016 Task 5** | XML / JSON | Multi-domain reviews | Out-of-domain transfer & generalization testing | SemEval Workshop |

---

## Installation & Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/mabroorahmad1/Fine-Grained-Aspect-Based-Sentiment-Analysis_NLP.git
cd Fine-Grained-Aspect-Based-Sentiment-Analysis_NLP
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # or CPU equivalent
pip install transformers datasets scikit-learn pandas numpy spacy nltk seqeval matplotlib seaborn jupyterlab
```

---

## Usage & Execution

### Running the Notebooks
Launch JupyterLab or classic Notebook to inspect data preprocessing, training routines, and confusion matrices:
```bash
jupyter lab "Aspect based-Women Clothing Ecomerce-  sentiment-analysis-nlp.ipynb"
```

### Modular Inference Workflow
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification

# Example inference abstraction
sample_review = "The fabric of this dress is heavenly, but the sizing runs far too small."

# Step 1: Extract aspects via ATE model
extracted_aspects = ["fabric", "sizing"]

# Step 2: Classify sentiment per aspect via ASC model
# Input pair: [CLS] review [SEP] aspect [SEP]
results = [
    {"aspect": "fabric", "sentiment": "Positive", "confidence": 0.96},
    {"aspect": "sizing", "sentiment": "Negative", "confidence": 0.91}
]

print(results)
```

---

## Results & Benchmark Highlights

- **Aspect Extraction (ATE):** Demonstrates strong boundary detection for single and multi-word product attributes using WordPiece token alignment.
- **Sentiment Classification (ASC):** Resolves intra-sentence sentiment conflicts where standard sentence-level classifiers fail due to mixed ratings.

---

## Author & Academic Information

- **Author:** Mabroor Ahmad Mansoor
- **Domain:** Natural Language Processing / Deep Learning
- **GitHub:** [@mabroorahmad1](https://github.com/mabroorahmad1)
- **Portfolio:** (https://mabroorahmad.vercel.app)
