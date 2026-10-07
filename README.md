# 📚 BERT-Based Text Classification System

An end-to-end Natural Language Processing (NLP) pipeline designed to classify domain-specific text into multiple categories. This project leverages transfer learning by fine-tuning a pretrained **BERT (`bert-base-uncased`)** transformer model and benchmarks it against a traditional **TF-IDF + Logistic Regression** baseline.

## 📖 Project Overview

Traditional Bag-of-Words and TF-IDF models often struggle with context, polysemy, and long-range semantic dependencies in textual data. This project implements a modern deep learning pipeline using **BERT (Bidirectional Encoder Representations from Transformers)** to capture bidirectional contextual information and achieve superior classification performance across complex categories.

### Highlights:
* End-to-end modular pipeline: data preprocessing -> tokenization -> model training -> evaluation -> error analysis.
* Strict benchmarking between classical statistical NLP and deep transformer architectures.
* Dynamic padding and attention mask handling using Hugging Face's `BertTokenizer`.

---

## 🚀 Key Features

* **Data Preprocessing & Cleaning:** URL removal, contraction expansion, casing normalization, and punctuation handling.
* **Tokenization Pipeline:** Subword tokenization using `BertTokenizerFast`, including truncation and padding to fixed sequence lengths.
* **Baseline Benchmarking:** Implemented an n-gram `TfidfVectorizer` paired with a tuned `LogisticRegression` classifier for baseline comparisons.
* **Fine-Tuning Architecture:** Layer fine-tuning on `BertForSequenceClassification` using the `AdamW` optimizer with linear learning rate warmup schedules.
* **Hyperparameter Tuning:** Systematic experiments across batch sizes (16, 32), learning rates (2e-5, 3e-5, 5e-5), and sequence lengths (128, 256).
* **Comprehensive Evaluation:** Precision, Recall, Macro/Weighted F1-score, Confusion Matrix generation, and classification error profiling.

---

## 🏗️ System Architecture

```text
Raw Text Input
      │
      ▼
┌──────────────────────────────────────────────┐
│       Text Preprocessing & Cleaning          │
└──────────────────────┬───────────────────────┘
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
[ Baseline Path ]              [ Transformer Path ]
 TF-IDF Vectorizer               BertTokenizer (Tokens + Attention Masks)
       │                               │
 Logistic Regression             Pretrained BERT Encoder (bert-base-uncased)
       │                               │
       │                         Dropout + Linear Classification Head
       │                               │
       └───────────────┬───────────────┘
                       ▼
        Evaluation, Metrics & Error Analysis
        (Accuracy, F1-Score, Confusion Matrix)

🛠️ Tech Stack
Language: Python (3.8+)
Deep Learning Framework: PyTorch
NLP & Transformers: Hugging Face transformers, tokenizers
Machine Learning & Baselines: Scikit-learn
Data Manipulation: Pandas, NumPy
Visualization: Matplotlib, Seaborn

├── data/
│   ├── raw/                  # Raw dataset files (CSV/JSON)
│   └── processed/            # Cleaned train/val/test splits
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb
│   └── 02_error_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── dataset.py            # PyTorch Dataset & DataLoader implementation
│   ├── preprocess.py         # Text cleaning and preprocessing routines
│   ├── baseline.py           # TF-IDF + Logistic Regression baseline
│   ├── train.py              # BERT fine-tuning script
│   ├── evaluate.py           # Evaluation metrics & confusion matrix plot
│   └── predict.py            # Inference script for new text samples
├── saved_models/             # Checkpoints and serialized model weights
├── requirements.txt          # Python dependencies
└── README.md
