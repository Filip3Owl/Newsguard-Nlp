# Newsguard NLP

> **Detecting fake news with progressively complex NLP models — from classical TF-IDF baselines to transformer fine-tuning.**

---

## The Story

Every day, millions of news articles circulate online. Some are carefully reported facts. Others are fabricated to manipulate, provoke, or mislead. The challenge: **can a machine learn to tell the difference — and explain its reasoning?**

This project tackles that question through a progressive pipeline of NLP models, each layer more sophisticated than the last. We start with the simplest possible approach and build up, always asking: *does the added complexity actually earn its cost?*

The dataset covers **~45,000 political news articles** from 2015–2017 — a period of intense media polarization surrounding the US election. Real articles come from Reuters; fake ones from catalogued disinformation websites.

Before training a single model, we discovered two critical **data leakage** traps hidden in the dataset — the kind that inflate metrics on paper while building models useless in the real world. Identifying and neutralizing them is the first act of the story.

---

## Pipeline

| Level | Approach | Status |
|-------|----------|--------|
| **1** | TF-IDF + Logistic Regression + LinearSVC | ✅ Done |
| 2 | Stylometric Features + XGBoost + SHAP | 🔜 Next |
| 3 | BiLSTM + GloVe / TextCNN | 🔜 Planned |
| 4 | DistilBERT / RoBERTa Fine-tuning | 🔜 Planned |
| 5 | Heterogeneous Ensemble | 🔜 Planned |

---

## Level 1 — TF-IDF + Linear Models

### The Approach

Text is converted into sparse high-dimensional vectors using **TF-IDF** (Term Frequency–Inverse Document Frequency), then fed into two linear classifiers:

- **Logistic Regression** — probabilistic model, well-calibrated output, interpretable coefficients
- **LinearSVC** — maximum-margin classifier, faster convergence, stronger regularization

Both models use `ngram_range=(1,2)` — capturing not just individual words but bigrams like *"fake news"*, *"breaking news"*, *"white house"* — with 100,000 features and sublinear TF scaling.

### Data Leakage — The Hidden Traps

Two leakage sources were identified and neutralized before training:

**1. Reuters Byline (99.82% signal)**
Real articles from Reuters always begin with `"CITY (Reuters) -"`. Any bag-of-words model trivially learns `reuters → real` — a shortcut that would fail completely on real-world data.

**2. `subject` Column (100% signal)**
The metadata categories are mutually exclusive between classes (`left-news`, `Government News` for fake vs. `politicsNews`, `worldnews` for real). A naive classifier using only this column achieves perfect accuracy — without reading a single word.

Both were removed. The quantified impact: keeping leakage inflates F1 by **+0.64 pp** — modest in absolute terms, but built on a lie.

### Results

Evaluated on a held-out test set of **6,735 articles** (15% of data), stratified split, no leakage.

| Model | Accuracy | F1 Macro | ROC-AUC | Train Time |
|-------|----------|----------|---------|------------|
| Logistic Regression | 0.9878 | **0.9878** | **0.9991** | 0.70s |
| LinearSVC | 0.9939 | **0.9939** | **0.9997** | 3.94s |

5-fold cross-validation on training set:

| Model | Accuracy CV | F1 Macro CV |
|-------|-------------|-------------|
| Logistic Regression | 0.9865 ± 0.0018 | 0.9865 ± 0.0018 |
| LinearSVC | 0.9941 ± 0.0010 | 0.9941 ± 0.0010 |

**LinearSVC wins** on every metric and shows lower variance across folds — more stable generalization. Both models train in under 4 seconds on CPU.

### Exploratory Analysis

![EDA Overview](results/level1/eda_overview.png)

Key findings from EDA:
- Classes are near-balanced: **52.3% fake / 47.7% real**
- Fake articles have a wider distribution of text lengths — more variance in style
- Fake titles are significantly longer on average (94 chars vs. 64 chars) — a hint of clickbait behavior

### Model Evaluation

**Logistic Regression**

![LR Evaluation](results/level1/lr_evaluation.png)

**LinearSVC**

![SVM Evaluation](results/level1/svm_evaluation.png)

### Model Comparison

![Model Comparison](results/level1/model_comparison.png)

Both ROC curves hug the top-left corner — AUC > 0.999. The zoomed view reveals LinearSVC maintains a slight edge at low false-positive rates, the most operationally critical region.

### What the Model Learned — Feature Interpretability

![Feature Importance](results/level1/feature_importance.png)

The coefficients tell a clear story:

**Real news** → formal, attributive, institutional language:
`said`, `reuters`, `president donald`, country names, verbs of attribution

**Fake news** → emotional, polarizing, sensationalist language:
charged political terms, informal framing, language of urgency and outrage

> Note: `reuters` still appears as a top feature for real news even after byline removal — because Reuters is cited by name throughout real article bodies (*"according to Reuters"*). This residual signal is unavoidable without aggressive content filtering.

---

## Project Structure

```
newsguard-nlp/
│
├── data/                    # Dataset CSVs (not versioned — download from Kaggle)
│   ├── Fake.csv
│   └── True.csv
│
├── notebooks/               # One notebook per pipeline level
│   └── 01_baseline_tfidf_linear_models.ipynb
│
├── results/                 # Saved figures, organized by level
│   └── level1/
│       ├── eda_overview.png
│       ├── lr_evaluation.png
│       ├── svm_evaluation.png
│       ├── model_comparison.png
│       └── feature_importance.png
│
├── models/                  # Serialized models (.pkl / .joblib)
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Getting Started

```bash
# 1. Clone and enter the repo
git clone https://github.com/Filip3Owl/Newsguard-Nlp.git
cd Newsguard-Nlp

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Register the Jupyter kernel
python3 -m ipykernel install --user --name "fakenews-venv" --display-name "Python (fakeNews)"

# 5. Download the dataset
# → https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset
# Place files at: data/Fake.csv and data/True.csv

# 6. Run the notebook
jupyter notebook notebooks/01_baseline_tfidf_linear_models.ipynb
```

Select kernel **"Python (fakeNews)"** when prompted.

---

## References

- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825–2830.
- Joachims, T. (1998). *Text Categorization with Support Vector Machines*. ECML.
- Salton, G. & Buckley, C. (1988). *Term-weighting approaches in automatic text retrieval*. Information Processing & Management.
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers*. NAACL.
- Reuters Institute Digital News Report (2023). University of Oxford.
