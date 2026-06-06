# Fake News Detection — Pipeline Progressiva de NLP

Projeto de detecção automática de fake news utilizando técnicas progressivas de NLP, partindo de modelos lineares clássicos até transformers de última geração.

## Problema

Dada uma notícia com `título` e `corpo de texto`, classificar se ela é **real** (1) ou **falsa** (0).

## Dataset

**Kaggle — Fake and Real News Dataset**  
→ [https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

| Atributo | Detalhe |
|----------|---------|
| Artigos falsos | 23.481 |
| Artigos reais | 21.417 |
| Total | 44.898 |
| Features | `title`, `text`, `subject`, `date` |
| Período | 2015–2017 |
| Fonte dos reais | Reuters News Agency |

> **Importante**: os arquivos CSV não estão incluídos no repositório por questão de tamanho.  
> Faça o download e coloque-os em `data/Fake.csv` e `data/True.csv`.

## Estrutura do Projeto

```
fakeNewVsReal/
│
├── data/                        # CSVs do dataset (não versionados)
│   ├── Fake.csv
│   └── True.csv
│
├── notebooks/                   # Notebooks por nível de complexidade
│   └── 01_baseline_tfidf_linear_models.ipynb   ← Nível 1 (atual)
│
├── models/                      # Modelos serializados (.pkl/.joblib)
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Pipeline de Modelagem (Progressiva)

| Nível | Técnica | Status |
|-------|---------|--------|
| **1** | TF-IDF + Regressão Logística + LinearSVC | ✅ Implementado |
| 2 | Feature Engineering Estilométrico + XGBoost | 🔜 Em breve |
| 3 | BiLSTM + GloVe / TextCNN | 🔜 Em breve |
| 4 | DistilBERT / RoBERTa Fine-tuning | 🔜 Em breve |
| 5 | Ensemble Heterogêneo | 🔜 Em breve |

## Como Executar

### 1. Clonar o repositório e criar o ambiente

```bash
git clone <repo-url>
cd fakeNewVsReal
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Baixar os dados

Faça o download do dataset no Kaggle e salve em:
```
data/Fake.csv
data/True.csv
```

### 4. Executar o notebook

```bash
jupyter notebook notebooks/01_baseline_tfidf_linear_models.ipynb
```

## Resultados — Nível 1 (TF-IDF + Modelos Lineares)

> Avaliação no conjunto de teste (15% = 6.735 amostras), **sem data leakage**.

| Modelo | F1 Macro | ROC-AUC | Tempo Treino |
|--------|----------|---------|--------------|
| Logistic Regression | **0.9878** | **0.9991** | 0.70s |
| LinearSVC | **0.9939** | **0.9997** | 3.94s |

Validação cruzada 5-fold:

| Modelo | Accuracy CV | F1 Macro CV |
|--------|-------------|-------------|
| Logistic Regression | 0.9865 ± 0.0018 | 0.9865 ± 0.0018 |
| LinearSVC | 0.9941 ± 0.0010 | 0.9941 ± 0.0010 |

## Alertas de Data Leakage

O dataset contém dois vazamentos críticos identificados e tratados:

1. **Byline Reuters**: 99,82% dos artigos reais contêm "Reuters" — sinal trivial para qualquer modelo bag-of-words.
2. **Coluna `subject`**: categorias completamente distintas entre fake e real — classificação perfeita sem entender o conteúdo.

Ambos são **removidos** antes do treinamento para garantir avaliação honesta e generalizável.

## Requisitos

- Python ≥ 3.9
- Ver `requirements.txt` para dependências completas

## Referências

- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL.
- Kim, Y. (2014). *Convolutional Neural Networks for Sentence Classification*. EMNLP.
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825–2830.
- Reuters Institute Digital News Report (2023). University of Oxford.
