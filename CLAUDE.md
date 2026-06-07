# CLAUDE.md — Newsguard NLP

Contexto do projeto para o Claude Code. Leia antes de qualquer tarefa.

---

## O Projeto

Detecção automática de fake news via pipeline progressiva de NLP. Classificação binária: dado o título + corpo de um artigo jornalístico, predizer se é **real (1)** ou **falso (0)**.

**Dataset**: Kaggle Fake and Real News Dataset — ~44.898 artigos de política americana (2015–2017). Arquivos em `data/Fake.csv` e `data/True.csv` (não versionados).

**Repositório**: https://github.com/Filip3Owl/Newsguard-Nlp

---

## Estrutura de Arquivos

```
fakeNewVsReal/
├── data/                  # CSVs brutos (não versionados — .gitignore)
│   ├── Fake.csv           # 23.481 artigos falsos, label=0
│   └── True.csv           # 21.417 artigos reais (Reuters), label=1
│
├── notebooks/             # Um notebook por nível da pipeline
│   ├── 01_baseline_tfidf_linear_models.ipynb
│   └── 02_stylometric_xgboost.ipynb
│
├── results/               # Figuras geradas pelos notebooks
│   ├── level1/            # Saídas do notebook 01
│   │   ├── eda_overview.png
│   │   ├── lr_evaluation.png
│   │   ├── svm_evaluation.png
│   │   ├── model_comparison.png
│   │   └── feature_importance.png
│   └── level2/            # Saídas do notebook 02
│       ├── eda_stylometric.png
│       ├── xgb_stylometric_evaluation.png
│       ├── xgb_hybrid_evaluation.png
│       ├── shap_beeswarm.png
│       ├── shap_importance.png
│       └── model_comparison.png
│
├── models/                # Modelos serializados (ainda vazio)
├── venv/                  # Ambiente virtual Python (não versionado)
├── .gitignore
├── requirements.txt
├── README.md
└── CLAUDE.md
```

---

## Ambiente Python

- **Venv**: `venv/` na raiz do projeto
- **Ativar**: `source venv/bin/activate`
- **Instalar**: `pip install -r requirements.txt`
- **Kernel Jupyter registrado**: `fakenews-venv` → `Python (fakeNews)`
  - Registrar: `python3 -m ipykernel install --user --name "fakenews-venv" --display-name "Python (fakeNews)"`
- **Executar notebook**: `jupyter notebook notebooks/<nome>.ipynb`

**Versões fixadas** (ver `requirements.txt`):
- Python 3.14 (venv local)
- pandas 3.0.3, numpy 2.4.6, scikit-learn 1.9.0
- matplotlib 3.10.9, seaborn 0.13.2, nltk 3.9.4
- xgboost 3.2.0, shap 0.52.0
- jupyter 1.1.1, ipykernel 7.2.0

---

## Pipeline de Modelagem (Progressiva)

| Nível | Notebook | Técnica | Status |
|-------|----------|---------|--------|
| 1 | `01_baseline_tfidf_linear_models.ipynb` | TF-IDF + Logistic Regression + LinearSVC | ✅ Completo |
| 2 | `02_stylometric_xgboost.ipynb` | Feature Engineering Estilométrico + XGBoost + SHAP | ✅ Completo |
| 3 | `03_bilstm_embeddings.ipynb` | BiLSTM + GloVe / TextCNN | 🔜 Próximo |
| 4 | `04_transformers_finetuning.ipynb` | DistilBERT / RoBERTa Fine-tuning | 🔜 Planejado |
| 5 | `05_ensemble.ipynb` | Ensemble Heterogêneo (stacking) | 🔜 Planejado |

---

## Convenções do Projeto

### Notebooks
- **Trabalhar exclusivamente com notebooks** — não criar scripts `.py` soltos
- Nomeação: `NN_descricao_nivel.ipynb` (ex: `02_stylometric_xgboost.ipynb`)
- Todo código deve ser **comentado** com `#`
- Células markdown devem explicar a **teoria aplicada** (fórmulas LaTeX incluídas)
- Figuras sempre salvas em `results/levelN/<nome>.png` via `plt.savefig()` antes de `plt.show()`
- Variável `RESULTS_DIR = Path('../results/levelN')` definida no início de cada notebook
- `RANDOM_STATE = 42` em todas as operações aleatórias

### Dados
- Features usadas: `title` + `text` (concatenados com separador `titleend`)
- Colunas **descartadas**: `subject`, `date` — ambas causam data leakage
- Labels: `0 = Fake`, `1 = Real`
- Split estratificado: **70% treino / 15% validação / 15% teste**
- TF-IDF sempre fitado **apenas no treino**, transform nos demais

### Data Leakage — Alertas Permanentes
Dois vazamentos críticos identificados neste dataset:
1. **Byline Reuters** — 99,82% dos artigos reais contêm `"CIDADE (Reuters) -"` no início. Remover via regex no pré-processamento.
2. **Coluna `subject`** — categorias mutuamente exclusivas entre classes. Nunca usar como feature.

### Avaliação
- Métrica principal: **F1 Macro**
- Métricas secundárias: Accuracy, F1 Weighted, Precision/Recall Macro, ROC-AUC
- Sempre rodar **validação cruzada 5-fold** antes da avaliação final no teste
- Sempre comparar o cenário **com leakage vs. sem leakage** para quantificar inflação

### Paleta de Cores (consistência visual)
```python
PALETTE = {
    'fake':    '#E74C3C',   # vermelho — classe 0
    'real':    '#2ECC71',   # verde    — classe 1
    'neutral': '#3498DB',   # azul     — uso geral
}
```

### Resultados — Nível 1 (benchmark)
| Modelo | F1 Macro | ROC-AUC | Treino |
|--------|----------|---------|--------|
| Logistic Regression | 0.9878 | 0.9991 | 0.70s |
| LinearSVC | 0.9939 | 0.9997 | 3.94s |

### Resultados — Nível 2
| Modelo | Features | F1 Macro | ROC-AUC | Treino |
|--------|----------|----------|---------|--------|
| XGBoost Estilométrico | 23 features de estilo | 0.9975 | 0.9999 | 1.5s |
| XGBoost Híbrido | TF-IDF (15k) + 23 estilométricas | **0.9990** | **1.0000** | 171s |

CV 5-fold (Modelo A): 0.9977 ± 0.0009

Features estilométricas (23): word_count, unique_word_ratio, avg_word_len, sent_count, avg_sent_len, std_sent_len, exclamation_count, question_count, ellipsis_count, comma_ratio, quote_count, number_ratio, url_count, caps_ratio, caps_word_ratio, title_word_count, title_char_count, title_avg_word_len, title_caps_ratio, title_caps_word_ratio, title_has_exclamation, title_has_question, title_word_ratio.

Top SHAP features: title_caps_ratio (6.85), title_char_count (1.50), quote_count (0.86), question_count (0.64), title_caps_word_ratio (0.63).

**Nota:** O modelo híbrido usa TF-IDF de 15k features (não 100k) pois XGBoost com sparse matrices de 100k features causa timeout na CV. Modelos lineares (SVM, LR) exploram espaços TF-IDF de 100k melhor.

Qualquer novo modelo deve superar **F1 Macro > 0.9990** para justificar complexidade adicional.

---

## Git

- Branch principal: `main`
- Remote: `https://github.com/Filip3Owl/Newsguard-Nlp.git`
- O que versionar: notebooks, results/, README.md, requirements.txt, .gitignore, CLAUDE.md
- O que **não** versionar: `venv/`, `data/*.csv`, `models/`, `__pycache__/`, `.ipynb_checkpoints/`
- Padrão de commit: `"Level N: <descrição breve da implementação>"`
