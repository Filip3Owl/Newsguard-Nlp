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
│   └── 01_baseline_tfidf_linear_models.ipynb
│
├── results/               # Figuras geradas pelos notebooks
│   └── level1/            # Saídas do notebook 01
│       ├── eda_overview.png
│       ├── lr_evaluation.png
│       ├── svm_evaluation.png
│       ├── model_comparison.png
│       └── feature_importance.png
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
- jupyter 1.1.1, ipykernel 7.2.0

---

## Pipeline de Modelagem (Progressiva)

| Nível | Notebook | Técnica | Status |
|-------|----------|---------|--------|
| 1 | `01_baseline_tfidf_linear_models.ipynb` | TF-IDF + Logistic Regression + LinearSVC | ✅ Completo |
| 2 | `02_stylometric_xgboost.ipynb` | Feature Engineering Estilométrico + XGBoost + SHAP | 🔜 Próximo |
| 3 | `03_bilstm_embeddings.ipynb` | BiLSTM + GloVe / TextCNN | 🔜 Planejado |
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

Qualquer novo modelo deve superar **F1 Macro > 0.9939** para justificar complexidade adicional.

---

## Git

- Branch principal: `main`
- Remote: `https://github.com/Filip3Owl/Newsguard-Nlp.git`
- O que versionar: notebooks, results/, README.md, requirements.txt, .gitignore, CLAUDE.md
- O que **não** versionar: `venv/`, `data/*.csv`, `models/`, `__pycache__/`, `.ipynb_checkpoints/`
- Padrão de commit: `"Level N: <descrição breve da implementação>"`
