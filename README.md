# Newsguard NLP

> **Detecção automática de fake news com modelos de NLP progressivamente complexos — do TF-IDF clássico ao fine-tuning de transformers.**

---

## A História

Todos os dias, milhões de artigos jornalísticos circulam online. Alguns são fatos cuidadosamente apurados. Outros são fabricados para manipular, provocar ou enganar. O desafio: **uma máquina consegue aprender a diferença — e explicar o seu raciocínio?**

Este projeto enfrenta essa questão através de uma pipeline progressiva de modelos de NLP, onde cada camada é mais sofisticada que a anterior. Começamos pela abordagem mais simples possível e evoluímos, sempre perguntando: *a complexidade adicional realmente se justifica?*

O dataset cobre **~45.000 artigos de política americana** entre 2015 e 2017 — um período de intensa polarização midiática em torno das eleições americanas. Artigos reais vêm da Reuters; os falsos, de sites de desinformação catalogados.

Antes de treinar qualquer modelo, identificamos duas armadilhas críticas de **data leakage** escondidas no dataset — o tipo que infla métricas no papel enquanto produz modelos inúteis na prática. Identificá-las e neutralizá-las é o primeiro ato da história.

---

## Pipeline

| Nível | Abordagem | Status |
|-------|-----------|--------|
| **1** | TF-IDF + Regressão Logística + LinearSVC | ✅ Concluído |
| 2 | Features Estilométricas + XGBoost + SHAP | 🔜 Próximo |
| 3 | BiLSTM + GloVe / TextCNN | 🔜 Planejado |
| 4 | DistilBERT / RoBERTa Fine-tuning | 🔜 Planejado |
| 5 | Ensemble Heterogêneo | 🔜 Planejado |

---

## Nível 1 — TF-IDF + Modelos Lineares

### A Abordagem

O texto é convertido em vetores esparsos de alta dimensão via **TF-IDF** (Term Frequency–Inverse Document Frequency) e alimentado em dois classificadores lineares:

- **Regressão Logística** — modelo probabilístico com saída calibrada e coeficientes interpretáveis
- **LinearSVC** — classificador de máxima margem, convergência rápida, regularização mais forte

Ambos usam `ngram_range=(1,2)` — capturando não apenas palavras individuais, mas bigramas como *"fake news"*, *"breaking news"*, *"white house"* — com 100.000 features e escala logarítmica do TF.

### Data Leakage — As Armadilhas Ocultas

Duas fontes de vazamento foram identificadas e neutralizadas antes do treinamento:

**1. Byline Reuters (sinal de 99,82%)**
Artigos reais da Reuters sempre começam com `"CIDADE (Reuters) -"`. Qualquer modelo bag-of-words aprende trivialmente `reuters → real` — um atalho que falharia completamente em dados do mundo real.

**2. Coluna `subject` (sinal de 100%)**
As categorias de metadados são mutuamente exclusivas entre as classes (`left-news`, `Government News` nos fakes vs. `politicsNews`, `worldnews` nos reais). Um classificador ingênuo usando apenas essa coluna alcança acurácia perfeita — sem ler uma única palavra.

Ambas foram removidas. O impacto quantificado: manter o leakage infla o F1 em **+0,64 pp** — modesto em termos absolutos, mas construído sobre uma mentira.

### Resultados

Avaliação no conjunto de teste com **6.735 artigos** (15% dos dados), split estratificado, sem leakage.

| Modelo | Accuracy | F1 Macro | ROC-AUC | Tempo de Treino |
|--------|----------|----------|---------|-----------------|
| Regressão Logística | 0,9878 | **0,9878** | **0,9991** | 0,70s |
| LinearSVC | 0,9939 | **0,9939** | **0,9997** | 3,94s |

Validação cruzada 5-fold no conjunto de treino:

| Modelo | Accuracy CV | F1 Macro CV |
|--------|-------------|-------------|
| Regressão Logística | 0,9865 ± 0,0018 | 0,9865 ± 0,0018 |
| LinearSVC | 0,9941 ± 0,0010 | 0,9941 ± 0,0010 |

**LinearSVC vence** em todas as métricas e apresenta menor variância entre os folds — generalização mais estável. Ambos os modelos treinam em menos de 4 segundos na CPU.

### Análise Exploratória

![EDA Overview](results/level1/eda_overview.png)

Principais achados da EDA:
- Classes quase balanceadas: **52,3% fake / 47,7% real**
- Artigos falsos têm distribuição de comprimento mais ampla — maior variância de estilo
- Títulos de fake news são significativamente mais longos em média (94 chars vs. 64 chars) — indício de comportamento clickbait

### Avaliação dos Modelos

**Regressão Logística**

![LR Evaluation](results/level1/lr_evaluation.png)

**LinearSVC**

![SVM Evaluation](results/level1/svm_evaluation.png)

### Comparação dos Modelos

![Model Comparison](results/level1/model_comparison.png)

Ambas as curvas ROC abraçam o canto superior esquerdo — AUC > 0,999. A visão com zoom revela que o LinearSVC mantém uma leve vantagem em taxas baixas de falsos positivos, a região operacionalmente mais crítica.

### O que o Modelo Aprendeu — Interpretabilidade

![Feature Importance](results/level1/feature_importance.png)

Os coeficientes contam uma história clara:

**Notícias reais** → linguagem formal, atributiva, institucional:
`said`, `reuters`, `president donald`, nomes de países, verbos de atribuição

**Notícias falsas** → linguagem emocional, polarizadora, sensacionalista:
termos políticos carregados, enquadramento informal, linguagem de urgência e indignação

> Nota: `reuters` ainda aparece como top feature para notícias reais mesmo após a remoção da byline — porque a Reuters é citada pelo nome ao longo do corpo dos artigos reais (*"according to Reuters"*). Esse sinal residual é inevitável sem filtragem agressiva de conteúdo.

---

## Estrutura do Projeto

```
newsguard-nlp/
│
├── data/                    # CSVs do dataset (não versionados — baixar do Kaggle)
│   ├── Fake.csv
│   └── True.csv
│
├── notebooks/               # Um notebook por nível da pipeline
│   └── 01_baseline_tfidf_linear_models.ipynb
│
├── results/                 # Figuras salvas, organizadas por nível
│   └── level1/
│       ├── eda_overview.png
│       ├── lr_evaluation.png
│       ├── svm_evaluation.png
│       ├── model_comparison.png
│       └── feature_importance.png
│
├── models/                  # Modelos serializados (.pkl / .joblib)
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Como Executar

```bash
# 1. Clonar o repositório
git clone https://github.com/Filip3Owl/Newsguard-Nlp.git
cd Newsguard-Nlp

# 2. Criar o ambiente virtual
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Registrar o kernel Jupyter
python3 -m ipykernel install --user --name "fakenews-venv" --display-name "Python (fakeNews)"

# 5. Baixar o dataset
# → https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset
# Salvar em: data/Fake.csv e data/True.csv

# 6. Abrir o notebook
jupyter notebook notebooks/01_baseline_tfidf_linear_models.ipynb
```

Ao abrir, selecionar o kernel **"Python (fakeNews)"**.

---

## Referências

- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825–2830.
- Joachims, T. (1998). *Text Categorization with Support Vector Machines*. ECML.
- Salton, G. & Buckley, C. (1988). *Term-weighting approaches in automatic text retrieval*. Information Processing & Management.
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers*. NAACL.
- Reuters Institute Digital News Report (2023). University of Oxford.
