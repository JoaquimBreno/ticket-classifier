# IT Ticket Classifier — DHAUZ Challenge

Fluxo de automação que, dado o texto de um ticket, retorna a **classe** e uma **justificativa curta** da classificação.

**Saída:** `{"classe": "...", "justificativa": "..."}`

## Arquitetura

- **Orquestração:** LangGraph (StateGraph) — preprocess → embed → knn_classify → se confiança ≥ limiar → generate_justification (só justificativa); senão → llm_classify (classe + justificativa em uma chamada) → log_and_return
- **Classificação:** Embeddings (sentence-transformers) + FAISS + KNN; fallback com **LLM local** (llama-cpp-python) quando a confiança do KNN é baixa. Foi escolhido **KNN com all-MiniLM-L6-v2** por resultado satisfatório e melhor **explicabilidade** (não é caixa preta: a decisão se apoia nos k vizinhos mais próximos). As análises que fundamentam essa escolha estão em **[labs/architecture-comparison.ipynb](labs/architecture-comparison.ipynb)**: comparação de arquiteturas (RNN, LSTM, GRU, BiLSTM, BiGRU, CNN+BiGRU e KNN com all-MiniLM-L6-v2), tabela de métricas (accuracy, F1 macro/weighted), benchmark de tempo de inferência e seção de discussão em que se conclui que o KNN é o mais transparente (os vizinhos *são* a explicação), enquanto modelos neurais exigem camadas adicionais (ex.: saliency) para interpretabilidade.
- **LLM e justificativa:** Pydantic (schema) + **llama-cpp-python** (inferência local). Sem API externa: modelo baixado do Hugging Face ou via `LLAMA_MODEL_PATH`; saída validada (classe + justificativa em 1–3 frases em português)

## Requisitos

- Python 3.10+
- Dataset: [Kaggle – IT Service Ticket Classification](https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset) (obrigatório; deve ser baixado via kagglehub/API do Kaggle)

## Setup

1. Clone o repositório e crie o ambiente com **Conda**:

```bash
cd ticket-classifier
conda env create -f environment.yml
conda activate ticket-classifier
```

Para atualizar o ambiente depois de mudar `requirements.txt`:

```bash
conda activate ticket-classifier
pip install -r requirements.txt
```

2. Configure variáveis de ambiente (copie `.env.example` para `.env`):

```bash
cp .env.example .env
```

Edite `.env` e defina:

- `KAGGLE_API_TOKEN` — token da API do Kaggle (usado pelo [kagglehub](https://github.com/Kaggle/kagglehub) para baixar o dataset). Obtenha em [Kaggle – Settings – API](https://www.kaggle.com/settings), clique em "Generate New Token" e copie o valor para o `.env`.

Opcionais: `LABEL_COLUMN`, `TEXT_COLUMNS`, `KNN_K`, `KNN_CONFIDENCE_THRESHOLD`, `SAMPLE_SIZE`, `EMBEDDING_MODEL`, `JUSTIFICATION_MAX_TOKENS`. Para o LLM local: se o modelo for baixado do Hugging Face na primeira execução, defina `HF_TOKEN` (token em [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)); ou baixe o `.gguf` manualmente e defina `LLAMA_MODEL_PATH`. Opcionais: `LLAMA_N_CTX`, `LLAMA_N_GPU_LAYERS` (ex.: `-1` para Metal no Mac).

1. Baixe o dataset do Kaggle (obrigatório):

   - Crie uma conta em [Kaggle](https://www.kaggle.com) e aceite as regras do dataset [IT Service Ticket Classification](https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset).
   - Com `KAGGLE_API_TOKEN` já definido no `.env`, no terminal (ambiente ativado) rode:

```bash
conda activate ticket-classifier
python -c "from src.prep.loader import download_from_kaggle; download_from_kaggle()"
```

   O CSV será baixado em `data/raw/`. Sem esse passo o notebook e o pipeline não rodam.

## Como rodar

Execute o notebook de ponta a ponta:

```bash
jupyter notebook notebook.ipynb
```

Ou, no Jupyter: *Run All*.

O notebook:

1. Carrega o dataset e identifica a coluna de rótulo (e colunas de texto)
2. Faz amostragem estratificada (200 tickets ou o tamanho da base)
3. Monta o índice de embeddings (FAISS) e o RAG
4. Constrói o grafo LangGraph e roda inferência em exemplos
5. Processa a amostra e salva `outputs/results_sample.jsonl`
6. Calcula métricas (accuracy, F1 macro/weighted, relatório por classe) e salva `outputs/metrics_report.json`

## Estrutura do projeto

```
ticket-classifier/
├── README.md
├── requirements.txt
├── environment.yml      # ambiente Conda (conda env create -f environment.yml)
├── .env.example
├── config.py
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── prep/           (loader, sampler)
│   ├── rag/             (embedder, vector_store)
│   ├── classification/  (knn_classifier)
│   ├── llm_local/       (Pydantic + llama-cpp-python: agent_classifier, agent_justify)
│   ├── graph/           (state, pipeline LangGraph)
│   ├── justification/   (wrapper que chama llm_local)
│   ├── metrics/         (evaluator)
│   └── logging_utils/
├── outputs/
├── models/              # modelo GGUF (criado no primeiro run se não houver LLAMA_MODEL_PATH)
└── notebook.ipynb
```

**Apple Silicon (M1/M2):** para usar Metal, instale o llama-cpp-python com suporte a GPU:

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
```

## Reprodutibilidade

- Seed fixo em `config.SEED` (42); usado na amostragem.
- Amostra de 200 (ou menor se a base for menor) salva em `data/processed/sample_200.csv`.

