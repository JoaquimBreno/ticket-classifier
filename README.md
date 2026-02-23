# IT Ticket Classifier — DHAUZ Challenge

Fluxo de automação que, dado o texto de um ticket, retorna a **classe** e uma **justificativa curta** da classificação.

**Saída:** `{"classe": "...", "justificativa": "..."}`

## Arquitetura

- **Orquestração:** LangGraph (StateGraph) — preprocess → embed → knn_classify → se confiança ≥ limiar → generate_justification (só justificativa); senão → agent_classify_justify (classe + justificativa em uma chamada) → log_and_return
- **Classificação:** Embeddings (sentence-transformers) + FAISS + KNN; fallback com **LLM** (modular: local via llama-cpp-python ou Groq) quando a confiança do KNN é baixa. Foi escolhido **KNN com all-MiniLM-L6-v2** por resultado satisfatório e melhor **explicabilidade** (não é caixa preta: a decisão se apoia nos k vizinhos mais próximos). As análises que fundamentam essa escolha estão em **[labs/architecture-comparison.ipynb](labs/architecture-comparison.ipynb)**: comparação de arquiteturas (RNN, LSTM, GRU, BiLSTM, BiGRU, CNN+BiGRU e KNN com all-MiniLM-L6-v2), tabela de métricas (accuracy, F1 macro/weighted), benchmark de tempo de inferência e seção de discussão em que se conclui que o KNN é o mais transparente (os vizinhos *são* a explicação), enquanto modelos neurais exigem camadas adicionais (ex.: saliency) para interpretabilidade.
- **LLM e justificativa:** Pydantic (schema) + backends plugáveis: **llama_cpp** (local, Hugging Face / `LLAMA_MODEL_PATH`) ou **Groq** (API). Saída validada (classe + justificativa em 1–3 frases em português). Troca via `LLM_BACKEND` no `.env`; threshold de confiança via `KNN_CONFIDENCE_THRESHOLD`.

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

1. Configure variáveis de ambiente (copie `.env.example` para `.env`):

```bash
cp .env.example .env
```

Edite `.env` e defina:

- `KAGGLE_API_TOKEN` — token da API do Kaggle (usado pelo [kagglehub](https://github.com/Kaggle/kagglehub) para baixar o dataset). Obtenha em [Kaggle – Settings – API](https://www.kaggle.com/settings), clique em "Generate New Token" e copie o valor para o `.env`.

Opcionais: `LABEL_COLUMN`, `TEXT_COLUMNS`, `KNN_K`, `KNN_CONFIDENCE_THRESHOLD`, `SAMPLE_SIZE`, `EMBEDDING_MODEL`, `JUSTIFICATION_MAX_TOKENS`. Para o LLM local: se o modelo for baixado do Hugging Face na primeira execução, defina `HF_TOKEN` (token em [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)); ou baixe o `.gguf` manualmente e defina `LLAMA_MODEL_PATH`. Opcionais: `LLAMA_N_CTX`, `LLAMA_N_GPU_LAYERS` (ex.: `-1` para Metal no Mac).

**LLM modular:** é possível trocar o backend do LLM (fallback quando a confiança do KNN é baixa). No `.env`, defina `LLM_BACKEND=groq` para usar a API Groq (requer `GROQ_API_KEY`) ou `LLM_BACKEND=llama_cpp` para o modelo local (padrão). Opcional: `GROQ_MODEL` (ex.: `llama-3.3-70b-versatile`).

**Threshold de confiança:** o limiar que decide entre “usar só KNN + justificativa” e “chamar o LLM para classificar e justificar” é controlado por `KNN_CONFIDENCE_THRESHOLD` no `.env` (padrão `0.45`). Valores mais altos disparam mais o LLM; `0` força 100% KNN (sem fallback).

1. Baixe o dataset do Kaggle (obrigatório):
  - Crie uma conta em [Kaggle](https://www.kaggle.com) e aceite as regras do dataset [IT Service Ticket Classification](https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset).
  - Com `KAGGLE_API_TOKEN` já definido no `.env`, no terminal (ambiente ativado) rode:

```bash
conda activate ticket-classifier
python -c "from src.prep.loader import download_from_kaggle; download_from_kaggle()"
```

   O CSV será baixado em `data/raw/`. Sem esse passo o notebook e o pipeline não rodam.

## Como rodar

**Notebook (pipeline completo):**

```bash
jupyter notebook notebook.ipynb
```

Execute *Run All* para: baixar/carregar dados, montar o vector store, rodar inferência na amostra e gerar métricas.

**Interface Gradio (classificar um ticket na UI):**

```bash
python app.py
```

Abre em `http://127.0.0.1:7860`. Requer o vector store já construído (rode o notebook uma vez antes).

O notebook:

1. Carrega o dataset e identifica a coluna de rótulo (e colunas de texto)
2. Faz amostragem estratificada (200 tickets ou o tamanho da base)
3. Monta o índice de embeddings (FAISS) e o RAG
4. Constrói o grafo LangGraph e roda inferência em exemplos
5. Processa a amostra e salva `outputs/results_sample.jsonl`
6. Calcula métricas (accuracy, F1 macro/weighted, relatório por classe) e salva `outputs/metrics_report.json`

## Estrutura do repositório

```
ticket-classifier/
├── README.md
├── requirements.txt
├── environment.yml
├── .env.example
├── config.py
├── app.py                    # UI Gradio (python app.py)
├── notebook.ipynb            # Pipeline completo: dados → RAG → inferência → métricas
├── data/
│   ├── raw/                  # CSV do Kaggle (download via notebook ou loader)
│   └── processed/            # dataset_with_id.csv, sample_200.csv
├── src/
│   ├── prep/                 # loader (download, load_dataset, document_text, stable_id), sampler (stratified_sample)
│   ├── rag/                  # Embedder, VectorStore (FAISS + embeddings)
│   ├── classification/       # KNNClassifier
│   ├── llm_local/            # agent_classify_and_justify, agent_justify; schemas (Pydantic); backend (get_llm_backend); backends/llama_cpp
│   ├── graph/                # state (PipelineState, PipelineResult), nodes (nós do grafo), pipeline (build_pipeline, run_pipeline)
│   ├── metrics/              # compute_metrics, save_metrics_report
│   └── logging_utils/       # log_usage, log_inference, log_result
├── outputs/
│   ├── artifacts/            # vector store (index.faiss, labels.json, texts.json, ids.json, classes.json, manifest.json)
│   ├── results_sample.jsonl
│   └── metrics_report.json
├── models/                   # modelo GGUF (baixado na 1ª execução se não houver LLAMA_MODEL_PATH)
├── labs/                     # notebooks de experimentação (ex.: architecture-comparison.ipynb)
```

**Apple Silicon (M1/M2):** para usar Metal, instale o llama-cpp-python com suporte a GPU:

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
```

## Reprodutibilidade

- Seed fixo em `config.SEED` (42); usado na amostragem.
- Amostra de 200 (ou menor se a base for menor) salva em `data/processed/sample_200.csv`.

