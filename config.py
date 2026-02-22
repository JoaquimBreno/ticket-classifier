import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED = 42
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
ARTIFACTS_DIR = OUTPUTS / "artifacts"
RAW_CSV_FILENAME = "all_tickets_processed_improved_v3.csv"
PROCESSED_CSV_FILENAME = "dataset_with_id.csv"

LABEL_COLUMN = os.getenv("LABEL_COLUMN", "Topic_group")
TEXT_COLUMNS = [c.strip() for c in os.getenv("TEXT_COLUMNS", "Document").split(",")]

KNN_K = int(os.getenv("KNN_K", "7"))
KNN_CONFIDENCE_THRESHOLD = float(os.getenv("KNN_CONFIDENCE_THRESHOLD", "0.45"))
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "200"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

MODELS_DIR = ROOT / os.getenv("MODELS_DIR", "models")
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH") or None
LLAMA_N_CTX = int(os.getenv("LLAMA_N_CTX", "2048"))
LLAMA_N_GPU_LAYERS = int(os.getenv("LLAMA_N_GPU_LAYERS", "-1"))
LLAMA_N_BATCH = int(os.getenv("LLAMA_N_BATCH", "512"))
LLAMA_N_THREADS = int(os.getenv("LLAMA_N_THREADS", "-1"))
LLAMA_HF_REPO = os.getenv("LLAMA_HF_REPO", "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF")
LLAMA_HF_FILENAME = os.getenv("LLAMA_HF_FILENAME", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
JUSTIFICATION_MAX_TOKENS = int(os.getenv("JUSTIFICATION_MAX_TOKENS", "256"))
JUSTIFICATION_MAX_LENGTH = int(os.getenv("JUSTIFICATION_MAX_LENGTH", "2500"))

CLASSIFICATION_MAX_CHARS = int(os.getenv("CLASSIFICATION_MAX_CHARS", "1000"))
JUSTIFICATION_MAX_CHARS = int(os.getenv("JUSTIFICATION_MAX_CHARS", "13000"))

for d in (DATA_RAW, DATA_PROCESSED, OUTPUTS, ARTIFACTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
