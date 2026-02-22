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

KNN_K = int(os.getenv("KNN_K", "5"))
KNN_CONFIDENCE_THRESHOLD = float(os.getenv("KNN_CONFIDENCE_THRESHOLD", "0.75"))
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "200"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

MODELS_DIR = ROOT / os.getenv("MODELS_DIR", "models")
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH") or None
LLAMA_N_CTX = int(os.getenv("LLAMA_N_CTX", "2048"))
LLAMA_N_GPU_LAYERS = int(os.getenv("LLAMA_N_GPU_LAYERS", "-1"))
LLAMA_HF_REPO = os.getenv("LLAMA_HF_REPO", "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF")
LLAMA_HF_FILENAME = os.getenv("LLAMA_HF_FILENAME", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
JUSTIFICATION_MAX_TOKENS = int(os.getenv("JUSTIFICATION_MAX_TOKENS", "200"))
LOG_DISPLAY = os.getenv("LOG_DISPLAY", "0") == "1"

_CHARS_PER_TOKEN = 3
_RESERVED_CLASSIFICATION_TOKENS = 1300
_RESERVED_JUSTIFICATION_TOKENS = 500
CLASSIFICATION_MAX_CHARS = int(
    os.getenv("CLASSIFICATION_MAX_CHARS")
    or ((LLAMA_N_CTX - _RESERVED_CLASSIFICATION_TOKENS - 64) * _CHARS_PER_TOKEN)
)
JUSTIFICATION_MAX_CHARS = int(
    os.getenv("JUSTIFICATION_MAX_CHARS")
    or ((LLAMA_N_CTX - _RESERVED_JUSTIFICATION_TOKENS - JUSTIFICATION_MAX_TOKENS) * _CHARS_PER_TOKEN)
)

for d in (DATA_RAW, DATA_PROCESSED, OUTPUTS, ARTIFACTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
