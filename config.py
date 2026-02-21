import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED = 42
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
ARTIFACTS_DIR = OUTPUTS / "artifacts"
RAW_CSV_FILENAME = "all_tickets_processed_improved_v3.csv"

LABEL_COLUMN = os.getenv("LABEL_COLUMN", "Topic_group")
TEXT_COLUMNS = [c.strip() for c in os.getenv("TEXT_COLUMNS", "Document").split(",")]

KNN_K = int(os.getenv("KNN_K", "5"))
KNN_CONFIDENCE_THRESHOLD = float(os.getenv("KNN_CONFIDENCE_THRESHOLD", "0.75"))
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "200"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")
JUSTIFICATION_MAX_TOKENS = int(os.getenv("JUSTIFICATION_MAX_TOKENS", "200"))

for d in (DATA_RAW, DATA_PROCESSED, OUTPUTS, ARTIFACTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
