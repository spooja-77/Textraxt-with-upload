import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Database path
DB_PATH = DATA_DIR / "app.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

# Model Configurations
VISION_MODEL = "qwen/qwen3.6-27b"
CHAT_MODEL = "llama-3.3-70b-versatile"
MIN_TEXT_CHARS = 50

# OCR prompt
OCR_PROMPT = (
    "You are an expert document transcriber. Extract ALL text and data from this "
    "document image — every field label, value, number, date, amount, address, "
    "invoice/bill ID, and name, exactly as written. Transcribe handwriting as "
    "accurately as possible. Preserve the document structure using 'label: value' "
    "lines and simple tables where appropriate. Output only the transcription, "
    "no commentary."
)

load_dotenv(BASE_DIR / ".env")


def get_groq_client(api_key: str | None = None) -> Groq | None:
    key = api_key or os.getenv("GROQ_API_KEY", "")
    return Groq(api_key=key) if key else None
