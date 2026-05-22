import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

class Settings:
    PROJECT_NAME: str = "Premium RAG Chatbot"
    DATABASE_NAME: str = "rag_store.db"
    DATABASE_PATH: Path = BASE_DIR / DATABASE_NAME
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    
    # API Keys & Endpoints (with fallback to environment variables)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    TOGETHER_API_KEY: str = os.getenv("TOGETHER_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # RAG Default Settings
    DEFAULT_CHUNK_SIZE: int = 1000
    DEFAULT_CHUNK_OVERLAP: int = 200
    DEFAULT_TOP_K: int = 4
    
    def __init__(self):
        # Create directories if they do not exist
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()
