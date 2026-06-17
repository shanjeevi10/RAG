from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "Agentic PDF Intelligence Platform"
    DEBUG: bool = False
    
    # LLM Settings
    LLM_PROVIDER: str = "ollama"  # "ollama" or "anthropic"
    ANTHROPIC_API_KEY: Optional[str] = None
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama2"  # Can be "mistral", "neural-chat", etc.
    LLM_MODEL: str = "llama2"  # Fallback for backward compatibility
    
    # FAISS Vector Store Settings
    FAISS_INDEX_PATH: str = "./faiss_index"
    
    # Embedding Settings
    DENSE_EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # File Upload Settings
    UPLOAD_DIR: str = "./uploads"

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
