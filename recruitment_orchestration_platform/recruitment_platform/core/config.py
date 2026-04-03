from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    # Groq LLM
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # App
    app_name: str = "Recruitment Orchestration Platform"
    app_version: str = "1.0.0"
    debug: bool = True
    log_level: str = "INFO"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # File uploads
    upload_dir: str = "./uploads"
    max_file_size_mb: int = 10

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
