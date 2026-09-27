"""Application configuration loaded from environment variables (.env)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- LLM provider ---
    llm_provider: str = "anthropic"  # "anthropic" | "openai" | "ollama"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    # --- Database ---
    db_url: str = "sqlite:///./agent.db"

    # --- Auth ---
    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    access_token_expires_hours: int = 24

    # --- Memory / Vector DB ---
    chroma_path: str = "./chroma_db"
    memory_top_k: int = 3

    # --- Agent ---
    max_iterations: int = 10
    default_model: str = "claude-sonnet-4-6"
    max_tokens: int = 1024

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()