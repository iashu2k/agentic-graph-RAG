import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
  database_url: str
  groq_api_key: str
  openrouter_api_key: str
  embedding_model: str = "BAAI/bge-small-en-v1.5"
  embedding_dim: int = 384
  docs_dir: str = "data/raw/sec-10-q/docs"

  neo4j_uri: str
  neo4j_username: str
  neo4j_password: str
  neo4j_database: str
  aura_instanceid: str
  aura_instancename: str

  langfuse_public_key: str
  langfuse_secret_key: str
  langfuse_host: str = "https://cloud.langfuse.com"

  model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()

# Langfuse's get_client() reads raw os.environ, not this Settings object —
# bridge these three explicitly so the SDK picks them up regardless of
# which module imports app.config first.
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)
