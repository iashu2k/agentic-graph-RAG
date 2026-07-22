from pydantic_settings import BaseSettings


class Settings(BaseSettings):
  database_url: str
  groq_api_key: str
  embedding_model: str = "BAAI/bge-small-en-v1.5"
  embedding_dim: int = 384
  docs_dir: str = "data/raw/sec-10-q/docs"

  neo4j_uri: str
  neo4j_username: str
  neo4j_password: str
  neo4j_database: str
  aura_instanceid: str
  aura_instancename: str

  class Config:
    env_file = ".env"


settings = Settings()
