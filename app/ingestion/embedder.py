from sentence_transformers import SentenceTransformer
from app.config import settings

_model = SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]):
  return _model.encode(texts, normalize_embeddings=True).tolist()
