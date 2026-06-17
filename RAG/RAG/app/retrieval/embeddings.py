from sentence_transformers import SentenceTransformer, CrossEncoder
from app.config import settings

class EmbeddingService:
    def __init__(self):
        # Load BGE-Large for dense embeddings
        self.dense_model = SentenceTransformer(settings.DENSE_EMBEDDING_MODEL)
        
        # Load Cross-Encoder for reranking
        self.rerank_model = CrossEncoder(settings.CROSS_ENCODER_MODEL)
        
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate dense embeddings for documents/chunks."""
        return self.dense_model.encode(texts).tolist()
        
    def embed_query(self, query: str) -> list[float]:
        """
        Generate dense embedding for a query.
        BGE-Large requires a specific prefix for retrieval queries.
        """
        prefixed_query = f"Represent this sentence for searching relevant passages: {query}"
        return self.dense_model.encode([prefixed_query])[0].tolist()
        
    def rerank(self, query: str, passages: list[str]) -> list[float]:
        """
        Rerank a list of passages against a query using the cross-encoder.
        Returns a list of scores corresponding to the passages.
        """
        pairs = [[query, passage] for passage in passages]
        scores = self.rerank_model.predict(pairs)
        return scores.tolist()

# Singleton instance
embedding_service = EmbeddingService()
