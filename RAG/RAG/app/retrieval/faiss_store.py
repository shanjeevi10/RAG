import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple

class FAISSStore:
    """FAISS-based vector store for efficient similarity search."""
    
    def __init__(self, index_path: str = "./faiss_index"):
        self.index_path = index_path
        self.index = None
        self.embeddings = None
        self.metadata_store = []  # List to store metadata for each vector
        self.id_mapping = {}  # Maps internal IDs to external chunk IDs
        self.vector_count = 0
        
        # Create directory if it doesn't exist
        os.makedirs(index_path, exist_ok=True)
        
        # Try to load existing index
        self._load_index()
    
    def _load_index(self):
        """Load FAISS index from disk if it exists."""
        index_file = os.path.join(self.index_path, "index.faiss")
        metadata_file = os.path.join(self.index_path, "metadata.pkl")
        
        if os.path.exists(index_file) and os.path.exists(metadata_file):
            try:
                self.index = faiss.read_index(index_file)
                with open(metadata_file, 'rb') as f:
                    data = pickle.load(f)
                    self.metadata_store = data.get('metadata_store', [])
                    self.id_mapping = data.get('id_mapping', {})
                    self.vector_count = data.get('vector_count', 0)
                print(f"Loaded FAISS index with {self.vector_count} vectors")
            except Exception as e:
                print(f"Error loading index: {e}. Creating new index.")
                self.index = None
    
    def _save_index(self):
        """Save FAISS index to disk."""
        index_file = os.path.join(self.index_path, "index.faiss")
        metadata_file = os.path.join(self.index_path, "metadata.pkl")
        
        if self.index is not None:
            faiss.write_index(self.index, index_file)
            with open(metadata_file, 'wb') as f:
                pickle.dump({
                    'metadata_store': self.metadata_store,
                    'id_mapping': self.id_mapping,
                    'vector_count': self.vector_count
                }, f)
    
    def insert_chunks(self, chunks: List[Dict], dense_embeddings: List[List[float]]) -> None:
        """
        Insert chunks with their embeddings into FAISS index.
        
        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata' keys
            dense_embeddings: List of embedding vectors
        """
        if not chunks or not dense_embeddings:
            return
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(dense_embeddings, dtype=np.float32)
        
        # Initialize index on first insert
        if self.index is None:
            embedding_dim = embeddings_array.shape[1]
            self.index = faiss.IndexFlatL2(embedding_dim)
        
        # Add embeddings to index
        self.index.add(embeddings_array)
        
        # Store metadata
        for i, chunk in enumerate(chunks):
            chunk_id = self.vector_count + i
            self.id_mapping[chunk_id] = chunk_id
            self.metadata_store.append({
                "text": chunk["text"],
                "metadata": chunk.get("metadata", {})
            })
        
        self.vector_count += len(chunks)
        
        # Save index after insertion
        self._save_index()
    
    def search(self, query_embedding: List[float], limit: int = 20) -> List[Dict]:
        """
        Search for similar vectors in the FAISS index.
        
        Args:
            query_embedding: Query embedding vector
            limit: Number of results to return
            
        Returns:
            List of dictionaries with 'id', 'score', 'text', and 'metadata' keys
        """
        if self.index is None or self.vector_count == 0:
            return []
        
        # Convert query to numpy array
        query_array = np.array([query_embedding], dtype=np.float32)
        
        # Search
        distances, indices = self.index.search(query_array, min(limit, self.vector_count))
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # Invalid index
                continue
            
            metadata = self.metadata_store[idx]
            results.append({
                "id": str(idx),
                "score": float(1 / (1 + distances[0][i])),  # Convert distance to similarity score
                "text": metadata["text"],
                "metadata": metadata.get("metadata", {})
            })
        
        return results
    
    def clear(self) -> None:
        """Clear the index and metadata."""
        self.index = None
        self.embeddings = None
        self.metadata_store = []
        self.id_mapping = {}
        self.vector_count = 0
        self._save_index()


# Singleton instance
faiss_store = FAISSStore()
