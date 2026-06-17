from app.workflow.state import WorkflowState
from app.retrieval.embeddings import embedding_service
from app.retrieval.faiss_store import faiss_store

def retrieval_node(state: WorkflowState) -> dict:
    """
    Retrieval Agent:
    1. Indexes chunks (if newly extracted)
    2. Performs vector retrieval
    3. Reranks using cross-encoder
    """
    query = state.get("query", "")
    chunks = state.get("chunks", [])
    
    # 1. Indexing Phase (Only if we have fresh chunks that haven't been indexed)
    # In a real system, you'd check if this document is already indexed.
    if chunks:
        # Generate dense embeddings for chunks
        chunk_texts = [c["text"] for c in chunks]
        dense_embs = embedding_service.embed_documents(chunk_texts)
        faiss_store.insert_chunks(chunks, dense_embs)
    
    if not query:
        return {"retrieved_context": []}
        
    # 2. Retrieval Phase (Stage 1)
    # Embed the query
    query_emb = embedding_service.embed_query(query)
    
    # Retrieve top 50 candidates from FAISS
    candidates = faiss_store.search(query_emb, limit=50)
    
    if not candidates:
        return {"retrieved_context": []}
        
    # 3. Reranking Phase (Stage 2)
    candidate_texts = [c["text"] for c in candidates]
    rerank_scores = embedding_service.rerank(query, candidate_texts)
    
    # Attach scores and sort
    for i, candidate in enumerate(candidates):
        candidate["rerank_score"] = rerank_scores[i]
        
    # Sort descending by rerank score
    ranked_candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    
    # Return Top 5 after reranking
    top_k = 5
    final_context = ranked_candidates[:top_k]
    
    return {"retrieved_context": final_context}
