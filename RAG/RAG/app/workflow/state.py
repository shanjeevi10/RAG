from typing import TypedDict, List, Dict, Any, Annotated
import operator

class WorkflowState(TypedDict):
    """
    State dictionary for the LangGraph workflow.
    """
    query: str
    file_path: str
    
    # Ingestion & Chunking
    pdf_text: str
    chunks: List[Dict[str, Any]] # e.g., [{"text": "...", "metadata": {"page": 1}}]
    
    # Retrieval
    retrieved_context: List[Dict[str, Any]] # e.g., [{"text": "...", "score": 0.9}]
    
    # Reasoning
    draft_response: str
    citations: List[str]
    
    # Verification
    verification_status: str # "pending", "passed", "failed"
    verification_feedback: str # Output from NeMo Guardrails or LLM-as-a-judge
    
    # Final Output
    final_report: str
    
    # Add a message history if we need conversational capabilities later
    # messages: Annotated[list, operator.add]
