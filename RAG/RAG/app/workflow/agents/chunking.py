from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.workflow.state import WorkflowState

def chunking_node(state: WorkflowState) -> dict:
    """
    Chunking Agent: Splits the extracted PDF text into semantically meaningful chunks.
    """
    pdf_text = state.get("pdf_text", "")
    
    if not pdf_text or pdf_text.startswith("Error:"):
        return {"chunks": []}
        
    # Standard chunking parameters
    chunk_size = 1000
    chunk_overlap = 200
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Simple chunking. We can enhance this to preserve page metadata better if needed.
    # In a real app we'd parse the '--- PAGE X ---' headers to attach metadata.
    
    raw_chunks = text_splitter.split_text(pdf_text)
    
    # Store chunks as dicts to hold potential metadata (e.g., page numbers, chunk index)
    chunks = []
    for i, chunk in enumerate(raw_chunks):
        chunks.append({
            "id": f"chunk_{i}",
            "text": chunk,
            "metadata": {"chunk_index": i}
        })
        
    return {"chunks": chunks}
