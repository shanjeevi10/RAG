from pypdf import PdfReader
from app.workflow.state import WorkflowState
import os

def ingestion_node(state: WorkflowState) -> dict:
    """
    Ingestion Agent: Reads a PDF from the given file_path and extracts text.
    Handles potentially long documents.
    """
    file_path = state["file_path"]
    
    if not os.path.exists(file_path):
        return {"pdf_text": f"Error: File not found at {file_path}"}
        
    try:
        reader = PdfReader(file_path)
        text_pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                text_pages.append(f"--- PAGE {i+1} ---\n{text}")
                
        full_text = "\n\n".join(text_pages)
        return {"pdf_text": full_text}
    except Exception as e:
        return {"pdf_text": f"Error during PDF extraction: {str(e)}"}
