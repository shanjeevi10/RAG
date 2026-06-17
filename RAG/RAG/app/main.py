import os
import aiofiles
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from app.config import settings

app = FastAPI(title=settings.APP_NAME)

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    query: str
    filename: str

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="Only PDFs allowed")
    
    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    
    # Stream in chunks (1MB chunks)
    CHUNK_SIZE = 1024 * 1024
    
    try:
        async with aiofiles.open(file_path, "wb") as out_file:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                await out_file.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")
        
    return {"filename": file.filename, "status": "uploaded", "path": file_path}

@app.post("/ask/")
async def query_pdf(request: QueryRequest):
    file_path = os.path.join(settings.UPLOAD_DIR, request.filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found. Please upload it first.")
    
    # Lazy import to avoid circular dependencies and slow startup
    from app.workflow.graph import workflow_app
    from app.workflow.state import WorkflowState
    
    # Initialize state
    initial_state = WorkflowState(
        query=request.query,
        file_path=file_path,
        pdf_text="",
        chunks=[],
        retrieved_context=[],
        draft_response="",
        citations=[],
        verification_status="pending",
        final_report=""
    )
    
    # Run workflow
    try:
        final_state = workflow_app.invoke(initial_state)
        return {
            "status": "success",
            "report": final_state.get("final_report", ""),
            "verification_status": final_state.get("verification_status", "unknown")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy"}
