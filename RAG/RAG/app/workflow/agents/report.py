from app.workflow.state import WorkflowState

def report_node(state: WorkflowState) -> dict:
    """
    Report Generation Agent: Formats the final output.
    If verification failed, prepends a warning.
    """
    draft = state.get("draft_response", "")
    status = state.get("verification_status", "unknown")
    feedback = state.get("verification_feedback", "")
    
    if status == "failed":
        final_report = f"WARNING: The following response failed safety/factuality verification.\nReason: {feedback}\n\n---\n\n{draft}"
    else:
        final_report = f"### Final Answer\n\n{draft}"
        
    return {"final_report": final_report}
