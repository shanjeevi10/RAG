from langgraph.graph import StateGraph, END
from app.workflow.state import WorkflowState

# Import nodes
from app.workflow.agents.ingestion import ingestion_node
from app.workflow.agents.chunking import chunking_node
from app.workflow.agents.retrieval import retrieval_node
from app.workflow.agents.reasoning import reasoning_node
from app.workflow.agents.verification import verification_node
from app.workflow.agents.report import report_node

# Initialize graph
workflow = StateGraph(WorkflowState)

# Add nodes
workflow.add_node("ingestion", ingestion_node)
workflow.add_node("chunking", chunking_node)
workflow.add_node("retrieval", retrieval_node)
workflow.add_node("reasoning", reasoning_node)
workflow.add_node("verification", verification_node)
workflow.add_node("report", report_node)

# Define edges
# Assuming standard sequential flow. In a more complex setup, we might skip
# ingestion/chunking if the file is already processed, but we'll do it linearly here for simplicity.
workflow.add_edge("ingestion", "chunking")
workflow.add_edge("chunking", "retrieval")
workflow.add_edge("retrieval", "reasoning")
workflow.add_edge("reasoning", "verification")

# Conditional edge from verification
def check_verification(state: WorkflowState):
    status = state.get("verification_status")
    # If we wanted to self-correct, we could route back to reasoning here.
    # For now, we'll route to report in both cases, but the report will show a warning if failed.
    return "report"

workflow.add_conditional_edges(
    "verification",
    check_verification,
    {"report": "report"}
)

workflow.add_edge("report", END)

# Set entry point
workflow.set_entry_point("ingestion")

# Compile graph
workflow_app = workflow.compile()
