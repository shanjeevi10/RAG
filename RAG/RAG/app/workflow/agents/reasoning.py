from app.workflow.state import WorkflowState
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from app.config import settings
import json

def reasoning_node(state: WorkflowState) -> dict:
    """
    Reasoning Agent: Generates an answer based on the retrieved context and query,
    enforcing citation of sources.
    """
    query = state.get("query", "")
    context = state.get("retrieved_context", [])
    
    # Initialize LLM based on provider
    if settings.LLM_PROVIDER.lower() == "ollama":
        from langchain_community.llms import Ollama
        llm = Ollama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0
        )
    elif settings.LLM_PROVIDER.lower() == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model=settings.LLM_MODEL, 
            temperature=0, 
            api_key=settings.ANTHROPIC_API_KEY
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")
    
    # Format context with explicit source IDs
    formatted_context = ""
    for idx, doc in enumerate(context):
        # We assign an index as the source citation ID
        source_id = f"source_{idx+1}"
        formatted_context += f"--- {source_id} ---\n{doc['text']}\n\n"
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert analyst. Answer the user's question using ONLY the provided context.
You MUST cite your sources using the format [source_id].
If the context does not contain the answer, say 'I cannot answer this based on the provided documents.'
Ensure the response is detailed and well-structured.
"""),
        ("user", "Context:\n{context}\n\nQuestion: {query}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": formatted_context, "query": query})
    
    # Extract citations used (naive extraction for demonstration)
    citations_used = []
    for idx in range(len(context)):
        source_id = f"[source_{idx+1}]"
        if source_id in response.content:
            citations_used.append(source_id)
    
    return {
        "draft_response": response.content,
        "citations": citations_used
    }
