import os
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# Sample evaluation script for RAGAS
def run_evaluation():
    # Example dataset structure expected by RAGAS
    data = {
        "question": ["What is the main topic of the document?"],
        "answer": ["The main topic is AI engineering."],
        "contexts": [["AI engineering is a rapidly growing field.", "It involves building intelligent systems."]],
        "ground_truth": ["AI engineering."] # Optional, needed for context_recall
    }
    
    dataset = Dataset.from_dict(data)
    
    print("Running RAGAS evaluation...")
    result = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
    )
    
    print("Evaluation Results:")
    print(result)

if __name__ == "__main__":
    # Ragas supports Anthropic via langchain integration.
    # Note: Using non-OpenAI models in Ragas requires passing an explicit `llm` object to evaluate().
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not set.")
    run_evaluation()
