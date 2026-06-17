import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, ContextualPrecisionMetric

# Pytest-integrated evaluation script using DeepEval
def test_rag_pipeline():
    # Simulate a call to our LangGraph pipeline
    input_query = "How do I implement hybrid search?"
    
    # Mocking pipeline outputs
    actual_output = "You can implement hybrid search using Qdrant with dense and sparse vectors."
    retrieval_context = [
        "Qdrant supports hybrid search out of the box.",
        "You need to provide both dense and sparse vectors."
    ]
    
    test_case = LLMTestCase(
        input=input_query,
        actual_output=actual_output,
        retrieval_context=retrieval_context
    )
    
    faithfulness_metric = FaithfulnessMetric(threshold=0.7)
    context_precision_metric = ContextualPrecisionMetric(threshold=0.7)
    
    # Assertions will fail the pytest if metrics are below threshold
    assert_test(test_case, [faithfulness_metric, context_precision_metric])

if __name__ == "__main__":
    # Run with: pytest tests/eval_deepeval.py
    print("Run this file using: pytest eval_deepeval.py")
