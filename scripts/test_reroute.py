from app.agent.graph_builder import needs_retry
from app.agent.rewrite import rewrite_node
from app.agent.router import router_node

state = {
    "question": "Which companies share a risk factor with Intel?",
    "original_question": None,
    "rewrite_reasoning": None,
    "route": "hybrid",
    "cypher_query": None,
    "graph_results": [],
    "hybrid_results": [{
        "company": "Placeholder", "filing_type": "10-Q", "fiscal_quarter": "Q1",
        "fiscal_year": 2023, "section": "Test", "content": "irrelevant",
        "rerank_score": -999,
    }],
    "context": "", "answer": "", "retry_count": 0, "max_retries": 2,
}

print("needs_retry with bad hybrid score:", needs_retry(state))

state = rewrite_node(state)
print("After rewrite -> retry_count:", state["retry_count"], "| question:", state["question"])

state = router_node(state)
print("After router re-classification -> route:", state["route"])
