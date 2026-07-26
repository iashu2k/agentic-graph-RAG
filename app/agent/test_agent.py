from app.agent.graph_builder import agent


def run_test(question: str):
  result = agent.invoke({
      "question": question,
      "original_question": None,
      "rewrite_reasoning": None,
      "route": None,
      "cypher_query": None,
      "graph_results": [],
      "hybrid_results": [],
      "context": "",
      "answer": "",
      "retry_count": 0,
      "max_retries": 2,
  })

  print(f"\n{'=' * 70}")
  print(f"Original question: {result.get('original_question') or question}")
  print(f"Route: {result['route']}")
  print(f"Retry count: {result['retry_count']}")
  if result["retry_count"] > 0:
    print(f"Final question after rewrite: {result['question']}")
    print(f"Rewrite reasoning: {result.get('rewrite_reasoning')}")
  print(f"Answer: {result['answer']}")


if __name__ == "__main__":
  run_test("List every company that mentioned litigation in their filings")
  run_test("What restructuring charges did Google disclose?")
  run_test("Did Intel fire people?")
  run_test("Which companies share a risk factor with Intel?")
  run_test("What was Apple's revenue in Q2 2023?")
  run_test("What restructuring charges did Intel disclose and how does that compare to Amazon's?")
