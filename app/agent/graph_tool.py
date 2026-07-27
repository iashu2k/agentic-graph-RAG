import json
from groq import Groq
from app.config import settings
from app.services.graph_db import graph_db
from app.services import llm_client

client = Groq(api_key=settings.groq_api_key)

CYPHER_PROMPT = """Generate a Cypher query for this question about a SEC 10-Q knowledge graph.

Schema (ONLY use these labels/properties/relationships, nothing else):
- (Company {{ticker}}) -- ticker holds FULL company names: 'Apple', 'Amazon', 'Intel', 'Microsoft', 'NVIDIA'
- (Filing {{filing_id, fiscal_year, fiscal_quarter, filing_type}})
- (Section {{filing_id, name}})
- (Disclosure {{description, category}}) -- category in: risk_factor, restructuring, financial_metric, legal_proceeding, tax, segment, other
- (Company)-[:FILED]->(Filing)
- (Filing)-[:HAS_SECTION]->(Section)
- (Filing)-[:DISCLOSES]->(Disclosure)

CRITICAL RULES:
1. Always include a LIMIT clause (max 20).
2. Use toLower() + CONTAINS for text matching on description, never exact equality.
3. ALWAYS return the company ticker (e.g. c.ticker or c2.ticker) as one of the RETURN columns, even if the question doesn't explicitly ask "which company" -- this is required for citation purposes.
4. If you use a WITH clause, you MUST explicitly list every variable you still need afterward (Cypher does not carry variables through WITH automatically). Prefer avoiding WITH entirely when possible by using multiple MATCH clauses in sequence instead.

EXAMPLE -- cross-company shared disclosure question ("which companies share X with Company A"):
Question: Which companies share a risk factor about supply chain with Intel?
{{"cypher": "MATCH (c1:Company {{ticker: 'Intel'}})-[:FILED]->(:Filing)-[:DISCLOSES]->(d1:Disclosure {{category: 'risk_factor'}}) WHERE toLower(d1.description) CONTAINS 'supply chain' MATCH (c2:Company)-[:FILED]->(:Filing)-[:DISCLOSES]->(d2:Disclosure {{category: 'risk_factor'}}) WHERE c2.ticker <> 'Intel' AND toLower(d2.description) CONTAINS 'supply chain' RETURN DISTINCT c2.ticker AS company, d2.description AS description LIMIT 20"}}

Return ONLY valid JSON: {{"cypher": "MATCH ... RETURN ..."}}

Question: {question}
"""

ALLOWED_LABELS = {"Company", "Filing", "Section", "Disclosure"}
ALLOWED_RELS = {"FILED", "HAS_SECTION", "DISCLOSES"}


def generate_cypher(question: str) -> str | None:
  resp = client.chat.completions.create(
      model=llm_client.get_model(),
      messages=[
        {"role": "user", "content": CYPHER_PROMPT.format(question=question)}],
      response_format={"type": "json_object"},
      temperature=0,
  )
  try:
    data = json.loads(resp.choices[0].message.content)
    return data.get("cypher")
  except json.JSONDecodeError:
    return None


def validate_cypher(cypher: str) -> bool:
  if not cypher or "MATCH" not in cypher.upper():
    return False
  dangerous = ["CREATE", "DELETE", "MERGE", "SET", "REMOVE", "DROP", "DETACH"]
  if any(kw in cypher.upper() for kw in dangerous):
    return False
  if "LIMIT" not in cypher.upper():
    cypher += " LIMIT 20"
  return True


def run_graph_query(question: str) -> tuple[list[dict], str | None]:
  cypher = generate_cypher(question)
  if not cypher or not validate_cypher(cypher):
    return [], cypher
  try:
    records = graph_db.execute_read(cypher)
    return [dict(r) for r in records], cypher
  except Exception as e:
    print(f"Cypher execution failed: {e}")
    return [], cypher


def graph_node(state: dict) -> dict:
  results, cypher = run_graph_query(state["question"])
  return {**state, "graph_results": results, "cypher_query": cypher}
