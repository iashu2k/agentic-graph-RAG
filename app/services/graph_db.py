from neo4j import GraphDatabase
from app.config import settings


class GraphDB:
  def __init__(self):
    self._driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    self._database = settings.neo4j_database

  def verify(self):
    self._driver.verify_connectivity()

  def execute_write(self, query: str, params: dict | None = None):
    result = self._driver.execute_query(
        query, params or {}, database_=self._database
    )
    return result.records

  def execute_read(self, query: str, params: dict | None = None):
    result = self._driver.execute_query(
        query, params or {}, database_=self._database
    )
    return result.records

  def init_constraints(self):
    constraints = [
        "CREATE CONSTRAINT company_ticker IF NOT EXISTS "
        "FOR (c:Company) REQUIRE c.ticker IS UNIQUE",
        "CREATE CONSTRAINT filing_id IF NOT EXISTS "
        "FOR (f:Filing) REQUIRE f.filing_id IS UNIQUE",
        "CREATE CONSTRAINT section_key IF NOT EXISTS "
        "FOR (s:Section) REQUIRE (s.filing_id, s.name) IS UNIQUE",
    ]
    for c in constraints:
      self.execute_write(c)

  def close(self):
    self._driver.close()


graph_db = GraphDB()
