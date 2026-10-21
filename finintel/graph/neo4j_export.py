"""Neo4j export: Cypher generation (always available) and optional direct load.

The Cypher uses ``MERGE`` on stable keys so re-running a load is idempotent. Useful
read queries for the same schema are in :data:`EXAMPLE_QUERIES`.
"""

from __future__ import annotations

import json
from typing import Iterator

from .builder import FinancialKnowledgeGraph

CONSTRAINTS = [
    "CREATE CONSTRAINT company_key IF NOT EXISTS FOR (n:Company) REQUIRE n.key IS UNIQUE",
    "CREATE CONSTRAINT institution_key IF NOT EXISTS FOR (n:Institution) REQUIRE n.key IS UNIQUE",
    "CREATE CONSTRAINT security_key IF NOT EXISTS FOR (n:Security) REQUIRE n.key IS UNIQUE",
    "CREATE CONSTRAINT insider_key IF NOT EXISTS FOR (n:Insider) REQUIRE n.key IS UNIQUE",
    "CREATE CONSTRAINT risk_key IF NOT EXISTS FOR (n:RiskTheme) REQUIRE n.key IS UNIQUE",
]

EXAMPLE_QUERIES = {
    "capital_rotation": """
// Funds that cut $source and added $dest between two 13F periods
MATCH (i:Institution)-[a:HOLDS_EQUITY {period:$p0}]->(s:Company {ticker:$source}),
      (i)-[b:HOLDS_EQUITY {period:$p1}]->(s)
OPTIONAL MATCH (i)-[c:HOLDS_EQUITY {period:$p0}]->(d:Company {ticker:$dest})
MATCH (i)-[e:HOLDS_EQUITY {period:$p1}]->(d)
WHERE b.shares < a.shares AND e.shares > coalesce(c.shares, 0)
RETURN i.name, a.shares - b.shares AS sold, e.shares - coalesce(c.shares,0) AS bought
ORDER BY sold DESC""",
    "risk_gap": """
MATCH (s:Company {ticker:$source})-[r:DISCLOSES_RISK {period:'current'}]->(t:RiskTheme)
OPTIONAL MATCH (d:Company {ticker:$dest})-[q:DISCLOSES_RISK {period:'current'}]->(t)
RETURN t.name, r.intensity AS source, coalesce(q.intensity,0) AS dest, r.intensity - coalesce(q.intensity,0) AS gap
ORDER BY gap DESC""",
    "shared_suppliers": """
MATCH (sup:Company)-[:SUPPLIES]->(a:Company {ticker:$a}), (sup)-[:SUPPLIES]->(b:Company {ticker:$b})
RETURN sup.ticker""",
}


def _props(d: dict) -> str:
    clean = {k: v for k, v in d.items() if v is not None and not isinstance(v, (list, dict))}
    for k, v in d.items():
        if isinstance(v, list):
            clean[k] = [str(x) for x in v]
    return json.dumps(clean, default=str)


def cypher_statements(kg: FinancialKnowledgeGraph) -> Iterator[tuple[str, dict]]:
    for n, d in kg.g.nodes(data=True):
        label = d.get("label", "Entity")
        props = {k: v for k, v in d.items() if k != "label"}
        yield f"MERGE (n:{label} {{key:$key}}) SET n += $props", {"key": n, "props": json.loads(_props(props))}
    for u, v, k, d in kg.g.edges(keys=True, data=True):
        rel = d.get("type", "RELATED")
        props = {kk: vv for kk, vv in d.items() if kk != "type"}
        yield (
            f"MATCH (a {{key:$u}}), (b {{key:$v}}) MERGE (a)-[r:{rel} {{key:$k}}]->(b) SET r += $props",
            {"u": u, "v": v, "k": k, "props": json.loads(_props(props))},
        )


def to_cypher_script(kg: FinancialKnowledgeGraph) -> str:
    lines = [c + ";" for c in CONSTRAINTS]
    for stmt, params in cypher_statements(kg):
        lines.append(f":param batch => {json.dumps(params, default=str)};")
        lines.append(stmt.replace("$key", "$batch.key").replace("$props", "$batch.props").replace("$u", "$batch.u")
                     .replace("$v", "$batch.v").replace("$k", "$batch.k") + ";")
    return "\n".join(lines)


def load_into_neo4j(kg: FinancialKnowledgeGraph, uri: str, user: str, password: str, database: str | None = None) -> int:
    from neo4j import GraphDatabase  # optional dependency

    n = 0
    with GraphDatabase.driver(uri, auth=(user, password)) as driver, driver.session(database=database) as session:
        for c in CONSTRAINTS:
            session.run(c)
        for stmt, params in cypher_statements(kg):
            session.run(stmt, **params)
            n += 1
    return n
