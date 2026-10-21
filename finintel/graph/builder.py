"""Knowledge-graph construction (NetworkX MultiDiGraph; exportable to Neo4j).

Schema::

    (:Institution)-[:HOLDS_EQUITY {period, shares, value_usd, pct_outstanding, accession}]->(:Company)
    (:Institution)-[:HOLDS_DEBT {period, principal, cusip}]->(:Company)
    (:Institution)-[:HOLDS_DERIVATIVE {period, put_call, shares}]->(:Company)
    (:Company)-[:COMPETES_WITH {strength, basis}]->(:Company)        (both directions)
    (:Company)-[:SUPPLIES {revenue_share_of_supplier}]->(:Company)
    (:Company)-[:SUBSIDIARY_OF]->(:Company)
    (:Company)-[:ISSUED]->(:Security)
    (:Insider)-[:TRADED {date, code, shares, value_usd, is_10b5_1}]->(:Company)
    (:Company)-[:DISCLOSES_RISK {period, intensity, accession}]->(:RiskTheme)

Node keys are namespaced strings (``company:0000027419``) so labels never collide.
"""

from __future__ import annotations

from collections import Counter

import networkx as nx

from ..analytics.risk_themes import THEMES, profile_chunks, risk_factor_chunks
from ..ingestion.sources import CorpusBundle
from ..schemas.common import SecurityType
from ..schemas.graph import EdgeType, GraphStats, NodeLabel
from ..schemas.retrieval import Chunk


def company_node(cik: str) -> str:
    return f"company:{cik}"


def institution_node(cik: str) -> str:
    return f"institution:{cik}"


def insider_node(key: str) -> str:
    return f"insider:{key}"


def risk_node(theme: str) -> str:
    return f"risk:{theme}"


class FinancialKnowledgeGraph:
    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()

    # ------------------------------------------------------------------ construction
    def build(self, bundle: CorpusBundle, chunks: list[Chunk] | None = None) -> "FinancialKnowledgeGraph":
        reg = bundle.registry
        for iss in reg.issuers.values():
            self.g.add_node(company_node(iss.cik), label=NodeLabel.COMPANY.value, cik=iss.cik, name=iss.name, ticker=iss.primary_ticker,
                            sector=iss.sector, industry=iss.industry, sic=iss.sic)
            if iss.parent_cik:
                self.g.add_edge(company_node(iss.cik), company_node(iss.parent_cik), key=EdgeType.SUBSIDIARY_OF.value, type=EdgeType.SUBSIDIARY_OF.value)
        for sec in reg.securities_by_cusip.values():
            sn = f"security:{sec.cusip}"
            self.g.add_node(sn, label=NodeLabel.SECURITY.value, cusip=sec.cusip, ticker=sec.ticker, security_type=sec.security_type.value,
                            share_class=sec.share_class)
            self.g.add_edge(company_node(sec.issuer_cik), sn, key=EdgeType.ISSUED.value, type=EdgeType.ISSUED.value)
        for inst in bundle.institutions.values():
            self.g.add_node(institution_node(inst.cik), label=NodeLabel.INSTITUTION.value, cik=inst.cik, name=inst.name,
                            institution_type=inst.institution_type.value, is_activist=inst.is_known_activist, style=inst.style)
        shares_out = {cik: m.shares_outstanding for cik, m in bundle.market.items()}
        for h in bundle.holdings:
            if not h.issuer_cik:
                continue
            src, dst = institution_node(h.filer_cik), company_node(h.issuer_cik)
            if src not in self.g:
                self.g.add_node(src, label=NodeLabel.INSTITUTION.value, cik=h.filer_cik, name=h.filer_name, institution_type="other", is_activist=False)
            period = h.report_period.isoformat()
            if h.security_type in (SecurityType.OPTION_CALL, SecurityType.OPTION_PUT):
                et = EdgeType.HOLDS_DERIVATIVE
            elif h.counts_toward_equity_ownership:
                et = EdgeType.HOLDS_EQUITY
            else:
                et = EdgeType.HOLDS_DEBT
            key = f"{et.value}:{period}:{h.cusip}"
            attrs = dict(type=et.value, period=period, shares=h.shares, value_usd=h.value_usd, cusip=h.cusip, accession=h.accession.value,
                         put_call=h.put_call, pct_outstanding=(h.shares / shares_out[h.issuer_cik]) if (et == EdgeType.HOLDS_EQUITY and shares_out.get(h.issuer_cik)) else None)
            if self.g.has_edge(src, dst, key):
                e = self.g.edges[src, dst, key]
                e["shares"] += h.shares
                e["value_usd"] += h.value_usd
            else:
                self.g.add_edge(src, dst, key=key, **attrs)
        for c in bundle.competitors:
            for a, b in ((c.company_cik, c.peer_cik), (c.peer_cik, c.company_cik)):
                self.g.add_edge(company_node(a), company_node(b), key=EdgeType.COMPETES_WITH.value, type=EdgeType.COMPETES_WITH.value,
                                strength=c.strength, basis=c.basis)
        for s in bundle.suppliers:
            self.g.add_edge(company_node(s.supplier_cik), company_node(s.customer_cik), key=EdgeType.SUPPLIES.value, type=EdgeType.SUPPLIES.value,
                            revenue_share_of_supplier=s.revenue_share_of_supplier, note=s.note)
        for t in bundle.insider_transactions:
            key = t.insider_cik or t.insider_name
            n = insider_node(key)
            self.g.add_node(n, label=NodeLabel.INSIDER.value, name=t.insider_name, role=t.insider_role)
            self.g.add_edge(n, company_node(t.issuer_cik), key=f"{EdgeType.TRADED.value}:{t.accession.value}:{t.transaction_date}",
                            type=EdgeType.TRADED.value, date=t.transaction_date.isoformat(), code=t.code.value, shares=t.shares,
                            value_usd=t.value_usd, is_10b5_1=t.is_10b5_1, accession=t.accession.value)
        if chunks:
            self.add_risk_disclosures(bundle, chunks)
        return self

    def add_risk_disclosures(self, bundle: CorpusBundle, chunks: list[Chunk]) -> None:
        for theme in THEMES:
            self.g.add_node(risk_node(theme), label=NodeLabel.RISK_THEME.value, name=theme)
        for cik in bundle.fundamentals:
            for latest in (True, False):
                rc = risk_factor_chunks(chunks, cik, latest=latest)
                if not rc:
                    continue
                prof = profile_chunks(cik, rc)
                acc = rc[0].metadata.accession
                period = "current" if latest else "prior"
                for theme, inten in prof.intensity.items():
                    if inten > 0:
                        self.g.add_edge(company_node(cik), risk_node(theme), key=f"{EdgeType.DISCLOSES_RISK.value}:{period}",
                                        type=EdgeType.DISCLOSES_RISK.value, period=period, intensity=inten, accession=acc,
                                        chunk_ids=prof.evidence.get(theme, []))

    # ------------------------------------------------------------------ accessors
    def stats(self) -> GraphStats:
        nodes = Counter(d.get("label", "?") for _, d in self.g.nodes(data=True))
        edges = Counter(d.get("type", "?") for _, _, d in self.g.edges(data=True))
        return GraphStats(nodes_by_label=dict(nodes), edges_by_type=dict(edges))

    def edges_of_type(self, et: EdgeType, **match: object):
        for u, v, d in self.g.edges(data=True):
            if d.get("type") == et.value and all(d.get(k) == val for k, val in match.items()):
                yield u, v, d

    def peers_of(self, cik: str) -> list[tuple[str, float]]:
        n = company_node(cik)
        out = {}
        for _, v, d in self.g.out_edges(n, data=True):
            if d.get("type") == EdgeType.COMPETES_WITH.value:
                out[self.g.nodes[v]["cik"]] = max(out.get(self.g.nodes[v]["cik"], 0.0), d.get("strength", 1.0))
        return sorted(out.items(), key=lambda x: -x[1])

    def holders_of(self, cik: str, period: str) -> dict[str, dict]:
        out = {}
        for u, _, d in self.g.in_edges(company_node(cik), data=True):
            if d.get("type") == EdgeType.HOLDS_EQUITY.value and d.get("period") == period:
                out[u] = d
        return out

    def risk_intensity(self, cik: str, period: str = "current") -> dict[str, tuple[float, list[str]]]:
        out = {}
        for _, v, d in self.g.out_edges(company_node(cik), data=True):
            if d.get("type") == EdgeType.DISCLOSES_RISK.value and d.get("period") == period:
                out[self.g.nodes[v]["name"]] = (d["intensity"], d.get("chunk_ids", []))
        return out

    def suppliers_of(self, cik: str) -> list[str]:
        return [self.g.nodes[u]["cik"] for u, _, d in self.g.in_edges(company_node(cik), data=True) if d.get("type") == EdgeType.SUPPLIES.value]
