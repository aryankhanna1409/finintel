from datetime import date

import pytest

from finintel.advisory import select_peers
from finintel.analytics import capital as C
from finintel.analytics.fundamentals import benchmark, stat
from finintel.analytics.ownership import concentration, flows, insider_signal, price_neutral_value_change
from finintel.analytics.risk_themes import themes_in
from finintel.graph import activist_vulnerability, capital_rotation, risk_discrepancies
from finintel.graph.neo4j_export import EXAMPLE_QUERIES, to_cypher_script
from finintel.schemas import CompanyValuation
from finintel.schemas.graph import EdgeType

T, T1, T4 = date(2025, 6, 30), date(2025, 3, 31), date(2024, 6, 30)


@pytest.fixture(scope="module")
def vals(bundle):
    return {f.ticker: CompanyValuation(issuer_cik=c, ticker=f.ticker, name=bundle.name(c), fundamentals=f, market=bundle.market[c]) for c, f in bundle.fundamentals.items()}


def test_derived_ratios(vals):
    t = vals["TGT"].fundamentals
    assert t.ebitda == pytest.approx(t.operating_income + t.depreciation_amortization)
    assert t.cash_conversion_cycle == pytest.approx(t.dso + t.dio - t.dpo)
    assert 55 < t.dio < 65  # 12.74bn inventory / 76.5bn COGS


def test_peer_statistics_are_robust(vals):
    stats = benchmark(vals["TGT"], [vals[p] for p in ["WMT", "COST", "DG", "KR"]])
    ev = stat(stats, "ev_to_ebitda")
    assert ev.peer_median == pytest.approx(sorted(v for v in ev.peer_values.values())[1:3][0] / 2 + sorted(ev.peer_values.values())[1:3][1] / 2)
    assert ev.z_score < 0 and ev.percentile_rank == 0.0  # cheapest in the group


def test_hhi_excludes_debt_and_options(bundle):
    tgt = bundle.cik("TGT")
    oc = concentration(bundle.holdings, tgt, T, bundle.market[tgt].shares_outstanding, bundle.institutions)
    eq = [h for h in bundle.holdings if h.issuer_cik == tgt and h.report_period == T and h.counts_toward_equity_ownership]
    assert oc.institutional_shares == pytest.approx(sum(h.shares for h in eq))
    assert 0 < oc.hhi < 10000 and oc.activist_pct_outstanding == pytest.approx(0.012, rel=1e-3)


def test_price_neutral_flows(bundle):
    tgt = bundle.cik("TGT")
    fl = {f.filer_name: f for f in flows(bundle.holdings, T4, T) if f.issuer_cik == tgt}
    idx = fl["Meridian Index Advisors (synthetic)"]
    assert idx.share_change > 0 and idx.value_to < idx.value_from  # bought shares while value fell with price
    assert price_neutral_value_change(idx) > 0
    assert fl["Halcyon Ridge Capital (synthetic)"].status == "exited"
    assert fl["Granite Harbor Activist Partners (synthetic)"].status == "new"


def test_insider_cluster_signal(bundle):
    s = insider_signal(bundle.insider_transactions, bundle.cik("TGT"), bundle.as_of)
    assert s.cluster_buy and s.distinct_buyers == 4 and s.signal == "bullish"
    w = insider_signal(bundle.insider_transactions, bundle.cik("WMT"), bundle.as_of)
    assert not w.cluster_buy and w.planned_sale_share == 1.0 and w.signal != "bullish"


def test_synthetic_rating_table():
    assert C.synthetic_rating(10) == "AAA" and C.synthetic_rating(4.5) == "A" and C.synthetic_rating(1.0) == "CCC" and C.synthetic_rating(None) == "AAA"


def test_wacc_curve_shape_and_floor(bundle, vals):
    pts, beta_u = C.wacc_curve(vals["TGT"], bundle.macro)
    assert beta_u < vals["TGT"].market.beta
    assert all(b.levered_beta >= a.levered_beta for a, b in zip(pts, pts[1:]))
    assert all(C.notch(b.implied_rating) >= C.notch(a.implied_rating) for a, b in zip(pts, pts[1:]))  # never improves with leverage
    cs = C.capital_structure(vals["TGT"], bundle.macro, None, "A")
    assert C.notch(cs.optimal_rating) <= C.notch("A")
    loose = C.capital_structure(vals["TGT"], bundle.macro, None, "BBB-")
    assert loose.optimal_debt_to_capital >= cs.optimal_debt_to_capital


def test_rating_floor_inference():
    floor, src = C.infer_rating_floor(["We remain committed to our middle-A credit ratings."], "A")
    assert floor == "A" and src
    assert C.infer_rating_floor([], "A")[0] == "A-"
    assert C.infer_rating_floor([], None)[0] == "BBB-"


def test_risk_theme_detection():
    assert {"tariffs_trade", "brand_reputation"} <= themes_in("Tariffs and a consumer boycott hurt our brand and reputation.")


def test_graph_schema(kg):
    st = kg.stats()
    for et in ("HOLDS_EQUITY", "HOLDS_DEBT", "HOLDS_DERIVATIVE", "COMPETES_WITH", "SUPPLIES", "TRADED", "DISCLOSES_RISK", "ISSUED"):
        assert st.edges_by_type.get(et, 0) > 0, et
    debt = list(kg.edges_of_type(EdgeType.HOLDS_DEBT))
    assert all(d["cusip"] == "87612EBM7" for _, _, d in debt)


def test_capital_rotation_no_double_counting(kg, bundle):
    tgt = bundle.cik("TGT")
    peers = [bundle.cik(p) for p in ["WMT", "COST", "DG", "KR"]]
    rot = capital_rotation(kg, bundle, tgt, peers, T1, T)
    halcyon = [r for r in rot if r.filer_name.startswith("Halcyon")]
    assert {r.destination_ticker for r in halcyon} >= {"WMT", "COST"}
    outflow = -halcyon[0].source_value_change_usd
    assert sum(r.rotation_usd for r in halcyon) <= outflow + 1
    themes = {d.theme for d in halcyon[0].risk_discrepancies}
    assert "leadership_transition" in themes and "brand_reputation" in themes
    assert not any(r.filer_name.startswith(("Meridian", "Bluewater")) for r in rot)


def test_risk_discrepancy_new_themes(kg, bundle):
    vs_cost = {x.theme: x for x in risk_discrepancies(kg, bundle.cik("TGT"), bundle.cik("COST"))}
    assert vs_cost["brand_reputation"].new_in_source and not vs_cost["discretionary_demand"].new_in_source
    assert "tariffs_trade" not in vs_cost  # Costco discloses tariffs too, so no gap
    vs_kr = {x.theme: x for x in risk_discrepancies(kg, bundle.cik("TGT"), bundle.cik("KR"))}
    assert vs_kr["tariffs_trade"].new_in_source


def test_activist_vulnerability_ranks_target_highest(kg, bundle, vals):
    scores = {}
    for t in ["TGT", "WMT", "COST", "DG", "KR"]:
        peers = [vals[p] for p in ["TGT", "WMT", "COST", "DG", "KR"] if p != t]
        scores[t] = activist_vulnerability(kg, bundle, vals[t], peers, T)
    assert max(scores, key=lambda k: scores[k].score) == "TGT"
    assert scores["TGT"].band in ("elevated", "high")
    assert scores["TGT"].activists_present == ["Granite Harbor Activist Partners (synthetic)"]
    assert 0 <= scores["COST"].score < scores["TGT"].score


def test_peer_selection(kg, bundle):
    peers = select_peers(kg, bundle, bundle.cik("TGT"))
    assert 3 <= len(peers) <= 5 and bundle.cik("WMT") in peers
    assert all(p in bundle.fundamentals for p in peers)


def test_cypher_export(kg):
    script = to_cypher_script(kg)
    assert "CREATE CONSTRAINT company_key" in script and "MERGE (a)-[r:HOLDS_EQUITY" in script
    assert "HOLDS_EQUITY" in EXAMPLE_QUERIES["capital_rotation"]
