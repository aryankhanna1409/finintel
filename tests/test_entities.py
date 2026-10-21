from finintel.entities import EntityRegistry, is_valid_cusip
from finintel.entities.resolver import infer_security_type, normalize_name
from finintel.schemas import Issuer, Security, SecurityType, normalize_ticker


def _registry() -> EntityRegistry:
    reg = EntityRegistry()
    reg.add_issuer(Issuer(cik="1652044", name="Alphabet Inc.", tickers=("GOOGL", "GOOG")))
    reg.add_issuer(Issuer(cik="1067983", name="Berkshire Hathaway Inc", tickers=("BRK.B", "BRK.A")))
    # Hypothetical parent + financing subsidiary (names deliberately fictional).
    reg.add_issuer(Issuer(cik="9800000001", name="Example Parent Holdings Corp", tickers=("EXPH",)))
    reg.add_issuer(Issuer(cik="9800000002", name="Example Parent Finance LLC", parent_cik="9800000001"))
    reg.add_security(Security(cusip="02079K305", ticker="GOOGL", issuer_cik="1652044", security_type=SecurityType.COMMON_EQUITY, share_class="A"))
    reg.add_security(Security(cusip="02079K107", ticker="GOOG", issuer_cik="1652044", security_type=SecurityType.COMMON_EQUITY, share_class="C", votes_per_share=0))
    reg.add_security(Security(cusip="084670702", ticker="BRK.B", issuer_cik="1067983", security_type=SecurityType.COMMON_EQUITY, share_class="B"))
    reg.add_security(Security(cusip="084670108", ticker="BRK.A", issuer_cik="1067983", security_type=SecurityType.COMMON_EQUITY, share_class="A", economic_conversion_ratio=1500))
    reg.add_security(Security(cusip="990000010", issuer_cik="9800000001", security_type=SecurityType.COMMON_EQUITY, ticker="EXPH"))
    reg.add_security(Security(cusip="990000AB7", issuer_cik="9800000002", security_type=SecurityType.SENIOR_DEBT, description="5% notes"))
    return reg


def test_cusip_check_digits():
    for c in ["87612E106", "931142103", "22160K105", "02079K305", "084670702"]:
        assert is_valid_cusip(c)
    assert not is_valid_cusip("87612E107")
    assert not is_valid_cusip("12345")


def test_ticker_normalization_and_resolution():
    reg = _registry()
    assert normalize_ticker("brk-b") == normalize_ticker("BRK/B") == "BRK.B"
    r = reg.resolve("brk-b")
    assert r.issuer_cik == "0001067983" and r.security and r.security.share_class == "B"


def test_share_classes_map_to_one_issuer_with_economic_conversion():
    reg = _registry()
    a, c = reg.resolve("GOOGL"), reg.resolve("GOOG")
    assert a.issuer_cik == c.issuer_cik == "0001652044"
    assert a.security.cusip != c.security.cusip
    assert len(reg.equity_classes("1652044")) == 2
    # One BRK.A share is 1,500 BRK.B-equivalent economic units.
    assert reg.economic_equity_shares("084670108", 2) == 3000
    assert reg.economic_equity_shares("084670702", 2) == 2


def test_subsidiary_debt_does_not_become_parent_equity():
    reg = _registry()
    res = reg.resolve_holding("990000AB7", "EXAMPLE PARENT FINANCE LLC", "NOTE 5.000% 2030", share_type="PRN")
    assert res.issuer_cik == "9800000002"
    assert not any("check digit" in w for w in res.warnings)
    assert res.security.security_type == SecurityType.SENIOR_DEBT
    assert reg.ultimate_parent(res.issuer_cik) == "9800000001"
    assert any("subsidiary" in w for w in res.warnings)
    assert reg.economic_equity_shares("990000AB7", 1_000_000) == 0.0


def test_unknown_issue_under_known_prefix_is_typed_from_title():
    reg = _registry()
    res = reg.resolve_holding("02079KAA5", "ALPHABET INC", "NOTE 1.998% 8/15/26", share_type="PRN")
    assert res.method == "cusip6_prefix" and res.issuer_cik == "0001652044"
    assert res.security.security_type == SecurityType.SENIOR_DEBT


def test_name_resolution_and_ambiguity():
    reg = _registry()
    assert normalize_name("The Kroger Co.") == "kroger"
    r = reg.resolve("Alphabet Inc Class A")
    assert r.issuer_cik == "0001652044" and r.method.startswith("name")
    bad = reg.resolve("Completely Unrelated Widgets")
    assert not bad.resolved


def test_security_type_inference():
    assert infer_security_type("COM") == SecurityType.COMMON_EQUITY
    assert infer_security_type("NOTE 4.5% 2034", "PRN") == SecurityType.SENIOR_DEBT
    assert infer_security_type("COM", put_call="Put") == SecurityType.OPTION_PUT
    assert infer_security_type("SPONSORED ADR") == SecurityType.ADR
    assert infer_security_type("CONV NOTE 0.5%", "PRN") == SecurityType.CONVERTIBLE_DEBT


def test_mentions_are_case_sensitive(bundle):
    reg = bundle.registry
    assert bundle.cik("TGT") in reg.mentions("Should Target add debt?")
    assert reg.mentions("our target leverage ratio") == []
    assert bundle.cik("WMT") in reg.mentions("compare with WMT")
