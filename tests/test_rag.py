from datetime import date

import numpy as np

from finintel.rag import BM25Index, EvidenceLedger, ExtractiveAnswerer, FinancialRAGIndex, GroundingValidator, HashingEmbedder, HierarchicalChunker
from finintel.rag.chunking import ChunkerConfig
from finintel.rag.text import extract_numbers, split_sentences, tokenize
from finintel.schemas import (
    Accession,
    DataProvenance,
    FilingSection,
    FiscalPeriod,
    FormType,
    ParsedDocument,
    RawFiling,
    RetrievalFilter,
    RetrievalQuery,
)
from finintel.schemas.filings import ContentBlock, TableData


def test_tokenizer_canonicalizes_finance_synonyms():
    assert "repurchase" in tokenize("stock buybacks")
    assert "repurchase" in tokenize("share repurchases")
    assert tokenize("theft") == tokenize("shrink")
    assert "10-k" in tokenize("our 10-K filing")
    assert "7.35" in tokenize("EPS of $7.35")


def test_sentence_splitter_respects_decimals_and_abbreviations():
    s = split_sentences("Target Corp. reported EPS of $2.05 in Q2. Sales fell 1.9 percent. See Item 1A. Risk Factors follow.")
    assert s[0].startswith("Target Corp. reported EPS of $2.05")
    assert len(s) == 3


def test_extract_numbers_handles_parentheses_and_percent():
    assert extract_numbers("(1.9)% and $1,234 and -3") == [-1.9, 1234.0, -3.0]


def test_bm25_prefers_exact_rare_terms():
    docs = ["revenue grew strongly", "tariff exposure increased costs", "revenue and margin"]
    s = BM25Index().build(docs).scores("tariff")
    assert int(np.argmax(s)) == 1 and s[0] == 0


def test_hashing_embedder_maps_synonyms_together():
    e = HashingEmbedder().fit(["share repurchases continued", "dividends paid", "inventory shrink"])
    a, b, c = e.embed_documents(["stock buybacks", "share repurchases", "inventory shrink"])
    assert float(a @ b) > float(a @ c)


def _doc(rows: int) -> ParsedDocument:
    table = TableData(caption="Quarterly results (in millions)", header=["Metric", "Q1", "Q2"], rows=[[f"Line item {i}", f"${i},000", f"${i},500"] for i in range(rows)], units="$ in millions")
    sec = FilingSection(item="7", title="MD&A", blocks=[
        ContentBlock(kind="paragraph", text="Sales increased. Margins expanded on lower markdowns. " * 3, page=4),
        ContentBlock(kind="table", text=table.to_markdown(), table=table, page=5),
    ])
    qa = FilingSection(item="Q&A", title="Q&A", blocks=[
        ContentBlock(kind="speaker_turn", text="What about tariffs?", speaker="Analyst, A", speaker_role="analyst"),
        ContentBlock(kind="speaker_turn", text="We are mitigating.", speaker="CFO", speaker_role="management"),
        ContentBlock(kind="speaker_turn", text="And buybacks?", speaker="Analyst, B", speaker_role="analyst"),
        ContentBlock(kind="speaker_turn", text="Later.", speaker="CFO", speaker_role="management"),
    ])
    f = RawFiling(accession=Accession(value="FIXTURE-X-1"), cik="1", company_name="Example Co", ticker="EXM", form_type=FormType.TEN_Q,
                  filed_date=date(2025, 8, 30), period=FiscalPeriod(fiscal_year=2025, fiscal_quarter=2, period_end=date(2025, 8, 2)),
                  content="", provenance=DataProvenance(source="fixture"))
    return ParsedDocument(filing=f, sections=[sec, qa])


def test_tables_split_by_rows_with_header_repeated():
    ch = HierarchicalChunker(ChunkerConfig(max_table_tokens=120))
    children, parents = ch.chunk_document(_doc(30), anchor_quarter_index=2025 * 4 + 1)
    tables = [c for c in children if c.metadata.chunk_type == "table"]
    assert len(tables) > 1
    for t in tables:
        assert "| Metric | Q1 | Q2 |" in t.text  # header repeated in every part
        assert "Quarterly results" in t.text  # caption carried
        for line in t.text.splitlines():
            if line.startswith("| Line item"):
                assert line.count("|") == 4  # never split mid-row
    covered = sum(t.text.count("| Line item") for t in tables)
    assert covered == 30
    assert all(c.metadata.relative_period == "T" for c in children)
    assert all(c.index_text.startswith("[EXM | Example Co | 10-Q Q2 FY2025") for c in children)
    assert len(parents) == 2 and all(c.metadata.parent_id in {p.metadata.chunk_id for p in parents} for c in children)


def test_qa_pairs_keep_question_with_answer():
    children, _ = HierarchicalChunker().chunk_document(_doc(2))
    qa = [c for c in children if c.metadata.chunk_type == "qa_pair"]
    assert len(qa) == 2
    assert "tariffs" in qa[0].text and "mitigating" in qa[0].text
    assert qa[0].metadata.speaker == "Analyst, A"


def test_relative_periods_on_fixture_index(index):
    tgt = [c for c in index.chunks if c.metadata.ticker == "TGT" and c.metadata.form_type == FormType.TEN_K]
    labels = {c.metadata.fiscal_year: c.metadata.relative_period for c in tgt}
    assert labels[2024] == "T-2" and labels[2023] == "T-6"


def test_hybrid_search_finds_tariff_risk(index, bundle):
    hits = index.search(RetrievalQuery(text="tariffs on imported merchandise", top_k=3, filters=RetrievalFilter(issuer_ciks=[bundle.cik("TGT")])), as_of=bundle.as_of)
    assert any("tariff" in h.chunk.text.lower() for h in hits[:2])
    assert all(h.chunk.metadata.issuer_cik == bundle.cik("TGT") for h in hits)


def test_point_in_time_filter_blocks_look_ahead(index, bundle):
    q = RetrievalQuery(text="CEO succession transition", top_k=10, filters=RetrievalFilter(filed_on_or_before=date(2025, 6, 1)))
    hits = index.search(q, as_of=date(2025, 6, 1))
    assert hits and all(h.chunk.metadata.filed_date <= date(2025, 6, 1) for h in hits)
    assert not any(h.chunk.metadata.accession == "FIXTURE-TGT-8K-20250820" for h in hits)


def test_authority_weighting_prefers_filings_over_news(index, bundle):
    q = RetrievalQuery(text="Target dividend commitment", top_k=6, filters=RetrievalFilter(issuer_ciks=[bundle.cik("TGT")]))
    hits = index.search(q, as_of=bundle.as_of)
    forms = [h.chunk.metadata.form_type for h in hits]
    if FormType.NEWS in forms:
        news_rank = forms.index(FormType.NEWS)
        assert any(f in (FormType.TEN_K, FormType.TEN_Q, FormType.EIGHT_K, FormType.TRANSCRIPT) for f in forms[:news_rank + 1])
    # Raising the authority floor removes news entirely.
    q2 = q.model_copy(update={"filters": RetrievalFilter(issuer_ciks=[bundle.cik("TGT")], min_authority=0.7)})
    assert all(h.chunk.metadata.form_type != FormType.NEWS for h in index.search(q2, as_of=bundle.as_of))


def test_mmr_returns_distinct_chunks(index):
    hits = index.search(RetrievalQuery(text="discretionary demand traffic", top_k=5))
    ids = [h.chunk.metadata.chunk_id for h in hits]
    assert len(ids) == len(set(ids)) == 5


def test_index_persistence_roundtrip(index, tmp_path):
    index.save(tmp_path)
    loaded = FinancialRAGIndex.load(tmp_path)
    q = RetrievalQuery(text="share repurchase authorization", top_k=3)
    a = [h.chunk.metadata.chunk_id for h in index.search(q)]
    b = [h.chunk.metadata.chunk_id for h in loaded.search(q)]
    assert a == b and len(loaded.parents) == len(index.parents)


def test_grounding_validator_flags_unsupported_numbers_and_unknown_ids(index):
    led = EvidenceLedger()
    eid = led.add_chunk(index.chunks[0])
    fid = led.add_fact("metric", "Dividends of approximately 2046 million", {"dividends_paid": 2.046e9}, ["FIXTURE-TGT-10K-FY2024"])
    v = GroundingValidator(led)
    _, rep = v.validate(f"Dividends were about $2.0 billion [{fid}]. Buybacks were $9.9 billion [{fid}]. Shares rose 12.5% [E999]. Revenue grew 4.4%.")
    assert any("9.9" in u for u in rep.unsupported_numbers)
    assert not any(u.startswith("2 ") or u.startswith("2.0") for u in rep.unsupported_numbers)
    assert rep.unknown_evidence_ids == ["E999"]
    assert rep.uncited_factual and rep.grounding_rate < 1
    cleaned, rep2 = v.validate(f"Dividends were about $2.0 billion [{fid}]. Buybacks were $9.9 billion [{fid}].", strict=True)
    assert "9.9" not in cleaned and "2.0 billion" in cleaned and rep2.removed
    assert eid == "E1"


def test_extractive_answerer_quotes_or_abstains(index, bundle):
    ans = ExtractiveAnswerer(index).answer("What did management say about tariffs?", as_of=bundle.as_of)
    assert not ans.abstained and ans.citations
    assert all("tariff" in c.claim.lower() or "trade" in c.claim.lower() for c in ans.claims[:2])
    assert ExtractiveAnswerer(index).answer("What is Nvidia's data center revenue?").abstained
    assert ExtractiveAnswerer(index).answer("boiling point of liquid helium on jupiter").abstained
