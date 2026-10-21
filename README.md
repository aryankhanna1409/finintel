# finintel: Financial Intelligence & RAG System

An end-to-end system that ingests SEC filings, 13F holdings, Form 4 insider trades, XBRL fundamentals and earnings-call transcripts, indexes them for hybrid retrieval with strict citation lineage, builds a cross-company knowledge graph, and produces a grounded **Strategic Advisory Report** for a CFO benchmarking a target against 3–5 peers.

The worked example is **Target (TGT)** against **Walmart, Costco, Dollar General and Kroger**, with Procter & Gamble, PepsiCo and Kimberly-Clark as shared suppliers.

```
pip install -e ".[dev]"            # core: pydantic, numpy, networkx, httpx
pytest                             # 64 tests, ~3s, fully offline
finintel report TGT --peers WMT COST DG KR --out reports
```

---

## 1. Architecture

```
            ┌──────────────────────── ingestion ────────────────────────┐
 SEC EDGAR  │ EdgarClient (UA, 8 req/s limiter, retries, disk cache)   │
 Stooq/FRED │ parsers: 13F infotable · Form 4 · XBRL companyfacts ·     │
 fixtures   │ 10-K/10-Q/8-K HTML (Item sectionizer, tables) · transcripts│
            │ EntityRegistry: CIK ↔ ticker ↔ CUSIP(6/9) ↔ name, parents  │
            └──────────────┬───────────────────────────┬────────────────┘
                           │ CorpusBundle (point-in-time, as_of)
         ┌─────────────────▼──────────┐   ┌────────────▼──────────────┐
         │ RAG                         │   │ Knowledge graph (NetworkX) │
         │ hierarchical, table-aware   │   │ HOLDS_EQUITY/DEBT/DERIV,   │
         │ chunks + temporal/authority │   │ COMPETES_WITH, SUPPLIES,   │
         │ metadata; BM25 + dense →    │   │ TRADED, DISCLOSES_RISK…    │
         │ weighted RRF → rerank → MMR │   │ → Neo4j Cypher export      │
         └─────────────┬──────────────┘   └────────────┬──────────────┘
                       │                                │
              ┌────────▼────────────────────────────────▼────────┐
              │ Analytics: HHI · price-neutral flows · insider    │
              │ signal · robust-z peer stats · WACC curve ·       │
              │ capital rotation · activist vulnerability         │
              └────────────────────────┬──────────────────────────┘
                                       │ EvidenceLedger (E# with lineage)
              ┌────────────────────────▼──────────────────────────┐
              │ StrategicPrescriptiveEngine → recommendations R1–R6│
              │ ClaudeAdvisoryAgent (tool calling + structured     │
              │ output) → GroundingValidator → merge or fall back  │
              └────────────────────────┬──────────────────────────┘
                                       ▼
                    Markdown board pack + JSON (StrategicAdvisoryReport)
```

| Package | What it does |
|---|---|
| `finintel/schemas` | Pydantic v2 models for everything: filings, securities, holdings, insider trades, fundamentals, chunks, citations, evidence, graph findings, the advisory report, and the LLM's structured output. |
| `finintel/entities` | `EntityRegistry`: CUSIP check-digit validation, CUSIP-6 issuer matching, name fuzzing, subsidiary → ultimate parent roll-up, share-class aggregation. |
| `finintel/ingestion` | `EdgarClient`, parsers, market data (Stooq prices, FRED Treasury and ICE BofA OAS curves), `FixtureSource` and `LiveEdgarSource`. |
| `finintel/rag` | Finance-aware tokenizer, chunker, BM25, embedders, rerankers, `FinancialRAGIndex`, `EvidenceLedger`, `GroundingValidator`, `ExtractiveAnswerer`. |
| `finintel/analytics` | Ownership, fundamentals/peer statistics, capital structure and capital return math, risk-theme scoring. |
| `finintel/graph` | Graph builder, cross-company inference (rotation, vulnerability, common ownership, supply chain), Cypher export. |
| `finintel/advisory` | Peer selection and the prescriptive engine. |
| `finintel/agents` | Claude tool definitions, the agent loop, grounding-gated merge, and the `FinancialIntelligencePipeline` façade. |
| `finintel/reporting`, `finintel/cli.py` | Markdown/JSON rendering and the `finintel` CLI. |

---

## 2. Ingestion and entity resolution

**Ownership.** 13F-HR information tables are discovered from each filing's index, parsed (handling the pre-2023 "value in thousands" convention), and resolved to issuers by CUSIP. Rows are typed: `SH` vs `PRN` and `PUT`/`CALL` are kept apart so `HoldingRecord.counts_toward_equity_ownership` excludes debt and options from ownership math. Concentration is the Herfindahl-Hirschman Index over institutional holders, alongside top-10 share, activist % and index %.

**Insiders.** Form 4 XML is parsed into `InsiderTransaction`s; only open-market `P`/`S` codes count as discretionary, and sales flagged as 10b5-1 are discounted. A *cluster buy* is 3+ distinct insiders buying within 30 days.

**Fundamentals.** XBRL `companyfacts` is mapped to a `FinancialStatementSnapshot` with tag fallbacks, picking only facts **filed on or before `as_of`** so the report is point-in-time.

**Text.** 10-K/10-Q/8-K HTML is flattened by a block parser that skips hidden XBRL headers, keeps tables as `TableData` (cleaned of `$`/`%`/spacer cells), and splits on Item headings (1A risk factors, 7 MD&A, 2.02/5.02 for 8-Ks). Transcripts are split into prepared remarks vs. Q&A, with each analyst question/answer pair kept together.

**Entity resolution.** The registry maps CIK ↔ ticker ↔ CUSIP ↔ names and walks `parent_cik` to the ultimate parent. That's what lets **subsidiary debt** roll up to the parent's credit profile while **equity ownership** counts only that issuer's share classes (`economic_equity_shares` converts class B/C at their ratios). Cross-holdings appear as `HOLDS_EQUITY` edges between companies.

---

## 3. Retrieval (hybrid RAG with citation lineage)

* **Chunking** is hierarchical: a parent per Item section, child chunks that never split a table; each table is also emitted as its own Markdown chunk with its caption and the paragraph before it, so numbers stay with their row and column labels.
* **Metadata** on every chunk: issuer, form type, fiscal period, filing date, accession, item, page, and a **temporal tag** (`T`, `T-1`, `T-4` relative to the latest period) plus an **authority weight** (10-K > 10-Q > 8-K > transcript > press release).
* **Retrieval** runs BM25 (with finance phrase canonicalisation such as "free cash flow" → `fcf`, filler-word removal, and ×2.5 weighting of finance concepts) and dense search (feature-hashing embedder by default, or any sentence-transformers model), fuses them with weighted reciprocal rank fusion, applies authority and recency priors, reranks (a feature reranker by default, or a cross-encoder), diversifies with MMR, and expands to the parent section.
* **Point-in-time filtering**: nothing filed after `as_of` can be retrieved.
* **Lineage**: every retrieved chunk becomes an `E#` evidence item carrying a `Citation` (company, form, period, accession, item, page, quote). Computed metrics are also `E#` items, with `derived_from` listing the accessions they came from.
* **Grounding**: `GroundingValidator` checks every factual sentence cites a known `E#` and that every number in it appears in the cited evidence (with scale and rounding tolerance). `ExtractiveAnswerer` (`finintel ask`) answers only with quotes and abstains when coverage is low or the question names an entity the corpus doesn't know.

---

## 4. GraphRAG

Nodes are namespaced (`company:`, `institution:`, `insider:`, `security:`, `risk:`); edges include `HOLDS_EQUITY`, `HOLDS_DEBT`, `HOLDS_DERIVATIVE`, `COMPETES_WITH`, `SUPPLIES`, `SUBSIDIARY_OF`, `INSIDER_OF`, `TRADED`, `DISCLOSES_RISK`.

* **Capital rotation.** For each fund, price-neutral flows (share change × price) out of the target between two 13F periods are matched to inflows into peers in the same window and allocated pro rata so nothing is double counted. Each rotation is annotated with **risk-factor discrepancies**: themes where the target's Item 1A intensity exceeds the destination peer's, flagged *new* when absent from the target's prior-year 10-K.
* **Activist vulnerability** (0–100) blends valuation discount, TSR gap, cash conversion cycle and dividend-yield gaps vs peers, balance-sheet capacity, register dispersion, activist/hedge-fund presence and disclosed leadership change. It returns the drivers alongside the score.
* **Common ownership** is the cosine similarity of *active* holder vectors (index funds excluded, since they would push every pair toward 1.0).
* **Supply chain**: shared upstream suppliers across target and peers.
* `finintel graph --cypher out.cypher` writes `MERGE` statements for Neo4j.

---

## 5. Prescriptive engine

| Rec | Logic |
|---|---|
| **R1 Capital structure** | Synthetic ratings from interest coverage (Damodaran tables), calibrated to the actual rating; a WACC curve across debt/capital with Hamada re-levering and a distress-adjusted value index; a **rating floor** inferred from management text ("middle-A" → A). Optimum = best WACC within the floor. |
| **R2 Capital return** | Valuation discount and robust z vs peers; FCF yield normalised by the consensus EPS revision plus long-run growth as an implied buyback return vs cost of equity; insider signal; excess cash. Chooses the buyback / special dividend / dividend-growth mix and execution (ASR share, 10b5-1 grid, blackout timing). |
| **R3 Investor relations** | Concern severity from analyst Q&A frequency, risk-factor intensity and its change vs the prior 10-K, and rotation pressure. Produces talking points, KPI commitments, things to avoid, and holders to engage. |
| **R4 Activist defense** | Triggered by the vulnerability band; named drivers drive the playbook. |
| **R5 Working capital** | DIO/CCC gap vs peers converted to releasable cash. |
| **R6 M&A posture** | Stock-funded deals discouraged at a trough multiple. |

Every recommendation carries `evidence_ids`, a `quantitative_basis`, KPIs, risks, priority, horizon and confidence.

---

## 6. Agentic LLM layer (Claude)

`ClaudeAdvisoryAgent` uses the Anthropic Python SDK with `claude-opus-5-5`, adaptive thinking, `effort: high` and server-side model fallbacks. It runs a manual tool loop over eight **strict** tools (`search_filings`, `get_peer_benchmark`, `get_ownership_rotation`, `get_activist_vulnerability`, `get_capital_structure`, `get_capital_return`, `get_investor_concerns`, `get_engine_recommendations`) whose outputs reference ledger `E#` ids only, and asks for a structured `LLMNarrative` via `messages.parse`. Refusals, `pause_turn`, `max_tokens` and step budgets are handled explicitly.

The model can **rewrite wording and priorities but can't introduce facts**: `merge_narrative` runs the grounding validator, drops unsupported sentences, keeps the engine's text for any action with unsupported numbers, ignores unknown recommendation ids, and reverts the whole summary if fewer than 85% of its sentences are grounded. No credentials, an API error, or a refusal all produce the deterministic report, and the trace says why.

---

## 7. Running it

**Offline (default).** Uses bundled fixtures; no network or API key.

```
finintel report TGT --peers WMT COST DG KR --out reports      # Markdown + JSON
finintel ask "What did Walmart say about tariffs?"
finintel graph --cypher graph.cypher
```

**Live data.** SEC requires a descriptive User-Agent with a contact email.

```
finintel report TGT --peers WMT COST DG KR --live --user-agent "Your Firm you@example.com"
```

`LiveEdgarSource` pulls submissions, 10-K/10-Q/8-K documents, Form 4s, XBRL company facts, 13F-HRs for a default universe of large filers, Stooq prices and FRED curves. Supply credit ratings and CUSIPs if you have them (constructor arguments); transcripts and consensus are not in EDGAR, so plug your vendor in through `CorpusBundle`.

**Claude narrative.** `pip install -e ".[llm]"` and set `ANTHROPIC_API_KEY`; it's used automatically (`--no-llm` to disable, `--model` to override).

**Better retrieval models.** `pip install -e ".[embeddings]"`, then `--embedder BAAI/bge-base-en-v1.5 --reranker BAAI/bge-reranker-base`.

**Python API.**

```python
from finintel.agents import FinancialIntelligencePipeline, PipelineConfig
from finintel.reporting import to_markdown

pipe = FinancialIntelligencePipeline.from_fixtures(PipelineConfig(use_llm=False))
report = pipe.advise("TGT", ["WMT", "COST", "DG", "KR"])
print(to_markdown(report))
ans = pipe.ask("How is Target thinking about buybacks?")
for claim in ans.claims:
    print(claim.claim, claim.evidence_ids)
for c in ans.citations:
    print(c.render())
```

Rebuild fixtures with `python scripts/build_fixtures.py`.

---

## 8. Data honesty

The offline fixtures use **real** CIKs, tickers and CUSIPs. Fundamentals and prices **approximate** public figures. Institutions (CIK `99000000xx`), 13F positions, insider trades, filing excerpts and transcripts are **synthetic**, with accessions prefixed `FIXTURE-`. The report states this in its data notice. Nothing here is investment advice; use `--live` and your own data before acting on any output.

## 9. Tests

`pytest` runs 64 tests covering entity resolution, every parser, retrieval quality and abstention, analytics and graph inference, the engine's grounding (100% of factual sentences cited, no unsupported numbers), the Claude agent loop against a fake client (tool calls, error results, refusal, step budget, merge rules), the CLI, and `LiveEdgarSource` end to end against a mocked network.
