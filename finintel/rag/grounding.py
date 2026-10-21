"""Citation lineage and zero-hallucination guards.

* :class:`EvidenceLedger` assigns stable evidence ids (``E1``, ``E2`` …) to every unit
  of evidence a downstream generator may use: retrieved chunks *and* computed facts
  (metrics, graph inferences, ownership flows). Computed facts carry ``derived_from``
  lineage back to the accessions they were computed from.
* :class:`GroundingValidator` checks generated prose sentence by sentence: every
  factual sentence must cite at least one known evidence id, and every number in it must
  be supported (within rounding tolerance, across %/decimal and $bn/$mm scale changes)
  by the evidence it cites. Unsupported sentences are reported or removed.
* :class:`ExtractiveAnswerer` is a deterministic answerer that can only quote retrieved
  text, and abstains when retrieval confidence is too low.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from ..schemas.common import FormType
from ..schemas.retrieval import Chunk, Citation, EvidenceItem, GroundedAnswer, GroundedClaim, RetrievalQuery, ScoredChunk
from .text import extract_numbers, query_tokens, split_sentences, tokenize

_EID_RE = re.compile(r"\[(E\d+(?:\s*,\s*E\d+)*)\]")


def citation_from_chunk(c: Chunk, citation_id: str) -> Citation:
    md = c.metadata
    fp = None
    if md.fiscal_year is not None:
        fp = f"FY{md.fiscal_year}" if md.fiscal_quarter == 0 else f"Q{md.fiscal_quarter} FY{md.fiscal_year}"
    return Citation(
        citation_id=citation_id, chunk_id=md.chunk_id, accession=md.accession, form_type=md.form_type, company=md.company_name,
        ticker=md.ticker, fiscal_period=fp, item=md.item, page=md.page, filed_date=md.filed_date, source_url=md.source_url,
        is_synthetic=md.is_synthetic, quote=c.text[:300],
    )


class EvidenceLedger:
    def __init__(self) -> None:
        self.items: dict[str, EvidenceItem] = {}
        self._by_chunk: dict[str, str] = {}
        self._n = 0

    def _next(self) -> str:
        self._n += 1
        return f"E{self._n}"

    def add_chunk(self, c: Chunk | ScoredChunk) -> str:
        chunk = c.chunk if isinstance(c, ScoredChunk) else c
        cid = chunk.metadata.chunk_id
        if cid in self._by_chunk:
            return self._by_chunk[cid]
        eid = self._next()
        cit = citation_from_chunk(chunk, eid)
        self.items[eid] = EvidenceItem(evidence_id=eid, kind="chunk", summary=chunk.text, citation=cit, derived_from=[chunk.metadata.accession])
        self._by_chunk[cid] = eid
        return eid

    def add_fact(self, kind: str, summary: str, values: dict[str, float | str | None], derived_from: list[str]) -> str:
        eid = self._next()
        self.items[eid] = EvidenceItem(evidence_id=eid, kind=kind, summary=summary, values=values, derived_from=sorted(set(derived_from)))  # type: ignore[arg-type]
        return eid

    def citations(self) -> list[Citation]:
        return [e.citation for e in self.items.values() if e.citation is not None]

    def render_for_prompt(self, max_chars: int = 900) -> str:
        lines = []
        for e in self.items.values():
            if e.kind == "chunk" and e.citation:
                head = e.citation.render()
                lines.append(f"<evidence id=\"{e.evidence_id}\" kind=\"chunk\" source=\"{head}\">\n{e.summary[:max_chars]}\n</evidence>")
            else:
                vals = ", ".join(f"{k}={_fmt(v)}" for k, v in e.values.items())
                lines.append(f"<evidence id=\"{e.evidence_id}\" kind=\"{e.kind}\" derived_from=\"{', '.join(e.derived_from[:6])}\">\n{e.summary}\nvalues: {vals}\n</evidence>")
        return "\n".join(lines)


def _fmt(v: float | str | None) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


@dataclass
class GroundingReport:
    sentences: int = 0
    factual_sentences: int = 0
    cited_sentences: int = 0
    unknown_evidence_ids: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    uncited_factual: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    @property
    def grounding_rate(self) -> float:
        if self.factual_sentences == 0:
            return 1.0
        bad = len(self.uncited_factual) + len({s for s in self.unsupported_numbers})
        return max(0.0, 1 - bad / self.factual_sentences)

    def as_dict(self) -> dict[str, int | float | list[str]]:
        return {
            "sentences": self.sentences, "factual_sentences": self.factual_sentences, "cited_sentences": self.cited_sentences,
            "grounding_rate": round(self.grounding_rate, 4), "unknown_evidence_ids": self.unknown_evidence_ids,
            "unsupported_numbers": self.unsupported_numbers, "uncited_factual": self.uncited_factual, "removed": self.removed,
        }


class GroundingValidator:
    """Sentence-level citation and numeric-support checker."""

    def __init__(self, ledger: EvidenceLedger, rel_tol: float = 0.02) -> None:
        self.ledger = ledger
        self.rel_tol = rel_tol

    def _evidence_numbers(self, eids: list[str]) -> list[float]:
        nums: list[float] = []
        for eid in eids:
            e = self.ledger.items.get(eid)
            if not e:
                continue
            nums.extend(extract_numbers(e.summary))
            for v in e.values.values():
                if isinstance(v, (int, float)):
                    nums.append(float(v))
                elif isinstance(v, str):
                    nums.extend(extract_numbers(v))
        return nums

    def _supported(self, x: float, pool: list[float], sentence: str) -> bool:
        if abs(x) < 1e-12:
            return True
        # Small integers (counts, item numbers, years, horizons) are structural, not claims.
        if float(x).is_integer() and (abs(x) <= 12 or 1990 <= abs(x) <= 2100):
            return True
        decimals = 0
        m = re.search(rf"{re.escape(f'{abs(x):g}')}", sentence)
        if m and "." in m.group(0):
            decimals = len(m.group(0).split(".")[1])
        rounding = 0.5 * 10 ** (-decimals) if decimals else 0.5
        scales = (1.0, 100.0, 0.01, 1e3, 1e-3, 1e6, 1e-6, 1e9, 1e-9)
        for y in pool:
            for s in scales:
                t = y * s
                if abs(abs(t) - abs(x)) <= max(self.rel_tol * abs(t), rounding if abs(x) >= 1 else self.rel_tol * abs(t) + 1e-9):
                    return True
        return False

    def validate(self, text: str, strict: bool = False) -> tuple[str, GroundingReport]:
        rep = GroundingReport()
        kept: list[str] = []
        for para in text.split("\n"):
            out_sents = []
            for s in split_sentences(para) if para.strip() else []:
                rep.sentences += 1
                ids = [i.strip() for grp in _EID_RE.findall(s) for i in grp.split(",")]
                known = [i for i in ids if i in self.ledger.items]
                rep.unknown_evidence_ids.extend(i for i in ids if i not in self.ledger.items)
                body = _EID_RE.sub("", s)
                nums = [n for n in extract_numbers(body)]
                factual = bool(nums) or bool(ids)
                if factual:
                    rep.factual_sentences += 1
                ok = True
                if ids:
                    rep.cited_sentences += 1
                    pool = self._evidence_numbers(known)
                    for n in nums:
                        if not self._supported(n, pool, body):
                            rep.unsupported_numbers.append(f"{n:g} in: {s[:140]}")
                            ok = False
                    if not known:
                        ok = False
                elif nums:
                    rep.uncited_factual.append(s[:160])
                    ok = False
                if ok or not strict:
                    out_sents.append(s)
                else:
                    rep.removed.append(s[:160])
            kept.append(" ".join(out_sents))
        return "\n".join(kept), rep


class ExtractiveAnswerer:
    """Deterministic, quote-only QA with abstention."""

    def __init__(self, index, min_score: float = 0.2, min_coverage: float = 0.34, max_claims: int = 4) -> None:  # noqa: ANN001
        self.index = index
        self.min_score = min_score
        self.min_coverage = min_coverage
        self.max_claims = max_claims

    def _unknown_entities(self, question: str) -> list[str]:
        """Capitalised, non-initial words (names/tickers) that never occur in the corpus.
        Answering about an entity we hold no documents for is how RAG systems hallucinate."""
        out = []
        words = re.findall(r"(?<!^)(?<![.?!]\s)\b([A-Z][A-Za-z0-9&]{1,}(?:'s)?)", question.strip())
        for w in words:
            tok = tokenize(w.replace("'s", ""))
            if tok and all(t not in self.index.bm25.postings for t in tok):
                out.append(w.replace("'s", ""))
        return out

    def _content_terms(self, question: str) -> set[str]:
        """Query terms that are informative in this corpus: skip entity-name tokens that
        appear in most documents (company names) since they say nothing about the topic."""
        n = max(self.index.bm25.n_docs, 1)
        out = set()
        for t in query_tokens(question):
            df = len(self.index.bm25.postings.get(t, ()))
            if df / n < 0.4:
                out.add(t)
        return out

    def answer(self, question: str, query: RetrievalQuery | None = None, as_of: date | None = None) -> GroundedAnswer:
        q = query or RetrievalQuery(text=question, top_k=6)
        hits = self.index.search(q, as_of=as_of)
        unknown = self._unknown_entities(question)
        if unknown:
            return GroundedAnswer(question=question, claims=[], citations=[], abstained=True,
                                  abstain_reason=f"question names entities not present in the corpus: {', '.join(unknown)}")
        content = self._content_terms(question)
        best_cov = max((len(content & set(tokenize(h.chunk.text))) / len(content) for h in hits), default=0.0) if content else 0.0
        if not hits or max(h.rerank_score for h in hits) < self.min_score or best_cov < self.min_coverage:
            return GroundedAnswer(question=question, claims=[], citations=[], abstained=True,
                                  abstain_reason="no sufficiently relevant evidence retrieved; declining to answer rather than guess")
        ledger = EvidenceLedger()
        q_terms = content or set(query_tokens(question))
        scored: list[tuple[float, str, str]] = []
        for h in hits:
            eid = ledger.add_chunk(h)
            for s in split_sentences(h.chunk.text.replace("\n", " ")):
                overlap = len(q_terms & set(tokenize(s))) / (len(q_terms) or 1)
                authority = h.chunk.metadata.authority_weight
                scored.append((overlap * 0.6 + h.final_score * 0.3 + authority * 0.1, s, eid))
        scored.sort(key=lambda x: -x[0])
        claims, seen = [], set()
        for sc, s, eid in scored:
            key = s[:80]
            if key in seen or sc <= 0.15:
                continue
            seen.add(key)
            claims.append(GroundedClaim(claim=s, evidence_ids=[eid], verified=True, verification_notes=["verbatim extract"]))
            if len(claims) >= self.max_claims:
                break
        used = {e for c in claims for e in c.evidence_ids}
        cits = [c for c in ledger.citations() if c.citation_id in used]
        if not claims:
            return GroundedAnswer(question=question, claims=[], citations=[], abstained=True, abstain_reason="retrieved passages do not address the question")
        return GroundedAnswer(question=question, claims=claims, citations=cits)


def authority_rank(form: FormType) -> int:
    order = [FormType.TEN_K, FormType.TEN_Q, FormType.EIGHT_K, FormType.TRANSCRIPT, FormType.CONSENSUS, FormType.NEWS]
    return order.index(form) if form in order else len(order)
