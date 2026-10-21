"""Hierarchical, table-aware chunking with temporal and source-authority metadata.

Hierarchy::

    Document (accession)
      └── Section parent chunk (Item 1A / Item 7 / Q&A)       ← "big" context for expansion
            ├── paragraph chunks (semantic sentence grouping)  ← retrieval units
            ├── table chunks (atomic; row-split w/ header)     ← retrieval units
            └── qa_pair chunks (analyst Q + mgmt answers)      ← retrieval units

Rules that matter for finance:
* Tables are never split mid-row and every split repeats the header row and caption,
  so a retrieved fragment always carries its column semantics (period, units).
* Every chunk's ``index_text`` is prefixed with a contextual header (company, form,
  fiscal period, item) so isolated fragments remain unambiguous to both retrievers.
* Paragraph chunks break where adjacent-sentence similarity drops (topic shift) once a
  minimum size is reached, otherwise at a hard token ceiling.
* Q&A transcripts are chunked as question→answer exchanges, never splitting an answer
  from the question it responds to.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date

import numpy as np

from ..schemas.common import AUTHORITY_WEIGHTS, FormType, tier_for
from ..schemas.filings import ContentBlock, FilingSection, ParsedDocument, TableData
from ..schemas.retrieval import Chunk, ChunkMetadata
from .embeddings import Embedder, HashingEmbedder
from .text import split_sentences

SECTION_NAMES = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments", "2": "Properties", "3": "Legal Proceedings",
    "5": "Market for Registrant's Common Equity", "7": "MD&A", "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements", "I-2": "MD&A", "II-1A": "Risk Factors", "Q&A": "Earnings Call Q&A",
    "PREPARED_REMARKS": "Earnings Call Prepared Remarks", "2.02": "Results of Operations", "5.02": "Officer Changes",
    "7.01": "Regulation FD", "8.01": "Other Events",
}


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _cid(*parts: object) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


@dataclass
class ChunkerConfig:
    max_tokens: int = 220
    min_tokens: int = 60
    overlap_sentences: int = 1
    max_table_tokens: int = 450
    semantic_break_threshold: float = 0.12  # adjacent-sentence cosine below this = topic shift


class HierarchicalChunker:
    def __init__(self, config: ChunkerConfig | None = None, sentence_embedder: Embedder | None = None) -> None:
        self.cfg = config or ChunkerConfig()
        self.embedder = sentence_embedder or HashingEmbedder(dim=1024)

    # ------------------------------------------------------------------ public
    def chunk_document(self, doc: ParsedDocument, anchor_quarter_index: int | None = None) -> tuple[list[Chunk], list[Chunk]]:
        """Return (retrieval_chunks, parent_chunks)."""
        children: list[Chunk] = []
        parents: list[Chunk] = []
        for s_idx, section in enumerate(doc.sections):
            base = self._base_meta(doc, section, anchor_quarter_index)
            parent_id = _cid(doc.filing.accession.value, section.item, s_idx)
            parent_text = "\n\n".join(b.text for b in section.blocks)
            parents.append(self._make(parent_text, base, parent_id, None, "section", 0, None))
            if section.item in ("Q&A",):
                children.extend(self._qa_chunks(section, base, parent_id))
                continue
            if section.item == "PREPARED_REMARKS":
                pos = 0
                for blk in section.blocks:
                    for c in self._flush_paragraphs([blk], base, parent_id, pos):
                        meta = c.metadata.model_copy(update={"chunk_type": "remarks", "speaker": blk.speaker, "speaker_role": blk.speaker_role})
                        text = f"{blk.speaker or 'Speaker'}: {c.text}" if blk.speaker else c.text
                        children.append(Chunk(text=text, metadata=meta, index_text=f"{self._header(base)}\n{text}"))
                        pos += 1
                continue
            pos = 0
            para_buffer: list[ContentBlock] = []
            for blk in section.blocks:
                if blk.kind == "table" and blk.table is not None:
                    for c in self._flush_paragraphs(para_buffer, base, parent_id, pos):
                        children.append(c)
                        pos += 1
                    para_buffer = []
                    for c in self._table_chunks(blk.table, blk.page, base, parent_id, pos):
                        children.append(c)
                        pos += 1
                else:
                    para_buffer.append(blk)
            for c in self._flush_paragraphs(para_buffer, base, parent_id, pos):
                children.append(c)
                pos += 1
        return children, parents

    # ------------------------------------------------------------------ metadata
    def _base_meta(self, doc: ParsedDocument, section: FilingSection, anchor_qi: int | None) -> dict:
        f = doc.filing
        period = f.period
        rel = None
        if period is not None and anchor_qi is not None:
            lag = anchor_qi - period.quarter_index()
            rel = "T" if lag <= 0 else f"T-{lag}"
        return dict(
            doc_id=f.accession.value,
            issuer_cik=f.cik,
            ticker=f.ticker,
            company_name=f.company_name,
            form_type=f.form_type,
            source_tier=tier_for(f.form_type),
            authority_weight=AUTHORITY_WEIGHTS[f.form_type],
            accession=f.accession.value,
            is_synthetic=f.provenance.is_synthetic or f.accession.is_fixture,
            filed_date=f.filed_date,
            period_end=period.period_end if period else None,
            fiscal_year=period.fiscal_year if period else None,
            fiscal_quarter=period.fiscal_quarter if period else None,
            relative_period=rel,
            item=section.item,
            section_title=SECTION_NAMES.get(section.item, section.title),
            source_url=f.primary_document_url,
        )

    def _header(self, base: dict) -> str:
        fp = ""
        if base.get("fiscal_year") is not None:
            fp = f"FY{base['fiscal_year']}" if base.get("fiscal_quarter") == 0 else f"Q{base['fiscal_quarter']} FY{base['fiscal_year']}"
        form = base["form_type"].value if isinstance(base["form_type"], FormType) else base["form_type"]
        return f"[{base.get('ticker') or ''} | {base['company_name']} | {form} {fp} | {base.get('section_title') or base.get('item')} | filed {base['filed_date']}]"

    def _make(self, text: str, base: dict, chunk_id: str, parent_id: str | None, ctype: str, pos: int, page: int | None, extra_index: str = "", **kw) -> Chunk:
        meta = ChunkMetadata(chunk_id=chunk_id, parent_id=parent_id, chunk_type=ctype, position=pos, page=page, **base, **kw)  # type: ignore[arg-type]
        return Chunk(text=text, metadata=meta, index_text=f"{self._header(base)}\n{text}" + (f"\n{extra_index}" if extra_index else ""))

    # ------------------------------------------------------------------ paragraphs
    def _flush_paragraphs(self, blocks: list[ContentBlock], base: dict, parent_id: str, pos0: int) -> list[Chunk]:
        sents: list[tuple[str, int | None]] = []
        for b in blocks:
            if b.kind == "heading":
                continue
            for s in split_sentences(b.text):
                sents.append((s, b.page))
        if not sents:
            return []
        texts = [s for s, _ in sents]
        sims = self._adjacent_similarity(texts)
        groups: list[list[int]] = []
        cur: list[int] = []
        cur_tok = 0
        for i, s in enumerate(texts):
            t = _est_tokens(s)
            topic_shift = i > 0 and sims[i - 1] < self.cfg.semantic_break_threshold
            if cur and (cur_tok + t > self.cfg.max_tokens or (topic_shift and cur_tok >= self.cfg.min_tokens)):
                groups.append(cur)
                ov = cur[-self.cfg.overlap_sentences:] if self.cfg.overlap_sentences and not topic_shift else []
                cur = list(ov)
                cur_tok = sum(_est_tokens(texts[j]) for j in cur)
            cur.append(i)
            cur_tok += t
        if cur:
            groups.append(cur)
        out = []
        for k, g in enumerate(groups):
            text = " ".join(texts[j] for j in g)
            page = sents[g[0]][1]
            out.append(self._make(text, base, _cid(parent_id, "p", pos0 + k, text[:40]), parent_id, "paragraph", pos0 + k, page))
        return out

    def _adjacent_similarity(self, sents: list[str]) -> list[float]:
        if len(sents) < 2:
            return []
        m = self.embedder.embed_documents(sents)
        return [float(np.dot(m[i], m[i + 1])) for i in range(len(sents) - 1)]

    # ------------------------------------------------------------------ tables
    def _table_chunks(self, table: TableData, page: int | None, base: dict, parent_id: str, pos0: int) -> list[Chunk]:
        rows = table.rows
        full = table.to_markdown()
        if _est_tokens(full) <= self.cfg.max_table_tokens or len(rows) <= 1:
            groups = [rows]
        else:
            header_tok = _est_tokens(table.to_markdown(rows=[]))
            groups, cur, cur_tok = [], [], header_tok
            for r in rows:
                rt = _est_tokens(" | ".join(r)) + 2
                if cur and cur_tok + rt > self.cfg.max_table_tokens:
                    groups.append(cur)
                    cur, cur_tok = [], header_tok
                cur.append(r)
                cur_tok += rt
            if cur:
                groups.append(cur)
        out = []
        for k, g in enumerate(groups):
            md = table.to_markdown(rows=g)
            if len(groups) > 1:
                md += f"\n(table part {k + 1} of {len(groups)})"
            linear = "\n".join(
                "; ".join(f"{h or 'col'}: {c}" for h, c in zip(table.header, r) if c) for r in g
            )
            out.append(self._make(md, base, _cid(parent_id, "t", pos0 + k, md[:40]), parent_id, "table", pos0 + k, page, extra_index=linear))
        return out

    # ------------------------------------------------------------------ transcripts
    def _qa_chunks(self, section: FilingSection, base: dict, parent_id: str) -> list[Chunk]:
        exchanges: list[list[ContentBlock]] = []
        cur: list[ContentBlock] = []
        for b in section.blocks:
            if b.speaker_role == "analyst" and cur:
                exchanges.append(cur)
                cur = []
            cur.append(b)
        if cur:
            exchanges.append(cur)
        out = []
        for k, ex in enumerate(exchanges):
            text = "\n".join(f"{b.speaker or 'Unknown'} ({b.speaker_role or 'n/a'}): {b.text}" for b in ex)
            q = next((b for b in ex if b.speaker_role == "analyst"), None)
            out.append(
                self._make(text, base, _cid(parent_id, "qa", k, text[:40]), parent_id, "qa_pair", k, None,
                           speaker=q.speaker if q else None, speaker_role="analyst" if q else "management")
            )
        return out


def quarter_index_for(d: date, fiscal_year: int, fiscal_quarter: int) -> int:
    q = 4 if fiscal_quarter == 0 else fiscal_quarter
    return fiscal_year * 4 + (q - 1)
