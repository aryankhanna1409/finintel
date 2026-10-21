"""The financial RAG index: chunk store, dense vectors, BM25, and hybrid retrieval.

Retrieval pipeline for a :class:`RetrievalQuery`:

1. **Metadata pre-filter** (issuer, form, item, chunk type, authority floor and the
   point-in-time ``filed_on_or_before`` guard against look-ahead bias).
2. **Dense** cosine search and **sparse** BM25 over the filtered set, each truncated to
   ``candidate_pool``.
3. **Fusion**: weighted Reciprocal Rank Fusion (default) or min-max convex fusion.
4. **Priors**: source authority (10-K > transcript > news) and recency decay.
5. **Rerank** with a cross-encoder (or the feature reranker) and blend with priors.
6. **MMR** diversification so the top-k isn't five near-identical paragraphs.
7. Optional **small-to-big** parent expansion.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np

from ..ingestion.parsers import parse_document
from ..ingestion.sources import CorpusBundle
from ..schemas.retrieval import Chunk, RetrievalFilter, RetrievalQuery, ScoredChunk
from .chunking import HierarchicalChunker
from .embeddings import Embedder, HashingEmbedder
from .rerank import FinancialFeatureReranker, Reranker
from .sparse import BM25Index

log = logging.getLogger(__name__)


class FinancialRAGIndex:
    def __init__(self, embedder: Embedder | None = None, reranker: Reranker | None = None, chunker: HierarchicalChunker | None = None) -> None:
        self.embedder: Embedder = embedder or HashingEmbedder()
        self.chunker = chunker or HierarchicalChunker()
        self._reranker = reranker
        self.chunks: list[Chunk] = []
        self.parents: dict[str, Chunk] = {}
        self.by_id: dict[str, int] = {}
        self.vectors = np.zeros((0, getattr(self.embedder, "dim", 1)), dtype=np.float32)
        self.bm25 = BM25Index()

    # ------------------------------------------------------------------ build
    def build_from_bundle(self, bundle: CorpusBundle) -> "FinancialRAGIndex":
        anchors: dict[str, int] = {}
        for f in bundle.filings:
            if f.period and f.filed_date <= bundle.as_of and f.form_type.value in ("10-K", "10-Q", "TRANSCRIPT", "8-K"):
                anchors[f.cik] = max(anchors.get(f.cik, 0), f.period.quarter_index())
        children: list[Chunk] = []
        for f in bundle.filings:
            doc = parse_document(f)
            c, p = self.chunker.chunk_document(doc, anchors.get(f.cik))
            children.extend(c)
            for pc in p:
                self.parents[pc.metadata.chunk_id] = pc
        return self.add_chunks(children)

    def add_chunks(self, chunks: Sequence[Chunk]) -> "FinancialRAGIndex":
        self.chunks.extend(chunks)
        self.by_id = {c.metadata.chunk_id: i for i, c in enumerate(self.chunks)}
        texts = [c.index_text or c.text for c in self.chunks]
        self.embedder.fit(texts)
        self.vectors = self.embedder.embed_documents(texts)
        self.bm25.build(texts)
        if self._reranker is None or isinstance(self._reranker, FinancialFeatureReranker):
            self._reranker = FinancialFeatureReranker(self.bm25.idf)
        return self

    @property
    def reranker(self) -> Reranker:
        assert self._reranker is not None
        return self._reranker

    def get(self, chunk_id: str) -> Chunk | None:
        i = self.by_id.get(chunk_id)
        if i is not None:
            return self.chunks[i]
        return self.parents.get(chunk_id)

    # ------------------------------------------------------------------ filtering
    def _mask(self, f: RetrievalFilter) -> np.ndarray:
        m = np.ones(len(self.chunks), dtype=bool)
        for i, c in enumerate(self.chunks):
            md = c.metadata
            if f.issuer_ciks is not None and md.issuer_cik not in f.issuer_ciks:
                m[i] = False
            elif f.form_types is not None and md.form_type not in f.form_types:
                m[i] = False
            elif f.items is not None and (md.item or "") not in f.items:
                m[i] = False
            elif f.chunk_types is not None and md.chunk_type not in f.chunk_types:
                m[i] = False
            elif f.filed_on_or_before is not None and md.filed_date > f.filed_on_or_before:
                m[i] = False
            elif f.filed_on_or_after is not None and md.filed_date < f.filed_on_or_after:
                m[i] = False
            elif f.relative_periods is not None and md.relative_period not in f.relative_periods:
                m[i] = False
            elif md.authority_weight < f.min_authority:
                m[i] = False
            elif f.exclude_synthetic and md.is_synthetic:
                m[i] = False
        return m

    # ------------------------------------------------------------------ search
    def search(self, q: RetrievalQuery, as_of: date | None = None) -> list[ScoredChunk]:
        if not self.chunks:
            return []
        mask = self._mask(q.filters)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return []
        qv = self.embedder.embed_query(q.text)
        dense_all = self.vectors[idx] @ qv
        sparse_all = self.bm25.scores(q.text, mask)[idx]
        pool = min(q.candidate_pool, idx.size)
        d_order = np.argsort(-dense_all)[:pool]
        s_order = np.argsort(-sparse_all)[:pool]
        s_order = s_order[sparse_all[s_order] > 0]
        d_rank = {int(idx[j]): r for r, j in enumerate(d_order)}
        s_rank = {int(idx[j]): r for r, j in enumerate(s_order)}
        cand = sorted(set(d_rank) | set(s_rank))
        dense_map = {int(idx[j]): float(dense_all[j]) for j in range(idx.size)}
        sparse_map = {int(idx[j]): float(sparse_all[j]) for j in range(idx.size)}

        fused: dict[int, float] = {}
        if q.use_rrf:
            k = 60.0
            for i in cand:
                fused[i] = q.dense_weight * (1 / (k + d_rank[i] + 1) if i in d_rank else 0.0) + (1 - q.dense_weight) * (
                    1 / (k + s_rank[i] + 1) if i in s_rank else 0.0
                )
        else:
            dv = np.array([dense_map[i] for i in cand])
            sv = np.array([sparse_map[i] for i in cand])
            dn = (dv - dv.min()) / (np.ptp(dv) or 1)
            sn = (sv - sv.min()) / (np.ptp(sv) or 1)
            fused = {i: float(q.dense_weight * a + (1 - q.dense_weight) * b) for i, a, b in zip(cand, dn, sn)}

        ref = as_of or max(c.metadata.filed_date for c in self.chunks)
        prior: dict[int, float] = {}
        for i in cand:
            md = self.chunks[i].metadata
            age = max((ref - md.filed_date).days, 0)
            recency = 0.5 ** (age / q.recency_half_life_days)
            prior[i] = fused[i] * (1 + q.authority_boost * (md.authority_weight - 0.75) * 2) * (0.7 + 0.3 * recency)

        top = sorted(cand, key=lambda i: -prior[i])[: max(q.top_k * 4, 20)]
        rr = self.reranker.score(q.text, [self.chunks[i] for i in top])
        pv = np.array([prior[i] for i in top])
        pn = (pv - pv.min()) / (np.ptp(pv) or 1)
        final = {i: 0.65 * r + 0.35 * float(p) for i, r, p in zip(top, rr, pn)}
        rerank_map = dict(zip(top, rr))

        chosen = self._mmr(top, final, q.top_k, q.mmr_lambda)
        out = []
        for i in chosen:
            ch = self.chunks[i]
            if q.expand_parents and ch.metadata.parent_id and ch.metadata.parent_id in self.parents:
                parent = self.parents[ch.metadata.parent_id]
                ch = ch.model_copy(update={"text": ch.text + "\n\n[section context]\n" + parent.text[:2000]})
            out.append(
                ScoredChunk(
                    chunk=ch, dense_score=dense_map[i], sparse_score=sparse_map[i], fused_score=fused[i],
                    rerank_score=rerank_map[i], final_score=final[i], dense_rank=d_rank.get(i), sparse_rank=s_rank.get(i),
                )
            )
        return out

    def _mmr(self, cand: list[int], score: dict[int, float], k: int, lam: float) -> list[int]:
        chosen: list[int] = []
        remaining = sorted(cand, key=lambda i: -score[i])
        while remaining and len(chosen) < k:
            if not chosen:
                chosen.append(remaining.pop(0))
                continue
            cv = self.vectors[chosen]
            best, best_val = None, -1e9
            for i in remaining:
                sim = float(np.max(cv @ self.vectors[i]))
                val = lam * score[i] - (1 - lam) * sim
                if val > best_val:
                    best, best_val = i, val
            chosen.append(best)  # type: ignore[arg-type]
            remaining.remove(best)  # type: ignore[arg-type]
        return chosen

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.vectors)
        (d / "chunks.jsonl").write_text("\n".join(c.model_dump_json() for c in self.chunks), encoding="utf-8")
        (d / "parents.jsonl").write_text("\n".join(c.model_dump_json() for c in self.parents.values()), encoding="utf-8")
        (d / "meta.json").write_text(json.dumps({"embedder": getattr(self.embedder, "name", "?"), "n": len(self.chunks)}), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder | None = None) -> "FinancialRAGIndex":
        d = Path(directory)
        idx = cls(embedder=embedder)
        chunks = [Chunk.model_validate_json(line) for line in (d / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line]
        idx.parents = {
            c.metadata.chunk_id: c
            for c in (Chunk.model_validate_json(line) for line in (d / "parents.jsonl").read_text(encoding="utf-8").splitlines() if line)
        }
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if meta.get("embedder") == getattr(idx.embedder, "name", None):
            # Same embedder: reuse persisted vectors, rebuild only the cheap structures.
            idx.chunks = chunks
            idx.by_id = {c.metadata.chunk_id: i for i, c in enumerate(chunks)}
            texts = [c.index_text or c.text for c in chunks]
            idx.embedder.fit(texts)
            idx.vectors = np.load(d / "vectors.npy")
            idx.bm25.build(texts)
            idx._reranker = FinancialFeatureReranker(idx.bm25.idf)
        else:
            log.info("embedder changed (%s -> %s); re-embedding", meta.get("embedder"), getattr(idx.embedder, "name", None))
            idx.add_chunks(chunks)
        return idx
