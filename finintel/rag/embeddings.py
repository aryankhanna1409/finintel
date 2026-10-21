"""Dense embedding backends.

* :class:`SentenceTransformerEmbedder`: any sentence-transformers model. Recommended
  domain-adapted choices: ``BAAI/bge-base-en-v1.5`` (strong general retrieval, use the
  query instruction) or a finance-tuned model such as
  ``FinLang/finance-embeddings-investopedia``. Optional dependency.
* :class:`HashingEmbedder`: dependency-free fallback. A signed feature-hashing
  projection of IDF-weighted word unigrams, bigrams and character n-grams over a
  finance-canonicalized vocabulary (synonyms collapsed by :mod:`finintel.rag.text`).
  This is the "domain adaptation" available without model weights: concepts like
  buyback/repurchase or theft/shrink land on identical features.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import Counter
from typing import Protocol, Sequence

import numpy as np

from .text import tokenize

log = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str
    dim: int

    def fit(self, corpus: Sequence[str]) -> "Embedder": ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _h(feature: str, dim: int) -> tuple[int, float]:
    d = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    v = int.from_bytes(d, "little")
    return v % dim, (1.0 if (v >> 63) & 1 else -1.0)


class HashingEmbedder:
    def __init__(self, dim: int = 2048, char_ngrams: tuple[int, ...] = (4,), bigram_weight: float = 0.7, char_weight: float = 0.25) -> None:
        self.name = f"hashing-{dim}"
        self.dim = dim
        self.char_ngrams = char_ngrams
        self.bigram_weight = bigram_weight
        self.char_weight = char_weight
        self._idf: dict[str, float] = {}
        self._default_idf = 1.0

    def _features(self, text: str) -> Counter[str]:
        toks = [t for t in tokenize(text) if not t.replace(".", "").isdigit()]
        feats: Counter[str] = Counter()
        for t in toks:
            feats["w:" + t] += 1.0
            if len(t) > 5 and "_" not in t:
                padded = f"#{t}#"
                for n in self.char_ngrams:
                    for i in range(len(padded) - n + 1):
                        feats["c:" + padded[i : i + n]] += self.char_weight
        for a, b in zip(toks, toks[1:]):
            feats[f"b:{a}|{b}"] += self.bigram_weight
        return feats

    def fit(self, corpus: Sequence[str]) -> "HashingEmbedder":
        df: Counter[str] = Counter()
        for doc in corpus:
            df.update({f for f in self._features(doc) if f.startswith(("w:", "b:"))})
        n = max(len(corpus), 1)
        self._idf = {f: math.log(1 + (n - c + 0.5) / (c + 0.5)) for f, c in df.items()}
        self._default_idf = math.log(1 + (n + 0.5) / 0.5)
        return self

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for f, w in self._features(text).items():
            idf = self._idf.get(f, self._default_idf) if f[0] in "wb" else 1.0
            i, sign = _h(f, self.dim)
            v[i] += sign * (1.0 + math.log(w)) * idf if w >= 1 else sign * w * idf
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class SentenceTransformerEmbedder:
    QUERY_PREFIX = {"bge": "Represent this sentence for searching relevant passages: ", "e5": "query: "}

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", device: str | None = None, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model = SentenceTransformer(model_name, device=device)
        self.name = model_name
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.batch_size = batch_size
        low = model_name.lower()
        self.query_prefix = next((p for k, p in self.QUERY_PREFIX.items() if k in low), "")
        self.doc_prefix = "passage: " if "e5" in low else ""

    def fit(self, corpus: Sequence[str]) -> "SentenceTransformerEmbedder":
        return self

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode([self.doc_prefix + t for t in texts], batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(self.model.encode([self.query_prefix + text], normalize_embeddings=True)[0], dtype=np.float32)


def get_embedder(spec: str | None = None) -> Embedder:
    """``None``/``"hashing"`` → HashingEmbedder; anything else is a sentence-transformers
    model id, falling back to hashing (with a warning) if the package or weights are missing."""
    if not spec or spec == "hashing":
        return HashingEmbedder()
    try:
        return SentenceTransformerEmbedder(spec)
    except Exception as exc:  # noqa: BLE001 - optional backend
        log.warning("sentence-transformers backend %r unavailable (%s); using HashingEmbedder", spec, exc)
        return HashingEmbedder()
