"""Multimodal ingestion: EDGAR, market/macro adapters, parsers and corpus assembly."""
from .sources import CorpusBundle, FixtureSource, LiveEdgarSource

__all__ = ["CorpusBundle", "FixtureSource", "LiveEdgarSource"]
