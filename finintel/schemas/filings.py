"""Raw and parsed filing / document models."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import Field, computed_field

from .common import AUTHORITY_WEIGHTS, Accession, DataProvenance, FiscalPeriod, FormType, StrictModel


class RawFiling(StrictModel):
    """A filing (or filing-like document such as a transcript) before parsing."""

    accession: Accession
    cik: str
    company_name: str
    ticker: str | None = None
    form_type: FormType
    filed_date: date
    period: FiscalPeriod | None = None
    primary_document_url: str | None = None
    content_type: Literal["html", "text", "xml", "json"] = "html"
    content: str
    provenance: DataProvenance

    @computed_field  # type: ignore[prop-decorator]
    @property
    def authority_weight(self) -> float:
        return AUTHORITY_WEIGHTS[self.form_type]


class FilingSection(StrictModel):
    """A logical section of a filing (e.g. Item 1A) holding ordered blocks."""

    item: str  # "1A", "7", "7A", "Q&A", "PREPARED_REMARKS", "8.01" ...
    title: str
    blocks: list["ContentBlock"] = Field(default_factory=list)
    page_start: int | None = None


class TableData(StrictModel):
    caption: str | None = None
    header: list[str]
    rows: list[list[str]]
    units: str | None = None  # "$ in millions", "%" ...

    def to_markdown(self, rows: list[list[str]] | None = None) -> str:
        rows = self.rows if rows is None else rows
        width = max([len(self.header)] + [len(r) for r in rows]) if (self.header or rows) else 0
        header = self.header + [""] * (width - len(self.header))
        lines = []
        if self.caption:
            lines.append(f"**{self.caption}**" + (f" ({self.units})" if self.units else ""))
        lines.append("| " + " | ".join(h.replace("|", "\\|") for h in header) + " |")
        lines.append("|" + "|".join(["---"] * width) + "|")
        for r in rows:
            r = r + [""] * (width - len(r))
            lines.append("| " + " | ".join(c.replace("|", "\\|") for c in r) + " |")
        return "\n".join(lines)


class ContentBlock(StrictModel):
    kind: Literal["paragraph", "table", "heading", "speaker_turn"]
    text: str = ""
    table: TableData | None = None
    page: int | None = None
    speaker: str | None = None
    speaker_role: str | None = None  # "management" | "analyst" | "operator"


class ParsedDocument(StrictModel):
    filing: RawFiling
    sections: list[FilingSection]


FilingSection.model_rebuild()
