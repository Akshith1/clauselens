"""Clause store interface and the flat record every backend stores."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Protocol

from clauselens.models import (
    Citation,
    EndorsementAction,
    EndorsementChange,
    PolicyForm,
    SectionType,
)


@dataclass
class ClauseRecord:
    """One clause, flattened for indexing. Lists are stored as JSON strings by backends
    that only accept scalar columns."""

    record_id: str
    form_number: str
    form_title: str
    is_endorsement: bool
    modifies: list[str]
    section_type: str
    section_heading: str
    clause_id: str
    text: str
    full_text: str
    parent_path: list[str] = field(default_factory=list)
    defines: str = ""
    terms_used: list[str] = field(default_factory=list)
    change: dict | None = None

    @property
    def search_text(self) -> str:
        """What gets embedded/searched: heading and parents give a sub-clause its context."""
        return " | ".join([self.section_heading, *self.parent_path, self.full_text])

    def citation(self) -> Citation:
        return Citation(self.form_number, self.section_heading, self.clause_id, self.full_text)

    def endorsement_change(self) -> EndorsementChange | None:
        if not self.change:
            return None
        section = self.change.get("target_section")
        return EndorsementChange(
            EndorsementAction(self.change["action"]),
            SectionType(section) if section else None,
            self.change.get("target_clause_id"),
        )

    def to_row(self) -> dict:
        """Scalar-only dict for Delta tables / Chroma metadata."""
        row = asdict(self)
        row["modifies"] = json.dumps(self.modifies)
        row["parent_path"] = json.dumps(self.parent_path)
        row["terms_used"] = json.dumps(self.terms_used)
        row["change"] = json.dumps(self.change) if self.change else ""
        row["search_text"] = self.search_text
        return row

    @classmethod
    def from_row(cls, row: dict) -> ClauseRecord:
        def load(value, default):
            if value in (None, ""):
                return default
            return json.loads(value) if isinstance(value, str) else value

        return cls(
            record_id=row["record_id"],
            form_number=row["form_number"],
            form_title=row.get("form_title", ""),
            is_endorsement=bool(row.get("is_endorsement")),
            modifies=load(row.get("modifies"), []),
            section_type=row["section_type"],
            section_heading=row["section_heading"],
            clause_id=row["clause_id"],
            text=row["text"],
            full_text=row.get("full_text") or row["text"],
            parent_path=load(row.get("parent_path"), []),
            defines=row.get("defines") or "",
            terms_used=load(row.get("terms_used"), []),
            change=load(row.get("change"), None),
        )


@dataclass
class ScoredClause:
    record: ClauseRecord
    score: float


class ClauseStore(Protocol):
    def upsert(self, records: list[ClauseRecord]) -> None: ...

    def search(
        self,
        query: str,
        k: int,
        form_numbers: list[str] | None = None,
        section_types: list[str] | None = None,
    ) -> list[ScoredClause]: ...

    def definitions(self, terms: list[str], form_numbers: list[str]) -> dict[str, ClauseRecord]: ...

    def form_clauses(self, form_number: str) -> list[ClauseRecord]: ...


def form_to_records(form: PolicyForm) -> list[ClauseRecord]:
    records = []
    for index, section in enumerate(form.sections):
        for clause in section.clauses:
            change = None
            if clause.change:
                change = {
                    "action": clause.change.action.value,
                    "target_section": clause.change.target_section.value
                    if clause.change.target_section else None,
                    "target_clause_id": clause.change.target_clause_id,
                }
            records.append(ClauseRecord(
                record_id=f"{form.form_number}::{index}::{clause.clause_id}",
                form_number=form.form_number,
                form_title=form.title,
                is_endorsement=form.is_endorsement,
                modifies=form.modifies,
                section_type=section.section_type.value,
                section_heading=section.heading,
                clause_id=clause.clause_id,
                text=clause.text,
                full_text=clause.full_text or clause.text,
                parent_path=clause.parent_path,
                defines=clause.defines or "",
                terms_used=clause.terms_used,
                change=change,
            ))
    return records
