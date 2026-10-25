"""Domain model for ClauseLens. See docs/DESIGN.md section 4."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class SectionType(StrEnum):
    DECLARATIONS = "declarations"
    INSURING_AGREEMENT = "insuring_agreement"
    DEFINITIONS = "definitions"
    EXCLUSIONS = "exclusions"
    CONDITIONS = "conditions"
    LIMITS = "limits"
    ENDORSEMENT = "endorsement"
    OTHER = "other"


class Verdict(StrEnum):
    COVERED = "covered"
    NOT_COVERED = "not_covered"
    DEPENDS = "depends"
    CANNOT_DETERMINE = "cannot_determine"
    ANSWERED = "answered"        # answered from the customer's policy record


class QuestionType(StrEnum):
    COVERAGE = "coverage"
    DEFINITION = "definition"
    LIMIT = "limit"
    CONDITION = "condition"
    ACCOUNT = "account"          # the customer's own policy record: dates, premium, payments
    OUT_OF_SCOPE = "out_of_scope"


class EndorsementAction(StrEnum):
    ADDS = "adds"
    DELETES = "deletes"
    REPLACES = "replaces"
    LIMITS = "limits"


@dataclass
class EndorsementChange:
    """One endorsement instruction, e.g. "Paragraph A.2.b. in Section I - Exclusions is deleted"."""

    action: EndorsementAction
    target_section: SectionType | None = None   # section of the base form it changes
    target_clause_id: str | None = None         # e.g. "A.2.b"; None = whole section


@dataclass
class Clause:
    clause_id: str                   # outline path within its section, e.g. "A.2.b"
    text: str                        # this clause's own text, without the outline marker
    full_text: str = ""              # this clause plus all sub-clauses
    parent_path: list[str] = field(default_factory=list)   # ancestor clause texts, outermost first
    defines: str | None = None       # defined term, for clauses in a Definitions section
    terms_used: list[str] = field(default_factory=list)     # quoted defined terms used
    change: EndorsementChange | None = None                 # for endorsement clauses


@dataclass
class Section:
    section_type: SectionType
    heading: str
    clauses: list[Clause] = field(default_factory=list)


@dataclass
class PolicyForm:
    form_number: str                 # e.g. "CL-HO-01"
    edition: str                     # e.g. "10 26"
    line_of_business: str
    title: str
    sections: list[Section] = field(default_factory=list)
    is_endorsement: bool = False
    modifies: list[str] = field(default_factory=list)   # base form numbers, for endorsements


@dataclass
class Citation:
    form_number: str
    section_heading: str
    clause_id: str
    text: str
    ref: str = ""        # label used in the answer text, e.g. "2" or "E1"

    def label(self) -> str:
        return f"{self.form_number}, {self.section_heading}, {self.clause_id}"


@dataclass
class EndorsementEffect:
    endorsement_form: str
    action: EndorsementAction
    summary: str
    citation: Citation


@dataclass
class CoverageAnswer:
    question_type: QuestionType
    verdict: Verdict
    explanation: str
    citations: list[Citation] = field(default_factory=list)
    definitions_used: dict[str, str] = field(default_factory=dict)
    endorsement_effects: list[EndorsementEffect] = field(default_factory=list)
    base_verdict: Verdict | None = None   # verdict before endorsements were applied
