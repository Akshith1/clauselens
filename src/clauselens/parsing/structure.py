"""Structure-aware parser for insurance policy forms.

Policy forms follow strong conventions: ALL-CAPS section headings, outline-numbered
clauses (A. / 1. / a. / (1)), and quoted defined terms. This parser turns plain form
text into a PolicyForm tree. See docs/DESIGN.md section 5.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from clauselens.models import Clause, PolicyForm, Section, SectionType
from clauselens.parsing.endorsement import parse_change, parse_modifies

# Outline markers, outermost first. The index is the clause's nesting level.
_MARKERS = [
    re.compile(r"^([A-Z])\.\s+(.*)$"),      # A.
    re.compile(r"^(\d{1,2})\.\s+(.*)$"),    # 1.
    re.compile(r"^([a-z])\.\s+(.*)$"),      # a.
    re.compile(r"^\((\d{1,2})\)\s+(.*)$"),  # (1)
]

# Heading keyword → section type. Checked in order; first match wins.
_HEADING_TYPES: list[tuple[str, SectionType]] = [
    ("ENDORSEMENT PROVISIONS", SectionType.ENDORSEMENT),
    ("DEFINITIONS", SectionType.DEFINITIONS),
    ("EXCLUSIONS", SectionType.EXCLUSIONS),
    ("CONDITIONS", SectionType.CONDITIONS),
    ("LIMITS OF INSURANCE", SectionType.LIMITS),
    ("LIMIT OF LIABILITY", SectionType.LIMITS),
    ("DECLARATIONS", SectionType.DECLARATIONS),
    ("SCHEDULE", SectionType.DECLARATIONS),
    ("INSURING AGREEMENT", SectionType.INSURING_AGREEMENT),
    ("PERILS INSURED", SectionType.INSURING_AGREEMENT),
    ("COVERAGES", SectionType.INSURING_AGREEMENT),
]

_QUOTED = re.compile(r"[\"“]([^\"”]{1,40})[\"”]")
_SMALL_WORDS = {"of", "and", "or", "the", "to", "for", "in"}


def section_type_for(heading: str) -> SectionType | None:
    """Map a heading such as 'SECTION I - EXCLUSIONS' to its section type."""
    upper = heading.upper()
    for keyword, section_type in _HEADING_TYPES:
        if keyword in upper:
            return section_type
    if upper.startswith("SECTION "):
        return SectionType.OTHER
    return None


def quoted_terms(text: str) -> list[str]:
    """Defined terms used in a clause, lower-cased, in order of first use."""
    seen: list[str] = []
    for term in _QUOTED.findall(text):
        term = term.strip().lower()
        if term not in seen:
            seen.append(term)
    return seen


@dataclass
class _Open:
    """A clause still being read: continuation lines are appended to it."""

    clause: Clause
    level: int
    lines: list[str] = field(default_factory=list)


def parse_form(
    text: str,
    form_number: str,
    edition: str = "",
    line_of_business: str = "",
    title: str = "",
    is_endorsement: bool = False,
    known_forms: list[str] | None = None,
) -> PolicyForm:
    """Parse plain form text into a PolicyForm.

    Lines before the first recognised section heading (form number, title, notices)
    are treated as preamble and skipped, apart from an endorsement's "modifies" block.
    """
    form = PolicyForm(form_number, edition, line_of_business, title,
                      is_endorsement=is_endorsement)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if is_endorsement:
        form.modifies = parse_modifies(lines, known_forms or [])

    section: Section | None = None
    stack: list[_Open] = []
    opened: list[_Open] = []

    for line in lines:
        if _is_heading(line):
            section = Section(section_type_for(line) or SectionType.OTHER, _title_case(line))
            form.sections.append(section)
            stack = []
            continue
        if section is None:
            continue

        marker = _match_marker(line)
        if marker is None:
            if not stack:   # text before the section's first numbered clause
                entry = _Open(Clause(clause_id="intro", text=""), level=-1)
                section.clauses.append(entry.clause)
                opened.append(entry)
                stack.append(entry)
            stack[-1].lines.append(line)
            continue

        level, label, body = marker
        while stack and stack[-1].level >= level:
            stack.pop()
        parents = [o for o in stack if o.level >= 0]
        clause_id = f"{parents[-1].clause.clause_id}.{label}" if parents else label
        clause = Clause(clause_id=clause_id, text="",
                        parent_path=[" ".join(o.lines) for o in parents])
        entry = _Open(clause, level, [body])
        section.clauses.append(clause)
        opened.append(entry)
        stack.append(entry)

    for entry in opened:
        entry.clause.text = " ".join(entry.lines)
    for section in form.sections:
        _finish_section(form, section)
    return form


def _finish_section(form: PolicyForm, section: Section) -> None:
    for clause in section.clauses:
        prefix = clause.clause_id + "."
        children = [c for c in section.clauses if c.clause_id.startswith(prefix)]
        clause.full_text = " ".join(
            [clause.text] + [f"({c.clause_id}) {c.text}" for c in children])
        clause.terms_used = quoted_terms(clause.text)
        if section.section_type == SectionType.DEFINITIONS and clause.clause_id != "intro":
            terms = _QUOTED.findall(clause.text[:60])
            if terms:
                clause.defines = terms[0].strip().lower()
        if form.is_endorsement:
            clause.change = parse_change(clause.text)


def _is_heading(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    if len(letters) < 4 or len(line) > 90 or _match_marker(line):
        return False
    return line == line.upper() and section_type_for(line) is not None


def _match_marker(line: str) -> tuple[int, str, str] | None:
    for level, pattern in enumerate(_MARKERS):
        m = pattern.match(line)
        if m:
            return level, m.group(1), m.group(2).strip()
    return None


def _title_case(heading: str) -> str:
    out = []
    for i, word in enumerate(heading.lower().split()):
        if re.fullmatch(r"[ivx]+", word):
            out.append(word.upper())
        elif i > 0 and word in _SMALL_WORDS:
            out.append(word)
        else:
            out.append(word.capitalize())
    return " ".join(out)
