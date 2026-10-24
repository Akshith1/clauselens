"""Endorsement-specific parsing: which forms an endorsement modifies, and how.

Endorsements state their changes in a small set of stock phrases, for example:
  "Paragraph A.2.b. in Section I - Exclusions is deleted."
  "Paragraph B.1. in Section I - Conditions is replaced by the following:"
  "The following is added to Section I - Perils Insured Against, Paragraph B.:"
  "The most we will pay for loss covered by this endorsement is ..."
"""

from __future__ import annotations

import re

from clauselens.models import EndorsementAction, EndorsementChange, SectionType

_MODIFIES = re.compile(r"endorsement modifies insurance provided under", re.I)
_PARAGRAPH = r"[Pp]aragraph\s+([A-Za-z0-9.()]+?)\.?"
_SECTION = r"(Section\s+[^,.:]+?(?:\s-\s[^,.:]+)?)"

_DELETE = re.compile(_PARAGRAPH + r"\s+(?:in|of)\s+" + _SECTION + r"\s+is\s+deleted", re.I)
_REPLACE = re.compile(_PARAGRAPH + r"\s+(?:in|of)\s+" + _SECTION + r"\s+is\s+replaced", re.I)
_ADD = re.compile(r"following\s+is\s+added\s+to\s+" + _SECTION
                  + r"(?:,\s*" + _PARAGRAPH + r")?\s*:", re.I)
_LIMIT = re.compile(r"\bthe most we will pay\b|\bdeductible of\b|\bdeductible applies\b", re.I)


def parse_modifies(lines: list[str], known_forms: list[str]) -> list[str]:
    """Form numbers listed after "This endorsement modifies insurance provided under…"."""
    for i, line in enumerate(lines):
        if not _MODIFIES.search(line):
            continue
        block = []
        for following in lines[i + 1:i + 6]:
            if following.isupper() and any(k in following for k in ("SCHEDULE", "PROVISIONS")):
                break
            block.append(following)
        text = " ".join(block)
        found = [f for f in known_forms if f in text]
        if found:
            return found
        return [ln.split()[0] for ln in block if ln.split()]
    return []


def parse_change(text: str) -> EndorsementChange | None:
    """Recognise the change instruction in one endorsement clause, if any."""
    from clauselens.parsing.structure import section_type_for

    for pattern, action in ((_DELETE, EndorsementAction.DELETES),
                            (_REPLACE, EndorsementAction.REPLACES)):
        m = pattern.search(text)
        if m:
            return EndorsementChange(action, section_type_for(m.group(2)) or SectionType.OTHER,
                                     _clause_id(m.group(1)))
    m = _ADD.search(text)
    if m:
        target = _clause_id(m.group(2)) if m.group(2) else None
        return EndorsementChange(EndorsementAction.ADDS,
                                 section_type_for(m.group(1)) or SectionType.OTHER, target)
    if _LIMIT.search(text):
        return EndorsementChange(EndorsementAction.LIMITS)
    return None


def _clause_id(raw: str) -> str:
    """'A.2.b.' → 'A.2.b'; '(1)' style parts are kept as-is."""
    return raw.strip().rstrip(".")
