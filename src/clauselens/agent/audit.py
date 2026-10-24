"""Citation audit: the last gate before an answer reaches the user.

Every sentence must cite at least one clause that was actually given to the model.
Sentences without a valid citation are dropped; if nothing survives, the answer becomes a
refusal. Citations the model invented (labels never shown to it) are stripped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_REF = re.compile(r"\[(E?\d+)\]")
_GROUPED = re.compile(r"\[((?:E?\d+)(?:\s*,\s*E?\d+)+)\]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"])")

REFUSAL = ("The policy wording provided doesn't answer this question, so ClauseLens can't "
           "give a grounded answer. Try naming the form, peril or coverage more specifically.")


@dataclass
class AuditResult:
    answer: str
    refs: list[str]          # labels cited by the surviving sentences, first-use order
    dropped: list[str]       # sentences removed for lacking a valid citation

    @property
    def refused(self) -> bool:
        return not self.refs


def audit(answer: str, valid_refs: set[str]) -> AuditResult:
    answer = _GROUPED.sub(lambda m: "".join(f"[{r.strip()}]" for r in m.group(1).split(",")),
                          answer or "")
    kept, dropped, refs = [], [], []
    for sentence in (s.strip() for s in _SENTENCE_END.split(answer)):
        if not sentence:
            continue
        cited = [r for r in _REF.findall(sentence) if r in valid_refs]
        cleaned = _REF.sub(lambda m: m.group(0) if m.group(1) in valid_refs else "", sentence)
        cleaned = re.sub(r"\s+([.,;:])", r"\1", re.sub(r"\s{2,}", " ", cleaned)).strip()
        if cited:
            kept.append(cleaned)
            refs.extend(r for r in cited if r not in refs)
        else:
            dropped.append(sentence)
    if not kept:
        return AuditResult(REFUSAL, [], dropped)
    return AuditResult(" ".join(kept), refs, dropped)
