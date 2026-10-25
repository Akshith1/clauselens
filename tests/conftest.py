from __future__ import annotations

import re
from pathlib import Path

import pytest

from clauselens.agent import prompts
from clauselens.parsing.loader import load_forms
from clauselens.retrieval.base import form_to_records
from clauselens.retrieval.memory_store import MemoryClauseStore

FORMS_DIR = Path(__file__).resolve().parents[1] / "data" / "forms"


@pytest.fixture(scope="session")
def forms():
    return {f.form_number: f for f in load_forms(FORMS_DIR)}


@pytest.fixture()
def store(forms):
    s = MemoryClauseStore()
    for form in forms.values():
        s.upsert(form_to_records(form))
    return s


class FakeLLM:
    """Deterministic stand-in for the model: answers the sewer-backup question by citing
    whichever numbered clause mentions sewers, so tests exercise the real graph."""

    def __init__(self, question_type: str = "coverage") -> None:
        self.question_type = question_type
        self.calls: list[str] = []

    @staticmethod
    def _ref(prompt: str, needle: str, prefix: str = "") -> str | None:
        for line in prompt.splitlines():
            m = re.match(rf"\[({prefix}\d+)\]", line)
            if m and needle in line.lower():
                return m.group(1)
        return None

    def complete_json(self, system: str, user: str) -> dict:
        if system == prompts.CLASSIFY_SYSTEM:
            self.calls.append("classify")
            return {"question_type": self.question_type,
                    "search_query": "water backs up through sewers or drains"}
        if system == prompts.DRAFT_SYSTEM:
            self.calls.append("draft")
            ref = self._ref(user, "sewers")
            if ref is None:
                return {"verdict": "cannot_determine", "answer": "Not addressed."}
            return {"verdict": "not_covered",
                    "answer": f"Sewer backup is excluded [{ref}]. This sentence has no citation. "
                              f"Invented reference [99]."}
        if system == prompts.ENDORSEMENT_SYSTEM:
            self.calls.append("endorsement")
            ref = self._ref(user, "is deleted", prefix="E")
            if ref is None:
                return {"changes_answer": False, "effects": []}
            return {"changes_answer": True, "verdict": "covered",
                    "answer": f"The endorsement deletes the sewer backup exclusion [{ref}].",
                    "effects": [{"ref": ref, "action": "deletes",
                                 "summary": "Removes the water backup exclusion."}]}
        if system == prompts.RECITE_SYSTEM:
            self.calls.append("recite")
            return {"answer": user.split("Answer to fix:\n", 1)[1].split("\n\n", 1)[0]}
        if system == prompts.ACCOUNT_SYSTEM:
            self.calls.append("account")
            ref = self._ref(user, "expiration date", prefix="P")
            return {"answer": f"Your policy expires on the date shown [{ref}]. "
                              f"Nothing to cite here."}
        raise AssertionError("unexpected prompt")
