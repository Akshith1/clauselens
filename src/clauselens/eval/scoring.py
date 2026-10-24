"""Scoring for the ClauseLens question set. Plain functions, so they run with or without MLflow."""

from __future__ import annotations

import json
from pathlib import Path

from clauselens.models import CoverageAnswer, QuestionType

QUESTIONS = Path(__file__).with_name("questions.jsonl")


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def answer_to_dict(answer: CoverageAnswer) -> dict:
    """Serializable view of an answer (MLflow traces and reports need plain data)."""
    return {
        "question_type": answer.question_type.value,
        "verdict": answer.verdict.value,
        "base_verdict": answer.base_verdict.value if answer.base_verdict else None,
        "answer": answer.explanation,
        "citations": [{"form": c.form_number, "section": c.section_heading,
                       "clause": c.clause_id, "ref": c.ref} for c in answer.citations],
        "endorsement_effects": [{"form": e.endorsement_form, "action": e.action.value,
                                 "summary": e.summary} for e in answer.endorsement_effects],
    }


def verdict_match(output: dict, expected: dict) -> bool:
    return output["verdict"] in expected["expected_verdict"]


def _same_clause(cited: dict, want: dict) -> bool:
    """A citation counts if it's the expected clause, a sub-clause of it, or its parent."""
    if cited["form"] != want["form"] or want["section"].lower() not in cited["section"].lower():
        return False
    a, b = cited["clause"], want["clause"]
    return a == b or a.startswith(b + ".") or b.startswith(a + ".")


def citation_recall(output: dict, expected: dict) -> float | None:
    wanted = expected.get("expected_clauses") or []
    if not wanted:
        return None
    hits = sum(any(_same_clause(c, w) for c in output["citations"]) for w in wanted)
    return hits / len(wanted)


def refusal_correct(output: dict, expected: dict) -> bool | None:
    if not expected.get("expect_refusal"):
        return None
    return output["question_type"] == QuestionType.OUT_OF_SCOPE.value or not output["citations"]


def endorsement_detected(output: dict, expected: dict) -> bool | None:
    if not expected.get("expect_endorsement_effect"):
        return None
    return bool(output["endorsement_effects"])


SCORERS = {
    "verdict_match": verdict_match,
    "citation_recall": citation_recall,
    "refusal_correct": refusal_correct,
    "endorsement_detected": endorsement_detected,
}

# CI fails if any metric's mean drops below its threshold.
THRESHOLDS = {
    "verdict_match": 0.75,
    "citation_recall": 0.70,
    "refusal_correct": 1.0,
    "endorsement_detected": 1.0,
}


def summarize(rows: list[dict]) -> dict[str, float]:
    summary = {}
    for name in SCORERS:
        values = [r[name] for r in rows if r.get(name) is not None]
        if values:
            summary[name] = sum(float(v) for v in values) / len(values)
    return summary
