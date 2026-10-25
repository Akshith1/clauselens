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


def fact_match(output: dict, expected: dict) -> bool | None:
    """Policy-record answers must state the right values. Each entry in expected_facts is a
    list of acceptable spellings, e.g. ["2027-03-15", "March 15, 2027"]."""
    facts = expected.get("expected_facts")
    if not facts:
        return None
    text = output["answer"].lower()
    return all(any(alt.lower() in text for alt in options) for options in facts)


SCORERS = {
    "verdict_match": verdict_match,
    "citation_recall": citation_recall,
    "refusal_correct": refusal_correct,
    "endorsement_detected": endorsement_detected,
    "fact_match": fact_match,
}

# CI fails if any metric's mean drops below its threshold.
THRESHOLDS = {
    "verdict_match": 0.75,
    "citation_recall": 0.70,
    "refusal_correct": 1.0,
    "endorsement_detected": 1.0,
    "fact_match": 1.0,
}


def summarize(rows: list[dict]) -> dict[str, float]:
    summary = {}
    for name in SCORERS:
        values = [r[name] for r in rows if r.get(name) is not None]
        if values:
            summary[name] = sum(float(v) for v in values) / len(values)
    return summary


# ---- running and saving results (used by the CLI and the app's Test results page) ----

EXPECTED_LABEL = {
    "covered": "Covered", "not_covered": "Not covered", "depends": "Depends",
    "cannot_determine": "Can't determine", "answered": "Answered from record",
}


def setup_label(q: dict) -> str:
    if q.get("policy"):
        return f"Policy {q['policy']}"
    forms = ", ".join(q.get("forms") or []) or "no policy picked"
    endorsements = q.get("endorsements") or []
    return forms + (f" + {', '.join(endorsements)}" if endorsements else "")


def expected_label(q: dict) -> str:
    return " or ".join(EXPECTED_LABEL.get(v, v) for v in q["expected_verdict"])


def run_one(agent, q: dict) -> dict:
    """Ask one question and score it. Returns a flat, JSON-safe row."""
    output = answer_to_dict(agent.ask(q["question"], q.get("forms"), q.get("endorsements"),
                                      policy_number=q.get("policy")))
    row = {"id": q["id"], "question": q["question"], "setup": setup_label(q),
           "expected": expected_label(q), "verdict": output["verdict"],
           "answer": output["answer"]}
    row |= {name: fn(output, q) for name, fn in SCORERS.items()}
    checks = [row[name] for name in SCORERS if row[name] is not None]
    row["passed"] = all(float(c) >= 1.0 for c in checks)
    return row


def save_results(rows: list[dict], model: str, path: Path | None = None) -> Path:
    from datetime import datetime

    from clauselens.config import REPO_ROOT

    path = path or REPO_ROOT / "data" / "eval_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"run_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": model,
               "summary": summarize(rows), "rows": rows}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_results(path: Path | None = None) -> dict | None:
    from clauselens.config import REPO_ROOT

    path = path or REPO_ROOT / "data" / "eval_results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
