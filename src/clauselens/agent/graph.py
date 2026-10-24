"""The coverage agent: a LangGraph pipeline over the clause store.

    classify ─┬─> retrieve -> define -> draft -> endorsements -> audit -> END
              └─> refuse (out of scope) ─────────────────────────────────> END

See docs/DESIGN.md section 7.
"""

from __future__ import annotations

import logging
import re
from typing import TypedDict

from langgraph.graph import END, StateGraph

from clauselens.agent import prompts
from clauselens.agent.audit import audit
from clauselens.agent.llm import LLM
from clauselens.models import (
    Citation,
    CoverageAnswer,
    EndorsementAction,
    EndorsementEffect,
    QuestionType,
    SectionType,
    Verdict,
)
from clauselens.retrieval.base import ClauseRecord, ClauseStore

log = logging.getLogger(__name__)

# Which sections to search, and how many clauses from each, per question type.
# Exclusions get their own budget for coverage questions so they're never crowded out.
RETRIEVAL_PLAN: dict[QuestionType, list[tuple[list[SectionType], int]]] = {
    QuestionType.COVERAGE: [
        ([SectionType.INSURING_AGREEMENT], 4),
        ([SectionType.EXCLUSIONS], 4),
        ([SectionType.CONDITIONS, SectionType.DEFINITIONS], 2),
    ],
    QuestionType.DEFINITION: [
        ([SectionType.DEFINITIONS], 3),
        ([SectionType.INSURING_AGREEMENT, SectionType.EXCLUSIONS], 2),
    ],
    QuestionType.LIMIT: [
        ([SectionType.DECLARATIONS, SectionType.LIMITS], 4),
        ([SectionType.INSURING_AGREEMENT, SectionType.CONDITIONS], 3),
    ],
    QuestionType.CONDITION: [
        ([SectionType.CONDITIONS], 4),
        ([SectionType.DEFINITIONS, SectionType.INSURING_AGREEMENT], 2),
    ],
}

OUT_OF_SCOPE_REPLY = ("ClauseLens answers questions about what policy wording says: coverage, "
                      "definitions and limits. This question is outside that scope.")


class AgentState(TypedDict, total=False):
    question: str
    forms: list[str]
    endorsements: list[str]
    question_type: QuestionType
    search_query: str
    clauses: list[ClauseRecord]                 # numbered [1]..[n] in prompts
    definitions: dict[str, ClauseRecord]
    draft_verdict: Verdict
    draft_answer: str
    endorsement_clauses: list[ClauseRecord]     # numbered [E1]..[En]
    verdict: Verdict
    answer_text: str
    effects: list[dict]
    result: CoverageAnswer


class CoverageAgent:
    def __init__(self, store: ClauseStore, llm: LLM, top_k: int = 6) -> None:
        self.store, self.llm, self.top_k = store, llm, top_k
        self.graph = self._build()

    # ---- public API ---------------------------------------------------------------

    def ask(self, question: str, forms: list[str],
            endorsements: list[str] | None = None) -> CoverageAnswer:
        state = self.graph.invoke(
            {"question": question, "forms": forms, "endorsements": endorsements or []})
        return state["result"]

    # ---- graph --------------------------------------------------------------------

    def _build(self):
        g = StateGraph(AgentState)
        for name in ("classify", "retrieve", "define", "draft", "endorsements", "audit",
                     "refuse"):
            g.add_node(name, getattr(self, f"_{name}"))
        g.set_entry_point("classify")
        g.add_conditional_edges(
            "classify",
            lambda s: "refuse" if s["question_type"] == QuestionType.OUT_OF_SCOPE else "retrieve",
            {"refuse": "refuse", "retrieve": "retrieve"})
        g.add_edge("retrieve", "define")
        g.add_edge("define", "draft")
        g.add_edge("draft", "endorsements")
        g.add_edge("endorsements", "audit")
        g.add_edge("audit", END)
        g.add_edge("refuse", END)
        return g.compile()

    # ---- nodes --------------------------------------------------------------------

    def _classify(self, state: AgentState) -> AgentState:
        question = state["question"]
        try:
            out = self.llm.complete_json(prompts.CLASSIFY_SYSTEM, question)
            qtype = QuestionType(out.get("question_type", "coverage"))
            query = out.get("search_query") or question
        except Exception as exc:  # classification failure shouldn't sink the question
            log.warning("classify failed, falling back to heuristics: %s", exc)
            qtype, query = _heuristic_type(question), question
        return {"question_type": qtype, "search_query": query}

    def _retrieve(self, state: AgentState) -> AgentState:
        query = f"{state['search_query']} {state['question']}"
        seen: dict[str, ClauseRecord] = {}
        for section_types, k in RETRIEVAL_PLAN[state["question_type"]]:
            hits = self.store.search(query, k=max(1, round(k * self.top_k / 6)),
                                     form_numbers=state["forms"],
                                     section_types=[s.value for s in section_types])
            for hit in hits:
                seen.setdefault(hit.record.record_id, hit.record)
        return {"clauses": list(seen.values())}

    def _define(self, state: AgentState) -> AgentState:
        already = {c.defines for c in state["clauses"] if c.defines}
        terms = {t for c in state["clauses"] for t in c.terms_used} - already
        terms |= {t for t in _quoted(state["question"]) if t not in already}
        found = self.store.definitions(sorted(terms), state["forms"]) if terms else {}
        return {"definitions": found}

    def _draft(self, state: AgentState) -> AgentState:
        clauses = state["clauses"] + list(state["definitions"].values())
        if not clauses:
            return {"clauses": [], "draft_verdict": Verdict.CANNOT_DETERMINE,
                    "draft_answer": ""}
        user = (f"Question: {state['question']}\n\nPolicy clauses:\n"
                + _number(clauses, ""))
        out = self.llm.complete_json(prompts.DRAFT_SYSTEM, user)
        return {"clauses": clauses, "draft_verdict": _verdict(out.get("verdict")),
                "draft_answer": out.get("answer", "")}

    def _endorsements(self, state: AgentState) -> AgentState:
        base = {"verdict": state["draft_verdict"], "answer_text": state["draft_answer"],
                "effects": [], "endorsement_clauses": []}
        if not state["endorsements"] or not state["draft_answer"]:
            return base
        candidates = self._endorsement_candidates(state)
        if not candidates:
            return base
        user = (f"Question: {state['question']}\n\n"
                f"Base-form answer ({state['draft_verdict'].value}): {state['draft_answer']}\n\n"
                f"Base-form clauses:\n{_number(state['clauses'], '')}\n\n"
                f"Endorsement clauses:\n{_number(candidates, 'E')}")
        out = self.llm.complete_json(prompts.ENDORSEMENT_SYSTEM, user)
        if not out.get("changes_answer"):
            return {**base, "endorsement_clauses": candidates}
        return {"verdict": _verdict(out.get("verdict")),
                "answer_text": out.get("answer") or state["draft_answer"],
                "effects": out.get("effects") or [], "endorsement_clauses": candidates}

    def _endorsement_candidates(self, state: AgentState) -> list[ClauseRecord]:
        """Endorsement clauses that target a retrieved base clause by instruction, plus the
        endorsement clauses most similar to the question."""
        cited = {(c.section_type, c.clause_id) for c in state["clauses"]}
        picked: dict[str, ClauseRecord] = {}
        for form in state["endorsements"]:
            clauses = self.store.form_clauses(form)
            targeted = False
            for record in clauses:
                change = record.endorsement_change()
                if change and _targets(change.target_section, change.target_clause_id, cited):
                    picked[record.record_id] = record
                    targeted = True
            for hit in self.store.search(f"{state['search_query']} {state['question']}",
                                         k=4, form_numbers=[form]):
                picked.setdefault(hit.record.record_id, hit.record)
            # Include the endorsement's schedule (limits, deductibles) when it bears on the
            # answer or when the question is about limits.
            if targeted or state.get("question_type") == QuestionType.LIMIT:
                for record in clauses:
                    if record.section_type == SectionType.DECLARATIONS.value:
                        picked.setdefault(record.record_id, record)
        return _most_specific(list(picked.values()))

    def _audit(self, state: AgentState) -> AgentState:
        clauses, extra = state["clauses"], state.get("endorsement_clauses", [])
        labels = {str(i): c for i, c in enumerate(clauses, 1)}
        labels |= {f"E{i}": c for i, c in enumerate(extra, 1)}
        checked = audit(state.get("answer_text", ""), set(labels))
        verdict = Verdict.CANNOT_DETERMINE if checked.refused else state["verdict"]

        effects, seen_refs = [], set()
        for effect in [] if checked.refused else state.get("effects", []):
            ref = str(effect.get("ref", ""))
            record = labels.get(ref)
            if record is None or not ref.startswith("E") or ref in seen_refs:
                continue
            seen_refs.add(ref)
            try:
                action = EndorsementAction(effect.get("action"))
            except ValueError:
                action = EndorsementAction.ADDS
            effects.append(EndorsementEffect(record.form_number, action,
                                             effect.get("summary", ""), record.citation()))

        definitions = {term: rec.full_text for term, rec in state["definitions"].items()}
        result = CoverageAnswer(
            question_type=state["question_type"],
            verdict=verdict,
            explanation=checked.answer,
            citations=[_labelled(labels[r], r) for r in checked.refs],
            definitions_used=definitions,
            endorsement_effects=effects,
            base_verdict=state["draft_verdict"] if effects else None,
        )
        return {"result": result}

    def _refuse(self, state: AgentState) -> AgentState:
        return {"result": CoverageAnswer(QuestionType.OUT_OF_SCOPE, Verdict.CANNOT_DETERMINE,
                                         OUT_OF_SCOPE_REPLY)}


# ---- helpers ------------------------------------------------------------------------

def _number(records: list[ClauseRecord], prefix: str) -> str:
    return "\n".join(
        f"[{prefix}{i}] {r.form_number} | {r.section_heading} | {r.clause_id}: {r.full_text}"
        for i, r in enumerate(records, 1))


def _labelled(record: ClauseRecord, ref: str) -> Citation:
    citation = record.citation()
    citation.ref = ref
    return citation


def _verdict(value) -> Verdict:
    try:
        return Verdict(value)
    except ValueError:
        return Verdict.CANNOT_DETERMINE


def _targets(section: SectionType | None, clause_id: str | None,
             cited: set[tuple[str, str]]) -> bool:
    if section is None:
        return False
    for cited_section, cited_id in cited:
        if cited_section != section.value:
            continue
        if clause_id is None or cited_id == clause_id or cited_id.startswith(clause_id + ".") \
                or clause_id.startswith(cited_id + "."):
            return True
    return False


def _is_descendant(child: ClauseRecord, parent: ClauseRecord) -> bool:
    return (child.form_number == parent.form_number
            and child.section_heading == parent.section_heading
            and child.clause_id.startswith(parent.clause_id + "."))


def _most_specific(records: list[ClauseRecord]) -> list[ClauseRecord]:
    """Avoid showing the model the same wording twice. A clause carrying a change
    instruction already includes its sub-clauses (full_text), so those are dropped; a parent
    heading whose children include an instruction clause is dropped too."""
    instructions = [r for r in records if r.change]
    kept = []
    for r in records:
        if any(_is_descendant(r, i) for i in instructions):
            continue
        if not r.change and any(_is_descendant(i, r) for i in instructions):
            continue
        kept.append(r)
    return kept


def _quoted(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[\"“]([^\"”]{1,40})[\"”]", text)]


_LIMIT_WORDS = ("limit", "how much", "deductible", "maximum", "most we will pay", "$", "percent")
_CONDITION_WORDS = ("how long", "deadline", "notice", "proof of loss", "duties", "must i",
                    "do i have to", "within", "cancel", "settle")


def _heuristic_type(question: str) -> QuestionType:
    q = question.lower()
    if any(w in q for w in ("what does", "mean", "definition", "defined")):
        return QuestionType.DEFINITION
    if any(w in q for w in _CONDITION_WORDS):
        return QuestionType.CONDITION
    if any(w in q for w in _LIMIT_WORDS):
        return QuestionType.LIMIT
    return QuestionType.COVERAGE
