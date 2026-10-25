"""The coverage agent: a LangGraph pipeline over the clause store.

    classify ─┬─> retrieve -> define -> draft -> endorsements -> audit -> END
              ├─> account (the customer's policy record) ────────────────> END
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
from clauselens.policies.records import PolicyRecord, PolicySystem, record_fields
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
NO_POLICY_REPLY = ("To answer questions about your own policy, like dates, premium or payments, "
                   "pick your policy under \"Your policy\" in the sidebar.")


class AgentState(TypedDict, total=False):
    question: str
    forms: list[str]
    endorsements: list[str]
    policy_number: str | None
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
    def __init__(self, store: ClauseStore, llm: LLM, top_k: int = 6,
                 policies: PolicySystem | None = None) -> None:
        self.store, self.llm, self.top_k, self.policies = store, llm, top_k, policies
        self.graph = self._build()

    # ---- public API ---------------------------------------------------------------

    def ask(self, question: str, forms: list[str] | None = None,
            endorsements: list[str] | None = None,
            policy_number: str | None = None) -> CoverageAnswer:
        """Answer a question. With a policy number and no forms given, the policy's own
        form and endorsements are used."""
        if policy_number and not forms:
            record = self._policy(policy_number)
            if record:
                forms, endorsements = [record.form], list(record.endorsements)
        state = self.graph.invoke(
            {"question": question, "forms": forms or [], "endorsements": endorsements or [],
             "policy_number": policy_number})
        return state["result"]

    def _policy(self, policy_number: str) -> PolicyRecord | None:
        if not self.policies:
            return None
        return self.policies.get_policy(policy_number)

    # ---- graph --------------------------------------------------------------------

    def _build(self):
        g = StateGraph(AgentState)
        for name in ("classify", "retrieve", "define", "draft", "endorsements", "audit",
                     "account", "refuse"):
            g.add_node(name, getattr(self, f"_{name}"))
        g.set_entry_point("classify")
        g.add_conditional_edges("classify", _route,
                                {"refuse": "refuse", "account": "account",
                                 "retrieve": "retrieve"})
        g.add_edge("retrieve", "define")
        g.add_edge("define", "draft")
        g.add_edge("draft", "endorsements")
        g.add_edge("endorsements", "audit")
        g.add_edge("audit", END)
        g.add_edge("account", END)
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
                    change = record.endorsement_change()
                    is_limit = change is not None and change.action == EndorsementAction.LIMITS
                    if record.section_type == SectionType.DECLARATIONS.value or is_limit:
                        picked.setdefault(record.record_id, record)
        return _most_specific(list(picked.values()))

    def _audit(self, state: AgentState) -> AgentState:
        clauses, extra = state["clauses"], state.get("endorsement_clauses", [])
        labels = {str(i): c for i, c in enumerate(clauses, 1)}
        labels |= {f"E{i}": c for i, c in enumerate(extra, 1)}
        text = state.get("answer_text", "")
        checked = audit(text, set(labels))
        if checked.dropped and text:
            # Some sentences had no citation. Ask the model once to cite (or remove) them,
            # then audit again; keep whichever version loses fewer sentences.
            fixed = self._recite(state["question"], text, clauses, extra)
            if fixed:
                rechecked = audit(fixed, set(labels))
                if not rechecked.refused and len(rechecked.dropped) < len(checked.dropped):
                    checked = rechecked
        checked = _collapse_parents(checked, labels)
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

    def _recite(self, question: str, text: str, clauses: list[ClauseRecord],
                extra: list[ClauseRecord]) -> str:
        user = (f"Question: {question}\n\nAnswer to fix:\n{text}\n\nPolicy clauses:\n"
                + _number(clauses, ""))
        if extra:
            user += "\n\nEndorsement clauses:\n" + _number(extra, "E")
        try:
            return self.llm.complete_json(prompts.RECITE_SYSTEM, user).get("answer", "")
        except Exception as exc:  # the audited original is still a safe answer
            log.warning("re-citation failed: %s", exc)
            return ""

    def _account(self, state: AgentState) -> AgentState:
        """Answer from the customer's policy record, citing each field used as [Pn]."""
        def reply(text: str) -> AgentState:
            return {"result": CoverageAnswer(QuestionType.ACCOUNT, Verdict.CANNOT_DETERMINE,
                                             text)}

        number = state.get("policy_number")
        if not number or not self.policies:
            return reply(NO_POLICY_REPLY)
        try:
            record = self.policies.get_policy(number)
        except Exception as exc:
            log.warning("policy system error: %s", exc)
            return reply("ClauseLens couldn't reach the policy system just now. "
                         "Please try again in a moment.")
        if record is None:
            return reply(f"Policy {number} wasn't found in the policy system.")

        fields = record_fields(record)
        labels = {f"P{i}": f for i, f in enumerate(fields, 1)}
        user = (f"Question: {state['question']}\n\nPolicy record:\n"
                + "\n".join(f"[{ref}] {label}: {value}"
                            for ref, (_, label, value) in labels.items()))
        out = self.llm.complete_json(prompts.ACCOUNT_SYSTEM, user)
        checked = audit(out.get("answer", ""), set(labels))
        citations = [Citation(record.policy_number, "Policy record", labels[r][1],
                              f"{labels[r][1]}: {labels[r][2]}", ref=r)
                     for r in checked.refs]
        verdict = Verdict.CANNOT_DETERMINE if checked.refused else Verdict.ANSWERED
        return {"result": CoverageAnswer(QuestionType.ACCOUNT, verdict, checked.answer,
                                         citations=citations)}

    def _refuse(self, state: AgentState) -> AgentState:
        return {"result": CoverageAnswer(QuestionType.OUT_OF_SCOPE, Verdict.CANNOT_DETERMINE,
                                         OUT_OF_SCOPE_REPLY)}


# ---- helpers ------------------------------------------------------------------------

def _route(state: AgentState) -> str:
    if state["question_type"] == QuestionType.OUT_OF_SCOPE:
        return "refuse"
    if state["question_type"] == QuestionType.ACCOUNT:
        return "account"
    return "retrieve"


def _collapse_parents(checked, labels: dict[str, ClauseRecord]):
    """When an answer cites both a clause and its parent heading (A.2.b and A.2 or A), point
    the parent's references at the most specific cited clause, so each citation shows
    distinct text instead of the same wording three times."""
    def descendant(ref: str) -> str | None:
        rec = labels[ref]
        children = [r for r in checked.refs if r != ref
                    and labels[r].form_number == rec.form_number
                    and labels[r].section_heading == rec.section_heading
                    and labels[r].clause_id.startswith(rec.clause_id + ".")]
        return max(children, key=lambda r: len(labels[r].clause_id)) if children else None

    mapping = {r: d for r in checked.refs if (d := descendant(r))}
    if not mapping:
        return checked
    text = re.sub(r"\[([EP]?\d+)\]", lambda m: f"[{mapping.get(m.group(1), m.group(1))}]",
                  checked.answer)
    text = re.sub(r"(\[[EP]?\d+\])(?:\1)+", r"\1", text)
    return type(checked)(text, [r for r in checked.refs if r not in mapping], checked.dropped)


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
_ACCOUNT_WORDS = ("expire", "expiration", "renewal", "renew", "premium", "payment",
                  "policy number", "my policy start", "when does my", "how much do i pay")
_CONDITION_WORDS = ("how long", "deadline", "notice", "proof of loss", "duties", "must i",
                    "do i have to", "within", "cancel", "settle")


def _heuristic_type(question: str) -> QuestionType:
    q = question.lower()
    if any(w in q for w in ("what does", "mean", "definition", "defined")):
        return QuestionType.DEFINITION
    if any(w in q for w in _ACCOUNT_WORDS):
        return QuestionType.ACCOUNT
    if any(w in q for w in _CONDITION_WORDS):
        return QuestionType.CONDITION
    if any(w in q for w in _LIMIT_WORDS):
        return QuestionType.LIMIT
    return QuestionType.COVERAGE
