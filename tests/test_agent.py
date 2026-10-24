from clauselens.agent.audit import REFUSAL, audit
from clauselens.agent.graph import OUT_OF_SCOPE_REPLY, CoverageAgent
from clauselens.eval.scoring import answer_to_dict, citation_recall
from clauselens.models import EndorsementAction, QuestionType, Verdict

from .conftest import FakeLLM

QUESTION = "Is water that backs up through a floor drain covered?"


def test_audit_drops_uncited_and_invented_refs():
    result = audit("Covered [1]. No cite here. Fake [7]. Grouped [1, 2].", {"1", "2"})
    assert result.answer == "Covered [1]. Grouped [1][2]."
    assert result.refs == ["1", "2"]
    assert len(result.dropped) == 2


def test_audit_refuses_when_nothing_is_cited():
    result = audit("It is probably fine.", {"1"})
    assert result.refused and result.answer == REFUSAL


def test_base_form_answer_is_cited_and_audited(store):
    answer = CoverageAgent(store, FakeLLM()).ask(QUESTION, forms=["CL-HO-01"])
    assert answer.verdict == Verdict.NOT_COVERED
    assert "no citation" not in answer.explanation and "[99]" not in answer.explanation
    assert answer.citations[0].section_heading == "Section I - Exclusions"
    expected = {"expected_clauses": [{"form": "CL-HO-01", "section": "Exclusions",
                                      "clause": "A.2.b"}]}
    assert citation_recall(answer_to_dict(answer), expected) == 1.0


def test_endorsement_overrides_base_answer(store):
    llm = FakeLLM()
    answer = CoverageAgent(store, llm).ask(QUESTION, forms=["CL-HO-01"],
                                           endorsements=["CL-HO-E1"])
    assert llm.calls == ["classify", "draft", "endorsement"]
    assert answer.base_verdict == Verdict.NOT_COVERED
    assert answer.verdict == Verdict.COVERED
    effect = answer.endorsement_effects[0]
    assert effect.endorsement_form == "CL-HO-E1"
    assert effect.action == EndorsementAction.DELETES
    assert any(c.ref.startswith("E") for c in answer.citations)


def test_out_of_scope_skips_retrieval(store):
    llm = FakeLLM(question_type="out_of_scope")
    answer = CoverageAgent(store, llm).ask("Where is my claim check?", forms=["CL-HO-01"])
    assert llm.calls == ["classify"]
    assert answer.question_type == QuestionType.OUT_OF_SCOPE
    assert answer.explanation == OUT_OF_SCOPE_REPLY


def test_classifier_failure_falls_back_to_heuristics(store):
    class Broken(FakeLLM):
        def complete_json(self, system, user):
            if "route questions" in system:
                raise ValueError("bad json")
            return super().complete_json(system, user)

    answer = CoverageAgent(store, Broken()).ask(QUESTION, forms=["CL-HO-01"])
    assert answer.question_type == QuestionType.COVERAGE


def test_endorsement_candidates_are_most_specific_and_include_schedule(store):
    agent = CoverageAgent(store, FakeLLM())
    clauses = [h.record for h in store.search("sewers drains backs up", k=4,
                                              form_numbers=["CL-HO-01"],
                                              section_types=["exclusions"])]
    picked = agent._endorsement_candidates({
        "clauses": clauses, "endorsements": ["CL-HO-E1"],
        "search_query": "water backs up through sewers or drains",
        "question": "Is sewer backup covered?"})
    ids = {(r.section_heading, r.clause_id) for r in picked}
    assert ("Endorsement Provisions", "A.1") in ids          # the deletion instruction
    assert ("Schedule", "A") in ids                          # the endorsement's limit
    assert ("Endorsement Provisions", "A") not in ids        # parent heading (duplicate text)
    assert ("Endorsement Provisions", "A.2.a") not in ids    # already inside A.2's text


def test_limit_question_includes_endorsement_schedule(store):
    agent = CoverageAgent(store, FakeLLM())
    picked = agent._endorsement_candidates({
        "clauses": [], "endorsements": ["CL-HO-E1"], "question_type": QuestionType.LIMIT,
        "search_query": "limit for water backup", "question": "What is the most I can recover?"})
    assert ("Schedule", "A") in {(r.section_heading, r.clause_id) for r in picked}


def test_condition_questions_search_conditions(store):
    class Conditions(FakeLLM):
        def complete_json(self, system, user):
            if "route questions" in system:
                return {"question_type": "condition", "search_query": "proof of loss days"}
            ref = self._ref(user, "proof of loss")
            return {"verdict": "covered", "answer": f"Within 60 days of the request [{ref}]."}

    answer = CoverageAgent(store, Conditions()).ask(
        "How long do I have to send a proof of loss?", forms=["CL-HO-01"])
    assert answer.question_type == QuestionType.CONDITION
    assert answer.citations[0].section_heading == "Section I - Conditions"
    assert answer.citations[0].clause_id == "A.3"
