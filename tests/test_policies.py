import threading

import pytest

from clauselens.agent.graph import NO_POLICY_REPLY, CoverageAgent
from clauselens.config import get_settings
from clauselens.models import QuestionType, Verdict
from clauselens.policies.api import serve
from clauselens.policies.records import (
    FilePolicySystem,
    HttpPolicySystem,
    PolicyRecord,
    record_fields,
)

from .conftest import FakeLLM

EXPIRY = "When does my policy expire?"


@pytest.fixture()
def policies():
    return FilePolicySystem(get_settings().policies_file)


def test_sample_policies_load(policies):
    records = policies.list_policies()
    assert [r.policy_number for r in records] == ["CLP-1001", "CLP-1002", "CLP-1003"]
    assert policies.get_policy("CLP-1001").endorsements == ["CL-HO-E1"]
    assert policies.get_policy("NOPE") is None


def test_record_fields_format_money_and_blanks(policies):
    shown = {label: value for _, label, value in record_fields(policies.get_policy("CLP-1003"))}
    assert shown["Deductible"] == "$2,500.00"
    assert shown["Renewal premium (quoted for next term)"] == "not on file"


def test_endorsements_stored_as_json_text():
    record = PolicyRecord.from_dict({"policy_number": "X", "insured_name": "Y",
                                     "form": "CL-HO-01", "endorsements": '["CL-HO-E1"]'})
    assert record.endorsements == ["CL-HO-E1"]


def test_http_policy_system_against_sample_api(policies):
    server = serve(0, policies)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = HttpPolicySystem(f"http://127.0.0.1:{server.server_address[1]}")
        assert len(client.list_policies()) == 3
        assert client.get_policy("CLP-1002").renewal_premium == 1610
        assert client.get_policy("CLP-9999") is None
    finally:
        server.shutdown()
        server.server_close()


def test_account_question_answers_from_record(store, policies):
    llm = FakeLLM(question_type="account")
    answer = CoverageAgent(store, llm, policies=policies).ask(EXPIRY, policy_number="CLP-1001")
    assert llm.calls == ["classify", "account"]
    assert answer.question_type == QuestionType.ACCOUNT
    assert answer.verdict == Verdict.ANSWERED
    assert "Nothing to cite" not in answer.explanation
    assert answer.citations[0].clause_id == "Expiration date"
    assert answer.citations[0].text == "Expiration date: 2027-03-15"


def test_account_question_without_policy_asks_to_pick_one(store, policies):
    answer = CoverageAgent(store, FakeLLM(question_type="account"), policies=policies).ask(
        EXPIRY, forms=["CL-HO-01"])
    assert answer.verdict == Verdict.CANNOT_DETERMINE
    assert answer.explanation == NO_POLICY_REPLY


def test_unknown_policy_number(store, policies):
    answer = CoverageAgent(store, FakeLLM(question_type="account"), policies=policies).ask(
        EXPIRY, policy_number="CLP-9999")
    assert "wasn't found" in answer.explanation


def test_policy_brings_its_own_forms_and_endorsements(store, policies):
    llm = FakeLLM()
    answer = CoverageAgent(store, llm, policies=policies).ask(
        "Is water that backs up through a floor drain covered?", policy_number="CLP-1001")
    assert llm.calls == ["classify", "draft", "endorsement"]
    assert answer.verdict == Verdict.COVERED
