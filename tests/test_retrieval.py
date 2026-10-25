from clauselens.retrieval.base import ClauseRecord


def test_search_filters_by_section(store):
    hits = store.search("sewer backup drains", k=3, form_numbers=["CL-HO-01"],
                        section_types=["exclusions"])
    assert hits and all(h.record.section_type == "exclusions" for h in hits)
    assert hits[0].record.clause_id in {"A.2", "A.2.b"}


def test_search_respects_form_filter(store):
    hits = store.search("sewer backup", k=10, form_numbers=["CL-HO-E1"])
    assert hits and {h.record.form_number for h in hits} == {"CL-HO-E1"}


def test_definitions_lookup(store):
    found = store.definitions(["occurrence", "flood", "unknown"], ["CL-HO-01"])
    assert set(found) == {"occurrence", "flood"}


def test_row_round_trip(store):
    record = next(r for r in store.form_clauses("CL-HO-E1") if r.change)
    again = ClauseRecord.from_row(record.to_row())
    assert again == record
    assert again.endorsement_change() is not None


def test_search_client_uses_app_service_principal(monkeypatch):
    from clauselens.retrieval.vector_search_store import client_auth

    for name in ("DATABRICKS_CLIENT_ID", "DATABRICKS_CLIENT_SECRET", "DATABRICKS_HOST"):
        monkeypatch.delenv(name, raising=False)
    assert client_auth() == {}                      # notebook: automatic auth
    monkeypatch.setenv("DATABRICKS_CLIENT_ID", "id")
    monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "secret")
    monkeypatch.setenv("DATABRICKS_HOST", "dbc-123.cloud.databricks.com")
    assert client_auth() == {"workspace_url": "https://dbc-123.cloud.databricks.com",
                             "service_principal_client_id": "id",
                             "service_principal_client_secret": "secret"}
