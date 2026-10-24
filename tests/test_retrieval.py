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
