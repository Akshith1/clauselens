from clauselens.models import EndorsementAction, SectionType
from clauselens.parsing.structure import parse_form, section_type_for


def _clause(form, section_type, clause_id):
    for section in form.sections:
        if section.section_type == section_type:
            for clause in section.clauses:
                if clause.clause_id == clause_id:
                    return clause
    raise KeyError(clause_id)


def test_section_types(forms):
    base = forms["CL-HO-01"]
    types = [s.section_type for s in base.sections]
    assert types == [SectionType.DECLARATIONS, SectionType.DEFINITIONS,
                     SectionType.INSURING_AGREEMENT, SectionType.INSURING_AGREEMENT,
                     SectionType.EXCLUSIONS, SectionType.CONDITIONS]
    assert base.sections[4].heading == "Section I - Exclusions"


def test_nested_clause_ids_and_parent_path(forms):
    clause = _clause(forms["CL-HO-01"], SectionType.EXCLUSIONS, "A.2.b")
    assert "backs up through sewers" in clause.text
    assert clause.parent_path[-1] == "Water, meaning:"


def test_full_text_includes_sub_clauses(forms):
    water = _clause(forms["CL-HO-01"], SectionType.EXCLUSIONS, "A.2")
    assert "(A.2.a)" in water.full_text and "(A.2.c)" in water.full_text


def test_continuation_lines_join_their_clause(forms):
    duties = _clause(forms["CL-HO-01"], SectionType.CONDITIONS, "A")
    assert duties.text.startswith("Duties After Loss In case of a loss")


def test_defined_terms(forms):
    occurrence = _clause(forms["CL-HO-01"], SectionType.DEFINITIONS, "C")
    assert occurrence.defines == "occurrence"
    dwelling = _clause(forms["CL-HO-01"], SectionType.INSURING_AGREEMENT, "A.1")
    assert dwelling.terms_used == ["residence premises"]


def test_endorsement_modifies_and_changes(forms):
    endorsement = forms["CL-HO-E1"]
    assert endorsement.modifies == ["CL-HO-01"]
    delete = _clause(endorsement, SectionType.ENDORSEMENT, "A.1").change
    assert delete.action == EndorsementAction.DELETES
    assert delete.target_section == SectionType.EXCLUSIONS
    assert delete.target_clause_id == "A.2.b"
    add = _clause(endorsement, SectionType.ENDORSEMENT, "A.2").change
    assert add.action == EndorsementAction.ADDS
    assert add.target_section == SectionType.INSURING_AGREEMENT
    assert _clause(endorsement, SectionType.ENDORSEMENT, "B.1").change.action \
        == EndorsementAction.LIMITS


def test_preamble_is_skipped():
    form = parse_form("XX-1\nSOME TITLE\nDEFINITIONS\nA. \"Thing\" means a thing.", "XX-1")
    assert len(form.sections) == 1 and form.sections[0].clauses[0].defines == "thing"


def test_section_type_for_unknown_heading():
    assert section_type_for("SAMPLE HOMEOWNERS POLICY") is None
    assert section_type_for("SECTION II - LIABILITY") == SectionType.OTHER
