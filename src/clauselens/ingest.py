"""Local ingestion: parse every form in the catalog and index it into the local store.

    python -m clauselens.ingest            # index into Chroma (data/chroma)
    python -m clauselens.ingest --show     # also print each form's parsed outline

On Databricks, use notebooks/ingest_forms.py instead.
"""

from __future__ import annotations

import argparse

from clauselens.config import get_settings
from clauselens.factory import build_store
from clauselens.models import PolicyForm
from clauselens.parsing.loader import load_forms
from clauselens.retrieval.base import form_to_records


def print_outline(form: PolicyForm) -> None:
    kind = "endorsement" if form.is_endorsement else "form"
    print(f"\n{form.form_number} ({kind}) {form.title}")
    if form.modifies:
        print(f"  modifies: {', '.join(form.modifies)}")
    for section in form.sections:
        print(f"  [{section.section_type.value}] {section.heading}")
        for clause in section.clauses:
            extra = f"  defines '{clause.defines}'" if clause.defines else ""
            if clause.change:
                extra += f"  {clause.change.action.value} {clause.change.target_clause_id or ''}"
            print(f"    {clause.clause_id:<8} {clause.text[:70]}{extra}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="print each parsed form outline")
    args = parser.parse_args()

    settings = get_settings()
    if settings.backend != "local":
        raise SystemExit("Local ingestion only; on Databricks run notebooks/ingest_forms.py.")
    forms = load_forms(settings.forms_dir)
    store = build_store(settings)
    total = 0
    for form in forms:
        records = form_to_records(form)
        store.upsert(records)
        total += len(records)
        if args.show:
            print_outline(form)
    print(f"Indexed {total} clauses from {len(forms)} forms into the {settings.store} store.")


if __name__ == "__main__":
    main()
