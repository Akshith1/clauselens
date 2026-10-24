"""Load forms listed in a catalog.json (and data/forms/private/catalog.json if present)."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from clauselens.models import PolicyForm
from clauselens.parsing.structure import parse_form

_PAGE_FOOTER = re.compile(r"^(page\s+\d+(\s+of\s+\d+)?|\d+\s+of\s+\d+)$", re.I)


def read_form_text(path: Path) -> str:
    """Plain text from a .txt or .pdf form, with running headers/footers removed."""
    if path.suffix.lower() != ".pdf":
        return path.read_text(encoding="utf-8")
    from pypdf import PdfReader

    pages = [(page.extract_text() or "").splitlines() for page in PdfReader(str(path)).pages]
    # A line repeated on most pages is a running header or footer (form number, copyright).
    counts = Counter(ln.strip() for page in pages for ln in set(page) if ln.strip())
    repeated = {ln for ln, n in counts.items() if len(pages) > 2 and n >= 0.6 * len(pages)}
    kept = [ln for page in pages for ln in page
            if ln.strip() and ln.strip() not in repeated and not _PAGE_FOOTER.match(ln.strip())]
    return "\n".join(kept)


def load_catalog(forms_dir: Path) -> list[tuple[dict, Path]]:
    entries = []
    for catalog in (forms_dir / "catalog.json", forms_dir / "private" / "catalog.json"):
        if catalog.exists():
            for entry in json.loads(catalog.read_text(encoding="utf-8")):
                entries.append((entry, catalog.parent / entry["file"]))
    return entries


def load_forms(forms_dir: Path) -> list[PolicyForm]:
    entries = load_catalog(forms_dir)
    known = [entry["form_number"] for entry, _ in entries]
    return [
        parse_form(
            read_form_text(path),
            form_number=entry["form_number"],
            edition=entry.get("edition", ""),
            line_of_business=entry.get("line_of_business", ""),
            title=entry.get("title", ""),
            is_endorsement=bool(entry.get("endorsement")),
            known_forms=known,
        )
        for entry, path in entries
    ]
