"""Databricks clause store: a Unity Catalog Delta table served by a Vector Search index.

Rows are written by the ingestion notebook (notebooks/ingest_forms.py); this class only
reads. The index is a Delta Sync index with Databricks-managed embeddings computed from
the `search_text` column, so no embedding code runs in the app.
"""

from __future__ import annotations

import os

from clauselens.retrieval.base import ClauseRecord, ScoredClause

_COLUMNS = ["record_id", "form_number", "form_title", "is_endorsement", "modifies",
            "section_type", "section_heading", "clause_id", "text", "full_text",
            "parent_path", "defines", "terms_used", "change"]


def client_auth() -> dict:
    """Credentials for the search client. In a notebook it authenticates automatically; a
    Databricks App instead gets its service principal through DATABRICKS_CLIENT_ID /
    DATABRICKS_CLIENT_SECRET / DATABRICKS_HOST, which the client must be given explicitly."""
    client_id = os.getenv("DATABRICKS_CLIENT_ID")
    secret = os.getenv("DATABRICKS_CLIENT_SECRET")
    host = os.getenv("DATABRICKS_HOST", "")
    if not (client_id and secret and host):
        return {}
    if not host.startswith("http"):
        host = "https://" + host
    return {"workspace_url": host.rstrip("/"), "service_principal_client_id": client_id,
            "service_principal_client_secret": secret}


class VectorSearchClauseStore:
    def __init__(self, endpoint: str, index_name: str) -> None:
        from databricks.vector_search.client import VectorSearchClient

        self._index = VectorSearchClient(disable_notice=True, **client_auth()).get_index(
            endpoint_name=endpoint, index_name=index_name)

    def upsert(self, records: list[ClauseRecord]) -> None:
        raise NotImplementedError(
            "On Databricks, clauses are written to the Delta table by notebooks/ingest_forms.py "
            "and synced into the index; the app is read-only.")

    def search(self, query, k, form_numbers=None, section_types=None):
        filters = {}
        if form_numbers:
            filters["form_number"] = form_numbers
        if section_types:
            filters["section_type"] = section_types
        result = self._index.similarity_search(
            query_text=query, columns=_COLUMNS, num_results=k, filters=filters or None)
        return [ScoredClause(ClauseRecord.from_row(row), row["score"]) for row in _rows(result)]

    def definitions(self, terms, form_numbers):
        if not terms:
            return {}
        result = self._index.similarity_search(
            query_text=" ".join(terms), columns=_COLUMNS, num_results=max(len(terms) * 2, 5),
            filters={"form_number": form_numbers, "defines": [t.lower() for t in terms]})
        return {row["defines"]: ClauseRecord.from_row(row) for row in _rows(result)}

    def form_clauses(self, form_number):
        result = self._index.similarity_search(
            query_text=form_number, columns=_COLUMNS, num_results=500,
            filters={"form_number": form_number})
        return [ClauseRecord.from_row(row) for row in _rows(result)]


def _rows(result: dict) -> list[dict]:
    """Turn a Vector Search response into dicts; the last column is the similarity score."""
    names = [c["name"] for c in result["manifest"]["columns"]]
    return [dict(zip(names, values, strict=True))
            for values in result.get("result", {}).get("data_array", [])]
