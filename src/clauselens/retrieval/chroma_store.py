"""Local persistent clause store on Chroma (embeds with Chroma's default MiniLM model)."""

from __future__ import annotations

from pathlib import Path

from clauselens.retrieval.base import ClauseRecord, ScoredClause

_COLLECTION = "clauses"


class ChromaClauseStore:
    def __init__(self, path: Path) -> None:
        import chromadb

        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            _COLLECTION, metadata={"hnsw:space": "cosine"})

    def upsert(self, records: list[ClauseRecord]) -> None:
        if not records:
            return
        rows = [r.to_row() for r in records]
        self._collection.upsert(
            ids=[row["record_id"] for row in rows],
            documents=[row.pop("search_text") for row in rows],
            metadatas=[{k: ("" if v is None else v) for k, v in row.items()} for row in rows],
        )

    def search(self, query, k, form_numbers=None, section_types=None):
        result = self._collection.query(
            query_texts=[query], n_results=k,
            where=_where(form_numbers=form_numbers, section_type=section_types))
        out = []
        for meta, distance in zip(result["metadatas"][0], result["distances"][0], strict=True):
            out.append(ScoredClause(ClauseRecord.from_row(meta), 1.0 - distance))
        return out

    def definitions(self, terms, form_numbers):
        if not terms:
            return {}
        result = self._collection.get(
            where=_where(form_number=form_numbers, defines=[t.lower() for t in terms]))
        return {m["defines"]: ClauseRecord.from_row(m) for m in result["metadatas"]}

    def form_clauses(self, form_number):
        result = self._collection.get(where={"form_number": form_number})
        return [ClauseRecord.from_row(m) for m in result["metadatas"]]


def _where(**filters: list[str] | None) -> dict | None:
    clauses = [{key: {"$in": values}} for key, values in filters.items() if values]
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}
