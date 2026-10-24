"""In-memory BM25 clause store. No dependencies; used by tests and as a quick local option."""

from __future__ import annotations

import math
import re
from collections import Counter

from clauselens.retrieval.base import ClauseRecord, ScoredClause

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "or", "and", "to", "in", "is", "for", "by", "on", "we",
         "our", "us", "this", "that", "it", "be", "are", "any", "from", "under", "with",
         "if", "my", "i", "does", "do", "what", "which"}


def _fold(token: str) -> str:
    """Light plural folding so 'sewers' matches 'sewer' and 'drains' matches 'drain'."""
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(text: str) -> list[str]:
    return [_fold(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class MemoryClauseStore:
    def __init__(self, k1: float = 1.4, b: float = 0.75) -> None:
        self._records: dict[str, ClauseRecord] = {}
        self._k1, self._b = k1, b

    def upsert(self, records: list[ClauseRecord]) -> None:
        for record in records:
            self._records[record.record_id] = record

    def search(self, query, k, form_numbers=None, section_types=None):
        candidates = [r for r in self._records.values()
                      if (not form_numbers or r.form_number in form_numbers)
                      and (not section_types or r.section_type in section_types)]
        if not candidates:
            return []
        docs = {r.record_id: _tokens(r.search_text) for r in candidates}
        avg_len = sum(len(d) for d in docs.values()) / len(docs)
        df = Counter(t for d in docs.values() for t in set(d))
        n = len(docs)
        scored = []
        for record in candidates:
            tf = Counter(docs[record.record_id])
            length = len(docs[record.record_id]) or 1
            score = 0.0
            for term in set(_tokens(query)):
                if term not in tf:
                    continue
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                norm = tf[term] * (self._k1 + 1) / (
                    tf[term] + self._k1 * (1 - self._b + self._b * length / avg_len))
                score += idf * norm
            if score > 0:
                scored.append(ScoredClause(record, score))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]

    def definitions(self, terms, form_numbers):
        wanted = {t.lower() for t in terms}
        return {r.defines: r for r in self._records.values()
                if r.defines in wanted and r.form_number in form_numbers}

    def form_clauses(self, form_number):
        return [r for r in self._records.values() if r.form_number == form_number]
