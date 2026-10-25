"""Builds the clause store, LLM client and agent for the configured backend."""

from __future__ import annotations

from clauselens.config import Settings, get_settings
from clauselens.retrieval.base import ClauseStore


def build_store(settings: Settings | None = None) -> ClauseStore:
    settings = settings or get_settings()
    if settings.backend == "databricks":
        from clauselens.retrieval.vector_search_store import VectorSearchClauseStore

        return VectorSearchClauseStore(settings.vs_endpoint, settings.vs_index)
    if settings.store == "memory":
        from clauselens.parsing.loader import load_forms
        from clauselens.retrieval.base import form_to_records
        from clauselens.retrieval.memory_store import MemoryClauseStore

        store = MemoryClauseStore()
        for form in load_forms(settings.forms_dir):
            store.upsert(form_to_records(form))
        return store
    from clauselens.retrieval.chroma_store import ChromaClauseStore

    return ChromaClauseStore(settings.chroma_dir)


def build_llm(settings: Settings | None = None):
    settings = settings or get_settings()
    from clauselens.agent.llm import GatewayLLM, GroqLLM

    if settings.backend == "databricks":
        return GatewayLLM(settings.gateway_endpoint)
    return GroqLLM(settings.groq_api_key, settings.groq_model)


def build_policy_system(settings: Settings | None = None):
    from clauselens.policies.records import (
        DatabricksPolicySystem,
        FilePolicySystem,
        HttpPolicySystem,
    )

    settings = settings or get_settings()
    if settings.policy_source == "http":
        return HttpPolicySystem(settings.policy_api_url)
    if settings.policy_source == "databricks":
        return DatabricksPolicySystem(settings.policies_table, settings.warehouse_id)
    return FilePolicySystem(settings.policies_file)


def build_agent(settings: Settings | None = None):
    from clauselens.agent.graph import CoverageAgent

    settings = settings or get_settings()
    return CoverageAgent(build_store(settings), build_llm(settings), top_k=settings.top_k,
                         policies=build_policy_system(settings))
