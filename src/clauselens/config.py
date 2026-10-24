"""Environment-driven settings. `CL_BACKEND` selects local or Databricks backends."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # optional: load .env when python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    backend: str                 # "local" or "databricks"
    # LLM
    groq_api_key: str
    groq_model: str
    gateway_endpoint: str        # Databricks serving endpoint (AI Gateway) in front of Groq
    # Local clause store
    store: str                   # "chroma" or "memory"
    chroma_dir: Path
    forms_dir: Path
    # Databricks clause store
    catalog: str
    schema: str
    vs_endpoint: str
    # Retrieval
    top_k: int

    @property
    def clauses_table(self) -> str:
        return f"{self.catalog}.{self.schema}.clauses"

    @property
    def vs_index(self) -> str:
        return f"{self.catalog}.{self.schema}.clauses_index"


def get_settings() -> Settings:
    env = os.getenv
    return Settings(
        backend=env("CL_BACKEND", "local"),
        groq_api_key=env("GROQ_API_KEY", ""),
        groq_model=env("CL_GROQ_MODEL", "llama-3.3-70b-versatile"),
        gateway_endpoint=env("CL_GATEWAY_ENDPOINT", "clauselens-llm"),
        store=env("CL_LOCAL_STORE", "chroma"),
        chroma_dir=Path(env("CL_CHROMA_DIR", str(REPO_ROOT / "data" / "chroma"))),
        forms_dir=Path(env("CL_FORMS_DIR", str(REPO_ROOT / "data" / "forms"))),
        catalog=env("CL_CATALOG", "clauselens_dev"),
        schema=env("CL_SCHEMA", "policies"),
        vs_endpoint=env("CL_VS_ENDPOINT", "clauselens-vs"),
        top_k=int(env("CL_TOP_K", "6")),
    )
