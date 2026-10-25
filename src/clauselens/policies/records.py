"""Customer policy records: the "policy system" side of ClauseLens.

Policy wording (forms, endorsements) says what a policy covers in general. A customer's
policy record says what applies to *them*: which form and endorsements, dates, premium,
renewal offer, payments. Real insurers keep these in a policy administration system that
other applications reach through an API; ClauseLens talks to it through `PolicySystem`.

Implementations:
  FilePolicySystem        reads data/policies/sample_policies.json (default, no server needed)
  HttpPolicySystem        calls a policy API over HTTP (see clauselens.policies.api for a mock)
  DatabricksPolicySystem  reads a Unity Catalog table through a SQL warehouse

All bundled records are synthetic sample data.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class PolicyRecord:
    policy_number: str
    insured_name: str
    form: str
    endorsements: list[str] = field(default_factory=list)
    status: str = ""
    effective_date: str | None = None
    expiration_date: str | None = None
    annual_premium: float | None = None
    renewal_premium: float | None = None
    renewal_offer_date: str | None = None
    deductible: float | None = None
    coverage_a_limit: float | None = None
    payment_plan: str | None = None
    next_payment_due: str | None = None
    next_payment_amount: float | None = None

    @classmethod
    def from_dict(cls, data: dict) -> PolicyRecord:
        known = {k: data.get(k) for k in cls.__dataclass_fields__ if k in data}
        endorsements = known.get("endorsements") or []
        if isinstance(endorsements, str):  # stored as JSON text in a table
            endorsements = json.loads(endorsements) if endorsements.strip() else []
        known["endorsements"] = list(endorsements)
        return cls(**known)

    def to_dict(self) -> dict:
        return asdict(self)

    def label(self) -> str:
        return f"{self.policy_number} — {self.insured_name}"


# Field → human label, in the order shown to the model and in citations.
FIELD_LABELS: dict[str, str] = {
    "policy_number": "Policy number",
    "insured_name": "Named insured",
    "status": "Policy status",
    "form": "Policy form",
    "endorsements": "Attached endorsements",
    "effective_date": "Effective date",
    "expiration_date": "Expiration date",
    "annual_premium": "Current annual premium",
    "renewal_premium": "Renewal premium (quoted for next term)",
    "renewal_offer_date": "Renewal offer date",
    "deductible": "Deductible",
    "coverage_a_limit": "Coverage A - Dwelling limit",
    "payment_plan": "Payment plan",
    "next_payment_due": "Next payment due date",
    "next_payment_amount": "Next payment amount",
}

_MONEY = {"annual_premium", "renewal_premium", "deductible", "coverage_a_limit",
          "next_payment_amount"}


def record_fields(record: PolicyRecord) -> list[tuple[str, str, str]]:
    """(field name, label, display value) for every field, with "not on file" for blanks."""
    out = []
    for name, label in FIELD_LABELS.items():
        value = getattr(record, name)
        if value in (None, "", []):
            shown = "not on file"
        elif name == "endorsements":
            shown = ", ".join(value)
        elif name in _MONEY:
            shown = f"${float(value):,.2f}"
        else:
            shown = str(value)
        out.append((name, label, shown))
    return out


class PolicySystem(Protocol):
    def list_policies(self) -> list[PolicyRecord]: ...

    def get_policy(self, policy_number: str) -> PolicyRecord | None: ...


class FilePolicySystem:
    def __init__(self, path: Path) -> None:
        self._path = path

    def list_policies(self) -> list[PolicyRecord]:
        if not self._path.exists():
            return []
        rows = json.loads(self._path.read_text(encoding="utf-8"))
        return [PolicyRecord.from_dict(r) for r in rows]

    def get_policy(self, policy_number: str) -> PolicyRecord | None:
        return next((p for p in self.list_policies() if p.policy_number == policy_number),
                    None)


class HttpPolicySystem:
    """Client for a policy API: GET /policies and GET /policies/{number}."""

    def __init__(self, base_url: str, timeout: float = 5.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    def _get(self, path: str):
        with urllib.request.urlopen(self._base + path, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def list_policies(self) -> list[PolicyRecord]:
        return [PolicyRecord.from_dict(r) for r in self._get("/policies")]

    def get_policy(self, policy_number: str) -> PolicyRecord | None:
        try:
            return PolicyRecord.from_dict(
                self._get("/policies/" + urllib.parse.quote(policy_number)))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise


class DatabricksPolicySystem:
    """Reads `<catalog>.<schema>.policy_records` through a SQL warehouse."""

    def __init__(self, table: str, warehouse_id: str) -> None:
        from databricks.sdk import WorkspaceClient

        self._client = WorkspaceClient()
        self._table, self._warehouse = table, warehouse_id

    def _query(self, where: str = "", params: list | None = None) -> list[PolicyRecord]:
        result = self._client.statement_execution.execute_statement(
            warehouse_id=self._warehouse, wait_timeout="30s",
            statement=f"SELECT * FROM {self._table} {where}", parameters=params or None)
        names = [c.name for c in result.manifest.schema.columns]
        rows = (result.result.data_array if result.result else None) or []
        return [PolicyRecord.from_dict(dict(zip(names, row, strict=True))) for row in rows]

    def list_policies(self) -> list[PolicyRecord]:
        return self._query("ORDER BY policy_number")

    def get_policy(self, policy_number: str) -> PolicyRecord | None:
        from databricks.sdk.service.sql import StatementParameterListItem

        rows = self._query("WHERE policy_number = :p",
                           [StatementParameterListItem(name="p", value=policy_number)])
        return rows[0] if rows else None
