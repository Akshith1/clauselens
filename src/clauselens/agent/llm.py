"""LLM clients. The agent only depends on the `LLM` protocol: JSON in, JSON out."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Protocol

log = logging.getLogger(__name__)

RATE_LIMIT_RETRIES = 6        # waits up to about a minute in total on the free tier
_RETRY_AFTER = re.compile(r"try again in ([\d.]+)\s*(ms|s)", re.I)


class LLM(Protocol):
    def complete_json(self, system: str, user: str) -> dict: ...


def parse_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating code fences or leading prose."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in model output: {text[:200]!r}")
    return json.loads(text[start:end + 1])


class _ChatJSON:
    """Shared logic for OpenAI-style chat clients."""

    model: str

    def _create(self, messages: list[dict], json_mode: bool):  # pragma: no cover - network
        raise NotImplementedError

    def complete_json(self, system: str, user: str) -> dict:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            response = self._create_with_retry(messages, json_mode=True)
        except Exception as exc:  # some models/endpoints reject response_format
            if "response_format" not in str(exc):
                raise
            response = self._create_with_retry(messages, json_mode=False)
        return parse_json(response.choices[0].message.content or "")

    def _create_with_retry(self, messages: list[dict], json_mode: bool):
        """Wait and retry when the provider's per-minute token limit is hit (HTTP 429)."""
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            try:
                return self._create(messages, json_mode=json_mode)
            except Exception as exc:
                if not is_rate_limit(exc) or attempt == RATE_LIMIT_RETRIES:
                    raise
                wait = retry_delay(exc, attempt)
                log.warning("Rate limited; retrying in %.1fs", wait)
                time.sleep(wait)
        raise RuntimeError("unreachable")


def is_rate_limit(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 429 or "rate_limit" in str(exc).lower()


def retry_delay(exc: Exception, attempt: int) -> float:
    """Use the provider's "try again in Xs" hint when present, else back off exponentially."""
    m = _RETRY_AFTER.search(str(exc))
    if m:
        seconds = float(m.group(1)) / (1000 if m.group(2).lower() == "ms" else 1)
        return min(seconds + 0.5, 30.0)
    return min(2.0 * 2 ** attempt, 30.0)


class GroqLLM(_ChatJSON):
    """Llama via Groq, called directly. Used for local development."""

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set (copy .env.example to .env).")
        from groq import Groq

        self._client = Groq(api_key=api_key)
        self.model = model

    def _create(self, messages, json_mode):  # pragma: no cover - network
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        return self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=0, **extra)


class GatewayLLM(_ChatJSON):
    """Llama via Groq, called through a Databricks serving endpoint with AI Gateway
    (guardrails, usage tracking, payload logging). The Groq key lives in a Databricks
    secret referenced by the endpoint, never in the app. See resources/llm_endpoint.yml."""

    def __init__(self, endpoint: str) -> None:
        from databricks.sdk import WorkspaceClient

        self._client = WorkspaceClient().serving_endpoints.get_open_ai_client()
        self.model = endpoint

    def _create(self, messages, json_mode):  # pragma: no cover - network
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        return self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=0, **extra)
