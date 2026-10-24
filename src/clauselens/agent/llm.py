"""LLM clients. The agent only depends on the `LLM` protocol: JSON in, JSON out."""

from __future__ import annotations

import json
import re
from typing import Protocol


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
            response = self._create(messages, json_mode=True)
        except Exception as exc:  # some models/endpoints reject response_format
            if "response_format" not in str(exc):
                raise
            response = self._create(messages, json_mode=False)
        return parse_json(response.choices[0].message.content or "")


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
