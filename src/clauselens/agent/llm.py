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
        return parse_json(message_text(response.choices[0].message.content))

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


def message_text(content) -> str:
    """Reasoning models on Databricks (e.g. gpt-oss) return content as a list of blocks,
    [{"type": "reasoning", ...}, {"type": "text", "text": "..."}]; keep only the text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        block = block if isinstance(block, dict) else getattr(block, "__dict__", {})
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


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
    """A Databricks model serving endpoint, called by name through the workspace's
    OpenAI-compatible API. On the trial target this is the Databricks-hosted
    `databricks-gpt-oss-120b`; on the azure target it's our own endpoint fronting Groq with
    AI Gateway (key kept in a Databricks secret). See databricks.yml."""

    def __init__(self, endpoint: str, workspace=None) -> None:
        if workspace is None:
            from databricks.sdk import WorkspaceClient

            workspace = WorkspaceClient()
        self._config = workspace.config
        self.model = endpoint

    def _client(self):
        """An OpenAI client for <host>/serving-endpoints. Built per call so the OAuth token
        (refreshed by the SDK, valid about an hour) is always current. Built here rather
        than with get_open_ai_client(), which not every SDK version has."""
        from openai import OpenAI

        auth = self._config.authenticate().get("Authorization", "")
        token = auth.split(" ", 1)[1] if " " in auth else auth
        return OpenAI(base_url=self._config.host.rstrip("/") + "/serving-endpoints",
                      api_key=token)

    def _create(self, messages, json_mode):  # pragma: no cover - network
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        return self._client().chat.completions.create(
            model=self.model, messages=messages, temperature=0, **extra)
