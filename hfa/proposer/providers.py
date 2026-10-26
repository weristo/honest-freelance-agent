"""LLM providers. Provider, model and key come from the environment:

  HFA_LLM_PROVIDER   anthropic | openai_compat            (default: anthropic)
  HFA_MODEL          default claude-haiku-4-5
  key                ONLY from the key file (see keys.py): HFA_ENV_FILE or ./.env;
                     shell environment keys are never used
  HFA_OPENAI_BASE_URL  for openai_compat, e.g. https://api.openai.com/v1

The mock is used only when explicitly requested (CLI --mock).

Keys are never logged, never put in prompts, traces or the UI.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Callable, Optional

from .base import LLMResponse
from .keys import lookup_key

DEFAULT_MODEL = "claude-haiku-4-5"


class MockClient:
    """Deterministic stand-in. `responder(system, user) -> JSON text`."""

    provider = "mock"

    def __init__(self, responder: Callable[[str, str], str], model: str = "mock"):
        self.responder = responder
        self.model = model

    @property
    def label(self) -> str:
        return "mock"

    def complete(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        return LLMResponse(self.responder(system, user), 0, 0)


class AnthropicClient:
    """Uses the official `anthropic` SDK (optional dependency, imported lazily)."""

    provider = "anthropic"

    def __init__(self, model: str, api_key: str):
        self.model = model
        self._api_key = api_key

    @property
    def label(self) -> str:
        return f"anthropic:{self.model}"

    def complete(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        import anthropic  # pip install anthropic

        client = anthropic.Anthropic(api_key=self._api_key)
        msg = client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        return LLMResponse(text, msg.usage.input_tokens, msg.usage.output_tokens)


class OpenAICompatClient:
    """Any OpenAI-compatible /chat/completions endpoint (stdlib HTTP)."""

    provider = "openai_compat"

    def __init__(self, model: str, api_key: str, base_url: str):
        if not base_url.startswith("https://"):
            raise ValueError("HFA_OPENAI_BASE_URL must be https")
        self.model = model
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")

    @property
    def label(self) -> str:
        return f"openai_compat:{self.model}"

    def complete(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        body = json.dumps({
            "model": self.model, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }).encode()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers={
            "Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        usage = data.get("usage") or {}
        return LLMResponse(data["choices"][0]["message"]["content"],
                           int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)))


class UnavailableClient:
    """Explicit 'no LLM' state: the proposer reports status 'unavailable'."""

    provider = "unavailable"
    model = "none"

    def __init__(self, reason: str):
        self.reason = reason

    @property
    def label(self) -> str:
        return "unavailable"

    def complete(self, system: str, user: str, max_tokens: int):
        raise RuntimeError(f"llm unavailable: {self.reason}")


def client_from_env(mock: bool = False, mock_responder: Optional[Callable[[str, str], str]] = None):
    """Real client from the key file, or UnavailableClient. The mock is used
    only when the caller passes mock=True (the CLI's --mock switch)."""
    if mock:
        if mock_responder is None:
            raise ValueError("mock needs a responder")
        return MockClient(mock_responder)
    provider = os.environ.get("HFA_LLM_PROVIDER", "anthropic").strip().lower()
    model = os.environ.get("HFA_MODEL", DEFAULT_MODEL).strip()
    if provider == "mock":
        return UnavailableClient("HFA_LLM_PROVIDER=mock is not accepted; use the --mock switch")
    if provider not in ("anthropic", "openai_compat"):
        return UnavailableClient(f"unknown HFA_LLM_PROVIDER: {provider}")
    found = lookup_key(provider)
    if not found.key:
        return UnavailableClient(found.reason)
    if provider == "anthropic":
        return AnthropicClient(model, found.key)
    base = os.environ.get("HFA_OPENAI_BASE_URL", "")
    if not base.startswith("https://"):
        return UnavailableClient("openai_compat needs an https HFA_OPENAI_BASE_URL")
    return OpenAICompatClient(model, found.key, base)
