"""LLM providers. Provider, model and key come from the environment:

  HFA_LLM_PROVIDER   mock | anthropic | openai_compat     (default: mock)
  HFA_MODEL          default claude-haiku-4-5
  HFA_API_KEY        the judge's own key (falls back to ANTHROPIC_API_KEY /
                     OPENAI_API_KEY for the respective provider)
  HFA_OPENAI_BASE_URL  for openai_compat, e.g. https://api.openai.com/v1

Keys are never logged, never put in prompts, traces or the UI.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Callable, Optional

from .base import LLMResponse

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


def client_from_env(mock_responder: Optional[Callable[[str, str], str]] = None):
    provider = os.environ.get("HFA_LLM_PROVIDER", "mock").strip().lower()
    model = os.environ.get("HFA_MODEL", DEFAULT_MODEL).strip()
    if provider == "mock":
        if mock_responder is None:
            raise ValueError("mock provider needs a responder")
        return MockClient(mock_responder)
    if provider == "anthropic":
        key = os.environ.get("HFA_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("set HFA_API_KEY (or ANTHROPIC_API_KEY)")
        return AnthropicClient(model, key)
    if provider == "openai_compat":
        key = os.environ.get("HFA_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base = os.environ.get("HFA_OPENAI_BASE_URL", "")
        if not key or not base:
            raise ValueError("set HFA_API_KEY and HFA_OPENAI_BASE_URL")
        return OpenAICompatClient(model, key, base)
    raise ValueError(f"unknown HFA_LLM_PROVIDER: {provider}")
