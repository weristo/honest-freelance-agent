"""Where the LLM key comes from: ONE key file, nothing else.

  - the file named by HFA_ENV_FILE, if that variable is set;
  - otherwise `.env` in the current directory (git-ignored).

The key is read from that file only (ANTHROPIC_API_KEY=..., or
HFA_OPENAI_COMPAT_KEY=... for the openai_compat provider). Process
environment variables such as ANTHROPIC_API_KEY or OPENAI_API_KEY are NEVER
used, so a key that happens to be set in the shell cannot be spent by
accident. No file, or no key in it -> the client is "unavailable" (an
explicit state), never a silent mock. The key value is never logged.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

KEY_NAMES = {"anthropic": "ANTHROPIC_API_KEY", "openai_compat": "HFA_OPENAI_COMPAT_KEY"}


@dataclass
class KeyLookup:
    key: Optional[str]
    source: str            # which file was consulted (path only, never the value)
    reason: str = ""       # why there is no key


def key_file_path() -> str:
    return os.path.expanduser(os.environ.get("HFA_ENV_FILE") or ".env")


def read_env_file(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip().startswith("export "):
                k = k.strip()[7:]
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def lookup_key(provider: str) -> KeyLookup:
    name = KEY_NAMES.get(provider)
    path = key_file_path()
    if not name:
        return KeyLookup(None, path, f"no key file entry defined for provider {provider!r}")
    if not os.path.isfile(path):
        return KeyLookup(None, path, f"key file not found ({'HFA_ENV_FILE' if os.environ.get('HFA_ENV_FILE') else '.env'})")
    try:
        values = read_env_file(path)
    except OSError as e:
        return KeyLookup(None, path, f"key file unreadable: {type(e).__name__}")
    key = values.get(name, "")
    if not key:
        return KeyLookup(None, path, f"{name} missing or empty in key file")
    return KeyLookup(key, path)
