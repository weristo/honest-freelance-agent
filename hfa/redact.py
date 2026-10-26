"""Outbound log/trace redactor. Applied to every line before it is written.

Secrets, e-mail addresses, IP addresses and absolute home paths are masked.
Lines that mention health-related terms are dropped entirely (replaced by a
marker), because a decision trace must never carry that kind of data.
"""
from __future__ import annotations

import re

_RULES = [
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer [REDACTED_TOKEN]"),
    (re.compile(r"(?i)\bBasic\s+[A-Za-z0-9+/=]{8,}"), "Basic [REDACTED_TOKEN]"),
    (re.compile(r"(?i)(authorization\s*[:=]\s*)\S+(\s+\S+)?"), r"\1[REDACTED_TOKEN]"),
    (re.compile(r"(?i)((?:client_secret|api[_-]?key|access_token|secret)[\"']?\s*[:=]\s*[\"']?)[^\s\"',}]+"),
     r"\1[REDACTED_TOKEN]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"), "[REDACTED_TOKEN]"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[EMAIL]"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[IP]"),
    (re.compile(r"(?i)\b(?:[0-9a-f]{1,4}:){2,7}[0-9a-f]{1,4}\b"), "[IP]"),
    (re.compile(r"(?:/Users|/home)/[^\s\"']+"), "[PATH]"),
    (re.compile(r"\b[A-Za-z0-9+/_-]{40,}={0,2}"),
     lambda m: m.group(0) if re.fullmatch(r"[0-9a-f]+", m.group(0)) else "[REDACTED_TOKEN]"),
]
# (lower-case hex runs are kept: those are our own SHA-256 evidence hashes)

HEALTH_TERMS = [
    "diagnosis", "diagnose", "medical", "illness", "disease", "therapy",
    "hospital", "disability", "medication",
    "diagnózis", "betegség", "kórház", "fogyatékosság",
    "diagnose", "krankheit", "krankenhaus",
]
_HEALTH = re.compile(r"(?i)(?<!\w)(" + "|".join(map(re.escape, HEALTH_TERMS)) + r")(?!\w)")


def redact(line: str) -> str:
    if _HEALTH.search(line):
        return "[HEALTH_REDACTED]"
    for rx, repl in _RULES:
        line = rx.sub(repl, line)
    return line
