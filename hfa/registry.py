"""Small, auditable source registry: which sources are *policy documents*.

A KILL caused by a verified contradicting fact (gate rule G1) becomes
REFUSE (POLICY_REFUSAL) only when that fact's source is a policy document
listed here: a platform's terms / acceptable-use / legal / site-policy page,
or a tax-authority / legislation site. Anything else is RULE_VIOLATION.
An unknown source is never a REFUSE.
"""
from __future__ import annotations

import urllib.parse

# Path segments (case-insensitive substring of the URL path).
POLICY_PATH_MARKERS = (
    "terms", "acceptable-use", "legal", "site-policy",
)

# Tax-authority and legislation hosts (exact host or any subdomain).
POLICY_DOMAINS = (
    "irs.gov",                  # US Internal Revenue Service
    "nav.gov.hu",               # Hungarian tax authority
    "eur-lex.europa.eu",        # EU legislation
    "bzst.de",                  # German Federal Central Tax Office
    "gesetze-im-internet.de",   # German federal statutes
    "njt.hu",                   # Hungarian national legislation database
)


def _host(url: str) -> str:
    return (urllib.parse.urlsplit(url).hostname or "").lower().rstrip(".")


def is_policy_source(url: str) -> bool:
    if not url:
        return False
    host = _host(url)
    if any(host == d or host.endswith("." + d) for d in POLICY_DOMAINS):
        return True
    path = urllib.parse.urlsplit(url).path.lower()
    return any(m in path for m in POLICY_PATH_MARKERS)


def policy_reason(url: str) -> str:
    host = _host(url)
    for d in POLICY_DOMAINS:
        if host == d or host.endswith("." + d):
            return f"domain:{d}"
    path = urllib.parse.urlsplit(url).path.lower()
    for m in POLICY_PATH_MARKERS:
        if m in path:
            return f"path:{m}"
    return "not in registry"
