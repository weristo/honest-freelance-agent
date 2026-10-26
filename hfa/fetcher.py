"""Deterministic fetcher with an SSRF guard, plus a fixture (snapshot) fetcher.

The URL to fetch comes from an LLM, so it is treated as hostile:
 - https only; no credentials in the URL; default port only
 - the host must resolve exclusively to public addresses (private, loopback,
   link-local, reserved, multicast and ULA ranges are refused), checked again
   on every redirect hop (max 5)
 - 10 MB body cap, 20 s timeout, an identifying User-Agent
 - robots.txt is honoured; captchas, logins and paywalls are never bypassed

Known limitation: the DNS check and the actual connection resolve separately
(TOCTOU / DNS rebinding). Acceptable for a demo; documented in the README.
"""
from __future__ import annotations

import hashlib
import ipaddress
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional, Protocol

from .extract import extract_html

USER_AGENT = "HonestFreelanceAgent/0.1 (quote verifier; +see README)"
MAX_BYTES = 10 * 1024 * 1024
TIMEOUT_S = 20
MAX_REDIRECTS = 5


class UrlRefused(Exception):
    """The URL failed the SSRF policy. Never fetched."""


@dataclass
class SourceDoc:
    url: str
    final_url: str
    http_status: Optional[int]
    content_type: str = ""
    raw_sha256: Optional[str] = None
    fetched_at: Optional[str] = None
    visible: str = ""
    embedded: str = ""
    title: str = ""
    is_html: bool = True
    error: Optional[str] = None          # network error / refusal text
    refused: bool = False                # SSRF policy refusal
    robots_disallowed: bool = False
    last_modified: Optional[str] = None  # HTTP Last-Modified header, if any
    meta: dict = field(default_factory=dict)


class Fetcher(Protocol):
    def fetch(self, url: str) -> SourceDoc: ...


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- SSRF policy -------------------------------------------------------------

Resolver = Callable[[str], list[str]]


def _default_resolver(host: str) -> list[str]:
    infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    return sorted({i[4][0] for i in infos})


def check_url(url: str, resolver: Resolver = _default_resolver) -> None:
    """Raise UrlRefused unless `url` is an https URL to a public host."""
    try:
        p = urllib.parse.urlsplit(url)
    except ValueError as e:
        raise UrlRefused(f"unparseable URL: {e}") from None
    if p.scheme != "https":
        raise UrlRefused(f"scheme not allowed: {p.scheme or '(none)'}")
    if p.username or p.password:
        raise UrlRefused("credentials in URL are not allowed")
    host = (p.hostname or "").rstrip(".").lower()
    if not host:
        raise UrlRefused("missing host")
    if p.port not in (None, 443):
        raise UrlRefused(f"port not allowed: {p.port}")
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        raise UrlRefused(f"local host name refused: {host}")
    try:
        literal = ipaddress.ip_address(host)
        addrs = [str(literal)]
    except ValueError:
        try:
            addrs = resolver(host)
        except OSError as e:
            raise UrlRefused(f"DNS resolution failed: {e}") from None
    if not addrs:
        raise UrlRefused("host resolves to no address")
    for a in addrs:
        ip = ipaddress.ip_address(a.split("%")[0])
        if not ip.is_global or ip.is_multicast:
            raise UrlRefused(f"non-public address refused: {host} -> {ip}")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class LiveFetcher:
    """Fresh fetch on every call (no cache for verification)."""

    def __init__(self, resolver: Resolver = _default_resolver, respect_robots: bool = True):
        self.resolver = resolver
        self.respect_robots = respect_robots
        self._opener = urllib.request.build_opener(_NoRedirect)

    def _robots_allows(self, url: str) -> bool:
        p = urllib.parse.urlsplit(url)
        robots_url = f"https://{p.hostname}/robots.txt"
        try:
            check_url(robots_url, self.resolver)
            req = urllib.request.Request(robots_url, headers={"User-Agent": USER_AGENT})
            with self._opener.open(req, timeout=TIMEOUT_S) as r:
                body = r.read(512 * 1024).decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            return e.code not in (401, 403)   # missing robots.txt = allowed
        except (urllib.error.URLError, OSError, UrlRefused):
            return True
        rp = urllib.robotparser.RobotFileParser()
        rp.parse(body.splitlines())
        return rp.can_fetch(USER_AGENT, url)

    def fetch(self, url: str) -> SourceDoc:
        fetched_at = utcnow_iso()
        current = url
        try:
            check_url(current, self.resolver)
        except UrlRefused as e:
            return SourceDoc(url, url, None, fetched_at=fetched_at, error=str(e), refused=True)
        if self.respect_robots and not self._robots_allows(current):
            return SourceDoc(url, url, None, fetched_at=fetched_at,
                             error="robots.txt disallows this URL", robots_disallowed=True)
        for _ in range(MAX_REDIRECTS + 1):
            req = urllib.request.Request(current, headers={
                "User-Agent": USER_AGENT,
                "Accept-Language": "en-US,en;q=0.9",
            })
            try:
                resp = self._opener.open(req, timeout=TIMEOUT_S)
                status = resp.status
            except urllib.error.HTTPError as e:
                resp, status = e, e.code
            except (urllib.error.URLError, OSError) as e:
                return SourceDoc(url, current, None, fetched_at=fetched_at, error=str(e))
            if status in (301, 302, 303, 307, 308):
                loc = resp.headers.get("Location", "")
                resp.close()
                nxt = urllib.parse.urljoin(current, loc)
                try:
                    check_url(nxt, self.resolver)
                except UrlRefused as e:
                    return SourceDoc(url, nxt, status, fetched_at=fetched_at,
                                     error=f"redirect refused: {e}", refused=True)
                current = nxt
                continue
            raw = resp.read(MAX_BYTES + 1)
            ctype = resp.headers.get("Content-Type", "")
            last_mod = resp.headers.get("Last-Modified")
            resp.close()
            if len(raw) > MAX_BYTES:
                return SourceDoc(url, current, status, ctype, fetched_at=fetched_at,
                                 error="body exceeds 10 MB cap")
            doc = doc_from_bytes(url, current, status, ctype, raw, fetched_at)
            doc.last_modified = last_mod
            return doc
        return SourceDoc(url, current, None, fetched_at=fetched_at, error="too many redirects")


def doc_from_bytes(url: str, final_url: str, status: int, ctype: str,
                   raw: bytes, fetched_at: str) -> SourceDoc:
    sha = hashlib.sha256(raw).hexdigest()
    charset = "utf-8"
    if "charset=" in ctype:
        charset = ctype.split("charset=")[-1].split(";")[0].strip() or "utf-8"
    text = raw.decode(charset, "replace")
    is_html = "html" in ctype.lower() or text.lstrip()[:15].lower().startswith(("<!doctype", "<html"))
    if is_html:
        visible, embedded, title = extract_html(text)
    else:
        visible, embedded, title = text, "", ""
    return SourceDoc(url, final_url, status, ctype, sha, fetched_at,
                     visible, embedded, title, is_html)


# --- fixture / snapshot fetcher ---------------------------------------------

EMBEDDED_MARKER = "=== EMBEDDED LAYER ==="


def load_snapshot(path: str) -> SourceDoc:
    """Snapshot format: '# key=value' header lines, blank line, visible text,
    optional '=== EMBEDDED LAYER ===' line followed by the embedded layer."""
    with open(path, encoding="utf-8") as f:
        content = f.read()
    meta: dict[str, str] = {}
    lines = content.split("\n")
    i = 0
    while i < len(lines) and lines[i].startswith("#"):
        for tok in lines[i][1:].strip().split(" "):
            if "=" in tok:
                k, v = tok.split("=", 1)
                meta[k] = v
        # 'ctype=text/html; charset=utf-8' contains a space; keep it whole
        if "ctype=" in lines[i]:
            meta["ctype"] = lines[i].split("ctype=", 1)[1].strip()
        i += 1
    body = "\n".join(lines[i:])
    if EMBEDDED_MARKER in body:
        visible, embedded = body.split(EMBEDDED_MARKER, 1)
    else:
        visible, embedded = body, ""
    http = meta.get("http")
    ctype = meta.get("ctype", "text/html")
    return SourceDoc(
        url=meta.get("url", ""),
        final_url=meta.get("final_url", meta.get("url", "")),
        http_status=int(http) if http and http.isdigit() else None,
        content_type=ctype,
        raw_sha256=meta.get("raw_sha256"),
        fetched_at=meta.get("fetched_at"),
        visible=visible, embedded=embedded,
        is_html="html" in ctype.lower(),
        last_modified=meta.get("last_modified"),
        meta=meta,
    )


class FixtureFetcher:
    """Maps URL -> snapshot file. Used by tests and by the offline eval."""

    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    @classmethod
    def from_dir(cls, directory: str) -> "FixtureFetcher":
        mapping: dict[str, str] = {}
        for name in sorted(os.listdir(directory)):
            if name.endswith(".txt"):
                p = os.path.join(directory, name)
                url = load_snapshot(p).url
                if url:
                    mapping.setdefault(url, p)
        return cls(mapping)

    def fetch(self, url: str) -> SourceDoc:
        path = self.mapping.get(url)
        if not path:
            return SourceDoc(url, url, None, error="no fixture for URL")
        return load_snapshot(path)
