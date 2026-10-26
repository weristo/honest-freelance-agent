"""Deterministic quote verifier. No LLM anywhere in this module.

verify(claim, fetcher) -> Evidence with state VERIFIED / UNVERIFIED /
NEEDS_HUMAN and a reason code. Order of checks (first hit wins):

  not a FACT                        -> NOT_APPLICABLE (gate handles it)
  quote shorter than 20 chars       -> UNVERIFIED  QUOTE_TOO_SHORT
  URL refused by SSRF policy        -> UNVERIFIED  URL_REFUSED
  network error / 404 / 410 / 5xx   -> UNVERIFIED  FETCH_FAILED
  401/403/429, captcha, login wall,
  robots.txt, empty 202             -> UNVERIFIED  FETCH_BLOCKED
  soft-404 page                     -> UNVERIFIED  SOFT_404
  HTML with < 2000 chars of text    -> UNVERIFIED  THIN_CONTENT
  no exact match                    -> UNVERIFIED  NO_MATCH (or SOURCE_CHANGED
                                       if the passage was seen before);
                                       nearest_span shown for display only
  match in >1 distinct context      -> NEEDS_HUMAN MULTI_MATCH if any context
                                       has a guard word or the contexts show
                                       different amounts; else VERIFIED
                                       MULTI_CONSISTENT (all snippets kept)
  redirected to another domain      -> NEEDS_HUMAN REDIRECTED
  legal source older than 365 days  -> NEEDS_HUMAN STALE_SOURCE
  guard word near a SUPPORTS quote  -> NEEDS_HUMAN AMBIGUOUS_CONTEXT
                                       ("near" = the quote's sentence + the
                                       next sentence; words inside the quote
                                       do not count)
  otherwise                         -> VERIFIED    EXACT_MATCH

A string match proves the text is in the source, not that the conclusion
drawn from it is right. That is why INFERENCE claims are never VERIFIED and
why the guard-word heuristic exists (it is a heuristic, not a proof).
"""
from __future__ import annotations

import difflib
import hashlib
import re
import urllib.parse
from datetime import datetime, timezone
from typing import Iterable, Optional

from .extract import BLOCK_PATTERNS, LOGIN_PATH_MARKERS, SOFT_404_PATTERNS
from .fetcher import Fetcher, SourceDoc, utcnow_iso
from .models import Claim, ClaimKind, Evidence, EvidenceState, Polarity, Reason
from .normalize import fold_with_map, norm

MIN_QUOTE_CHARS = 20
THIN_CONTENT_CHARS = 2000
SNIPPET_RADIUS = 300
DEDUP_RADIUS = 40          # see the merge rule in verify()
STALE_DAYS = 365
SOFT_404_MAX_CHARS = 40000

GUARD_WORDS = [
    "except", "unless", "not", "only if", "provided that", "excluding",
    "employer", "employers", "promotion", "kivéve", "nem", "kizárólag",
    "ausgenommen", "nicht",
]
_GUARD_RES = [re.compile(r"(?<!\w)" + re.escape(w) + r"(?!\w)") for w in GUARD_WORDS]
_DATE_RE = re.compile(r"(?<!\d)(20\d\d)[.\-/](0[1-9]|1[0-2])[.\-/](0[1-9]|[12]\d|3[01])(?!\d)")

LAYERS = ("visible", "embedded")


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _host(url: str) -> str:
    h = (urllib.parse.urlsplit(url).hostname or "").lower()
    return h[4:] if h.startswith("www.") else h


def _registrable(host: str) -> str:
    # Rough eTLD+1 (no public-suffix list in the stdlib): last two labels.
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def _find_all(hay: str, needle: str) -> list[int]:
    out, start = [], 0
    while True:
        i = hay.find(needle, start)
        if i < 0:
            return out
        out.append(i)
        start = i + 1


def _parse_iso(ts: Optional[str]) -> datetime:
    if ts:
        try:
            return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def latest_date_in(text: str) -> Optional[str]:
    """Most recent numeric date (YYYY.MM.DD / YYYY-MM-DD / YYYY/MM/DD) in text."""
    best = None
    for y, m, d in _DATE_RE.findall(text):
        cand = f"{y}-{m}-{d}"
        if best is None or cand > best:
            best = cand
    return best


def _blocked(doc: SourceDoc) -> Optional[str]:
    if doc.robots_disallowed:
        return "robots.txt disallows automated access"
    if doc.http_status in (401, 403, 429):
        return f"HTTP {doc.http_status}"
    path = urllib.parse.urlsplit(doc.final_url or "").path.lower()
    if any(m in path for m in LOGIN_PATH_MARKERS):
        return "redirected to a login page"
    text = norm(doc.visible + " " + doc.embedded)
    if doc.http_status == 202 and not text:
        return "HTTP 202 with empty body (bot check)"
    low = text.casefold()
    if len(low) < 5000:
        for p in BLOCK_PATTERNS:
            if p in low:
                return f"bot-check page ('{p}')"
    return None


def _soft_404(doc: SourceDoc) -> bool:
    low = norm(doc.title + " " + doc.visible).casefold()
    if len(low) > SOFT_404_MAX_CHARS:
        return False
    return any(p in low for p in SOFT_404_PATTERNS)


def _nearest_span(q: str, layers: dict[str, str]) -> Optional[str]:
    best, best_ratio = None, 0.0
    n = len(q)
    for text in layers.values():
        if not text:
            continue
        step = max(1, n // 4)
        for i in range(0, max(1, len(text) - n + 1), step):
            window = text[i:i + n + n // 2]
            r = difflib.SequenceMatcher(None, q.casefold(), window.casefold()).ratio()
            if r > best_ratio:
                best, best_ratio = window, r
    return best if best_ratio >= 0.5 else None


def verify(claim: Claim, fetcher: Fetcher, now: Optional[datetime] = None) -> Evidence:
    ev = lambda state, reason, **kw: Evidence(claim.claim_id, state, reason, **kw)  # noqa: E731

    if claim.kind != ClaimKind.FACT:
        return ev(EvidenceState.NOT_APPLICABLE, Reason.NOT_A_FACT,
                  detail=f"{claim.kind.value} claims are never verified")
    q = norm(claim.quote)
    if len(q) < MIN_QUOTE_CHARS:
        return ev(EvidenceState.UNVERIFIED, Reason.QUOTE_TOO_SHORT,
                  detail=f"quote has {len(q)} chars after normalization (min {MIN_QUOTE_CHARS})")

    doc = fetcher.fetch(claim.source_url)
    base = dict(raw_sha256=doc.raw_sha256, fetched_at=doc.fetched_at or utcnow_iso(),
                http_status=doc.http_status, final_url=doc.final_url)
    if doc.refused:
        return ev(EvidenceState.UNVERIFIED, Reason.URL_REFUSED, detail=doc.error or "", **base)
    blocked = _blocked(doc)
    if blocked:
        return ev(EvidenceState.UNVERIFIED, Reason.FETCH_BLOCKED,
                  detail=f"{blocked}; we do not bypass this", **base)
    if doc.error or doc.http_status is None or doc.http_status >= 400:
        return ev(EvidenceState.UNVERIFIED, Reason.FETCH_FAILED,
                  detail=doc.error or f"HTTP {doc.http_status}", **base)
    if _soft_404(doc):
        return ev(EvidenceState.UNVERIFIED, Reason.SOFT_404, detail="page says it was not found", **base)

    layers = {"visible": norm(doc.visible), "embedded": norm(doc.embedded)}
    if doc.is_html and len(layers["visible"]) + len(layers["embedded"]) < THIN_CONTENT_CHARS:
        return ev(EvidenceState.UNVERIFIED, Reason.THIN_CONTENT,
                  detail="page has too little static text (likely JS-rendered)", **base)

    qf = q.casefold()
    hits: list[tuple[str, int, int]] = []          # (layer, start, end) in normed text
    for name in LAYERS:
        folded, idx = fold_with_map(layers[name])
        for off in _find_all(folded, qf):
            end_f = off + len(qf) - 1
            hits.append((name, idx[off], idx[end_f] + 1))

    if not hits:
        reason = Reason.SOURCE_CHANGED if claim.first_seen_snippet_sha256 else Reason.NO_MATCH
        return ev(EvidenceState.UNVERIFIED, reason,
                  nearest_span=_nearest_span(q, layers),
                  detail="quote not found verbatim (nearest span is display-only)", **base)

    # Merge hits whose surrounding context is identical (same sentence shipped
    # in both layers, or a pricing table duplicated for mobile/desktop).
    # Across layers, one hit per layer is the same passage shipped twice
    # (rendered text + page data), so contexts are compared within a layer.
    # Within a layer, two hits are the same passage if the DEDUP_RADIUS chars
    # on the left OR on the right are identical (a repeated table block).
    per_layer: dict[str, list[tuple[str, int, int]]] = {}
    for layer, s, e in hits:
        text = layers[layer]
        left = text[max(0, s - DEDUP_RADIUS):s].casefold()
        right = text[e:e + DEDUP_RADIUS].casefold()
        kept = per_layer.setdefault(layer, [])
        if not any(left == text[max(0, ks - DEDUP_RADIUS):ks].casefold()
                   or right == text[ke:ke + DEDUP_RADIUS].casefold() for _, ks, ke in kept):
            kept.append((layer, s, e))
    contexts = max(per_layer.values(), key=len)
    if len(contexts) == 1 and "visible" in per_layer:
        contexts = per_layer["visible"]                   # prefer the rendered layer

    def snip(h):
        layer, s, e = h
        t = layers[layer]
        return t[max(0, s - SNIPPET_RADIUS):e + SNIPPET_RADIUS]

    layer, s, e = contexts[0]
    snippet = snip(contexts[0])
    ok = dict(match_layer=layer, match_offset=s, context_snippet=snippet,
              snippet_sha256=sha256_hex(layers[layer][s:e]), **base)

    multi_detail = ""
    if len(contexts) > 1:
        # Several distinct places. Accept only if no place carries a guard
        # word and the places do not disagree on amounts (prices, %, fees).
        guards = [g for h in contexts for g in _guard_hits(layers[h[0]], h[1], h[2])]
        amounts = {frozenset(_amounts(layers[h[0]], h[1], h[2])) for h in contexts}
        if guards or len(amounts) > 1:
            why = (f"guard words {sorted(set(guards))}" if guards else "different amounts nearby")
            return ev(EvidenceState.NEEDS_HUMAN, Reason.MULTI_MATCH, guard_hits=sorted(set(guards)),
                      other_snippets=[snip(h) for h in contexts[1:]],
                      detail=f"quote appears in {len(contexts)} different contexts; {why}", **ok)
        ok["other_snippets"] = [snip(h) for h in contexts[1:]]
        multi_detail = f"{len(contexts)} consistent matches"

    if doc.final_url and _registrable(_host(doc.final_url)) != _registrable(_host(claim.source_url)):
        return ev(EvidenceState.NEEDS_HUMAN, Reason.REDIRECTED,
                  detail=f"redirected to {_host(doc.final_url)}", **ok)

    if claim.source_domain_type == "legal":
        src_date = latest_date_in(layers["visible"])
        if src_date:
            ref = now or _parse_iso(doc.fetched_at)
            age = (ref - datetime.strptime(src_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)).days
            if age > STALE_DAYS:
                return ev(EvidenceState.NEEDS_HUMAN, Reason.STALE_SOURCE, source_date=src_date,
                          detail=f"source dated {src_date} ({age} days old) - may be outdated", **ok)
            ok["source_date"] = src_date

    guard_hits = _guard_hits(layers[layer], s, e)
    if guard_hits and claim.polarity == Polarity.SUPPORTS:
        return ev(EvidenceState.NEEDS_HUMAN, Reason.AMBIGUOUS_CONTEXT, guard_hits=guard_hits,
                  detail="words near the quote may narrow or reverse it (heuristic)", **ok)

    detail = "; ".join(x for x in (multi_detail,
                                   "found in page data (collapsed section)" if layer == "embedded" else "") if x)
    reason = Reason.MULTI_CONSISTENT if multi_detail else Reason.EXACT_MATCH
    return ev(EvidenceState.VERIFIED, reason, guard_hits=guard_hits, detail=detail, **ok)


_BOUNDARY = re.compile(r"[.!?](?=\s|$)")
_AMOUNT_RES = [re.compile(r"[$€£]\s?\d[\d.,]*\d|[$€£]\s?\d"),
               re.compile(r"\d[\d.,]*\s?%"),
               re.compile(r"\d[\d.,]*\s?(?:usd|eur|gbp|huf|ft)(?!\w)")]


def guard_window(t: str, s: int, e: int) -> tuple[str, str]:
    """Text of the sentence containing the quote (before / after the quote)
    plus the sentence right after it. The quote's own words are excluded."""
    start = 0
    for m in _BOUNDARY.finditer(t, 0, s):
        start = m.end()
    m1 = _BOUNDARY.search(t, max(e - 1, 0))
    end1 = m1.end() if m1 else len(t)
    m2 = _BOUNDARY.search(t, end1)
    end2 = m2.end() if m2 else len(t)
    return t[start:s], t[e:end2]


def _guard_hits(t: str, s: int, e: int) -> list[str]:
    before, after = guard_window(t, s, e)
    near = (before + " \u2016 " + after).casefold()
    return [w for w, rx in zip(GUARD_WORDS, _GUARD_RES) if rx.search(near)]


def _amounts(t: str, s: int, e: int) -> set[str]:
    around = (t[max(0, s - SNIPPET_RADIUS):s] + " \u2016 " + t[e:e + SNIPPET_RADIUS]).casefold()
    return {re.sub(r"\s", "", m.group(0)) for rx in _AMOUNT_RES for m in rx.finditer(around)}


def find_contradictions(claims: Iterable[Claim], evidence: list[Evidence]) -> list[Evidence]:
    """Two VERIFIED claims on the same rule_topic with opposite polarity ->
    both become NEEDS_HUMAN(CONTRADICTED). Returns a new evidence list."""
    by_id = {e.claim_id: e for e in evidence}
    topics: dict[str, list[Claim]] = {}
    for c in claims:
        if c.rule_topic and by_id.get(c.claim_id) and by_id[c.claim_id].state == EvidenceState.VERIFIED:
            topics.setdefault(c.rule_topic, []).append(c)
    flagged: set[str] = set()
    for group in topics.values():
        if {c.polarity for c in group} == {Polarity.SUPPORTS, Polarity.CONTRADICTS}:
            flagged.update(c.claim_id for c in group)
    out = []
    for e in evidence:
        if e.claim_id in flagged:
            e = Evidence(**{**e.__dict__, "state": EvidenceState.NEEDS_HUMAN,
                            "reason_code": Reason.CONTRADICTED,
                            "detail": "another verified source on the same topic says the opposite"})
        out.append(e)
    return out
