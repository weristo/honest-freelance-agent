"""Source-date detection for policy documents (legal / tax sources).

The source date is the LATEST of:
  - the HTTP Last-Modified header;
  - a year segment in the URL path ("/2022/" -> end of 2022), also inside a
    query value that is a path ("?path=/docs/2022/x");
  - explicit dates in the text, in Hungarian, German or English formats:
      2022.07.29.  2022-07-29  2025. január 1.  29.07.2022  1. Januar 2025
      January 1, 2025  1 January 2025  12-Aug-2025
      Stand: 03/2024   (month/year after a label -> last day of that month)

Partial dates are read as the LAST day of their period, so a vague date
never makes a page look older than it can be. No date at all -> UNDATED.
This is a heuristic; the README says so.
"""
from __future__ import annotations

import calendar
import re
import urllib.parse
from email.utils import parsedate_to_datetime
from typing import Optional

_HU = ["január", "február", "március", "április", "május", "június", "július",
       "augusztus", "szeptember", "október", "november", "december"]
_DE = ["januar", "februar", "märz", "april", "mai", "juni", "juli", "august",
       "september", "oktober", "november", "dezember"]
_EN = ["january", "february", "march", "april", "may", "june", "july", "august",
       "september", "october", "november", "december"]
_EN3 = [m[:3] for m in _EN]
_MONTH = {**{m: i + 1 for i, m in enumerate(_HU)}, **{m: i + 1 for i, m in enumerate(_DE)},
          **{m: i + 1 for i, m in enumerate(_EN)}, **{m: i + 1 for i, m in enumerate(_EN3)}}
_MN = "|".join(sorted(_MONTH, key=len, reverse=True))

Y = r"(?P<y>(?:19|20)\d\d)"
_PATTERNS = [
    re.compile(rf"(?<!\d){Y}[.\-/](?P<m>0?[1-9]|1[0-2])[.\-/](?P<d>0?[1-9]|[12]\d|3[01])(?!\d)"),
    re.compile(rf"(?<!\d){Y}\.\s*(?P<mn>{_MN})\s+(?P<d>[0-3]?\d)(?!\d)", re.I),
    re.compile(rf"(?<!\d)(?P<d>[0-3]?\d)\.(?P<m>0?[1-9]|1[0-2])\.{Y}(?!\d)"),
    re.compile(rf"(?<!\d)(?P<d>[0-3]?\d)\.?\s+(?P<mn>{_MN})\.?\s+{Y}(?!\d)", re.I),
    re.compile(rf"(?<!\w)(?P<mn>{_MN})\.?\s+(?P<d>[0-3]?\d),?\s+{Y}(?!\d)", re.I),
    re.compile(rf"(?<!\d)(?P<d>[0-3]?\d)-(?P<mn>{_MN})-{Y}(?!\d)", re.I),
]
_LABELED_MONTH_YEAR = re.compile(
    rf"(?:stand|as of|last updated|updated|effective|hatályos|állapot)\s*:?\s*(?P<m>0?[1-9]|1[0-2])/{Y}(?!\d)", re.I)
_URL_YEAR = re.compile(r"/(?P<y>(?:19|20)\d\d)(?=/|$)")


def _iso(y: int, m: int, d: Optional[int] = None) -> Optional[str]:
    last = calendar.monthrange(y, m)[1]
    d = last if d is None else d
    if not 1 <= d <= last:
        return None
    return f"{y:04d}-{m:02d}-{d:02d}"


def dates_in_text(text: str) -> list[str]:
    out = []
    for rx in _PATTERNS:
        for m in rx.finditer(text):
            g = m.groupdict()
            month = int(g["m"]) if g.get("m") else _MONTH.get(g["mn"].lower().rstrip("."))
            if not month:
                continue
            iso = _iso(int(g["y"]), month, int(g["d"]))
            if iso:
                out.append(iso)
    for m in _LABELED_MONTH_YEAR.finditer(text):
        iso = _iso(int(m.group("y")), int(m.group("m")))
        if iso:
            out.append(iso)
    return out


def date_from_url(url: str) -> Optional[str]:
    """Year segment in the URL path, or in a query value that is itself a path
    (e.g. file?path=/docs/2022/x)."""
    parts = urllib.parse.urlsplit(url or "")
    paths = [parts.path] + [v for _, v in urllib.parse.parse_qsl(parts.query) if v.startswith("/")]
    years = [int(m.group("y")) for p in paths for m in _URL_YEAR.finditer(p)]
    return _iso(max(years), 12, 31) if years else None


def date_from_last_modified(header: Optional[str]) -> Optional[str]:
    if not header:
        return None
    try:
        return parsedate_to_datetime(header).strftime("%Y-%m-%d")
    except (TypeError, ValueError, IndexError):
        return None


def source_date(text: str, urls: list[str], last_modified: Optional[str]) -> tuple[Optional[str], list[str]]:
    """Return (latest date or None, list of 'origin:date' found)."""
    found = []
    lm = date_from_last_modified(last_modified)
    if lm:
        found.append(f"last-modified:{lm}")
    for u in urls:
        ud = date_from_url(u)
        if ud:
            found.append(f"url:{ud}")
    found += [f"text:{d}" for d in dates_in_text(text)]
    if not found:
        return None, []
    return max(f.split(":", 1)[1] for f in found), found
