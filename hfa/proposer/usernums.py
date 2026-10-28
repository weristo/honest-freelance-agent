"""Which numbers of a USER_ATTESTED claim come from the user's own input?

The user's tick (CLI --user-confirm, the UI field, the eval case text) covers
what the USER said, not what the model wrote in the user's name. A model can
invent a "user" number ("my cost is 20 USD" while the request says 26); if
that claim were ticked, a CALC would compute from it. So the proposer layer
ticks a USER_ATTESTED claim only if EVERY number in it appears in the user's
own input text, compared as Decimal together with its unit:

  * value: Decimal after reading thousands/decimal separators the way
    hfa.calc.parse_literal does ("26" == "26.00", "1,642" == "1642",
    "25 000" == "25000"); an "M" before a currency means million
    ("6 M Ft" == "6 000 000 Ft"). A dot or comma followed by exactly three
    digits ("20.000") is ambiguous (thousands or decimals), so such a number
    also has to have the same digits in the input - never a looser match;
  * unit: "%" only matches "%" (26% is not 26); currencies are grouped
    ($ = USD = dollar, EUR = euro = €, £ = GBP, Ft = HUF = forint) and must be
    the same group, before or after the number ("USD 120" == "120 USD");
    a claim number WITH a currency needs that currency in the input
    ("26 USD" is not backed by a bare "26"); a bare claim number is backed
    by the same value with or without a currency (but never by a %).

A claim without numbers is ticked as before (the user's tick alone).
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import NamedTuple, Optional

from ..calc import CalcError, parse_literal

USER_NUMBER_NOT_IN_INPUT = "user number not in the user's input"

_CUR_WORDS = {"usd": "USD", "dollar": "USD", "dollars": "USD", "eur": "EUR", "euro": "EUR",
              "euros": "EUR", "gbp": "GBP", "huf": "HUF", "ft": "HUF", "forint": "HUF"}
_CUR_SIGNS = {"$": "USD", "€": "EUR", "£": "GBP"}

_NUM = r"\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)*"
_TOKEN = re.compile(
    r"(?<![\w.,])(?P<pre>[$€£]|(?:usd|eur|gbp|huf)(?=[\s\d]))?\s?(?P<num>" + _NUM + r")(?![\d])"
    r"(?P<post>\s?%|\s?[$€£]|\s?(?P<mult>[mM]\s?)?(?P<word>usd|eur|euros?|gbp|huf|ft|forint|dollars?)(?![^\W\d_]))?",
    re.I)


class Num(NamedTuple):
    value: Decimal
    unit: Optional[str]          # "%", "USD", "EUR", "GBP", "HUF" or None
    digits: str                  # digits only, for ambiguous separators
    ambiguous: bool
    raw: str


def _ambiguous(num: str) -> bool:
    seps = re.findall(r"[.,]", num)
    return len(set(seps)) == 1 and bool(re.search(r"[.,]\d{3}$", num))


def numbers(text: str) -> list[Num]:
    out = []
    for m in _TOKEN.finditer(text or ""):
        num = m.group("num")
        try:
            v = parse_literal(num)
        except CalcError:
            continue
        post = (m.group("post") or "").strip()
        unit = None
        if post == "%":
            unit = "%"
        elif post in _CUR_SIGNS:
            unit = _CUR_SIGNS[post]
        elif m.group("word"):
            unit = _CUR_WORDS[m.group("word").lower()]
            if m.group("mult"):
                v = v * 1_000_000
        pre = (m.group("pre") or "").lower()
        pre_unit = (_CUR_SIGNS.get(pre) or _CUR_WORDS[pre]) if pre else None
        if unit is None:
            unit = pre_unit
        elif pre_unit and pre_unit != unit:
            unit = "MIXED " + m.group(0).strip()     # "€26 USD": matches only itself
        try:
            v = Decimal(v)
        except InvalidOperation:
            continue
        out.append(Num(v, unit, re.sub(r"\D", "", num), _ambiguous(num), m.group(0).strip()))
    return out


def _backs(user: Num, claim: Num) -> bool:
    if user.value != claim.value:
        return False
    if (user.ambiguous or claim.ambiguous) and user.digits != claim.digits:
        return False
    if claim.unit == "%" or user.unit == "%":
        return claim.unit == user.unit
    if claim.unit is None:
        return True
    return user.unit == claim.unit


def missing_numbers(claim_text: str, user_input: str) -> list[str]:
    """Numbers of `claim_text` (as written) that the user's input does not contain."""
    have = numbers(user_input)
    return [n.raw for n in numbers(claim_text) if not any(_backs(u, n) for u in have)]
