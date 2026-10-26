"""CALC claims: arithmetic done by code, never by the LLM.

A CALC claim names its inputs and a few named steps:

    inputs : {"price": {"literal": "30",    "source": "u1"},
              "rate":  {"literal": "10%",   "source": "c1"},
              "minfee":{"literal": "$5.00", "source": "c1"},
              "cost":  {"literal": "26",    "source": "u1"}}
    steps  : [["fee", "max(rate*price, minfee)"], ["net", "price - fee"]]
    claimed: {"net": "25"}                       # what the proposer says
    check  : ["net", ">=", "cost"]               # must hold, else KILL
    on_fail: "NOT_WORTH"                         # or "RULE_VIOLATION"

Every input must point at a VERIFIED FACT claim whose quote contains the
literal, or at a USER_ATTESTED claim whose text contains it. The value is
parsed from the literal by code ("10%" -> 0.10, "$5.00" -> 5.00).

The expression language is deliberately tiny: decimal numbers, input/step
names, + - * / (also the symbols × ÷ −), parentheses, max(...) and min(...).
It has its own tokenizer and recursive-descent parser; Python's eval() is
never used. Arithmetic uses decimal.Decimal.

What this guarantees: the sums are right for the stated formula and inputs.
What it does not: that the formula is the right reading of the quoted rule.
That derivation stays an INFERENCE and is shown as such.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, DivisionByZero, InvalidOperation, localcontext
from typing import Optional

MAX_EXPR_CHARS = 300
MAX_TOKENS = 120
MAX_DEPTH = 20
MAX_STEPS = 12
MAX_ABS = Decimal("1e15")
FUNCS = ("max", "min")
OPS = (">=", "<=", ">", "<")

_TOKEN = re.compile(r"\s*(?:(?P<num>\d+(?:\.\d+)?)|(?P<name>[a-z][a-z0-9_]{0,31})|(?P<op>[-+*/(),]))")
_SYMBOLS = str.maketrans({"×": "*", "÷": "/", "−": "-", "–": "-"})


class CalcError(ValueError):
    pass


_AMOUNT = [re.compile(r"[$€£]\s?\d"), re.compile(r"\d\s?%"),
           re.compile(r"(?i)\d[\d.,\s]*\s?(?:usd|eur|euro|gbp|huf|ft|forint)(?!\w)")]


def has_amount(text: str) -> bool:
    """Does the text state a money amount or a percentage?"""
    return any(rx.search(text or "") for rx in _AMOUNT)


def tokenize(expr: str) -> list[tuple[str, str]]:
    if not isinstance(expr, str):
        raise CalcError("expression must be a string")
    if len(expr) > MAX_EXPR_CHARS:
        raise CalcError(f"expression longer than {MAX_EXPR_CHARS} chars")
    s = expr.translate(_SYMBOLS)
    pos, out = 0, []
    while pos < len(s):
        if s[pos:].strip() == "":
            break
        m = _TOKEN.match(s, pos)
        if not m or m.end() == pos:
            raise CalcError(f"unexpected character at {pos}: {s[pos]!r}")
        kind = m.lastgroup
        out.append((kind, m.group(kind)))
        pos = m.end()
        if len(out) > MAX_TOKENS:
            raise CalcError(f"more than {MAX_TOKENS} tokens")
    return out


class _Parser:
    def __init__(self, tokens, env: dict[str, Decimal]):
        self.t, self.i, self.env, self.depth = tokens, 0, env, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def take(self, value=None):
        tok = self.peek()
        if tok[0] is None or (value is not None and tok[1] != value):
            raise CalcError(f"expected {value or 'token'} at token {self.i}")
        self.i += 1
        return tok

    def _enter(self):
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise CalcError("expression nested too deeply")

    def expr(self) -> Decimal:
        self._enter()
        v = self.term()
        while self.peek()[1] in ("+", "-"):
            op = self.take()[1]
            r = self.term()
            v = v + r if op == "+" else v - r
        self.depth -= 1
        return _bounded(v)

    def term(self) -> Decimal:
        v = self.factor()
        while self.peek()[1] in ("*", "/"):
            op = self.take()[1]
            r = self.factor()
            if op == "*":
                v = v * r
            else:
                if r == 0:
                    raise CalcError("division by zero")
                v = v / r
            v = _bounded(v)
        return v

    def factor(self) -> Decimal:
        self._enter()
        kind, val = self.peek()
        if val == "-":
            self.take()
            v = -self.factor()
        elif val == "+":
            self.take()
            v = self.factor()
        elif kind == "num":
            self.take()
            v = Decimal(val)
        elif kind == "name" and val in FUNCS:
            self.take()
            self.take("(")
            args = [self.expr()]
            while self.peek()[1] == ",":
                self.take()
                args.append(self.expr())
            self.take(")")
            v = max(args) if val == "max" else min(args)
        elif kind == "name":
            self.take()
            if val not in self.env:
                raise CalcError(f"unknown name: {val}")
            v = self.env[val]
        elif val == "(":
            self.take()
            v = self.expr()
            self.take(")")
        else:
            raise CalcError(f"unexpected token {val!r}")
        self.depth -= 1
        return _bounded(v)


def _bounded(v: Decimal) -> Decimal:
    if not v.is_finite() or abs(v) > MAX_ABS:
        raise CalcError("value out of range")
    return v


def evaluate(expr: str, env: Optional[dict[str, Decimal]] = None) -> Decimal:
    """Evaluate a tiny arithmetic expression. Raises CalcError on anything else."""
    tokens = tokenize(expr)
    if not tokens:
        raise CalcError("empty expression")
    with localcontext() as ctx:
        ctx.prec = 34
        ctx.traps[DivisionByZero] = True
        ctx.traps[InvalidOperation] = True
        p = _Parser(tokens, dict(env or {}))
        try:
            v = p.expr()
        except (DivisionByZero, InvalidOperation, ArithmeticError) as e:
            raise CalcError(f"arithmetic error: {type(e).__name__}") from None
        except RecursionError:
            raise CalcError("expression nested too deeply") from None
        if p.i != len(tokens):
            raise CalcError(f"trailing tokens from {p.i}")
        return v


_LIT_NUM = re.compile(r"\d[\d\s.,]*")


def parse_literal(lit: str) -> Decimal:
    """'10%' -> 0.10, '$5.00 USD' -> 5.00, '3,40%' -> 0.034, '25 000' -> 25000,
    '1,642' -> 1642."""
    m = _LIT_NUM.search(lit or "")
    if not m:
        raise CalcError(f"no number in literal {lit!r}")
    raw = m.group(0).strip().replace(" ", "").replace(" ", "")
    if "," in raw and "." in raw:
        raw = raw.replace(",", "") if raw.rfind(".") > raw.rfind(",") else raw.replace(".", "").replace(",", ".")
    elif "," in raw:
        head, _, tail = raw.rpartition(",")
        raw = raw.replace(",", "") if len(tail) == 3 else head.replace(",", "") + "." + tail
    raw = raw.rstrip(".")
    try:
        v = Decimal(raw)
    except InvalidOperation:
        raise CalcError(f"cannot read number in {lit!r}") from None
    return v / 100 if "%" in lit[m.end():m.end() + 2] or lit.strip().endswith("%") else v


@dataclass
class CalcResult:
    claim_id: str
    status: str                       # OK | MISMATCH | INVALID
    values: dict[str, str] = field(default_factory=dict)
    check_passed: Optional[bool] = None
    on_fail: str = "NOT_WORTH"
    detail: str = ""


def run_calc(claim, claims_by_id: dict, evidence_by_id: dict) -> CalcResult:
    """Evaluate one CALC claim against the other claims and the evidence."""
    from .models import ClaimKind, EvidenceState   # local: avoid import cycle

    spec = claim.calc or {}
    cid = claim.claim_id
    on_fail = spec.get("on_fail", "NOT_WORTH")
    if on_fail not in ("NOT_WORTH", "RULE_VIOLATION"):
        return CalcResult(cid, "INVALID", detail=f"on_fail must be NOT_WORTH or RULE_VIOLATION, got {on_fail!r}")
    env: dict[str, Decimal] = {}
    try:
        for name, inp in (spec.get("inputs") or {}).items():
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", name) or name in FUNCS:
                raise CalcError(f"bad input name {name!r}")
            src = claims_by_id.get(inp.get("source"))
            lit = str(inp.get("literal", ""))
            if src is None:
                raise CalcError(f"input {name}: source claim {inp.get('source')!r} not found")
            if src.kind == ClaimKind.FACT:
                ev = evidence_by_id.get(src.claim_id)
                if ev is None or ev.state != EvidenceState.VERIFIED:
                    raise CalcError(f"input {name}: source {src.claim_id} is not VERIFIED")
                hay = src.quote
            elif src.kind == ClaimKind.USER_ATTESTED:
                hay = src.text
            else:
                raise CalcError(f"input {name}: source must be a FACT or USER_ATTESTED claim")
            if not lit or lit not in hay:
                raise CalcError(f"input {name}: literal {lit!r} not in the source claim")
            env[name] = parse_literal(lit)
        steps = spec.get("steps") or []
        if len(steps) > MAX_STEPS:
            raise CalcError("too many steps")
        for step in steps:
            name, expr = step
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", name) or name in FUNCS or name in env:
                raise CalcError(f"bad or duplicate step name {name!r}")
            env[name] = evaluate(expr, env)
    except (CalcError, TypeError, ValueError) as e:
        return CalcResult(cid, "INVALID", detail=str(e), on_fail=on_fail)

    values = {k: str(v.quantize(Decimal("0.01"))) if abs(v) < MAX_ABS else str(v) for k, v in env.items()}
    for name, claimed in (spec.get("claimed") or {}).items():
        if name not in env:
            return CalcResult(cid, "INVALID", values, detail=f"claimed value for unknown name {name!r}")
        try:
            cv = Decimal(str(claimed))
        except InvalidOperation:
            return CalcResult(cid, "INVALID", values, detail=f"claimed {name} is not a number")
        if abs(cv - env[name]) > Decimal("0.005"):
            return CalcResult(cid, "MISMATCH", values, on_fail=on_fail,
                              detail=f"{name}: proposer said {claimed}, code computed {values[name]}")
    chk = spec.get("check")
    passed = None
    if chk:
        try:
            left, op, right = chk
        except (TypeError, ValueError):
            return CalcResult(cid, "INVALID", values, detail="check must be [left, op, right]")
        if op not in OPS or left not in env or right not in env:
            return CalcResult(cid, "INVALID", values, detail=f"bad check {chk!r}")
        a, b = env[left], env[right]
        passed = {">=": a >= b, "<=": a <= b, ">": a > b, "<": a < b}[op]
    return CalcResult(cid, "OK", values, passed, on_fail,
                      detail=f"check {chk[0]} {chk[1]} {chk[2]}: {'holds' if passed else 'fails'}" if chk else "")
