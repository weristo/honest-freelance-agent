"""Proposer interface: an LLM client abstraction plus structured-output
parsing and validation. Every result carries an `llm` label: "mock" or
"<provider>:<model>". A result is never silently empty: it is either "ok",
"budget_exhausted" or "error" with a message.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional, Protocol

from ..models import Claim, KillClass, Verdict
from .budget import PRICE_SOURCE, Budget, BudgetExhausted, UnknownPrice, cost_usd, estimate_tokens
from ..normalize import norm
from .prompt import SOURCE_TEXT_LIMIT, SYSTEM_PROMPT, prompt_sha256, render_user
from .usernums import USER_NUMBER_NOT_IN_INPUT, missing_numbers

OUTSIDE_CONTEXT_MARK = "[quote outside the text given to the model] "
OUTSIDE_CONTEXT_REASON = "quote outside the text given to the model (verifier code: QUOTE_TOO_SHORT)"
OUTSIDE_CONTEXT_CATEGORY = "quote_outside_context"
# A source text passed to the model shorter than this is flagged as thin. Only
# a visible note (trace, card, eval row); it never changes the decision.
THIN_SOURCE_CHARS = 2000


def thin_note(passed_chars: int) -> Optional[str]:
    return f"source text thin ({passed_chars} chars)" if passed_chars < THIN_SOURCE_CHARS else None


def visible_reason(claim, evidence) -> tuple[str, str]:
    """(text shown to people, category for counting) for one claim's evidence.
    A claim marked by mark_outside_context() is shown with the proposer-layer
    reason, not with the frozen verifier's QUOTE_TOO_SHORT."""
    code = evidence.reason_code.value
    if claim is not None and claim.text.startswith(OUTSIDE_CONTEXT_MARK) and code == "QUOTE_TOO_SHORT":
        return OUTSIDE_CONTEXT_REASON, OUTSIDE_CONTEXT_CATEGORY
    return code, code


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int


class LLMClient(Protocol):
    provider: str
    model: str

    @property
    def label(self) -> str: ...

    def complete(self, system: str, user: str, max_tokens: int) -> LLMResponse: ...


@dataclass
class ProposerResult:
    status: str                         # "ok" | "unavailable" | "budget_exhausted" | "error"
    llm: str
    prompt_sha256: str
    decision: Optional[Verdict] = None
    kill_class: Optional[KillClass] = None
    rationale: str = ""
    claims: list[Claim] = field(default_factory=list)
    message: str = ""
    cost_usd: float = 0.0
    usage: Optional[dict] = None        # tokens + estimated USD; never a key or key id
    context: list[dict] = field(default_factory=list)       # per source: lengths, truncated?
    outside_context: list[str] = field(default_factory=list)  # claim ids quoted from cut-off text
    user_unticked: dict = field(default_factory=dict)  # claim id -> reason the user's tick was withheld


class ProposalInvalid(ValueError):
    pass


def parse_proposal(text: str) -> tuple[Verdict, Optional[KillClass], str, list[Claim]]:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("{"):]
    try:
        obj = json.loads(t[t.find("{"): t.rfind("}") + 1])
    except ValueError as e:
        raise ProposalInvalid(f"not JSON: {e}") from None
    if obj.get("decision") not in ("GO", "KILL", "NEEDS_HUMAN"):
        raise ProposalInvalid("missing/invalid 'decision'")
    kc = obj.get("kill_class")
    kill = KillClass(kc) if kc else None
    claims = []
    for i, c in enumerate(obj.get("claims") or []):
        if not isinstance(c, dict):
            raise ProposalInvalid(f"claim {i} is not an object")
        c.setdefault("claim_id", f"c_{i:02d}")
        c.setdefault("text", c.get("quote", ""))
        try:
            claim = Claim.from_dict(c)
        except (ValueError, TypeError) as e:
            raise ProposalInvalid(f"claim {i}: {e}") from None
        if claim.kind.value == "FACT" and (not claim.quote or not claim.source_url):
            raise ProposalInvalid(f"FACT claim {claim.claim_id} needs quote and source_url")
        claims.append(claim)
    return Verdict(obj["decision"]), kill, str(obj.get("rationale", "")), claims


def usage_record(model: str, input_tokens: int, output_tokens: int, llm: str) -> dict:
    """Spend of one call, from the provider's usage field. The USD figure is an
    ESTIMATE from the list price table; if the model is not priced, only
    tokens are recorded."""
    rec = {"llm": llm, "model": model, "input_tokens": input_tokens, "output_tokens": output_tokens}
    try:
        rec["cost_usd_estimate"] = round(cost_usd(model, input_tokens, output_tokens), 6)
        rec["price_source"] = (PRICE_SOURCE if not os.environ.get("HFA_PRICE_IN")
                               else "estimate from HFA_PRICE_IN/HFA_PRICE_OUT override (not the list price)")
    except UnknownPrice:
        rec["cost_usd_estimate"] = None
        rec["price_source"] = "unknown model price - tokens only"
    return rec


def mark_outside_context(claims, texts: dict[str, str]) -> list[str]:
    """A FACT quote that is NOT in the text passed to the model but IS in the
    part that was cut off can only come from somewhere other than the given
    context (memory, a guess). Such a claim is marked here, in the proposer
    layer: the quote is moved into the claim text with a visible mark and the
    quote field is emptied, so the frozen verifier returns UNVERIFIED
    (reason QUOTE_TOO_SHORT, the only existing reason that applies) and the
    gate cannot GO on it. Quotes found in neither part are left alone: the
    verifier reports them as NO_MATCH."""
    flagged = []
    for c in claims:
        full = texts.get(c.source_url)
        if c.kind.value != "FACT" or not full or not c.quote:
            continue
        q = norm(c.quote).casefold()
        passed = norm(full[:SOURCE_TEXT_LIMIT]).casefold()
        if q in passed:
            continue
        if q in norm(full).casefold():
            c.text = OUTSIDE_CONTEXT_MARK + c.quote
            c.quote = ""
            flagged.append(c.claim_id)
    return flagged


def apply_user_tick(claims, user_confirm: bool, user_input: str) -> dict:
    """Set the user's tick on USER_ATTESTED claims (the model never sets it).
    A claim with a number that is not in the user's own input stays unticked,
    whatever the user ticked: the model may have written its own number in
    the user's name. Returns {claim_id: reason} for the claims left unticked
    for that reason."""
    unticked = {}
    for c in claims:
        if c.kind.value != "USER_ATTESTED":
            continue
        c.user_confirmed = False
        if not user_confirm:
            continue
        missing = missing_numbers(c.text, user_input or "")
        if missing:
            unticked[c.claim_id] = f"{USER_NUMBER_NOT_IN_INPUT}: {', '.join(missing)}"
        else:
            c.user_confirmed = True
    return unticked


class Proposer:
    def __init__(self, client: LLMClient, budget: Optional[Budget] = None, max_tokens: int = 2000,
                 usage_log: Optional[str] = None, user_confirm: bool = False,
                 source_text: Optional[Callable[[str], str]] = None,
                 user_input: Optional[str] = None):
        """user_confirm: the USER's own tick (CLI --user-confirm / UI field) for
        the USER_ATTESTED claims of this run; never taken from the model. It
        lands only on a USER_ATTESTED claim whose every number appears in the
        user's own input (see usernums); the others stay unticked with the
        reason USER_NUMBER_NOT_IN_INPUT.
        user_input: the user's own text the numbers are checked against
        (default: the request passed to propose()).
        source_text: optional url -> text provider; the text is given to the
        model as context so it can quote verbatim (the verifier still fetches
        and checks every quote independently)."""
        self.client = client
        self.budget = budget
        self.max_tokens = max_tokens
        self.usage_log = usage_log
        self.user_confirm = user_confirm
        self.source_text = source_text
        self.user_input = user_input
        self.last_user_unticked: dict = {}
        self.last_usage: Optional[dict] = None
        self.last_context: list[dict] = []
        self.last_outside: list[str] = []

    def _log(self, rec: dict) -> None:
        if not self.usage_log:
            return
        os.makedirs(os.path.dirname(self.usage_log) or ".", exist_ok=True)
        rec = dict(rec, t=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        with open(self.usage_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def propose(self, request: str, sources: list[str]) -> ProposerResult:
        texts = {}
        if self.source_text is not None and self.client.provider not in ("mock", "unavailable"):
            for u in sources:
                try:
                    texts[u] = self.source_text(u) or ""
                except Exception as e:          # context is optional; the verifier decides
                    texts[u] = f"(could not fetch: {type(e).__name__})"
        user = render_user(request, sources, texts)
        label = self.client.label
        ph = prompt_sha256()
        context = []
        for u, t in texts.items():
            passed = min(len(t), SOURCE_TEXT_LIMIT)
            entry = {"source_url": u, "original_chars": len(t), "passed_chars": passed,
                     "truncated": len(t) > SOURCE_TEXT_LIMIT}
            note = thin_note(passed)
            if note:
                entry["thin"] = note
            context.append(entry)
        self.last_context, self.last_outside, self.last_user_unticked = context, [], {}
        if context:
            self._log({"llm": label, "event": "source_context", "limit_chars": SOURCE_TEXT_LIMIT,
                       "sources": context})
        thin = {c["source_url"]: c["thin"] for c in context if c.get("thin")}
        if thin:
            self._log({"llm": label, "event": "source_thin", "sources": thin,
                       "effect": "note only; the decision is not changed"})
        if self.client.provider == "unavailable":
            msg = f"llm: unavailable - {getattr(self.client, 'reason', '')}"
            self._log({"llm": "unavailable", "status": "unavailable", "message": msg})
            return ProposerResult("unavailable", "unavailable", ph, message=msg)
        metered = self.client.provider != "mock"
        if metered:
            if self.budget is None:
                return ProposerResult("error", label, ph, message="real LLM call without a budget is refused")
            try:
                self.budget.check(self.client.model, estimate_tokens(SYSTEM_PROMPT + user), self.max_tokens)
            except BudgetExhausted as e:
                self._log({"llm": label, "status": "budget_exhausted", "message": str(e)})
                return ProposerResult("budget_exhausted", label, ph, message=str(e))
            except UnknownPrice as e:
                self._log({"llm": label, "status": "error", "message": str(e)})
                return ProposerResult("error", label, ph, message=f"no call: {e}")
        try:
            resp = self.client.complete(SYSTEM_PROMPT, user, self.max_tokens)
        except Exception as e:  # provider errors are reported, not swallowed
            return ProposerResult("error", label, ph, message=f"{type(e).__name__}: {e}")
        cost = self.budget.record(self.client.model, resp.input_tokens, resp.output_tokens) if metered else 0.0
        usage = usage_record(self.client.model, resp.input_tokens, resp.output_tokens, label) if metered else None
        if usage:
            self.last_usage = usage
            self._log(dict(usage, status="ok"))
        try:
            verdict, kill, why, claims = parse_proposal(resp.text)
        except ProposalInvalid as e:
            return ProposerResult("error", label, ph, message=f"invalid proposal: {e}", cost_usd=cost, usage=usage)
        unticked = apply_user_tick(claims, self.user_confirm,
                                   request if self.user_input is None else self.user_input)
        self.last_user_unticked = unticked
        if unticked:
            self._log({"llm": label, "event": "user_unticked", "claims": unticked,
                       "effect": "user tick withheld -> G5 -> no GO"})
        outside = mark_outside_context(claims, texts)
        self.last_outside = outside
        if outside:
            self._log({"llm": label, "event": OUTSIDE_CONTEXT_CATEGORY, "claims": outside,
                       "reason": OUTSIDE_CONTEXT_REASON,
                       "effect": "quote removed before verification -> UNVERIFIED"})
        return ProposerResult("ok", label, ph, verdict, kill, why, claims, cost_usd=cost, usage=usage,
                              context=context, outside_context=outside, user_unticked=unticked)
