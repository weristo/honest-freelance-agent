"""Proposer interface: an LLM client abstraction plus structured-output
parsing and validation. Every result carries an `llm` label: "mock" or
"<provider>:<model>". A result is never silently empty: it is either "ok",
"budget_exhausted" or "error" with a message.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional, Protocol

from ..models import Claim, KillClass, Verdict
from .budget import Budget, BudgetExhausted, estimate_tokens
from .prompt import SYSTEM_PROMPT, prompt_sha256, render_user


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
    status: str                         # "ok" | "budget_exhausted" | "error"
    llm: str
    prompt_sha256: str
    decision: Optional[Verdict] = None
    kill_class: Optional[KillClass] = None
    rationale: str = ""
    claims: list[Claim] = field(default_factory=list)
    message: str = ""
    cost_usd: float = 0.0


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


class Proposer:
    def __init__(self, client: LLMClient, budget: Optional[Budget] = None, max_tokens: int = 2000):
        self.client = client
        self.budget = budget
        self.max_tokens = max_tokens

    def propose(self, request: str, sources: list[str]) -> ProposerResult:
        user = render_user(request, sources)
        label = self.client.label
        ph = prompt_sha256()
        metered = self.client.provider != "mock"
        if metered:
            if self.budget is None:
                return ProposerResult("error", label, ph, message="real LLM call without a budget is refused")
            try:
                self.budget.check(self.client.model, estimate_tokens(SYSTEM_PROMPT + user), self.max_tokens)
            except BudgetExhausted as e:
                return ProposerResult("budget_exhausted", label, ph, message=str(e))
        try:
            resp = self.client.complete(SYSTEM_PROMPT, user, self.max_tokens)
        except Exception as e:  # provider errors are reported, not swallowed
            return ProposerResult("error", label, ph, message=f"{type(e).__name__}: {e}")
        cost = self.budget.record(self.client.model, resp.input_tokens, resp.output_tokens) if metered else 0.0
        try:
            verdict, kill, why, claims = parse_proposal(resp.text)
        except ProposalInvalid as e:
            return ProposerResult("error", label, ph, message=f"invalid proposal: {e}", cost_usd=cost)
        return ProposerResult("ok", label, ph, verdict, kill, why, claims, cost_usd=cost)
