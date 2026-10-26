"""Core data types: Claim (proposed by an LLM), Evidence (produced by the
deterministic verifier) and Decision (produced by the deterministic gate).

The LLM never assigns an evidence state or a final decision. Those are
computed in code (see verifier.py and gate.py).
"""
from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

VERIFIER_VERSION = "0.1.0"


class ClaimKind(str, enum.Enum):
    FACT = "FACT"                    # quotable, checked by the verifier
    INFERENCE = "INFERENCE"          # reasoning; never VERIFIED
    USER_ATTESTED = "USER_ATTESTED"  # the user's own statement; never VERIFIED
    CALC = "CALC"                    # arithmetic over other claims, done by code


class Polarity(str, enum.Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"


class EvidenceState(str, enum.Enum):
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    NEEDS_HUMAN = "NEEDS_HUMAN"
    NOT_APPLICABLE = "NOT_APPLICABLE"  # INFERENCE / USER_ATTESTED claims


class Reason(str, enum.Enum):
    EXACT_MATCH = "EXACT_MATCH"
    MULTI_CONSISTENT = "MULTI_CONSISTENT"
    NO_MATCH = "NO_MATCH"
    QUOTE_TOO_SHORT = "QUOTE_TOO_SHORT"
    FETCH_FAILED = "FETCH_FAILED"
    FETCH_BLOCKED = "FETCH_BLOCKED"
    URL_REFUSED = "URL_REFUSED"
    SOFT_404 = "SOFT_404"
    THIN_CONTENT = "THIN_CONTENT"
    STALE_SOURCE = "STALE_SOURCE"
    CONTRADICTED = "CONTRADICTED"
    AMBIGUOUS_CONTEXT = "AMBIGUOUS_CONTEXT"
    MULTI_MATCH = "MULTI_MATCH"
    REDIRECTED = "REDIRECTED"
    SOURCE_CHANGED = "SOURCE_CHANGED"
    NOT_A_FACT = "NOT_A_FACT"


class Verdict(str, enum.Enum):
    GO = "GO"
    KILL = "KILL"
    NEEDS_HUMAN = "NEEDS_HUMAN"


class KillClass(str, enum.Enum):
    NOT_WORTH = "NOT_WORTH"
    RULE_VIOLATION = "RULE_VIOLATION"
    POLICY_REFUSAL = "POLICY_REFUSAL"


@dataclass
class Claim:
    claim_id: str
    text: str
    kind: ClaimKind = ClaimKind.FACT
    critical: bool = True
    quote: str = ""
    source_url: str = ""
    rule_id: str = "AD-HOC"
    rule_topic: Optional[str] = None
    polarity: Polarity = Polarity.SUPPORTS
    derived_from: list[str] = field(default_factory=list)
    source_domain_type: str = "general"   # "legal" sources get a staleness check
    first_seen_snippet_sha256: Optional[str] = None
    user_confirmed: bool = False          # USER_ATTESTED: explicit user tick
    calc: Optional[dict] = None           # CALC: inputs / steps / claimed / check

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Claim":
        """Parse a claim proposed by a model. `user_confirmed` is ALWAYS reset
        to False here, whatever the model sent: the tick on a USER_ATTESTED
        claim can only come from the user's own input (CLI flag / UI field),
        applied afterwards by the caller."""
        d = dict(d)
        d["user_confirmed"] = False
        d["kind"] = ClaimKind(d.get("kind", "FACT"))
        d["polarity"] = Polarity(d.get("polarity", "SUPPORTS"))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Evidence:
    claim_id: str
    state: EvidenceState
    reason_code: Reason
    match_layer: Optional[str] = None
    match_offset: Optional[int] = None
    context_snippet: Optional[str] = None
    guard_hits: list[str] = field(default_factory=list)
    other_snippets: list[str] = field(default_factory=list)
    raw_sha256: Optional[str] = None
    snippet_sha256: Optional[str] = None
    fetched_at: Optional[str] = None
    http_status: Optional[int] = None
    final_url: Optional[str] = None
    source_date: Optional[str] = None
    date_flag: Optional[str] = None      # policy sources: "DATED" | "UNDATED"
    date_origins: list[str] = field(default_factory=list)
    nearest_span: Optional[str] = None   # display only; never raises the state
    detail: str = ""
    verifier_version: str = VERIFIER_VERSION

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        d["reason_code"] = self.reason_code.value
        return d


@dataclass
class Decision:
    decision_id: str
    proposed: Verdict
    final: Verdict
    llm: str                                  # "mock" or "<provider>:<model>"
    kill_class: Optional[KillClass] = None
    gate_log: list[str] = field(default_factory=list)
    human_required: bool = False
    evidence: list[Evidence] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    calcs: list[dict] = field(default_factory=list)
    request_text: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "calcs": list(self.calcs),
            "llm": self.llm,
            "proposed": self.proposed.value,
            "final": self.final.value,
            "kill_class": self.kill_class.value if self.kill_class else None,
            "gate_log": list(self.gate_log),
            "human_required": self.human_required,
            "evidence": [e.to_dict() for e in self.evidence],
        }
