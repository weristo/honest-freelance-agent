"""One end-to-end decision: proposer -> fetch -> verify -> contradictions -> gate.
Writes an append-only, redacted JSONL trace when `trace_path` is given."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Optional

from .fetcher import Fetcher, utcnow_iso
from .gate import decide
from .models import Decision, Verdict
from .proposer.base import Proposer, ProposerResult
from .redact import redact
from .verifier import find_contradictions, verify


@dataclass
class RunResult:
    status: str               # "ok" | "budget_exhausted" | "error"
    llm: str
    prompt_sha256: str
    decision: Optional[Decision]
    message: str = ""

    def to_dict(self) -> dict:
        return {"status": self.status, "llm": self.llm, "prompt_sha256": self.prompt_sha256,
                "message": self.message,
                "decision": self.decision.to_dict() if self.decision else None}


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def write_trace(path: str, steps: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for s in steps:
            f.write(redact(json.dumps(s, ensure_ascii=False, default=str)) + "\n")


def decide_from_proposal(pr: ProposerResult, fetcher: Fetcher, request: Optional[str] = None) -> RunResult:
    if pr.status != "ok":
        return RunResult(pr.status, pr.llm, pr.prompt_sha256, None, pr.message)
    evidence = [verify(c, fetcher) for c in pr.claims]
    evidence = find_contradictions(pr.claims, evidence)
    d = decide(pr.decision or Verdict.NEEDS_HUMAN, pr.claims, evidence, pr.llm, pr.kill_class,
               request_text=request)
    return RunResult("ok", pr.llm, pr.prompt_sha256, d)


def run(request: str, sources: list[str], proposer: Proposer, fetcher: Fetcher,
        trace_path: Optional[str] = None) -> RunResult:
    pr = proposer.propose(request, sources)
    res = decide_from_proposal(pr, fetcher, request)
    if trace_path:
        t = utcnow_iso()
        steps = [{"t": t, "actor": "proposer", "llm": pr.llm, "status": pr.status,
                  "input_hash": _h([request, sources]), "prompt_sha256": pr.prompt_sha256,
                  "summary": pr.message or (pr.decision.value if pr.decision else "")}]
        if res.decision:
            for e in res.decision.evidence:
                steps.append({"t": t, "actor": "verifier", "llm": pr.llm, "claim_id": e.claim_id,
                              "state": e.state.value, "reason": e.reason_code.value,
                              "snippet_sha256": e.snippet_sha256, "raw_sha256": e.raw_sha256})
            steps.append({"t": t, "actor": "gate", "llm": pr.llm, "final": res.decision.final.value,
                          "gate_log": res.decision.gate_log})
        write_trace(trace_path, steps)
    return res
