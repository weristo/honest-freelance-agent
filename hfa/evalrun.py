"""Mini-eval: the same proposals with and without the verifier + gate.

Conditions
  no_verifier : final = what the proposer said
  verifier    : final = gate(proposer, verifier evidence)

Main metric: false-GO rate = #(pred GO and gold != GO) / #(gold != GO).
Reported next to it: false-KILL count and deferral rate (the price of
honesty), so "always NEEDS_HUMAN" can not look like a win.

Honesty rules enforced in code:
 - every row carries `llm` ("mock" or "<provider>:<model>");
 - aggregation refuses a run that mixes mock and real rows;
 - mock results are written only to the mock table and printed with a
   "MOCK - pipeline test, not a model measurement" banner; the real-results
   writer refuses any mock row;
 - holdout cases (split "holdout", ids R*) run only with --holdout and a
   --frozen-prompt-sha equal to the current proposer prompt hash.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Iterable, Optional

from .fetcher import FixtureFetcher, LiveFetcher
from .models import EvidenceState, KillClass, Verdict
from .pipeline import decide_from_proposal
from .proposer.base import Proposer
from .proposer.prompt import prompt_sha256
from .proposer.providers import MockClient

MOCK_BANNER = "MOCK - pipeline test, not a model measurement"


class MixedLLMError(ValueError):
    pass


class HoldoutLocked(PermissionError):
    pass


def load_cases(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    for c in cases:
        c.setdefault("split", "holdout" if c["id"].startswith("R") else "dev")
    return cases


def select_cases(cases: list[dict], holdout: bool, frozen_prompt_sha: Optional[str]) -> list[dict]:
    if not holdout:
        return [c for c in cases if c["split"] == "dev"]
    if not frozen_prompt_sha or frozen_prompt_sha != prompt_sha256():
        raise HoldoutLocked("holdout needs --frozen-prompt-sha equal to the current prompt hash "
                            "(freeze the prompt and gate first)")
    return [c for c in cases if c["split"] == "holdout"]


# --- naive mock proposer -----------------------------------------------------

HYPOTHETICAL_QUOTE = "Hypothetical rule text recalled from model memory (mock)"


def naive_proposal(case: dict) -> dict:
    """Stand-in for a model that trusts itself: proposes GO on every case
    flagged as a false-GO trap, otherwise the gold-consistent verdict. Its
    claims are the case's recorded quotes. This exercises the pipeline; it
    measures nothing about any real model."""
    gold = case["gold"]
    if case.get("naive_false_go_trap"):
        decision, kill = "GO", None
    elif gold in ("GO",):
        decision, kill = "GO", None
    elif gold == "REFUSE":
        decision, kill = "KILL", "POLICY_REFUSAL"
    elif gold == "KILL":
        decision, kill = "KILL", "NOT_WORTH"
    else:
        decision, kill = "NEEDS_HUMAN", None
    legal = case.get("domain", "").startswith("tax")
    claims = []
    for i, cl in enumerate(case.get("claims") or []):
        role = cl.get("role", "CONTEXT")
        claims.append({
            "claim_id": f"{case['id']}_c{i}", "text": cl["quote"], "kind": "FACT",
            "critical": role != "CONTEXT", "quote": cl["quote"], "source_url": cl["url"],
            "rule_id": "EVAL", "rule_topic": case["id"] if role != "CONTEXT" else None,
            "polarity": "CONTRADICTS" if role == "CONTRADICTS" else "SUPPORTS",
            "source_domain_type": "legal" if legal else "general",
        })
    if not claims and case.get("source"):
        src = case["source"]
        claims.append({"claim_id": f"{case['id']}_h0", "text": "recalled rule", "kind": "FACT",
                       "critical": True, "quote": HYPOTHETICAL_QUOTE, "source_url": src["url"],
                       "rule_id": "EVAL", "polarity": "SUPPORTS"})
    if case.get("user_attested"):
        claims.append({"claim_id": f"{case['id']}_u0", "text": case["user_attested"],
                       "kind": "USER_ATTESTED", "critical": True, "polarity": "SUPPORTS",
                       "user_confirmed": True})
    return {"decision": decision, "kill_class": kill, "rationale": "mock", "claims": claims}


def _snapshot_map(case: dict, snapshot_dir: str) -> dict[str, str]:
    m: dict[str, str] = {}
    refs = list(case.get("claims") or []) + [case[k] for k in ("source", "source2") if case.get(k)]
    for r in refs:
        snap = r.get("snapshot")
        if snap and r.get("url"):
            m.setdefault(r["url"], os.path.join(snapshot_dir, snap))
    return m


def to_label(final: Verdict, kill_class, evidence) -> str:
    if final == Verdict.GO:
        return "GO"
    if final == Verdict.KILL:
        return "REFUSE" if kill_class == KillClass.POLICY_REFUSAL else "KILL"
    if any(e.state == EvidenceState.UNVERIFIED for e in evidence):
        return "UNVERIFIED"
    return "NEEDS_HUMAN"


def run_case(case: dict, proposer: Proposer, fetcher) -> dict:
    pr = proposer.propose(case["input"], sorted({c["url"] for c in case.get("claims") or []}))
    row = {"id": case["id"], "llm": pr.llm, "gold": case["gold"], "status": pr.status}
    if pr.status != "ok":
        row.update(no_verifier=None, verifier=None, message=pr.message)
        return row
    proposed = pr.decision
    row["no_verifier"] = to_label(proposed, pr.kill_class, [])
    res = decide_from_proposal(pr, fetcher)
    d = res.decision
    row["verifier"] = to_label(d.final, d.kill_class, d.evidence)
    row["reasons"] = [f"{e.claim_id}:{e.state.value}:{e.reason_code.value}" for e in d.evidence]
    row["gate_log"] = d.gate_log
    return row


def aggregate(rows: list[dict]) -> dict:
    labels = {r["llm"] for r in rows}
    if len(labels) != 1:
        raise MixedLLMError(f"refusing to aggregate rows from different LLMs: {sorted(labels)}")
    llm = labels.pop()
    if any(r["llm"] == "mock" for r in rows) and llm != "mock":
        raise MixedLLMError("mock rows in a real run")
    ok = [r for r in rows if r["status"] == "ok"]
    out = {"llm": llm, "is_mock": llm == "mock", "n_cases": len(rows), "n_ok": len(ok),
           "n_budget_exhausted": sum(r["status"] == "budget_exhausted" for r in rows),
           "prompt_sha256": prompt_sha256()}
    non_go = [r for r in ok if r["gold"] != "GO"]
    go = [r for r in ok if r["gold"] == "GO"]
    decided = [r for r in ok if r["gold"] in ("GO", "KILL", "REFUSE")]
    for cond in ("no_verifier", "verifier"):
        fg = sum(r[cond] == "GO" for r in non_go)
        out[cond] = {
            "false_go": fg,
            "false_go_rate": round(fg / len(non_go), 3) if non_go else None,
            "false_kill_count": sum(r[cond] in ("KILL", "REFUSE") for r in go),
            "deferral_rate": round(sum(r[cond] in ("NEEDS_HUMAN", "UNVERIFIED") for r in decided)
                                   / len(decided), 3) if decided else None,
            "exact_label_agreement": sum(r[cond] == r["gold"] for r in ok),
            "labels": dict(Counter(r[cond] for r in ok)),
        }
    out["n_non_go"], out["n_go"] = len(non_go), len(go)
    return out


def write_results(rows: list[dict], summary: dict, out_dir: str, real: bool) -> str:
    if real and any(r["llm"] == "mock" for r in rows):
        raise MixedLLMError("the real-results table never accepts mock rows")
    if not real and any(r["llm"] != "mock" for r in rows):
        raise MixedLLMError("the mock table only takes mock rows")
    sub = "real" if real else "mock"
    os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    path = os.path.join(out_dir, sub, f"eval_{summary['llm'].replace(':', '_')}.json")
    payload = {"banner": None if real else MOCK_BANNER, "summary": summary, "rows": rows}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    return path


def format_table(rows: Iterable[dict], summary: dict) -> str:
    lines = []
    if summary["is_mock"]:
        lines.append(f"### {MOCK_BANNER} ###")
    lines.append(f"llm={summary['llm']}  cases={summary['n_cases']}  prompt={summary['prompt_sha256'][:12]}")
    lines.append(f"{'id':5} {'llm':6} {'gold':12} {'no_verifier':12} {'verifier':12}")
    for r in rows:
        lines.append(f"{r['id']:5} {r['llm'][:6]:6} {r['gold']:12} {str(r['no_verifier']):12} {str(r['verifier']):12}")
    for cond in ("no_verifier", "verifier"):
        s = summary[cond]
        lines.append(f"{cond:12} false_GO={s['false_go']}/{summary['n_non_go']} "
                     f"rate={s['false_go_rate']} false_KILL={s['false_kill_count']}/{summary['n_go']} "
                     f"deferral={s['deferral_rate']} exact={s['exact_label_agreement']}/{summary['n_ok']}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="hfa eval")
    ap.add_argument("--cases", default="data/eval/cases_public.jsonl")
    ap.add_argument("--snapshots", default="data/private/snapshots",
                    help="directory of full-text source snapshots (not in the repo)")
    ap.add_argument("--live", action="store_true", help="fetch sources live instead of snapshots")
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--frozen-prompt-sha")
    ap.add_argument("--out", default="data/eval/runs")
    a = ap.parse_args(argv)

    cases = select_cases(load_cases(a.cases), a.holdout, a.frozen_prompt_sha)
    if not a.live and not os.path.isdir(a.snapshots):
        print(f"snapshot dir {a.snapshots!r} not found; use --live to fetch sources fresh", file=sys.stderr)
        return 2
    # This round: mock proposer only (no real LLM calls from the eval yet).
    rows = []
    for case in cases:
        client = MockClient(lambda s, u, c=case: json.dumps(naive_proposal(c)))
        fetcher = LiveFetcher() if a.live else FixtureFetcher(_snapshot_map(case, a.snapshots))
        rows.append(run_case(case, Proposer(client), fetcher))
    summary = aggregate(rows)
    path = write_results(rows, summary, a.out, real=not summary["is_mock"])
    print(format_table(rows, summary))
    print(f"written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
