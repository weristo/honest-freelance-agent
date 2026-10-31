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
 - the holdout (separate file, pinned SHA-256) runs only after a freeze,
   only with a real model, and only once (see hfa/holdout.py).
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
from .proposer.base import Proposer, claims_record, proposal_from_text, visible_reason
from .proposer.prompt import SOURCE_TEXT_LIMIT, SYSTEM_PROMPT, prompt_sha256, render_user
from . import holdout as holdout_mod
from .normalize import norm
from .proposer.budget import Budget, UnknownPrice, cost_usd, estimate_tokens
from .proposer.providers import MockClient, client_from_env

MOCK_BANNER = "MOCK - pipeline test, not a model measurement"
DEFAULT_SNAPSHOTS = "data/private/snapshots"
DEFAULT_CASES = "data/eval/cases_public.jsonl"
PRE_E33 = "no raw model output stored in this run (pre-E33)"
MAX_TOKENS = 2000           # the Proposer's default output cap, also used for the estimate


class MixedLLMError(ValueError):
    pass


def load_cases(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    for c in cases:
        c.setdefault("split", "holdout" if c["id"].startswith("R") else "dev")
    return cases


def dev_cases(cases: list[dict]) -> list[dict]:
    """Only dev cases are ever run from the main file; the holdout has its own
    loader with a hash lock (hfa.holdout)."""
    return [c for c in cases if c["split"] == "dev"]


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
    for j, inf in enumerate(case.get("inferences") or []):
        claims.append({"claim_id": f"{case['id']}_i{j}", "text": inf["text"], "kind": "INFERENCE",
                       "critical": True, "polarity": "SUPPORTS",
                       "derived_from": [f"{case['id']}_c{k}" for k in inf.get("derived_from", [])]})
    if case.get("user_attested") or case.get("calc"):
        claims.append({"claim_id": f"{case['id']}_u0", "text": case.get("user_attested") or case["input"],
                       "kind": "USER_ATTESTED", "critical": True, "polarity": "SUPPORTS",
                       "user_confirmed": True})
    if case.get("calc"):
        spec = json.loads(json.dumps(case["calc"]))
        for inp in spec["inputs"].values():
            src = inp["source"]
            inp["source"] = f"{case['id']}_u0" if src == "user" else f"{case['id']}_c{src}"
        claims.append({"claim_id": f"{case['id']}_k0", "text": "calculation", "kind": "CALC",
                       "critical": True, "polarity": "SUPPORTS", "calc": spec})
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


def case_sources(case: dict) -> list[str]:
    return sorted({c["url"] for c in case.get("claims") or []})


def run_case(case: dict, proposer: Proposer, fetcher) -> dict:
    pr = proposer.propose(case["input"], case_sources(case))
    return decide_row(case, pr, fetcher)


def decide_row(case: dict, pr, fetcher) -> dict:
    """One eval row from a proposer result: verifier + gate on the case's
    fetcher. Shared by the live run and `--replay` (E33)."""
    row = {"id": case["id"], "llm": pr.llm, "gold": case["gold"], "status": pr.status}
    # E33: the model's raw answer, stored for a model-free replay (data/eval/runs only, git-ignored)
    row["raw_model_output"] = pr.raw_text
    if pr.status != "ok":
        row.update(no_verifier=None, verifier=None, message=pr.message)
        return row
    proposed = pr.decision
    row["no_verifier"] = to_label(proposed, pr.kill_class, [])
    row["claims_parsed"] = claims_record(pr.claims)
    row["proposed_kill_class"] = pr.kill_class.value if pr.kill_class else None
    res = decide_from_proposal(pr, fetcher, case["input"])
    d = res.decision
    row["kill_class"] = d.kill_class.value if d.kill_class else None
    row["verifier"] = to_label(d.final, d.kill_class, d.evidence)
    claims = {c.claim_id: c for c in d.claims}
    shown = [(e, *visible_reason(claims.get(e.claim_id), e)) for e in d.evidence]
    row["reasons"] = [f"{e.claim_id}:{e.state.value}:{cat}" for e, _, cat in shown]
    row["reason_texts"] = {e.claim_id: text for e, text, _ in shown}
    if pr.user_unticked:                 # user tick withheld: a number not in the user's input
        row["user_unticked"] = dict(pr.user_unticked)
        row["reason_texts"].update(pr.user_unticked)
    row["reason_categories"] = dict(Counter(cat for e, _, cat in shown if e.state.value != "VERIFIED"))
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
    errors = [r for r in rows if r["status"] == "error"]
    out["n_errors"] = len(errors)          # no decision, nothing done (e.g. invalid proposal)
    non_go = [r for r in ok if r["gold"] != "GO"]
    go = [r for r in ok if r["gold"] == "GO"]
    decided = [r for r in ok if r["gold"] in DECIDED_GOLD]
    decided_err = [r for r in errors if r["gold"] in DECIDED_GOLD]
    for cond in ("no_verifier", "verifier"):
        fg = sum(r[cond] == "GO" for r in non_go)
        out[cond] = {
            "false_go": fg,
            "false_go_rate": round(fg / len(non_go), 3) if non_go else None,
            "false_kill_count": sum(r[cond] in ("KILL", "REFUSE") for r in go),
            "deferral_rate": round(sum(r[cond] in ("NEEDS_HUMAN", "UNVERIFIED") for r in decided)
                                   / len(decided), 3) if decided else None,
            # an error on a gold GO/KILL/REFUSE case is also "no decision": counted as a deferral here
            "deferral_incl_errors": round((sum(r[cond] in ("NEEDS_HUMAN", "UNVERIFIED") for r in decided)
                                           + len(decided_err)) / (len(decided) + len(decided_err)), 3)
                                    if decided or decided_err else None,
            "exact_label_agreement": sum(r[cond] == r["gold"] for r in ok),
            "kill_refuse_mixups": sum({r[cond], r["gold"]} == {"KILL", "REFUSE"} for r in ok),
            "labels": dict(Counter(r[cond] for r in ok)),
            "confusion": confusion(ok, cond),
        }
    out["n_non_go"], out["n_go"] = len(non_go), len(go)
    # Non-VERIFIED evidence by visible category; quote_outside_context is its
    # own category (never counted as QUOTE_TOO_SHORT).
    cats: Counter = Counter()
    for r in ok:
        cats.update(r.get("reason_categories") or {})
    out["evidence_categories"] = dict(sorted(cats.items()))
    out["n_quote_outside_context"] = sum(len(r.get("quotes_outside_context") or []) for r in ok)
    out["n_user_unticked"] = sum(len(r.get("user_unticked") or {}) for r in ok)
    out["n_rows_source_thin"] = sum(bool(r.get("source_thin")) for r in rows)
    return out


LABELS = ("GO", "KILL", "REFUSE", "NEEDS_HUMAN", "UNVERIFIED")
DECIDED_GOLD = ("GO", "KILL", "REFUSE")


def confusion(ok_rows: list[dict], cond: str = "verifier") -> dict:
    """{gold: {predicted: n}} over the ok rows (errors have no prediction)."""
    m = {g: {p: 0 for p in LABELS} for g in LABELS}
    for r in ok_rows:
        m.setdefault(r["gold"], {p: 0 for p in LABELS})
        m[r["gold"]][r[cond]] = m[r["gold"]].get(r[cond], 0) + 1
    return m


def format_confusion(m: dict, cond: str = "verifier") -> str:
    cols = [p for p in LABELS]
    lines = [f"confusion (rows: gold, columns: {cond})",
             f"{'gold':12} " + " ".join(f"{c[:11]:>11}" for c in cols) + f" {'total':>6}"]
    for g in LABELS:
        row = m.get(g, {})
        lines.append(f"{g:12} " + " ".join(f"{row.get(c, 0):>11}" for c in cols) + f" {sum(row.values()):>6}")
    return "\n".join(lines)


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
        if "deferral_incl_errors" in s:
            lines.append(f"{'':12} deferral_incl_errors={s['deferral_incl_errors']} "
                         "(an error on a gold GO/KILL/REFUSE case counted as a deferral)")
        lines.append(f"{'':12} KILL<->REFUSE mix-ups={s['kill_refuse_mixups']} (not false GO; reported separately)")
    if "n_errors" in summary:
        lines.append(f"errors (no decision, nothing done): {summary['n_errors']}")
    if summary.get("evidence_categories") is not None:
        lines.append(f"non-VERIFIED evidence by category: {summary['evidence_categories']}")
    for k in ("n_quote_outside_context", "n_user_unticked", "n_rows_source_thin"):
        if k in summary:
            lines.append(f"{k[2:]}: {summary[k]}")
    if summary["verifier"].get("confusion"):
        lines.append(format_confusion(summary["verifier"]["confusion"], "verifier"))
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="hfa eval")
    ap.add_argument("--cases", default=None,
                    help="cases file (default data/eval/cases_public.jsonl; with --replay: the one the run recorded)")
    ap.add_argument("--snapshots", default=None,
                    help="directory of full-text source snapshots (not in the repo; the default "
                         f"{DEFAULT_SNAPSHOTS} does not exist in a fresh clone - pass the eval owner's "
                         "directory, e.g. --snapshots ~/ai-collab/002-cash/prep/eval_snapshots)")
    ap.add_argument("--live", action="store_true", help="fetch sources live instead of snapshots")
    ap.add_argument("--holdout", metavar="FILE", help="holdout file (pinned hash; frozen, real model, once)")
    ap.add_argument("--frozen-prompt-sha")
    ap.add_argument("--freeze", choices=["gate", "prompt"],
                    help="FINAL: record the decision-code (gate) or prompt hash; issued by the eval owner")
    ap.add_argument("--out", default="data/eval/runs")
    ap.add_argument("--mock", action="store_true",
                    help="run with the scripted mock proposer (labelled MOCK; not a measurement)")
    ap.add_argument("--replay", metavar="RUN_JSON",
                    help="model-free replay (E33): re-run the stored raw model answers of RUN_JSON through "
                         "parse -> verifier -> gate on the same snapshots; exit 1 on any difference. "
                         "No LLM client is created or called.")
    a = ap.parse_args(argv)

    if a.replay:
        return replay_main(a.replay, a.cases, a.snapshots)
    a.cases = a.cases or DEFAULT_CASES
    a.snapshots = a.snapshots or DEFAULT_SNAPSHOTS
    if a.freeze:
        print(json.dumps(holdout_mod.freeze(a.freeze), indent=1))
        return 0
    if a.mock:
        client = None
    else:
        client = client_from_env()
        if client.provider == "unavailable":
            print(f"llm: unavailable - {client.reason}. Nothing was sent to any model. "
                  "(Use --mock for the labelled pipeline test.)", file=sys.stderr)
            return 3
    if a.holdout:
        cases = holdout_mod.load_holdout(a.holdout)
        # Never mock; needs both freezes with matching hashes; only once.
        holdout_mod.authorize_run("mock" if a.mock else client.label, a.frozen_prompt_sha)
    else:
        cases = dev_cases(load_cases(a.cases))
    if not a.live and not os.path.isdir(a.snapshots):
        print(f"snapshot dir {a.snapshots!r} not found. The snapshots are not in the repository; "
              f"pass --snapshots DIR (the eval owner keeps them in ~/ai-collab/002-cash/prep/eval_snapshots) "
              f"or use --live to fetch sources fresh", file=sys.stderr)
        return 2
    if a.holdout and not a.live:
        missing = sorted({p for c in cases for p in _snapshot_map(c, a.snapshots).values()
                          if not os.path.isfile(p)})
        if missing:
            print(f"holdout not started (no lock written): {len(missing)} snapshot file(s) missing: "
                  f"{[os.path.basename(p) for p in missing]}", file=sys.stderr)
            return 2

    rows: list[dict] = []
    budget = None if a.mock else Budget()
    usage_log = None if a.mock else os.path.join(a.out, "real", "usage.jsonl")
    if a.holdout:
        # Budget pre-check BEFORE the lock and before any call: all cases x 1.5
        # must fit both the remaining daily and the remaining total budget.
        try:
            est = estimate_cost(cases, client.model)
            pre = holdout_mod.budget_precheck(est, budget.status())
        except (holdout_mod.BudgetTooSmall, UnknownPrice) as e:
            print(str(e), file=sys.stderr)
            return 4
        holdout_mod.start_run({"llm": client.label, "n_cases": len(cases), "prompt_sha256": prompt_sha256(),
                               "decision_code_sha256": holdout_mod.decision_code_sha256(),
                               "budget_precheck": pre})
        try:
            rows, complete = _run_cases(cases, a, client, budget, usage_log)
            summary = aggregate(rows)
        except BaseException as e:
            holdout_mod.finish_run("ABORTED", error=f"{type(e).__name__}: {e}")
            raise
        summary["complete"] = complete and len(rows) == len(cases)
        summary["split"] = "holdout"
        summary["replay_inputs"] = replay_inputs(a.holdout, True, a)
        try:
            path = write_results(rows, summary, a.out, real=True)
        finally:
            holdout_mod.finish_run("COMPLETE" if summary["complete"] else "INCOMPLETE", summary)
    else:
        rows, complete = _run_cases(cases, a, client, budget, usage_log)
        summary = aggregate(rows)
        summary["complete"] = complete and len(rows) == len(cases)
        summary["split"] = "dev"
        summary["replay_inputs"] = replay_inputs(a.cases, False, a)
        path = write_results(rows, summary, a.out, real=not summary["is_mock"])
    print(format_table(rows, summary))
    if not summary["complete"]:
        print("INCOMPLETE RUN - do not report these numbers as a full eval")
    print(f"written: {path}")
    return 0


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _portable(path: str) -> str:
    """E37: no absolute local path in a run file or the holdout lock. Inside the
    repository: the repo-relative path; outside: the file or directory name only."""
    full = os.path.abspath(path)
    rel = os.path.relpath(full, REPO_ROOT)
    if rel == os.pardir or rel.startswith(os.pardir + os.sep):
        return os.path.basename(full.rstrip(os.sep))
    return rel.replace(os.sep, "/")


def replay_inputs(cases_path: str, holdout: bool, a) -> dict:
    """What `--replay` needs to rebuild the same inputs (E33): portable names
    plus the cases file hash (E37). A file outside the repository is given to
    the replay with --cases / --snapshots."""
    return {"cases": _portable(cases_path), "cases_sha256": holdout_mod.file_sha256(cases_path),
            "holdout": holdout, "live": bool(a.live),
            "snapshots": None if a.live else _portable(a.snapshots)}


def _recorded_path(p: Optional[str]) -> Optional[str]:
    """A path from replay_inputs: absolute (runs before E37) as is, relative
    resolved against the repository; None if it does not exist."""
    if not p:
        return None
    full = p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)
    return full if os.path.exists(full) else None


# --- model-free replay (E33) ---------------------------------------------------

REPLAY_FIELDS = ("status", "no_verifier", "verifier", "proposed_kill_class", "kill_class",
                 "claims_parsed", "reasons", "gate_log", "user_unticked", "quotes_outside_context")


class ReplayError(ValueError):
    pass


def replay_rows(run: dict, cases_path: Optional[str] = None,
                snapshots: Optional[str] = None) -> tuple[list[dict], list[str], list[str]]:
    """Re-run the stored raw answers of a run through parse -> verifier ->
    gate on the same snapshots. Returns (replayed rows, differences, notes).
    Uses no Proposer and no LLM client: proposal_from_text() is the same
    model-free code the live Proposer runs after the call."""
    rows = run.get("rows") or []
    if not any("raw_model_output" in r for r in rows):
        raise ReplayError(PRE_E33)
    inputs = (run.get("summary") or {}).get("replay_inputs") or {}
    if inputs.get("live"):
        raise ReplayError("this run fetched sources live; there is no snapshot to replay against")
    cases_path = cases_path or _recorded_path(inputs.get("cases"))
    snapshots = snapshots or _recorded_path(inputs.get("snapshots"))
    if not cases_path or not snapshots:
        raise ReplayError(f"cases file {inputs.get('cases')!r} / snapshot dir {inputs.get('snapshots')!r} "
                          "not found from the run record; pass --cases and --snapshots")
    if not os.path.isdir(snapshots):
        raise ReplayError(f"snapshot dir {snapshots!r} not found")
    want = inputs.get("cases_sha256")
    if want and holdout_mod.file_sha256(cases_path) != want:
        raise ReplayError(f"cases file {cases_path!r} does not match the run's cases_sha256 {want[:12]}...")
    cases = holdout_mod.load_holdout(cases_path) if inputs.get("holdout") else load_cases(cases_path)
    by_id = {c["id"]: c for c in cases}
    out, diffs, notes = [], [], []
    for r in rows:
        case = by_id.get(r["id"])
        if case is None:
            diffs.append(f"{r['id']}: case not in {cases_path}")
            continue
        if case["gold"] != r.get("gold"):
            diffs.append(f"{r['id']}: gold {r.get('gold')!r} in the run, {case['gold']!r} in the cases file")
        raw = r.get("raw_model_output")
        if raw is None:
            notes.append(f"{r['id']}: no raw output (status {r.get('status')}); not replayed")
            continue
        fetcher = FixtureFetcher(_snapshot_map(case, snapshots))
        # The live run gave source text to a real model only (never to the mock).
        texts = {} if r.get("llm") == "mock" else _replay_texts(fetcher, case_sources(case))
        pr = proposal_from_text(raw, r["llm"], case_user_tick(case), case_user_input(case), texts)
        new = decide_row(case, pr, fetcher)
        if r.get("llm") != "mock":
            new["quotes_outside_context"] = pr.outside_context
        for k in REPLAY_FIELDS:
            if k not in r and k not in new:
                continue
            if r.get(k) != new.get(k):
                diffs.append(f"{r['id']}: {k}: stored {r.get(k)!r} != replay {new.get(k)!r}")
        out.append(new)
    return out, diffs, notes


def _replay_texts(fetcher, sources: list[str]) -> dict[str, str]:
    get = _text_of(fetcher)
    texts = {}
    for u in sources:                       # same fallback as Proposer.propose
        try:
            texts[u] = get(u) or ""
        except Exception as e:
            texts[u] = f"(could not fetch: {type(e).__name__})"
    return texts


def replay_main(path: str, cases_path: Optional[str], snapshots: Optional[str]) -> int:
    try:
        with open(path, encoding="utf-8") as f:
            run = json.load(f)
        rows, diffs, notes = replay_rows(run, cases_path, snapshots)
    except (OSError, ValueError, ReplayError) as e:
        print(f"replay: {e}", file=sys.stderr)
        return 2
    for n in notes:
        print(f"note: {n}")
    if diffs:
        print(f"REPLAY MISMATCH: {len(diffs)} difference(s) in {path}")
        for d in diffs:
            print(f"  - {d}")
        return 1
    print(f"replay OK: {len(rows)} row(s) replayed, decisions identical "
          f"(llm label {(run.get('summary') or {}).get('llm')}; no model was called)")
    return 0


def _run_cases(cases, a, client, budget, usage_log) -> tuple[list[dict], bool]:
    rows: list[dict] = []
    for case in cases:
        fetcher = LiveFetcher() if a.live else FixtureFetcher(_snapshot_map(case, a.snapshots))
        if a.mock:
            proposer = Proposer(MockClient(lambda s, u, c=case: json.dumps(naive_proposal(c))),
                                user_confirm=case_user_tick(case), user_input=case_user_input(case))
        else:
            proposer = Proposer(client, budget, max_tokens=MAX_TOKENS, usage_log=usage_log,
                                user_confirm=case_user_tick(case), user_input=case_user_input(case),
                                source_text=_text_of(fetcher))
        row = run_case(case, proposer, fetcher)
        if not a.mock:
            row["usage"] = proposer.last_usage
            row["source_context"] = proposer.last_context
            row["quotes_outside_context"] = proposer.last_outside
            thin = {c["source_url"]: c["thin"] for c in proposer.last_context or [] if c.get("thin")}
            if thin:
                row["source_thin"] = thin
            proposer.last_usage = None
        rows.append(row)
        if row["status"] == "budget_exhausted":
            print(f"budget_exhausted at {case['id']}: stopping; no further model calls", file=sys.stderr)
            return rows, False
    return rows, True


def estimate_cost(cases: list[dict], model: str, max_tokens: int = MAX_TOKENS) -> float:
    """Upper-bound USD estimate for one call per case: the prompt with every
    source at the full SOURCE_TEXT_LIMIT characters (the most the model is
    given), the conservative ~3 chars/token count of the budget module, and
    the whole max_tokens as output. Same price table as the Budget."""
    total = 0.0
    for c in cases:
        sources = sorted({cl["url"] for cl in c.get("claims") or []})
        user = render_user(c["input"], sources, {u: "x" * SOURCE_TEXT_LIMIT for u in sources})
        total += cost_usd(model, estimate_tokens(SYSTEM_PROMPT + user), max_tokens)
    return total


def case_user_input(case: dict) -> str:
    """The user's own words in an eval case: the request text plus the
    case's recorded user statement. USER_ATTESTED numbers are checked
    against this text only (never against the sources or the model)."""
    return " ".join(t for t in (case.get("input"), case.get("user_attested")) if t)


def case_user_tick(case: dict) -> bool:
    """The case text is the user's own input, so its stated facts count as the
    user's tick. The model's output never sets it (Claim.from_dict resets it)."""
    return bool(case.get("user_attested") or case.get("calc"))


def _text_of(fetcher):
    def get(url: str) -> str:
        doc = fetcher.fetch(url)
        return norm(doc.visible + " " + doc.embedded)
    return get


if __name__ == "__main__":
    raise SystemExit(main())
