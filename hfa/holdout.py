"""Holdout set: loader, hash lock, freeze and run-once lock.

Rules (enforced here, see EVAL_CHANGELOG.md):
 - the holdout file must hash to one of the pinned SHA-256 values below
   (private full file, or the public hash+<=200-char version in this repo);
 - a case whose claimed quote is known not to be in the source
   (claimed_quote_not_in_source) must have gold UNVERIFIED;
 - the freeze has two steps, issued by the eval owner (002), not by code:
     `python -m hfa eval --freeze gate`   -> data/eval/FREEZE_GATE.json
                                             (hash of the decision code)
     `python -m hfa eval --freeze prompt` -> data/eval/FREEZE_PROMPT.json
                                             (hash of the proposer prompt)
   each freeze is final (the file is never overwritten);
 - the holdout runs only when both freezes exist and both hashes still
   match, only with a real model (never mock), and only once: BEFORE the
   first model call the run creates data/eval/HOLDOUT_RUN.json atomically
   (open 'x') with state STARTED; at the end it is filled in (COMPLETE or
   INCOMPLETE, or ABORTED if the run raised). A lock in ANY state - also a
   STARTED lock left behind by a crash - refuses every later attempt;
 - nothing unlocks automatically. Only the eval owner (002) can decide to
   void a lock, by hand (README, "Voiding a holdout lock"): rename
   HOLDOUT_RUN.json to HOLDOUT_RUN.void-<UTC>.json and put the decision and
   the reason in HOLDOUT_RUN.void-<UTC>.reason.txt; a voided lock without a
   reason file blocks the run as well;
 - before the started lock is written, a budget pre-check must pass: the
   estimated cost of all cases x 1.5 must be below the remaining daily AND
   the remaining total budget, otherwise the run does not start.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Optional

from .proposer.prompt import prompt_sha256

PINNED_HOLDOUT_SHA256 = {
    "private": "0b7ae93043edd4693926f8af99b9819d36d6755e5e7b8cf385719fb577394994",
    "public": "886c9636b1915e2e5d81de6278542e86b70816fbc631082bb90be29506f989d3",
}
FREEZE_PATHS = {"gate": os.path.join("data", "eval", "FREEZE_GATE.json"),
                "prompt": os.path.join("data", "eval", "FREEZE_PROMPT.json")}
RUN_LOCK_PATH = os.path.join("data", "eval", "HOLDOUT_RUN.json")
BUDGET_SAFETY_FACTOR = 1.5
UNLOCK_HELP = ("only the eval owner (002) can void it, by hand: rename the lock to "
               "HOLDOUT_RUN.void-<UTC>.json and write the decision + reason into "
               "HOLDOUT_RUN.void-<UTC>.reason.txt (README: 'Voiding a holdout lock')")

# Source files whose behaviour decides the verdict. Their combined hash is frozen.
DECISION_MODULES = ("normalize.py", "extract.py", "fetcher.py", "verifier.py",
                    "registry.py", "calc.py", "dates.py", "gate.py", "pipeline.py")


class HoldoutLocked(PermissionError):
    pass


class HoldoutHashMismatch(ValueError):
    pass


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def decision_code_sha256() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for name in DECISION_MODULES:
        with open(os.path.join(here, name), "rb") as f:
            h.update(name.encode() + b"\0" + f.read() + b"\0")
    return h.hexdigest()


def _bool(v) -> bool:
    return v if isinstance(v, bool) else str(v).strip().lower() == "true"


def load_holdout(path: str) -> list[dict]:
    digest = file_sha256(path)
    if digest not in PINNED_HOLDOUT_SHA256.values():
        raise HoldoutHashMismatch(f"holdout file hash {digest[:12]}... is not pinned; refusing")
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            c["split"] = "holdout"
            c["naive_false_go_trap"] = _bool(c.get("naive_false_go_trap", False))
            for cl in c.get("claims") or []:
                cl["claimed_quote_not_in_source"] = _bool(cl.get("claimed_quote_not_in_source", False))
            if any(cl["claimed_quote_not_in_source"] for cl in c.get("claims") or []) \
                    and c["gold"] != "UNVERIFIED":
                raise ValueError(f"{c['id']}: claimed quote not in source, gold must be UNVERIFIED")
            cases.append(c)
    return cases


def fingerprint() -> dict:
    return {"prompt_sha256": prompt_sha256(), "decision_code_sha256": decision_code_sha256()}


def freeze(kind: str, paths: Optional[dict] = None, now: Optional[datetime] = None) -> dict:
    """Step 'gate' freezes the decision code, step 'prompt' the proposer prompt."""
    paths = paths or FREEZE_PATHS
    if kind not in ("gate", "prompt"):
        raise ValueError("freeze kind must be 'gate' or 'prompt'")
    path = paths[kind]
    if os.path.exists(path):
        raise HoldoutLocked(f"{path} already exists; a freeze is final")
    key = "decision_code_sha256" if kind == "gate" else "prompt_sha256"
    rec = {"kind": kind, key: fingerprint()[key],
           "frozen_at": (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    return rec


def authorize_run(llm: str, frozen_prompt_sha: Optional[str],
                  freeze_paths: Optional[dict] = None, lock_path: str = RUN_LOCK_PATH) -> dict:
    paths = freeze_paths or FREEZE_PATHS
    if llm == "mock":
        raise HoldoutLocked("the holdout is never run with the mock proposer")
    check_lock(lock_path)
    missing = [k for k in ("gate", "prompt") if not os.path.exists(paths[k])]
    if missing:
        raise HoldoutLocked(f"not frozen yet: {missing} (python -m hfa eval --freeze gate|prompt)")
    with open(paths["gate"], encoding="utf-8") as f:
        gate = json.load(f)
    with open(paths["prompt"], encoding="utf-8") as f:
        prompt = json.load(f)
    now = fingerprint()
    if frozen_prompt_sha != prompt["prompt_sha256"] or now["prompt_sha256"] != prompt["prompt_sha256"]:
        raise HoldoutLocked("prompt hash differs from the frozen one")
    if now["decision_code_sha256"] != gate["decision_code_sha256"]:
        raise HoldoutLocked("verifier/gate code changed since the freeze")
    return {"gate": gate, "prompt": prompt}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def check_lock(lock_path: str = RUN_LOCK_PATH) -> None:
    """Refuse if the holdout lock exists in any state, or if a voided lock
    has no reason file. Never removes or renames anything."""
    if os.path.exists(lock_path):
        try:
            with open(lock_path, encoding="utf-8") as f:
                st = json.load(f).get("state", "COMPLETE (old format)")
        except (OSError, ValueError):
            st = "unreadable"
        if st == "STARTED":
            raise HoldoutLocked(
                f"holdout run was STARTED and never finished ({lock_path}; crash or kill?). "
                f"No re-run: {UNLOCK_HELP}")
        raise HoldoutLocked(f"holdout already run once ({lock_path}, state {st}); no re-runs. {UNLOCK_HELP}")
    base, _ = os.path.splitext(lock_path)
    for v in sorted(glob.glob(glob.escape(base) + ".void-*.json")):
        reason = v[:-len(".json")] + ".reason.txt"
        text = ""
        if os.path.exists(reason):
            with open(reason, encoding="utf-8") as f:
                text = f.read().strip()
        if not text:
            raise HoldoutLocked(f"voided lock {v} has no (or an empty) reason file {reason}; "
                                "the eval owner (002) must record the decision and the reason")


def start_run(info: dict, lock_path: str = RUN_LOCK_PATH) -> dict:
    """Write the STARTED lock before the first model call. Atomic: open 'x'
    fails if any lock exists (also one created by a parallel run)."""
    check_lock(lock_path)
    rec = {"state": "STARTED", "started_at": _now(), **info}
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    try:
        with open(lock_path, "x", encoding="utf-8") as f:
            json.dump(rec, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
    except FileExistsError:
        raise HoldoutLocked(f"holdout lock {lock_path} appeared meanwhile; no re-runs") from None
    return rec


def finish_run(state: str, summary: Optional[dict] = None, error: Optional[str] = None,
               lock_path: str = RUN_LOCK_PATH) -> dict:
    """Fill in the STARTED lock (COMPLETE / INCOMPLETE / ABORTED). The lock
    stays; it is replaced atomically, never deleted."""
    with open(lock_path, encoding="utf-8") as f:
        rec = json.load(f)
    if rec.get("state") != "STARTED":
        raise HoldoutLocked(f"lock {lock_path} is {rec.get('state')}, not STARTED; not overwriting")
    rec.update(state=state, finished_at=_now(), ran_at=_now())
    if summary is not None:
        rec["summary"] = summary
    if error is not None:
        rec["error"] = error
    d = os.path.dirname(lock_path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".holdout_lock.", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, lock_path)
    return rec


def record_run(summary: dict, lock_path: str = RUN_LOCK_PATH) -> None:
    """One-step lock (kept for callers/tests that record a finished run)."""
    start_run({}, lock_path)
    finish_run("COMPLETE" if summary.get("complete", True) else "INCOMPLETE", summary, lock_path=lock_path)


class BudgetTooSmall(PermissionError):
    pass


def budget_precheck(estimated_usd: float, status, factor: float = BUDGET_SAFETY_FACTOR) -> dict:
    """estimated_usd x factor must be below the remaining daily AND total budget."""
    need = estimated_usd * factor
    left_day = status.daily_cap - status.spent_today
    left_total = status.total_cap - status.spent_total
    rec = {"estimated_usd": round(estimated_usd, 6), "factor": factor, "needed_usd": round(need, 6),
           "remaining_daily_usd": round(left_day, 6), "remaining_total_usd": round(left_total, 6)}
    if not (need < left_day and need < left_total):
        raise BudgetTooSmall(
            f"holdout not started: estimated ${estimated_usd:.4f} x {factor} = ${need:.4f} is not below "
            f"the remaining daily ${left_day:.4f} and total ${left_total:.4f} budget "
            "(no lock written, no model call made)")
    return rec
