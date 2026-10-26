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
   match, only with a real model (never mock), and only once: the first run
   writes data/eval/HOLDOUT_RUN.json and every later attempt is refused.
"""
from __future__ import annotations

import hashlib
import json
import os
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

# Source files whose behaviour decides the verdict. Their combined hash is frozen.
DECISION_MODULES = ("normalize.py", "extract.py", "fetcher.py", "verifier.py",
                    "registry.py", "calc.py", "gate.py", "pipeline.py")


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
    if os.path.exists(lock_path):
        raise HoldoutLocked(f"holdout already run once ({lock_path}); no re-runs")
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


def record_run(summary: dict, lock_path: str = RUN_LOCK_PATH) -> None:
    with open(lock_path, "x", encoding="utf-8") as f:   # 'x': fails if it exists
        json.dump({"ran_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "summary": summary}, f, indent=1)
