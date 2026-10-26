"""Deterministic decision gate (rules C0, G1-G6) and the GO enforcement point.

Order: C0 (CALC claims: invalid or mismatching arithmetic -> NEEDS_HUMAN; a
failed check -> KILL with the CALC's kill class, before G1 and without
consulting the source registry), then G1-G5, then G6 (a GO that rests on the
user's own amounts or thresholds needs a passing CALC).

The proposer (LLM) only *proposes* GO/KILL/NEEDS_HUMAN. The final verdict is
computed here from the claim list and the verifier's evidence. Any action
with side effects (e.g. sending a PayPal invoice) must call
`require_verified_go()`, which recomputes the condition from the evidence
instead of trusting the `final` field.
"""
from __future__ import annotations

import uuid
from typing import Optional

from .calc import has_money, has_number, run_calc
from .registry import is_policy_source, policy_reason
from .models import (Claim, ClaimKind, Decision, Evidence, EvidenceState,
                     KillClass, Polarity, Verdict)


class GoNotAllowed(Exception):
    """Raised when an action requires GO but the evidence does not support it."""


def _critical_facts(claims, ev):
    return [(c, ev.get(c.claim_id)) for c in claims if c.critical and c.kind == ClaimKind.FACT]


def _calcs(claims, ev):
    by_id = {c.claim_id: c for c in claims}
    return [run_calc(c, by_id, ev) for c in claims if c.critical and c.kind == ClaimKind.CALC]


def _amount_based(claims, request_text: Optional[str]) -> bool:
    """G6 trigger: the decision rests on the USER's numbers - a USER_ATTESTED
    claim with a numeric value, or a money amount in the request itself.
    A percentage inside a source quote alone does not trigger it."""
    if has_money(request_text or ""):
        return True
    return any(c.critical and c.kind == ClaimKind.USER_ATTESTED and has_number(c.text) for c in claims)


def _no_llm_refusal(kc: Optional[KillClass]) -> Optional[KillClass]:
    """POLICY_REFUSAL is only granted by G1 with a registry source."""
    return KillClass.RULE_VIOLATION if kc == KillClass.POLICY_REFUSAL else kc


def decide(proposed: Verdict, claims: list[Claim], evidence: list[Evidence], llm: str,
           proposed_kill_class: Optional[KillClass] = None,
           decision_id: Optional[str] = None, request_text: Optional[str] = None) -> Decision:
    if not llm:
        raise ValueError("decision must carry an 'llm' label ('mock' or '<provider>:<model>')")
    ev = {e.claim_id: e for e in evidence}
    log: list[str] = []
    facts = _critical_facts(claims, ev)

    calcs = _calcs(claims, ev)

    def done(final: Verdict, kill: Optional[KillClass] = None) -> Decision:
        return Decision(decision_id or f"d_{uuid.uuid4().hex[:10]}", proposed, final, llm,
                        kill_class=kill, gate_log=log, human_required=(final == Verdict.NEEDS_HUMAN),
                        evidence=evidence, claims=claims, calcs=[r.__dict__ for r in calcs],
                        request_text=request_text)

    # C0: arithmetic is done by code. Bad or disputed sums go to a human; a
    # failed check kills the plan whatever the proposer said, and the source
    # registry is not consulted (a fee table is an economic rule, not a ban).
    bad = [r for r in calcs if r.status == "INVALID"]
    if bad:
        log.append(f"C0 fail: CALC_INVALID {[(r.claim_id, r.detail) for r in bad]}")
        return done(Verdict.NEEDS_HUMAN)
    mism = [r for r in calcs if r.status == "MISMATCH"]
    if mism:
        log.append(f"C0 fail: CALC_MISMATCH {[(r.claim_id, r.detail) for r in mism]}")
        return done(Verdict.NEEDS_HUMAN)
    failed = [r for r in calcs if r.check_passed is False]
    if failed:
        log.append(f"C0 fail: check failed {[(r.claim_id, r.detail, r.values) for r in failed]}")
        return done(Verdict.KILL, KillClass(failed[0].on_fail))
    if calcs:
        log.append(f"C0 pass: {[(r.claim_id, r.detail) for r in calcs]}")

    # G1: a verified critical fact that contradicts the request kills it.
    contra = [c for c, e in facts if c.polarity == Polarity.CONTRADICTS
              and e is not None and e.state == EvidenceState.VERIFIED]
    if contra:
        log.append(f"G1 fail: verified contradicting fact(s) {[c.claim_id for c in contra]}")
        # REFUSE vs KILL comes from the source registry, never from the LLM.
        policy = [c for c in contra if is_policy_source(c.source_url)]
        for c in contra:
            log.append(f"G1 source {c.claim_id}: {policy_reason(c.source_url)}")
        return done(Verdict.KILL, KillClass.POLICY_REFUSAL if policy else KillClass.RULE_VIOLATION)
    log.append("G1 pass")

    # G3: anything needing a human, or a decision that rests on an inference.
    needs = [c.claim_id for c, e in facts if e is not None and e.state == EvidenceState.NEEDS_HUMAN]
    infer = [c.claim_id for c in claims if c.critical and c.kind == ClaimKind.INFERENCE]
    if needs or infer:
        log.append(f"G3 fail: needs_human={needs} inference_keys={infer}")
        return done(Verdict.NEEDS_HUMAN)
    log.append("G3 pass")

    # G4: an unverified critical fact forbids GO.
    unver = [c.claim_id for c, e in facts if e is None or e.state != EvidenceState.VERIFIED]
    if unver:
        log.append(f"G4 fail: unverified={unver}")
        if proposed == Verdict.KILL:
            return done(Verdict.KILL, _no_llm_refusal(proposed_kill_class) or KillClass.NOT_WORTH)
        return done(Verdict.NEEDS_HUMAN)
    log.append("G4 pass")

    # G5: the user's own statements need an explicit tick; never VERIFIED.
    unticked = [c.claim_id for c in claims if c.critical and c.kind == ClaimKind.USER_ATTESTED
                and not c.user_confirmed]
    if unticked:
        log.append(f"G5 fail: user-attested without tick={unticked}")
        return done(Verdict.NEEDS_HUMAN if proposed != Verdict.KILL else Verdict.KILL,
                    _no_llm_refusal(proposed_kill_class) if proposed == Verdict.KILL else None)
    log.append("G5 pass")

    if proposed == Verdict.KILL:
        log.append("proposer KILL kept (killing needs no verification)")
        return done(Verdict.KILL, _no_llm_refusal(proposed_kill_class) or KillClass.NOT_WORTH)
    if proposed == Verdict.NEEDS_HUMAN:
        log.append("proposer asked for a human")
        return done(Verdict.NEEDS_HUMAN)

    # G2: GO only if every critical fact is VERIFIED and there is at least one.
    if not facts:
        log.append("G2 fail: GO proposed with zero critical facts")
        return done(Verdict.NEEDS_HUMAN)
    log.append(f"G2 pass: {len(facts)} critical fact(s) VERIFIED")

    # G6: a GO that rests on the user's amounts/thresholds needs a passing CALC.
    if _amount_based(claims, request_text) and not any(r.check_passed for r in calcs):
        log.append("G6 fail: GO depends on the user's amount/threshold but no passing CALC")
        return done(Verdict.NEEDS_HUMAN)
    return done(Verdict.GO)


def require_verified_go(decision: Decision) -> None:
    """Enforcement point for side-effecting actions. Recomputes from evidence."""
    if decision.final != Verdict.GO:
        raise GoNotAllowed(f"decision {decision.decision_id} is {decision.final.value}, not GO")
    ev = {e.claim_id: e for e in decision.evidence}
    facts = _critical_facts(decision.claims, ev)
    if not facts:
        raise GoNotAllowed("GO without any critical fact is not allowed")
    bad = [c.claim_id for c, e in facts if e is None or e.state != EvidenceState.VERIFIED]
    if bad:
        raise GoNotAllowed(f"critical facts not VERIFIED: {bad}")
    if any(c.critical and c.kind == ClaimKind.INFERENCE for c in decision.claims):
        raise GoNotAllowed("GO rests on an inference")
    if any(c.critical and c.kind == ClaimKind.USER_ATTESTED and not c.user_confirmed
           for c in decision.claims):
        raise GoNotAllowed("user-attested fact without explicit confirmation")
    calcs = _calcs(decision.claims, ev)
    if any(r.status != "OK" or r.check_passed is False for r in calcs):
        raise GoNotAllowed("a CALC claim is invalid, disputed or fails its check")
    if _amount_based(decision.claims, decision.request_text) and not any(r.check_passed for r in calcs):
        raise GoNotAllowed("GO depends on an amount but has no passing CALC")
