"""Deterministic decision gate (rules G1-G5) and the GO enforcement point.

The proposer (LLM) only *proposes* GO/KILL/NEEDS_HUMAN. The final verdict is
computed here from the claim list and the verifier's evidence. Any action
with side effects (e.g. sending a PayPal invoice) must call
`require_verified_go()`, which recomputes the condition from the evidence
instead of trusting the `final` field.
"""
from __future__ import annotations

import uuid
from typing import Optional

from .models import (Claim, ClaimKind, Decision, Evidence, EvidenceState,
                     KillClass, Polarity, Verdict)


class GoNotAllowed(Exception):
    """Raised when an action requires GO but the evidence does not support it."""


def _critical_facts(claims, ev):
    return [(c, ev.get(c.claim_id)) for c in claims if c.critical and c.kind == ClaimKind.FACT]


def decide(proposed: Verdict, claims: list[Claim], evidence: list[Evidence], llm: str,
           proposed_kill_class: Optional[KillClass] = None,
           decision_id: Optional[str] = None) -> Decision:
    if not llm:
        raise ValueError("decision must carry an 'llm' label ('mock' or '<provider>:<model>')")
    ev = {e.claim_id: e for e in evidence}
    log: list[str] = []
    facts = _critical_facts(claims, ev)

    def done(final: Verdict, kill: Optional[KillClass] = None) -> Decision:
        return Decision(decision_id or f"d_{uuid.uuid4().hex[:10]}", proposed, final, llm,
                        kill_class=kill, gate_log=log, human_required=(final == Verdict.NEEDS_HUMAN),
                        evidence=evidence, claims=claims)

    # G1: a verified critical fact that contradicts the request kills it.
    contra = [c for c, e in facts if c.polarity == Polarity.CONTRADICTS
              and e is not None and e.state == EvidenceState.VERIFIED]
    if contra:
        log.append(f"G1 fail: verified contradicting fact(s) {[c.claim_id for c in contra]}")
        kc = proposed_kill_class if proposed_kill_class in (KillClass.RULE_VIOLATION,
                                                           KillClass.POLICY_REFUSAL) else KillClass.RULE_VIOLATION
        return done(Verdict.KILL, kc)
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
            return done(Verdict.KILL, proposed_kill_class or KillClass.NOT_WORTH)
        return done(Verdict.NEEDS_HUMAN)
    log.append("G4 pass")

    # G5: the user's own statements need an explicit tick; never VERIFIED.
    unticked = [c.claim_id for c in claims if c.critical and c.kind == ClaimKind.USER_ATTESTED
                and not c.user_confirmed]
    if unticked:
        log.append(f"G5 fail: user-attested without tick={unticked}")
        return done(Verdict.NEEDS_HUMAN if proposed != Verdict.KILL else Verdict.KILL,
                    proposed_kill_class if proposed == Verdict.KILL else None)
    log.append("G5 pass")

    if proposed == Verdict.KILL:
        log.append("proposer KILL kept (killing needs no verification)")
        return done(Verdict.KILL, proposed_kill_class or KillClass.NOT_WORTH)
    if proposed == Verdict.NEEDS_HUMAN:
        log.append("proposer asked for a human")
        return done(Verdict.NEEDS_HUMAN)

    # G2: GO only if every critical fact is VERIFIED and there is at least one.
    if not facts:
        log.append("G2 fail: GO proposed with zero critical facts")
        return done(Verdict.NEEDS_HUMAN)
    log.append(f"G2 pass: {len(facts)} critical fact(s) VERIFIED")
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
