"""E34: a proposal whose claims have no claim_id while its CALC/INFERENCE
references use the model's own names must fail visibly in parse, not later
as a silent CALC "source claim not found". With correct c_NN ids the CALC
runs through the (frozen) calc module to OK."""
import json
import unittest

from hfa import holdout
from hfa.calc import run_calc
from hfa.models import ClaimKind, Evidence, EvidenceState, Reason
from hfa.proposer.base import ProposalInvalid, parse_proposal

FEE = "The service fee is 10% or $5.00 USD, whichever is greater, on fixed price work."
URL = "https://example.org/fees"
GATE_SHA = "9bc97c853fe50e1db154d6498e7db0d4751fcb1d72ac2299c5ea636db3be5813"


def _real_shape():
    """Structure copied from a real model answer (T1_1029, haiku): no claim_id,
    the model's own ids under "id", references to those ids. Text invented."""
    return {
        "decision": "GO", "kill_class": None, "rationale": "short",
        "claims": [
            {"id": "FACT_1", "kind": "FACT", "polarity": "SUPPORTS", "quote": FEE,
             "source_url": URL, "rule_id": "example_fee", "critical": False},
            {"id": "USER_1", "kind": "USER_ATTESTED", "polarity": "SUPPORTS",
             "text": "Job price is 40 USD"},
            {"id": "USER_2", "kind": "USER_ATTESTED", "polarity": "SUPPORTS",
             "text": "My own cost is 20 USD"},
            {"id": "CALC_1", "kind": "CALC", "polarity": "SUPPORTS", "text": "net vs cost",
             "calc": {"inputs": {"price": {"literal": "40 USD", "source": "USER_1"},
                                 "pct": {"literal": "10%", "source": "FACT_1"},
                                 "minimum": {"literal": "$5.00", "source": "FACT_1"},
                                 "cost": {"literal": "20 USD", "source": "USER_2"}},
                      "steps": [["fee", "max(price * pct, minimum)"], ["net", "price - fee"]],
                      "claimed": {"fee": 5.0, "net": 35.0},
                      "check": ["net", ">=", "cost"], "on_fail": "NOT_WORTH"}},
            {"id": "INF_1", "kind": "INFERENCE", "polarity": "SUPPORTS",
             "text": "net covers cost", "derived_from": ["FACT_1", "USER_1", "CALC_1"]},
        ]}


def _with_cnn_ids(obj):
    o = json.loads(json.dumps(obj))
    old = {}
    for i, c in enumerate(o["claims"]):
        old[c.pop("id")] = f"c_{i:02d}"
        c["claim_id"] = f"c_{i:02d}"
    for c in o["claims"]:
        for inp in (c.get("calc") or {}).get("inputs", {}).values():
            inp["source"] = old[inp["source"]]
        if "derived_from" in c:
            c["derived_from"] = [old[r] for r in c["derived_from"]]
    return o


class E34ClaimIdTests(unittest.TestCase):
    def test_a_real_shape_missing_ids_is_invalid(self):
        text = "```json\n" + json.dumps(_real_shape(), indent=2) + "\n```"
        with self.assertRaises(ProposalInvalid) as cm:
            parse_proposal(text)
        self.assertIn("claim ids missing; references unresolved", str(cm.exception))

    def test_a2_only_derived_from_unresolved_is_invalid(self):
        o = _real_shape()
        o["claims"] = [o["claims"][0], o["claims"][4]]        # FACT + INFERENCE only
        with self.assertRaises(ProposalInvalid):
            parse_proposal(json.dumps(o))

    def test_b_cnn_ids_calc_ok(self):
        _, _, _, claims = parse_proposal(json.dumps(_with_cnn_ids(_real_shape())))
        self.assertEqual([c.claim_id for c in claims], [f"c_{i:02d}" for i in range(5)])
        by_id = {c.claim_id: c for c in claims}
        ev = {"c_00": Evidence("c_00", EvidenceState.VERIFIED, Reason.EXACT_MATCH)}
        calc = next(c for c in claims if c.kind == ClaimKind.CALC)
        r = run_calc(calc, by_id, ev)
        self.assertEqual(r.status, "OK", r.detail)
        self.assertTrue(r.check_passed)
        self.assertEqual(r.values["net"], "35.00")

    def test_missing_ids_without_references_unchanged(self):
        o = _real_shape()
        o["claims"] = o["claims"][:3]                          # no CALC, no INFERENCE
        _, _, _, claims = parse_proposal(json.dumps(o))
        self.assertEqual([c.claim_id for c in claims], ["c_00", "c_01", "c_02"])

    def test_missing_ids_with_positional_references_still_parse(self):
        o = _with_cnn_ids(_real_shape())
        for c in o["claims"]:
            del c["claim_id"]                                  # model omits ids but refs match c_NN
        _, _, _, claims = parse_proposal(json.dumps(o))
        self.assertEqual(len(claims), 5)

    def test_model_ids_kept(self):
        o = _real_shape()
        for c in o["claims"]:
            c["claim_id"] = c.pop("id")
        _, _, _, claims = parse_proposal(json.dumps(o))
        self.assertEqual(claims[1].claim_id, "USER_1")

    def test_c_gate_hash_unchanged(self):
        self.assertEqual(holdout.decision_code_sha256(), GATE_SHA)
        self.assertNotIn("base.py", holdout.DECISION_MODULES)


if __name__ == "__main__":
    unittest.main()
