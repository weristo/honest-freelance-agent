"""E36: the calc contract (as hfa/calc.py reads it) is part of OUTPUT_SCHEMA and so
of the system prompt (p8), plus one sentence: both sides of check are names.
Last structural change to the prompt; calc.py and the gate are unchanged."""
import json
import unittest

from hfa import calc as C
from hfa import holdout
from hfa.calc import run_calc
from hfa.models import Evidence, EvidenceState, Reason
from hfa.proposer import prompt as P
from hfa.proposer.base import parse_proposal

P8_SHA = "de5c1191515cee9c3904b7f3bdfd23b202ed19ad157ecc4781070ad990f8c7fc"
GATE_SHA = "9bc97c853fe50e1db154d6498e7db0d4751fcb1d72ac2299c5ea636db3be5813"
SENTENCE = "Both sides of check are names defined in inputs or steps, never numbers."
FEE = "A service fee of 10% applies to every payment, with a minimum of $5.00 per payment."
URL = "https://example.org/fees"


def _calc_schema():
    return P.OUTPUT_SCHEMA["properties"]["claims"]["items"]["properties"]["calc"]


def _conforming(calc):
    """Hand check of a calc object against the E36 subschema (no jsonschema dependency)."""
    s = _calc_schema()
    if set(calc) - set(s["properties"]) or not set(s["required"]) <= set(calc):
        return False
    for v in calc["inputs"].values():
        if set(v) != {"literal", "source"} or not all(isinstance(x, str) for x in v.values()):
            return False
    if not all(isinstance(st, list) and len(st) == 2 and all(isinstance(x, str) for x in st)
               for st in calc["steps"]):
        return False
    if not all(isinstance(x, (int, float, str)) for x in calc.get("claimed", {}).values()):
        return False
    chk = calc["check"]
    if not (isinstance(chk, list) and len(chk) == 3 and isinstance(chk[0], str) and isinstance(chk[2], str)
            and chk[1] in s["properties"]["check"]["prefixItems"][1]["enum"]):
        return False
    return calc["on_fail"] in s["properties"]["on_fail"]["enum"]


def _answer(check):
    return {
        "decision": "GO", "kill_class": None, "rationale": "short",
        "claims": [
            {"claim_id": "c_00", "text": "fee rule", "kind": "FACT", "critical": True,
             "polarity": "SUPPORTS", "quote": FEE, "source_url": URL, "rule_id": "example_fee"},
            {"claim_id": "c_01", "text": "Job price is 40 USD", "kind": "USER_ATTESTED",
             "critical": True, "polarity": "SUPPORTS"},
            {"claim_id": "c_02", "text": "My own cost is 20 USD", "kind": "USER_ATTESTED",
             "critical": True, "polarity": "SUPPORTS"},
            {"claim_id": "c_03", "text": "net vs cost", "kind": "CALC", "critical": True,
             "polarity": "SUPPORTS",
             "calc": {"inputs": {"price": {"literal": "40 USD", "source": "c_01"},
                                 "pct": {"literal": "10%", "source": "c_00"},
                                 "minimum": {"literal": "$5.00", "source": "c_00"},
                                 "cost": {"literal": "20 USD", "source": "c_02"}},
                      "steps": [["fee", "max(price * pct, minimum)"], ["net", "price - fee"]],
                      "claimed": {"fee": 5.0, "net": 35.0},
                      "check": check, "on_fail": "NOT_WORTH"}},
        ]}


def _run(check):
    _, _, _, claims = parse_proposal(json.dumps(_answer(check)))
    by_id = {c.claim_id: c for c in claims}
    ev = {"c_00": Evidence("c_00", EvidenceState.VERIFIED, Reason.EXACT_MATCH)}
    return run_calc(by_id["c_03"], by_id, ev)


class E36CalcSchemaTests(unittest.TestCase):
    def test_a_subschema_matches_calc_py(self):
        s = _calc_schema()
        self.assertEqual(s["type"], "object")
        self.assertEqual(set(s["required"]), {"inputs", "steps", "check", "on_fail"})
        self.assertEqual(set(s["properties"]), {"inputs", "steps", "claimed", "check", "on_fail"})
        inp = s["properties"]["inputs"]["additionalProperties"]
        self.assertEqual(set(inp["required"]), {"literal", "source"})
        self.assertEqual(tuple(s["properties"]["check"]["prefixItems"][1]["enum"]), C.OPS)
        self.assertEqual((s["properties"]["check"]["minItems"], s["properties"]["check"]["maxItems"]), (3, 3))
        self.assertEqual(set(s["properties"]["on_fail"]["enum"]), {"NOT_WORTH", "RULE_VIOLATION"})

    def test_b_schema_and_sentence_reach_the_model(self):
        self.assertIn(SENTENCE, P.SYSTEM_PROMPT)
        self.assertEqual(P.SYSTEM_PROMPT.count(SENTENCE), 1)
        self.assertTrue(P.SYSTEM_PROMPT.endswith(
            json.dumps(P.OUTPUT_SCHEMA, sort_keys=True, ensure_ascii=False)))
        self.assertIn('"prefixItems"', P.SYSTEM_PROMPT)

    def test_c_conforming_calc_is_ok_through_frozen_calc(self):
        a = _answer(["net", ">=", "cost"])
        self.assertTrue(_conforming(a["claims"][3]["calc"]))
        r = _run(["net", ">=", "cost"])
        self.assertEqual(r.status, "OK", r.detail)
        self.assertTrue(r.check_passed)

    def test_d_number_in_check_still_invalid(self):
        # the p7 'bad check' shape: a literal on the right; calc.py is unchanged and still refuses it
        r = _run(["net", ">=", "5.00"])
        self.assertEqual(r.status, "INVALID")
        self.assertIn("bad check", r.detail)

    def test_e_claimed_kept_mismatch_still_reported(self):
        a = _answer(["net", ">=", "cost"])
        a["claims"][3]["calc"]["claimed"]["fee"] = 4.5
        _, _, _, claims = parse_proposal(json.dumps(a))
        by_id = {c.claim_id: c for c in claims}
        ev = {"c_00": Evidence("c_00", EvidenceState.VERIFIED, Reason.EXACT_MATCH)}
        r = run_calc(by_id["c_03"], by_id, ev)
        self.assertEqual(r.status, "MISMATCH")

    def test_f_p8_hash_and_gate_unchanged(self):
        self.assertEqual(P.PROMPT_VERSION, "p8")
        self.assertEqual(P.prompt_sha256(), P8_SHA)
        self.assertEqual(holdout.decision_code_sha256(), GATE_SHA)


if __name__ == "__main__":
    unittest.main()
