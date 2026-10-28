"""E32: parse_proposal accepts a raw control character inside a JSON string
(json.loads strict=False); every other parse error stays visible."""
import json
import unittest

from hfa.proposer.base import ProposalInvalid, parse_proposal

from .test_proposer_budget_eval import GOOD


def _raw_newline_text(obj, field="rationale", value="line one\nline two"):
    o = json.loads(json.dumps(obj))
    o[field] = "@@PLACEHOLDER@@"
    # json.dumps would escape the newline; put a RAW one inside the string.
    return json.dumps(o).replace("@@PLACEHOLDER@@", value)


class E32ParseTests(unittest.TestCase):
    def test_raw_newline_in_string_parses(self):
        text = _raw_newline_text(GOOD)
        self.assertIn("\n", text)
        with self.assertRaises(ValueError):
            json.loads(text)                     # the old strict parse failed (T06)
        v, k, why, claims = parse_proposal(text)
        self.assertEqual(v.value, "GO")
        self.assertEqual(why, "line one\nline two")
        self.assertEqual(len(claims), 1)

    def test_raw_tab_in_claim_text_parses(self):
        o = json.loads(json.dumps(GOOD))
        o["claims"][0]["text"] = "@@T@@"
        text = json.dumps(o).replace("@@T@@", "a\tb")
        _, _, _, claims = parse_proposal(text)
        self.assertEqual(claims[0].text, "a\tb")

    def test_bad_schema_still_error(self):
        bad = json.loads(json.dumps(GOOD))
        bad["decision"] = "MAYBE"
        with self.assertRaises(ProposalInvalid):
            parse_proposal(_raw_newline_text(bad))
        bad = json.loads(json.dumps(GOOD))
        del bad["decision"]
        with self.assertRaises(ProposalInvalid):
            parse_proposal(_raw_newline_text(bad))

    def test_bad_enum_still_error(self):
        bad = json.loads(json.dumps(GOOD))
        bad["claims"][0]["polarity"] = "NEUTRAL"
        with self.assertRaises(ProposalInvalid):
            parse_proposal(_raw_newline_text(bad))
        bad = json.loads(json.dumps(GOOD))
        bad["claims"][0]["kind"] = "OPINION"
        with self.assertRaises(ProposalInvalid):
            parse_proposal(_raw_newline_text(bad))

    def test_not_json_still_error(self):
        with self.assertRaises(ProposalInvalid):
            parse_proposal('{"decision": "GO", "claims": [}')
        with self.assertRaises(ProposalInvalid):
            parse_proposal("I think GO")


if __name__ == "__main__":
    unittest.main()
