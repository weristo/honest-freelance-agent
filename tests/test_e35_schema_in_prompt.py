"""E35: the OUTPUT_SCHEMA reaches the model as part of the system prompt (p7);
before p7 it was only part of the prompt hash."""
import json
import unittest

from hfa import holdout
from hfa.proposer import prompt as P

INTRO = "The JSON object must match this JSON Schema exactly (key names included):"
P7_SHA = "1569d5ed153d6f463e95bbb14d770a16d342badc1971aafb21106a260c8e19ec"
GATE_SHA = "9bc97c853fe50e1db154d6498e7db0d4751fcb1d72ac2299c5ea636db3be5813"


def _keys(node, out):
    if isinstance(node, dict):
        for k, v in (node.get("properties") or {}).items():
            out.add(k)
            _keys(v, out)
        _keys(node.get("items"), out)
    return out


class E35SchemaInPromptTests(unittest.TestCase):
    def test_a_intro_and_schema_in_system_prompt(self):
        self.assertIn(INTRO, P.SYSTEM_PROMPT)
        tail = P.SYSTEM_PROMPT.split(INTRO, 1)[1].strip()
        self.assertEqual(json.loads(tail), P.OUTPUT_SCHEMA)       # the full schema, nothing after it
        keys = _keys(P.OUTPUT_SCHEMA, set())
        self.assertTrue({"decision", "kill_class", "rationale", "claims", "claim_id", "text", "kind",
                         "critical", "quote", "source_url", "rule_id", "rule_topic", "polarity",
                         "derived_from", "calc"} <= keys)
        for k in keys:
            self.assertIn(f'"{k}"', tail)
        req = P.OUTPUT_SCHEMA["properties"]["claims"]["items"]["required"]
        self.assertIn("claim_id", req)
        self.assertIn("text", req)

    def test_a2_schema_serialized_like_the_hash(self):
        self.assertTrue(P.SYSTEM_PROMPT.endswith(
            json.dumps(P.OUTPUT_SCHEMA, sort_keys=True, ensure_ascii=False)))

    def test_b_p7_superseded(self):
        # E36: p8 keeps the E35 schema-in-prompt rule; the p8 hash is pinned in test_e36
        self.assertEqual(P.PROMPT_VERSION, "p8")
        self.assertNotEqual(P.prompt_sha256(), P7_SHA)

    def test_c_gate_hash_unchanged(self):
        self.assertEqual(holdout.decision_code_sha256(), GATE_SHA)


if __name__ == "__main__":
    unittest.main()
