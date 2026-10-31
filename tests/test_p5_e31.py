"""E31: prompt p5 (polarity values), thin-source note, errors + confusion in the eval."""
import json
import os
import unittest

from hfa import evalrun
from hfa.demo import source_notes
from hfa.proposer import Proposer
from hfa.proposer import prompt as P
from hfa.proposer.base import THIN_SOURCE_CHARS, ProposalInvalid, parse_proposal, thin_note
from hfa.proposer.budget import Budget
from hfa.ui import render

from .test_proposer_budget_eval import GOOD, Clock, FakeReal, TmpDirCase

URL = "https://example.org/rules"


class P5PromptTests(unittest.TestCase):
    def test_polarity_rule_in_prompt(self):
        self.assertIn('polarity: exactly "SUPPORTS" or "CONTRADICTS", no other value. A background '
                      'fact that neither supports nor contradicts the plan: polarity "SUPPORTS" and '
                      'critical false.', P.SYSTEM_PROMPT)
        self.assertEqual(P.PROMPT_VERSION, "p8")                      # p5 rules kept in p6, p7, p8 (E34, E35, E36)
        self.assertNotEqual(P.prompt_sha256()[:12], "ffe4dedf69fa")   # p4

    def test_neutral_polarity_still_a_visible_error(self):
        bad = json.loads(json.dumps(GOOD))
        bad["claims"][0]["polarity"] = "NEUTRAL"
        with self.assertRaises(ProposalInvalid):
            parse_proposal(json.dumps(bad))


class ThinSourceTests(TmpDirCase):
    def _propose(self, text):
        log = os.path.join(self.tmp.name, f"trace_{len(text)}.jsonl")
        p = Proposer(FakeReal(), Budget(self.state, Clock()), usage_log=log, source_text=lambda u: text)
        r = p.propose("job", [URL])
        with open(log, encoding="utf-8") as f:
            events = [json.loads(line) for line in f]
        return p, r, events

    def test_thin_note_threshold(self):
        self.assertEqual(thin_note(THIN_SOURCE_CHARS - 1), f"source text thin ({THIN_SOURCE_CHARS - 1} chars)")
        self.assertIsNone(thin_note(THIN_SOURCE_CHARS))

    def test_thin_source_in_trace_and_card_decision_unchanged(self):
        p, r, events = self._propose("x" * 812)
        self.assertEqual(r.context[0]["thin"], "source text thin (812 chars)")
        thin = [e for e in events if e.get("event") == "source_thin"]
        self.assertEqual(thin[0]["sources"], {URL: "source text thin (812 chars)"})
        notes = source_notes(p.last_context)
        self.assertEqual(notes, [f"{URL}: source text thin (812 chars)"])
        self.assertIn("source text thin (812 chars)", render([{"llm": "anthropic:x", "source_notes": notes}]))
        # Only a note: the proposal is the same as with a long source text.
        _, r2, events2 = self._propose("x" * 5000)
        self.assertNotIn("thin", r2.context[0])
        self.assertFalse([e for e in events2 if e.get("event") == "source_thin"])
        self.assertEqual((r.status, r.decision, len(r.claims)), (r2.status, r2.decision, len(r2.claims)))


class ErrorsAndConfusionTests(unittest.TestCase):
    def _rows(self):
        def ok(i, gold, nv, v):
            return {"id": i, "llm": "anthropic:m", "gold": gold, "status": "ok", "no_verifier": nv, "verifier": v}
        return [ok("A", "GO", "GO", "GO"), ok("B", "KILL", "GO", "NEEDS_HUMAN"), ok("C", "KILL", "KILL", "KILL"),
                ok("D", "NEEDS_HUMAN", "GO", "NEEDS_HUMAN"),
                {"id": "E", "llm": "anthropic:m", "gold": "KILL", "status": "error", "no_verifier": None,
                 "verifier": None, "message": "invalid proposal"},
                {"id": "F", "llm": "anthropic:m", "gold": "NEEDS_HUMAN", "status": "error", "no_verifier": None,
                 "verifier": None, "message": "invalid proposal"}]

    def test_errors_counted_and_deferral_incl_errors(self):
        s = evalrun.aggregate(self._rows())
        self.assertEqual(s["n_errors"], 2)
        self.assertEqual(s["verifier"]["deferral_rate"], round(1 / 3, 3))       # B of A,B,C
        self.assertEqual(s["verifier"]["deferral_incl_errors"], 0.5)            # B + E of A,B,C,E
        t = evalrun.format_table(self._rows(), s)
        self.assertIn("errors (no decision, nothing done): 2", t)
        self.assertIn("deferral_incl_errors=0.5", t)

    def test_confusion_matrix(self):
        s = evalrun.aggregate(self._rows())
        m = s["verifier"]["confusion"]
        self.assertEqual(m["KILL"]["NEEDS_HUMAN"], 1)
        self.assertEqual(m["KILL"]["KILL"], 1)
        self.assertEqual(sum(sum(r.values()) for r in m.values()), 4)          # ok rows only
        self.assertEqual(s["no_verifier"]["confusion"]["NEEDS_HUMAN"]["GO"], 1)
        self.assertIn("confusion (rows: gold, columns: verifier)", evalrun.format_table(self._rows(), s))


if __name__ == "__main__":
    unittest.main()
